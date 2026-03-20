"""
VLM inference pipeline for natural language PCB inspection queries.
Combines Faster R-CNN detections with Qwen2-VL language understanding.
Includes confidence calibration and hallucination controls.
"""

import json
import logging
import os
import re
import time
from typing import Dict, List, Optional

import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel

from src.config import CLASSES, IDX_TO_CLASS, INFER
from src.model import build_model, load_checkpoint
from src.detect import detect_single

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"


class PCBInspector:
    """
    Production-grade PCB inspection system combining:
    1. Faster R-CNN for precise defect localization
    2. Qwen2-VL for natural language understanding and response generation
    3. Confidence calibration and hallucination controls
    """

    def __init__(
        self,
        detector_checkpoint: str,
        vlm_adapter_path: Optional[str] = None,
        device: Optional[torch.device] = None,
        score_threshold: float = INFER.score_threshold,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.score_threshold = score_threshold

        # Load Faster R-CNN detector
        logger.info("Loading Faster R-CNN detector...")
        self.detector = build_model()
        load_checkpoint(self.detector, detector_checkpoint, self.device)
        self.detector.to(self.device)
        self.detector.eval()

        # Load Qwen2-VL
        logger.info("Loading Qwen2-VL...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        self.vlm = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

        # Load LoRA adapter if available
        if vlm_adapter_path and os.path.isdir(vlm_adapter_path):
            logger.info(f"Loading LoRA adapter from {vlm_adapter_path}")
            self.vlm = PeftModel.from_pretrained(self.vlm, vlm_adapter_path)

        self.processor = AutoProcessor.from_pretrained(
            vlm_adapter_path or MODEL_ID, trust_remote_code=True
        )
        self.vlm.eval()
        logger.info("Models loaded successfully.")

    def _get_detector_context(self, image: Image.Image) -> Dict:
        """Run Faster R-CNN and format results as context."""
        result = detect_single(self.detector, image, self.device, self.score_threshold)
        return result

    def _build_grounded_prompt(self, question: str, detections: Dict) -> str:
        """
        Build a grounded prompt that includes detector results.
        This is the key hallucination control — the VLM must respond
        consistently with the detector's findings.
        """
        det_list = detections["detections"]

        if not det_list:
            context = "The Faster R-CNN detector found NO defects in this image."
        else:
            lines = []
            for i, d in enumerate(det_list, 1):
                bbox = d["bbox"]
                lines.append(
                    f"  {i}. {d['class']} (confidence: {d['confidence']:.2f}, "
                    f"severity: {d['severity']}, "
                    f"bbox: [{bbox[0]:.0f},{bbox[1]:.0f},{bbox[2]:.0f},{bbox[3]:.0f}])"
                )
            context = (
                f"The Faster R-CNN detector found {len(det_list)} defects:\n"
                + "\n".join(lines)
            )

        system_prompt = (
            "You are a PCB quality inspection assistant. You analyze circuit board images "
            "for manufacturing defects. You MUST base your answers on the detector results "
            "provided below. Do NOT hallucinate defects that the detector did not find. "
            "If the detector found no defects, confirm the board passes inspection.\n\n"
            f"DETECTOR RESULTS:\n{context}\n\n"
            "The valid defect types are: open, short, mousebite, spur, copper, pin-hole.\n"
            "Respond concisely and accurately."
        )

        return system_prompt

    @torch.no_grad()
    def query(
        self,
        image: Image.Image,
        question: str,
        max_new_tokens: int = 256,
        temperature: float = 0.1,
    ) -> Dict:
        """
        Query the PCB inspector with a natural language question.

        Returns:
            Dict with 'answer', 'detections', 'confidence', 'inference_time_s'
        """
        t0 = time.time()

        # Step 1: Run detector
        detections = self._get_detector_context(image)

        # Step 2: Build grounded prompt
        system_prompt = self._build_grounded_prompt(question, detections)

        # Step 3: Format for Qwen2-VL
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            },
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text], images=[image], return_tensors="pt"
        ).to(self.device)

        # Step 4: Generate response
        output_ids = self.vlm.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            top_p=0.9,
        )

        # Decode only the generated tokens
        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        answer = self.processor.decode(generated_ids, skip_special_tokens=True).strip()

        # Step 5: Confidence calibration
        confidence = self._calibrate_confidence(answer, detections)

        elapsed = time.time() - t0

        return {
            "answer": answer,
            "detections": detections["detections"],
            "confidence": confidence,
            "inference_time_s": elapsed,
        }

    def _calibrate_confidence(self, answer: str, detections: Dict) -> float:
        """
        Calibrate response confidence by checking consistency
        between the VLM answer and detector results.

        Returns confidence score 0.0 - 1.0.
        """
        det_classes = set(d["class"] for d in detections["detections"])
        answer_lower = answer.lower()

        # Check if answer mentions classes not found by detector (hallucination)
        mentioned_classes = set()
        for cls in CLASSES:
            if cls in answer_lower:
                mentioned_classes.add(cls)

        hallucinated = mentioned_classes - det_classes
        missed = det_classes - mentioned_classes

        # Base confidence from detector scores
        if detections["detections"]:
            avg_det_conf = sum(d["confidence"] for d in detections["detections"]) / len(
                detections["detections"]
            )
        else:
            avg_det_conf = 1.0  # High confidence in "no defects"

        # Penalize hallucinations heavily
        hallucination_penalty = len(hallucinated) * 0.2
        # Minor penalty for missed detections
        miss_penalty = len(missed) * 0.05

        confidence = max(0.0, min(1.0, avg_det_conf - hallucination_penalty - miss_penalty))

        if hallucinated:
            logger.warning(
                f"Potential hallucination: VLM mentioned {hallucinated} "
                f"but detector only found {det_classes}"
            )

        return round(confidence, 3)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="PCB VLM Inspector")
    parser.add_argument("--detector-checkpoint", type=str, required=True)
    parser.add_argument("--vlm-adapter", type=str, default=None)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--question", type=str, default="What defects are present in this PCB image?")
    parser.add_argument("--score-threshold", type=float, default=0.5)
    args = parser.parse_args()

    inspector = PCBInspector(
        detector_checkpoint=args.detector_checkpoint,
        vlm_adapter_path=args.vlm_adapter,
        score_threshold=args.score_threshold,
    )

    image = Image.open(args.image).convert("RGB")
    result = inspector.query(image, args.question)

    print("\n" + "=" * 60)
    print("PCB INSPECTION RESULT")
    print("=" * 60)
    print(f"Question: {args.question}")
    print(f"Answer: {result['answer']}")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"Detections: {len(result['detections'])}")
    print(f"Inference time: {result['inference_time_s']:.2f}s")
    for d in result["detections"]:
        print(f"  - {d['class']} ({d['confidence']:.2f}) [{d['severity']}]")
    print("=" * 60)


if __name__ == "__main__":
    main()
