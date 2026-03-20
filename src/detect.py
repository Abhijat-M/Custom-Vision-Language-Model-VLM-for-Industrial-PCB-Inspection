"""
Inference script — run defect detection on images with visualization.
Outputs annotated images with bounding boxes, class labels, scores, and severity.
"""

import argparse
import logging
import os
import time
from typing import Dict, List

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

from src.config import PATHS, INFER, SEVERITY, IDX_TO_CLASS, NUM_CLASSES
from src.model import build_model, load_checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Color palette for each class (BGR for OpenCV)
CLASS_COLORS = {
    "open": (0, 0, 255),       # Red
    "short": (0, 165, 255),    # Orange
    "mousebite": (0, 255, 255),  # Yellow
    "spur": (255, 0, 255),     # Magenta
    "copper": (255, 0, 0),     # Blue
    "pin-hole": (0, 255, 0),   # Green
}


def classify_severity(box: List[float], img_w: int, img_h: int) -> str:
    """Classify defect severity based on area ratio."""
    area = (box[2] - box[0]) * (box[3] - box[1])
    ratio = area / (img_w * img_h)
    if ratio >= SEVERITY.med_threshold:
        return "HIGH"
    elif ratio >= SEVERITY.low_threshold:
        return "MEDIUM"
    return "LOW"


def detect_single(
    model: torch.nn.Module,
    image: Image.Image,
    device: torch.device,
    score_threshold: float = INFER.score_threshold,
) -> Dict:
    """Run detection on a single PIL image."""
    to_tensor = T.ToTensor()
    img_tensor = to_tensor(image).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model([img_tensor])[0]

    # Filter by score
    keep = outputs["scores"] >= score_threshold
    boxes = outputs["boxes"][keep].cpu().numpy()
    labels = outputs["labels"][keep].cpu().numpy()
    scores = outputs["scores"][keep].cpu().numpy()

    w, h = image.size
    detections = []
    for box, label, score in zip(boxes, labels, scores):
        cls_name = IDX_TO_CLASS.get(int(label), "unknown")
        severity = classify_severity(box.tolist(), w, h)
        detections.append({
            "class": cls_name,
            "confidence": float(score),
            "bbox": box.tolist(),
            "severity": severity,
        })

    return {"detections": detections, "image_size": [w, h]}


def draw_detections(image_np: np.ndarray, result: Dict) -> np.ndarray:
    """Draw bounding boxes and labels on the image."""
    annotated = image_np.copy()

    for det in result["detections"]:
        cls = det["class"]
        score = det["confidence"]
        severity = det["severity"]
        x1, y1, x2, y2 = [int(c) for c in det["bbox"]]
        color = CLASS_COLORS.get(cls, (255, 255, 255))

        # Box
        thickness = 3 if severity == "HIGH" else 2
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        # Label background
        label = f"{cls} {score:.2f} [{severity}]"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            annotated, label, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        )

    return annotated


def main():
    parser = argparse.ArgumentParser(description="Detect PCB defects")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input", type=str, required=True, help="Image file or directory")
    parser.add_argument("--output-dir", type=str, default=PATHS.results_dir)
    parser.add_argument("--score-threshold", type=float, default=INFER.score_threshold)
    parser.add_argument("--save-json", action="store_true", help="Save detections as JSON")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = build_model()
    load_checkpoint(model, args.checkpoint, device)
    model.to(device)
    model.eval()
    logger.info(f"Model loaded from {args.checkpoint}")

    # Collect input images
    if os.path.isdir(args.input):
        img_paths = [
            os.path.join(args.input, f) for f in sorted(os.listdir(args.input))
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
    else:
        img_paths = [args.input]

    os.makedirs(args.output_dir, exist_ok=True)
    all_results = []
    total_time = 0.0

    for img_path in img_paths:
        image = Image.open(img_path).convert("RGB")

        t0 = time.time()
        result = detect_single(model, image, device, args.score_threshold)
        elapsed = time.time() - t0
        total_time += elapsed

        n_defects = len(result["detections"])
        logger.info(f"{os.path.basename(img_path)}: {n_defects} defects ({elapsed:.3f}s)")

        # Annotate and save
        img_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        annotated = draw_detections(img_np, result)
        out_name = "det_" + os.path.basename(img_path)
        cv2.imwrite(os.path.join(args.output_dir, out_name), annotated)

        result["file"] = os.path.basename(img_path)
        result["inference_time_s"] = elapsed
        all_results.append(result)

    if args.save_json:
        import json
        json_path = os.path.join(args.output_dir, "detections.json")
        with open(json_path, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"JSON results saved to {json_path}")

    fps = len(img_paths) / total_time if total_time > 0 else 0
    logger.info(f"Done. {len(img_paths)} images | {fps:.1f} FPS | Avg {total_time/len(img_paths):.3f}s/img")


if __name__ == "__main__":
    main()
