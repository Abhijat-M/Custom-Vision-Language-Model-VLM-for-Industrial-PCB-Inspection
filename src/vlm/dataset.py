"""
VLM Dataset for Qwen2-VL fine-tuning.
Loads synthetic QA pairs and formats them for the Qwen2-VL chat template.
"""

import json
from typing import Dict, List

import torch
from PIL import Image
from torch.utils.data import Dataset


class PCBVLMDataset(Dataset):
    """
    Dataset for VLM fine-tuning with Qwen2-VL.

    Each sample is a (image, question, answer) triplet formatted
    as a Qwen2-VL chat conversation.
    """

    def __init__(self, qa_json_path: str, processor=None, max_length: int = 1024):
        with open(qa_json_path, "r") as f:
            self.data = json.load(f)
        self.processor = processor
        self.max_length = max_length
        # Resize images to reduce vision token count (fits in 8GB VRAM)
        self.image_size = (224, 224)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]

        image = Image.open(item["image"]).convert("RGB")
        # Resize to reduce token count — Qwen2-VL generates many tokens for large images
        image = image.resize(self.image_size, Image.BILINEAR)

        question = item["question"]
        answer = item["answer"]

        # Format as Qwen2-VL chat messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": answer},
                ],
            },
        ]

        if self.processor is not None:
            # Use processor's chat template to format
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            inputs = self.processor(
                text=[text],
                images=[image],
                padding="max_length",
                max_length=self.max_length,
                truncation=False,
                return_tensors="pt",
            )
            # Squeeze batch dim
            inputs = {k: v.squeeze(0) for k, v in inputs.items()}

            # Truncate if exceeds max_length (without breaking image tokens)
            seq_len = inputs["input_ids"].shape[0]
            if seq_len > self.max_length:
                for k in inputs:
                    if inputs[k].dim() >= 1 and inputs[k].shape[0] == seq_len:
                        inputs[k] = inputs[k][:self.max_length]

            # Create labels: mask everything before the assistant response
            labels = inputs["input_ids"].clone()
            assistant_token = self.processor.tokenizer.encode("assistant", add_special_tokens=False)
            input_ids = inputs["input_ids"].tolist()

            # Find last occurrence of assistant marker
            mask_end = 0
            for i in range(len(input_ids) - len(assistant_token)):
                if input_ids[i:i + len(assistant_token)] == assistant_token:
                    mask_end = i + len(assistant_token)

            labels[:mask_end] = -100
            # Also mask padding
            if self.processor.tokenizer.pad_token_id is not None:
                labels[labels == self.processor.tokenizer.pad_token_id] = -100
            inputs["labels"] = labels

            return inputs

        # Return raw data if no processor
        return {
            "image": image,
            "question": question,
            "answer": answer,
        }
