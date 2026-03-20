"""
PCB Defect Dataset — Pascal VOC XML annotations.
Supports all 6 DeepPCB defect classes with train-time augmentation.
"""

import os
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from src.config import CLASS_TO_IDX, CLASSES


class PCBDefectDataset(Dataset):
    """
    Loads PCB images + Pascal VOC XML annotations from Roboflow export.

    Expected directory layout:
        root/
            images/     *.jpg
            annotations/  *.xml
    """

    def __init__(
        self,
        root: str,
        classes: Optional[List[str]] = None,
        train: bool = False,
    ):
        self.root = root
        self.train = train

        self.classes = [c.lower() for c in (classes or CLASSES)]
        self.class_to_idx = {c: CLASS_TO_IDX[c] for c in self.classes if c in CLASS_TO_IDX}

        self.img_dir = os.path.join(root, "images")
        self.ann_dir = os.path.join(root, "annotations")

        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")
        if not os.path.isdir(self.ann_dir):
            raise FileNotFoundError(f"Annotation directory not found: {self.ann_dir}")

        # Build matched pairs (image stem must have corresponding xml)
        img_stems = {
            os.path.splitext(f)[0]
            for f in os.listdir(self.img_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        }
        ann_stems = {
            os.path.splitext(f)[0]
            for f in os.listdir(self.ann_dir)
            if f.lower().endswith(".xml")
        }
        self.ids = sorted(img_stems & ann_stems)

        if not self.ids:
            raise RuntimeError(f"No matching image–annotation pairs in {root}")

        # Augmentation
        self.color_jitter = (
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
            if train
            else None
        )
        self.to_tensor = T.ToTensor()

    def __len__(self) -> int:
        return len(self.ids)

    def _find_image(self, stem: str) -> str:
        for ext in (".jpg", ".jpeg", ".png"):
            p = os.path.join(self.img_dir, stem + ext)
            if os.path.isfile(p):
                return p
        return os.path.join(self.img_dir, stem + ".jpg")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        stem = self.ids[idx]

        img_path = self._find_image(stem)
        ann_path = os.path.join(self.ann_dir, stem + ".xml")

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[WARN] Cannot load {img_path}: {e}")
            return self._empty_sample()

        try:
            tree = ET.parse(ann_path)
        except ET.ParseError as e:
            print(f"[WARN] Bad XML {ann_path}: {e}")
            return self._empty_sample()

        root = tree.getroot()
        boxes, labels = [], []
        w, h = image.size

        for obj in root.findall("object"):
            name = obj.find("name").text.lower().strip()
            if name not in self.class_to_idx:
                continue

            b = obj.find("bndbox")
            xmin = int(float(b.find("xmin").text))
            ymin = int(float(b.find("ymin").text))
            xmax = int(float(b.find("xmax").text))
            ymax = int(float(b.find("ymax").text))

            # Skip degenerate
            if xmax <= xmin or ymax <= ymin:
                continue

            # Clamp to image bounds
            xmin = max(0, min(xmin, w - 1))
            ymin = max(0, min(ymin, h - 1))
            xmax = max(1, min(xmax, w))
            ymax = max(1, min(ymax, h))
            if xmax <= xmin or ymax <= ymin:
                continue

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_to_idx[name])

        if boxes:
            boxes_t = torch.tensor(boxes, dtype=torch.float32)
            labels_t = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)

        # Random horizontal flip (training)
        if self.train and torch.rand(1).item() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            if boxes_t.shape[0] > 0:
                boxes_t[:, [0, 2]] = w - boxes_t[:, [2, 0]]

        if self.color_jitter is not None:
            image = self.color_jitter(image)

        image = self.to_tensor(image)

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": torch.tensor([idx]),
            "area": (boxes_t[:, 3] - boxes_t[:, 1]) * (boxes_t[:, 2] - boxes_t[:, 0])
            if boxes_t.shape[0] > 0
            else torch.zeros(0),
            "iscrowd": torch.zeros(len(labels), dtype=torch.int64),
        }
        return image, target

    def _empty_sample(self) -> Tuple[torch.Tensor, Dict]:
        return (
            torch.zeros(3, 640, 640),
            {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "image_id": torch.tensor([0]),
                "area": torch.zeros(0),
                "iscrowd": torch.zeros(0, dtype=torch.int64),
            },
        )


def collate_fn(batch):
    """Custom collate — Faster R-CNN expects list[Tensor], list[dict]."""
    return tuple(zip(*batch))
