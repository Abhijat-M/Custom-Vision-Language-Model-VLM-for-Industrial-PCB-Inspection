"""
Central configuration for PCB Defect Inspection System.
All hyperparameters, paths, and class definitions in one place.
"""

import os
from dataclasses import dataclass, field
from typing import Tuple

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class Paths:
    data_dir: str = os.path.join("C:/Users/Wizard/Documents/quality_inspection", "data")
    train_dir: str = ""
    valid_dir: str = ""
    checkpoint_dir: str = os.path.join(BASE_DIR, "checkpoints")
    results_dir: str = os.path.join(BASE_DIR, "results")

    def __post_init__(self):
        # Train images/annotations are directly under data_dir
        self.train_dir = self.data_dir
        self.valid_dir = os.path.join(self.data_dir, "valid")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)


# ── Classes (all 6 DeepPCB defect types) ───────────────────────────────────
CLASSES = ["open", "short", "mousebite", "spur", "copper", "pin-hole"]
NUM_CLASSES = len(CLASSES) + 1  # +1 for background

CLASS_TO_IDX = {c: i + 1 for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i + 1: c for i, c in enumerate(CLASSES)}
IDX_TO_CLASS[0] = "__background__"


@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 4
    lr: float = 0.005
    momentum: float = 0.9
    weight_decay: float = 0.0005
    lr_step_size: int = 15
    lr_gamma: float = 0.1
    num_workers: int = 2
    grad_clip: float = 5.0
    early_stop_patience: int = 10
    save_every: int = 5


@dataclass
class InferConfig:
    score_threshold: float = 0.5
    nms_threshold: float = 0.5
    image_size: int = 640


@dataclass
class SeverityConfig:
    """Defect severity based on (defect_area / image_area) ratio."""
    low_threshold: float = 0.01
    med_threshold: float = 0.05


PATHS = Paths()
TRAIN = TrainConfig()
INFER = InferConfig()
SEVERITY = SeverityConfig()
