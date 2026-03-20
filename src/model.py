"""
Faster R-CNN with ResNet-50 FPN backbone for PCB defect detection.
Uses torchvision's pre-trained model and swaps the classification head.
"""

import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator

from src.config import NUM_CLASSES


def build_model(
    num_classes: int = NUM_CLASSES,
    pretrained_backbone: bool = True,
    trainable_backbone_layers: int = 3,
    min_size: int = 640,
    max_size: int = 640,
) -> FasterRCNN:
    """
    Build Faster R-CNN with ResNet-50 FPN.

    Args:
        num_classes: Number of classes including background.
        pretrained_backbone: Use ImageNet pre-trained weights.
        trainable_backbone_layers: Number of backbone layers to fine-tune (0-5).
        min_size: Minimum image dimension for the transform.
        max_size: Maximum image dimension for the transform.

    Returns:
        Configured FasterRCNN model.
    """
    weights_backbone = "DEFAULT" if pretrained_backbone else None

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=weights_backbone,
        trainable_backbone_layers=trainable_backbone_layers,
        min_size=min_size,
        max_size=max_size,
    )

    # Replace the box predictor head for our number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def load_checkpoint(model: FasterRCNN, path: str, device: torch.device) -> dict:
    """Load model weights from checkpoint. Returns the checkpoint dict."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    return ckpt
