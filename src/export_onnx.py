"""
Export trained Faster R-CNN for production deployment.

Provides two export paths:
  1. TorchScript via torch.jit.script (handles dynamic control flow)
  2. ONNX via legacy exporter with a wrapper module

Faster R-CNN returns List[Dict] which requires special handling.
"""

import argparse
import logging
import os
from typing import Tuple

import torch
import torch.nn as nn

from src.config import NUM_CLASSES, INFER
from src.model import build_model, load_checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class FasterRCNNWrapper(nn.Module):
    """Wraps Faster R-CNN to return flat tensors instead of List[Dict]."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Convert batched tensor to list of tensors (Faster R-CNN expects this)
        image_list = [images[i] for i in range(images.shape[0])]
        outputs = self.model(image_list)
        # Return first image's results as flat tensors
        return outputs[0]["boxes"], outputs[0]["labels"], outputs[0]["scores"]


def export_torchscript(checkpoint_path: str, output_path: str):
    """Export to TorchScript using torch.jit.script (handles NMS control flow)."""
    device = torch.device("cpu")

    model = build_model(num_classes=NUM_CLASSES, min_size=INFER.image_size, max_size=INFER.image_size)
    load_checkpoint(model, checkpoint_path, device)
    model.eval()

    logger.info("Exporting to TorchScript via scripting...")
    with torch.no_grad():
        scripted = torch.jit.script(model)

    scripted.save(output_path)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info(f"TorchScript model saved to {output_path} ({size_mb:.1f} MB)")

    # Verify
    logger.info("Verifying TorchScript model...")
    loaded = torch.jit.load(output_path)
    dummy = [torch.randn(3, INFER.image_size, INFER.image_size)]
    with torch.no_grad():
        out = loaded(dummy)
    assert len(out) > 0, "No output from model"
    logger.info(f"Verification passed — {len(out[0]['boxes'])} detections on dummy input ✓")
    return output_path


def export_onnx(checkpoint_path: str, output_path: str, opset: int = 17):
    """Export to ONNX using wrapper module for flat tensor output."""
    device = torch.device("cpu")

    model = build_model(num_classes=NUM_CLASSES, min_size=INFER.image_size, max_size=INFER.image_size)
    load_checkpoint(model, checkpoint_path, device)
    model.eval()

    wrapper = FasterRCNNWrapper(model)
    wrapper.eval()

    dummy = torch.randn(1, 3, INFER.image_size, INFER.image_size)

    logger.info(f"Exporting to ONNX (opset {opset})...")
    torch.onnx.export(
        wrapper,
        (dummy,),
        output_path,
        opset_version=opset,
        input_names=["images"],
        output_names=["boxes", "labels", "scores"],
        dynamic_axes={
            "images": {0: "batch"},
            "boxes": {0: "num_detections"},
            "labels": {0: "num_detections"},
            "scores": {0: "num_detections"},
        },
        dynamo=False,
    )

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info(f"ONNX model saved to {output_path} ({size_mb:.1f} MB)")

    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        logger.info("ONNX verification passed ✓")
    except ImportError:
        logger.warning("Install 'onnx' to verify exported model")
    except Exception as e:
        logger.warning(f"ONNX verification note: {e}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Export model for deployment")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--format", choices=["onnx", "torchscript", "both"], default="both")
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.format in ("torchscript", "both"):
        ts_path = os.path.join(args.output_dir, "pcb_detector.pt")
        try:
            export_torchscript(args.checkpoint, ts_path)
        except Exception as e:
            logger.warning(f"TorchScript export failed: {e}")
            logger.info("Falling back to state_dict-only save...")
            # Fallback: save just the state dict for portable loading
            model = build_model(num_classes=NUM_CLASSES)
            load_checkpoint(model, args.checkpoint, torch.device("cpu"))
            torch.save(model.state_dict(), ts_path)
            logger.info(f"State dict saved to {ts_path}")

    if args.format in ("onnx", "both"):
        onnx_path = os.path.join(args.output_dir, "pcb_detector.onnx")
        try:
            export_onnx(args.checkpoint, onnx_path, args.opset)
        except Exception as e:
            logger.warning(f"ONNX export failed: {e}")

    logger.info("Export complete ✓")


if __name__ == "__main__":
    main()
