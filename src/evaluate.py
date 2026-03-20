"""
COCO-style mAP evaluation for PCB defect detection.
Computes AP@0.5, AP@0.75, and AP@[0.5:0.95] per class and overall.
"""

import argparse
import json
import logging
import os
from collections import defaultdict

import torch
import numpy as np
from torch.utils.data import DataLoader

from src.config import PATHS, INFER, NUM_CLASSES, CLASSES, IDX_TO_CLASS
from src.dataset import PCBDefectDataset, collate_fn
from src.model import build_model, load_checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_iou(box1, box2):
    """Compute IoU between two boxes [x1,y1,x2,y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def compute_ap(recalls, precisions):
    """Compute AP using 101-point interpolation (COCO style)."""
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([1.0], precisions, [0.0]))

    # Make precision monotonically decreasing
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    # 101-point interpolation
    recall_points = np.linspace(0, 1, 101)
    ap = 0.0
    for r in recall_points:
        prec_at_r = precisions[recalls >= r]
        ap += prec_at_r.max() if len(prec_at_r) > 0 else 0.0
    return ap / 101.0


def evaluate_map(
    all_predictions: list,
    all_targets: list,
    iou_thresholds: list = None,
) -> dict:
    """
    Compute mAP at various IoU thresholds.

    Args:
        all_predictions: List of dicts with 'boxes', 'labels', 'scores'.
        all_targets: List of dicts with 'boxes', 'labels'.
        iou_thresholds: IoU thresholds for AP computation.

    Returns:
        Dict with per-class and overall AP values.
    """
    if iou_thresholds is None:
        iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    results = {}

    for iou_thresh in iou_thresholds:
        class_aps = {}

        for cls_idx in range(1, NUM_CLASSES):
            cls_name = IDX_TO_CLASS[cls_idx]

            # Gather all predictions and ground truths for this class
            all_scores = []
            all_tp = []
            n_gt = 0

            for preds, gts in zip(all_predictions, all_targets):
                gt_boxes = gts["boxes"][gts["labels"] == cls_idx].numpy()
                gt_matched = [False] * len(gt_boxes)
                n_gt += len(gt_boxes)

                pred_mask = preds["labels"] == cls_idx
                pred_boxes = preds["boxes"][pred_mask].numpy()
                pred_scores = preds["scores"][pred_mask].numpy()

                # Sort by score descending
                sort_idx = np.argsort(-pred_scores)
                pred_boxes = pred_boxes[sort_idx]
                pred_scores = pred_scores[sort_idx]

                for pb, ps in zip(pred_boxes, pred_scores):
                    all_scores.append(ps)
                    best_iou = 0.0
                    best_gt = -1
                    for gi, gb in enumerate(gt_boxes):
                        iou = compute_iou(pb, gb)
                        if iou > best_iou:
                            best_iou = iou
                            best_gt = gi

                    if best_iou >= iou_thresh and not gt_matched[best_gt]:
                        all_tp.append(1)
                        gt_matched[best_gt] = True
                    else:
                        all_tp.append(0)

            if n_gt == 0:
                class_aps[cls_name] = 0.0
                continue

            # Sort all detections by score
            sort_idx = np.argsort(-np.array(all_scores))
            all_tp = np.array(all_tp)[sort_idx]

            cum_tp = np.cumsum(all_tp)
            cum_fp = np.cumsum(1 - all_tp)
            recalls = cum_tp / n_gt
            precisions = cum_tp / (cum_tp + cum_fp)

            class_aps[cls_name] = compute_ap(recalls, precisions)

        results[f"AP@{iou_thresh:.2f}"] = class_aps
        results[f"mAP@{iou_thresh:.2f}"] = np.mean(list(class_aps.values()))

    # Summary metrics
    results["mAP@0.5"] = results.get("mAP@0.50", 0.0)
    results["mAP@0.75"] = results.get("mAP@0.75", 0.0)
    map_values = [results[f"mAP@{t:.2f}"] for t in iou_thresholds]
    results["mAP@[0.5:0.95]"] = np.mean(map_values)

    return results


@torch.no_grad()
def run_evaluation(model, data_loader, device):
    """Run model on dataset and collect predictions + targets."""
    model.eval()
    all_preds = []
    all_targets = []

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        outputs = model(images)

        for out, tgt in zip(outputs, targets):
            all_preds.append({k: v.cpu() for k, v in out.items()})
            all_targets.append({k: v.cpu() for k, v in tgt.items()})

    return all_preds, all_targets


def main():
    parser = argparse.ArgumentParser(description="Evaluate PCB defect detector")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data-dir", type=str, default=PATHS.valid_dir)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--score-threshold", type=float, default=INFER.score_threshold)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset
    dataset = PCBDefectDataset(args.data_dir, train=False)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=2, collate_fn=collate_fn,
    )
    logger.info(f"Evaluating on {len(dataset)} images from {args.data_dir}")

    # Model
    model = build_model()
    load_checkpoint(model, args.checkpoint, device)
    model.to(device)
    model.eval()

    # Run inference
    all_preds, all_targets = run_evaluation(model, loader, device)

    # Filter by score threshold
    filtered_preds = []
    for pred in all_preds:
        mask = pred["scores"] >= args.score_threshold
        filtered_preds.append({
            "boxes": pred["boxes"][mask],
            "labels": pred["labels"][mask],
            "scores": pred["scores"][mask],
        })

    # Compute metrics
    results = evaluate_map(filtered_preds, all_targets)

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"  mAP@0.5       : {results['mAP@0.5']:.4f}")
    print(f"  mAP@0.75      : {results['mAP@0.75']:.4f}")
    print(f"  mAP@[0.5:0.95]: {results['mAP@[0.5:0.95]']:.4f}")
    print("-" * 60)
    print("Per-class AP@0.5:")
    for cls_name, ap in results["AP@0.50"].items():
        print(f"  {cls_name:12s}: {ap:.4f}")
    print("=" * 60)

    # Save results
    os.makedirs(PATHS.results_dir, exist_ok=True)
    out_path = os.path.join(PATHS.results_dir, "eval_results.json")
    serializable = {k: v if not isinstance(v, np.floating) else float(v)
                    for k, v in results.items()}
    # Convert nested dicts too
    for k, v in serializable.items():
        if isinstance(v, dict):
            serializable[k] = {kk: float(vv) for kk, vv in v.items()}
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
