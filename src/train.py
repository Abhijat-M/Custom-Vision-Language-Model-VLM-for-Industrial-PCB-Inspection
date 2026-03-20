"""
Training script for PCB defect detection with Faster R-CNN.
Features: mixed precision, LR scheduling, early stopping, best-model tracking, TensorBoard logging.
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime

import torch
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from src.config import PATHS, TRAIN, NUM_CLASSES, CLASSES
from src.dataset import PCBDefectDataset, collate_fn
from src.model import build_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def train_one_epoch(model, optimizer, data_loader, device, epoch, scaler):
    model.train()
    running_loss = 0.0
    n_batches = 0

    for batch_idx, (images, targets) in enumerate(data_loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Skip batches with no annotations
        if all(t["boxes"].shape[0] == 0 for t in targets):
            continue

        with torch.autocast(device_type="cuda", enabled=scaler.is_enabled()):
            loss_dict = model(images, targets)
            total_loss = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        running_loss += total_loss.item()
        n_batches += 1

        if (batch_idx + 1) % 50 == 0:
            avg = running_loss / n_batches
            logger.info(
                f"  Epoch {epoch} | Batch {batch_idx + 1}/{len(data_loader)} | "
                f"Loss: {avg:.4f} | "
                f"lr: {optimizer.param_groups[0]['lr']:.6f}"
            )

    return running_loss / max(n_batches, 1)


@torch.no_grad()
def validate(model, data_loader, device):
    model.train()  # Faster R-CNN computes losses only in train mode
    running_loss = 0.0
    n_batches = 0

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        if all(t["boxes"].shape[0] == 0 for t in targets):
            continue

        loss_dict = model(images, targets)
        total_loss = sum(loss for loss in loss_dict.values())
        running_loss += total_loss.item()
        n_batches += 1

    return running_loss / max(n_batches, 1)


def main():
    parser = argparse.ArgumentParser(description="Train PCB defect detector")
    parser.add_argument("--epochs", type=int, default=TRAIN.epochs)
    parser.add_argument("--batch-size", type=int, default=TRAIN.batch_size)
    parser.add_argument("--lr", type=float, default=TRAIN.lr)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--data-dir", type=str, default=PATHS.data_dir)
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Paths
    train_dir = args.data_dir
    valid_dir = os.path.join(args.data_dir, "valid")

    # Datasets
    train_ds = PCBDefectDataset(train_dir, train=True)
    valid_ds = PCBDefectDataset(valid_dir, train=False)
    logger.info(f"Train: {len(train_ds)} images | Valid: {len(valid_ds)} images")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=TRAIN.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=TRAIN.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Model
    model = build_model(num_classes=NUM_CLASSES)
    model.to(device)
    logger.info(f"Model: Faster R-CNN ResNet50-FPN | Classes: {NUM_CLASSES} ({CLASSES})")

    # Optimizer + scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=args.lr, momentum=TRAIN.momentum, weight_decay=TRAIN.weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=TRAIN.lr_step_size, gamma=TRAIN.lr_gamma
    )

    start_epoch = 0
    best_val_loss = float("inf")
    patience_counter = 0

    # Resume from checkpoint
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        lr_scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        logger.info(f"Resumed from epoch {start_epoch}")

    scaler = GradScaler(enabled=(not args.no_amp and device.type == "cuda"))
    os.makedirs(PATHS.checkpoint_dir, exist_ok=True)

    # Training history
    history = {"train_loss": [], "val_loss": [], "lr": []}

    logger.info(f"Starting training for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch, scaler)
        val_loss = validate(model, valid_loader, device)
        lr_scheduler.step()

        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)

        logger.info(
            f"Epoch {epoch}/{args.epochs - 1} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"LR: {current_lr:.6f} | Time: {elapsed:.1f}s"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_path = os.path.join(PATHS.checkpoint_dir, "best_model.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": lr_scheduler.state_dict(),
                    "best_val_loss": best_val_loss,
                    "classes": CLASSES,
                    "num_classes": NUM_CLASSES,
                },
                best_path,
            )
            logger.info(f"  -> New best model saved (val_loss={best_val_loss:.4f})")
        else:
            patience_counter += 1

        # Periodic checkpoint
        if (epoch + 1) % TRAIN.save_every == 0:
            ckpt_path = os.path.join(PATHS.checkpoint_dir, f"checkpoint_epoch{epoch}.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": lr_scheduler.state_dict(),
                    "best_val_loss": best_val_loss,
                },
                ckpt_path,
            )

        # Early stopping
        if patience_counter >= TRAIN.early_stop_patience:
            logger.info(f"Early stopping at epoch {epoch} (patience={TRAIN.early_stop_patience})")
            break

    # Save training history
    history_path = os.path.join(PATHS.results_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"Training complete. Best val loss: {best_val_loss:.4f}")
    logger.info(f"History saved to {history_path}")


if __name__ == "__main__":
    main()
