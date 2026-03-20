"""
LoRA fine-tuning script for Qwen2-VL-2B on PCB inspection QA.
Uses 4-bit quantization (QLoRA) to fit in 8GB VRAM.
"""

import argparse
import json
import logging
import os

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)

from src.vlm.dataset import PCBVLMDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"


def build_model_and_processor(model_id: str = MODEL_ID):
    """Load Qwen2-VL-2B with 4-bit quantization."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    logger.info(f"Loading {model_id} with 4-bit quantization...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    # Prepare for LoRA training
    model = prepare_model_for_kbit_training(model)

    return model, processor


def apply_lora(model, r: int = 16, alpha: int = 32, dropout: float = 0.05):
    """Apply LoRA adapters to the model."""
    # Target the attention layers in the language model
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]

    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model


class VLMDataCollator:
    """Custom data collator for variable-length VLM inputs."""

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        # Pad batch to same length
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [item["input_ids"] for item in batch],
            batch_first=True,
            padding_value=self.processor.tokenizer.pad_token_id,
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            [item["attention_mask"] for item in batch],
            batch_first=True,
            padding_value=0,
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            [item["labels"] for item in batch],
            batch_first=True,
            padding_value=-100,
        )

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        # Handle pixel values if present
        if "pixel_values" in batch[0]:
            pixel_values = torch.cat([item["pixel_values"].unsqueeze(0) for item in batch])
            result["pixel_values"] = pixel_values

        if "image_grid_thw" in batch[0]:
            image_grid_thw = torch.cat([item["image_grid_thw"].unsqueeze(0) for item in batch])
            result["image_grid_thw"] = image_grid_thw

        return result


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2-VL on PCB QA")
    parser.add_argument("--train-qa", type=str, required=True, help="Path to train QA JSON")
    parser.add_argument("--val-qa", type=str, default=None, help="Path to val QA JSON")
    parser.add_argument("--output-dir", type=str, default="checkpoints/vlm")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()

    # Build model
    model, processor = build_model_and_processor()
    model = apply_lora(model, r=args.lora_r, alpha=args.lora_alpha)

    # Datasets
    train_ds = PCBVLMDataset(args.train_qa, processor=processor, max_length=args.max_length)
    logger.info(f"Train dataset: {len(train_ds)} samples")

    val_ds = None
    if args.val_qa:
        val_ds = PCBVLMDataset(args.val_qa, processor=processor, max_length=args.max_length)
        logger.info(f"Val dataset: {len(val_ds)} samples")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if val_ds else "no",
        save_total_limit=2,
        load_best_model_at_end=True if val_ds else False,
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        gradient_checkpointing=True,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=VLMDataCollator(processor),
    )

    logger.info("Starting LoRA fine-tuning...")
    trainer.train()

    # Save LoRA adapter
    adapter_path = os.path.join(args.output_dir, "lora_adapter")
    model.save_pretrained(adapter_path)
    processor.save_pretrained(adapter_path)
    logger.info(f"LoRA adapter saved to {adapter_path}")


if __name__ == "__main__":
    main()
