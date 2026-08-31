"""
QLoRA Fine-Tuning Script for Hearthstone AI Assistant.
Trains a LoRA adapter on top of Qwen2.5-7B-Instruct / 1.5B using 4-bit NF4 quantization.
Optimized for 8GB VRAM (NVIDIA RTX 4060).
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger("hearthstone.qlora")


def parse_args():
    parser = argparse.ArgumentParser(description="QLoRA training for Hearthstone AI Assistant")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/qlora_config.json",
        help="Path to QLoRA configuration JSON file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override base model name/path (e.g. Qwen/Qwen2.5-1.5B-Instruct)",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a 1-step smoke test to verify dependencies, loading, and tokenization without full training",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def train_qlora(cfg: dict, smoke_test: bool = False):
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from trl import SFTTrainer, SFTConfig

    model_id = cfg.get("model_name_or_path", "Qwen/Qwen2.5-7B-Instruct")
    output_dir = cfg.get("output_dir", "models/qlora-hearthstone")
    train_file = cfg.get("train_file", "data/processed/sft_train_chatml.jsonl")
    eval_file = cfg.get("eval_file", "data/processed/sft_eval_chatml.jsonl")
    max_seq_length = cfg.get("max_seq_length", 1024)

    logger.info("=== Hearthstone QLoRA Trainer ===")
    logger.info("Base model: %s", model_id)
    logger.info("Output directory: %s", output_dir)
    logger.info("CUDA Available: %s (Device: %s)", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")
    if torch.cuda.is_available():
        vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
        logger.info("Total VRAM: %.1f MB", vram_mb)

    # 1. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Datasets
    logger.info("Loading datasets from %s and %s...", train_file, eval_file)
    data_files = {"train": train_file}
    if os.path.exists(eval_file):
        data_files["eval"] = eval_file

    raw_datasets = load_dataset("json", data_files=data_files)
    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets.get("eval")

    if smoke_test:
        train_dataset = train_dataset.select(range(min(4, len(train_dataset))))
        if eval_dataset:
            eval_dataset = eval_dataset.select(range(min(2, len(eval_dataset))))
        logger.info("Running in SMOKE-TEST mode with %d train samples.", len(train_dataset))

    # 3. 4-bit Quantization Config (NF4)
    use_cuda = torch.cuda.is_available()
    bnb_config = None
    if use_cuda:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    # 4. Load Base Model
    logger.info("Loading model weights (4-bit quantized)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config if use_cuda else None,
        device_map="auto" if use_cuda else None,
        torch_dtype=torch.bfloat16 if (use_cuda and torch.cuda.is_bf16_supported()) else torch.float32,
        trust_remote_code=True,
    )

    if use_cuda:
        model = prepare_model_for_kbit_training(model)

    # 5. LoRA Adapter Config
    peft_config = LoraConfig(
        r=cfg.get("lora_r", 16),
        lora_alpha=cfg.get("lora_alpha", 32),
        lora_dropout=cfg.get("lora_dropout", 0.05),
        target_modules=cfg.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 6. SFT Training Arguments
    sft_args = SFTConfig(
        output_dir=output_dir,
        max_seq_length=max_seq_length,
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 2),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 8),
        learning_rate=cfg.get("learning_rate", 2e-4),
        num_train_epochs=1 if smoke_test else cfg.get("num_train_epochs", 3),
        max_steps=2 if smoke_test else -1,
        warmup_ratio=cfg.get("warmup_ratio", 0.03),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        logging_steps=1 if smoke_test else cfg.get("logging_steps", 10),
        eval_strategy="no" if (smoke_test or not eval_dataset) else cfg.get("eval_strategy", "steps"),
        eval_steps=cfg.get("eval_steps", 100),
        save_strategy="no" if smoke_test else cfg.get("save_strategy", "steps"),
        save_steps=cfg.get("save_steps", 100),
        save_total_limit=cfg.get("save_total_limit", 2),
        bf16=torch.cuda.is_bf16_supported() if use_cuda else False,
        fp16=(use_cuda and not torch.cuda.is_bf16_supported()),
        gradient_checkpointing=cfg.get("gradient_checkpointing", True),
        optim=cfg.get("optim", "paged_adamw_8bit") if use_cuda else "adamw_torch",
        report_to="none",
        disable_tqdm=False,
    )

    # 7. Trainer setup
    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        tokenizer=tokenizer,
    )

    # 8. Train
    logger.info("Starting training loop...")
    train_result = trainer.train()

    # 9. Save
    if not smoke_test:
        logger.info("Saving trained LoRA adapter to %s...", output_dir)
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info("Training complete! Adapter saved.")
    else:
        logger.info("Smoke test passed successfully! (Loss: %.4f)", train_result.training_loss)

    return train_result


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    cfg = load_config(args.config)

    if args.model:
        cfg["model_name_or_path"] = args.model
    if args.output_dir:
        cfg["output_dir"] = args.output_dir

    try:
        train_qlora(cfg, smoke_test=args.smoke_test)
    except Exception as e:
        logger.error("Training failed with error: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
