"""
QLoRA Fine-Tuning Script for Hearthstone AI Assistant.
Trains a LoRA adapter on top of Qwen2.5-7B-Instruct / 1.5B using 4-bit NF4 quantization.
Optimized for 8GB VRAM (NVIDIA RTX 4060).
"""

import argparse
import importlib
import importlib.metadata
import inspect
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from .next_action_contract import build_next_action_prompt, format_next_action_completion
from .next_action_formatter import load_and_validate_manifest

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
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate schema-v2 train/eval files without importing the training stack",
    )
    parser.add_argument(
        "--check-environment",
        action="store_true",
        help="Report QLoRA dependency and CUDA readiness without loading a model",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_next_action_dataset(
    path: str | Path,
    *,
    expected_split: str | None = None,
    expected_game_ids: set[str] | None = None,
    expected_record_count: int | None = None,
) -> dict[str, int]:
    """Validates that a training file is schema-v2, formatted, and self-consistent."""
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found at {dataset_path}")

    records = 0
    games = set()
    with dataset_path.open("r", encoding="utf-8-sig") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                record: dict[str, Any] = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{dataset_path}:{line_number}: invalid JSON") from exc
            if record.get("schema_version") != 2 or record.get("dataset_contract") != "next_action_v2":
                raise ValueError(f"{dataset_path}:{line_number}: not a formatted schema-v2 record")
            if expected_split is not None and record.get("split") != expected_split:
                raise ValueError(f"{dataset_path}:{line_number}: record is not in split {expected_split}")
            if expected_game_ids is not None and record.get("game_id") not in expected_game_ids:
                raise ValueError(f"{dataset_path}:{line_number}: game_id is outside frozen split")
            candidates = record.get("candidates")
            chosen_id = record.get("chosen_candidate_id")
            if not isinstance(candidates, list) or not candidates:
                raise ValueError(f"{dataset_path}:{line_number}: candidates must be a non-empty list")
            candidate_ids = {candidate.get("id") for candidate in candidates}
            if chosen_id not in candidate_ids:
                raise ValueError(f"{dataset_path}:{line_number}: chosen candidate is absent")
            expected_prompt = build_next_action_prompt(record.get("state", {}), candidates)
            expected_completion = format_next_action_completion(chosen_id)
            if record.get("prompt") != expected_prompt or record.get("completion") != expected_completion:
                raise ValueError(f"{dataset_path}:{line_number}: prompt/completion drift")
            messages = record.get("messages")
            if not isinstance(messages, list) or len(messages) != 3:
                raise ValueError(f"{dataset_path}:{line_number}: invalid ChatML messages")
            if messages[1].get("content") != expected_prompt or messages[2].get("content") != expected_completion:
                raise ValueError(f"{dataset_path}:{line_number}: ChatML drift")
            records += 1
            games.add(record.get("game_id"))
    if not records:
        raise ValueError(f"Dataset file is empty: {dataset_path}")
    if expected_game_ids is not None and games != expected_game_ids:
        missing = sorted(expected_game_ids - games)
        extra = sorted(games - expected_game_ids)
        raise ValueError(f"{dataset_path}: frozen game membership mismatch; missing={missing}, extra={extra}")
    if expected_record_count is not None and records != expected_record_count:
        raise ValueError(
            f"{dataset_path}: frozen record count mismatch; expected={expected_record_count}, actual={records}"
        )
    return {"records": records, "games": len(games)}


def training_environment_report() -> dict[str, Any]:
    """Collects bounded, read-only readiness facts for the QLoRA environment."""
    package_names = ("torch", "transformers", "datasets", "trl", "peft", "bitsandbytes", "accelerate")
    packages: dict[str, dict[str, Any]] = {}
    for package_name in package_names:
        try:
            packages[package_name] = {"installed": True, "version": importlib.metadata.version(package_name)}
        except importlib.metadata.PackageNotFoundError:
            packages[package_name] = {"installed": False, "version": None}

    torch_info: dict[str, Any] = {"importable": False, "cuda_available": False}
    try:
        torch = importlib.import_module("torch")
        cuda_available = bool(torch.cuda.is_available())
        torch_info.update(
            {
                "importable": True,
                "cuda_available": cuda_available,
                "torch_version": getattr(torch, "__version__", None),
                "cuda_version": getattr(getattr(torch, "version", None), "cuda", None),
                "device_count": int(torch.cuda.device_count()) if cuda_available else 0,
                "device_name": torch.cuda.get_device_name(0) if cuda_available else None,
                "bf16_supported": bool(torch.cuda.is_bf16_supported()) if cuda_available else False,
            }
        )
    except Exception as exc:
        torch_info["error"] = f"{type(exc).__name__}: {exc}"

    required = ("torch", "transformers", "datasets", "trl", "peft", "bitsandbytes", "accelerate")
    missing = [name for name in required if not packages[name]["installed"]]
    return {
        "packages": packages,
        "torch": torch_info,
        "missing_packages": missing,
        "qlora_ready": not missing and bool(torch_info.get("cuda_available")),
    }


def _supported_kwargs(callable_obj: Any, values: dict[str, Any]) -> dict[str, Any]:
    """Keeps the trainer compatible with adjacent TRL releases."""
    parameters = inspect.signature(callable_obj).parameters
    return {key: value for key, value in values.items() if key in parameters}


def train_qlora(cfg: dict, smoke_test: bool = False):
    environment = training_environment_report()
    if environment["missing_packages"]:
        missing = ", ".join(environment["missing_packages"])
        raise RuntimeError(f"QLoRA dependencies are missing: {missing}; run --check-environment first")
    if not environment["torch"].get("cuda_available"):
        raise RuntimeError("QLoRA requires a CUDA-enabled PyTorch runtime; current environment has no CUDA device")

    import torch
    from datasets import load_dataset
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from trl import SFTTrainer, SFTConfig

    if cfg.get("dataset_contract") != "next_action_v2":
        raise ValueError("QLoRA config must explicitly select dataset_contract=next_action_v2")

    model_id = cfg.get("model_name_or_path", "Qwen/Qwen2.5-7B-Instruct")
    output_dir = cfg.get("output_dir", "models/qlora-hearthstone")
    train_file = cfg.get("train_file", "data/processed/next_action_train_chatml.jsonl")
    eval_file = cfg.get("eval_file", "data/processed/next_action_validation_chatml.jsonl")
    max_seq_length = cfg.get("max_seq_length", 1024)

    manifest_file = cfg.get("split_manifest")
    if not manifest_file:
        raise ValueError("QLoRA config must provide split_manifest")
    manifest = json.loads(Path(manifest_file).read_text(encoding="utf-8-sig"))
    source_dataset = manifest.get("source", {}).get("dataset")
    if not source_dataset:
        raise ValueError("Frozen split manifest does not identify its source dataset")
    manifest, _ = load_and_validate_manifest(source_dataset, manifest_file)
    train_game_ids = set(manifest["splits"]["train_game_ids"]["game_ids"])
    validation_game_ids = set(manifest["splits"]["validation_game_ids"]["game_ids"])
    train_stats = validate_next_action_dataset(
        train_file,
        expected_split="train",
        expected_game_ids=train_game_ids,
        expected_record_count=manifest["splits"]["train_game_ids"]["records"],
    )
    eval_stats = (
        validate_next_action_dataset(
            eval_file,
            expected_split="validation",
            expected_game_ids=validation_game_ids,
            expected_record_count=manifest["splits"]["validation_game_ids"]["records"],
        )
        if os.path.exists(eval_file)
        else None
    )

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
        if eval_dataset is not None:
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
    sft_arg_values = {
        "output_dir": output_dir,
        "max_length": max_seq_length,
        "per_device_train_batch_size": cfg.get("per_device_train_batch_size", 2),
        "gradient_accumulation_steps": cfg.get("gradient_accumulation_steps", 8),
        "learning_rate": cfg.get("learning_rate", 2e-4),
        "num_train_epochs": 1 if smoke_test else cfg.get("num_train_epochs", 3),
        "max_steps": 2 if smoke_test else -1,
        "warmup_ratio": cfg.get("warmup_ratio", 0.03),
        "lr_scheduler_type": cfg.get("lr_scheduler_type", "cosine"),
        "logging_steps": 1 if smoke_test else cfg.get("logging_steps", 10),
        "eval_strategy": "no" if (smoke_test or not eval_dataset) else cfg.get("eval_strategy", "steps"),
        "eval_steps": cfg.get("eval_steps", 100),
        "save_strategy": "no" if smoke_test else cfg.get("save_strategy", "steps"),
        "save_steps": cfg.get("save_steps", 100),
        "save_total_limit": cfg.get("save_total_limit", 2),
        "bf16": torch.cuda.is_bf16_supported() if use_cuda else False,
        "fp16": (use_cuda and not torch.cuda.is_bf16_supported()),
        "gradient_checkpointing": cfg.get("gradient_checkpointing", True),
        "optim": cfg.get("optim", "paged_adamw_8bit") if use_cuda else "adamw_torch",
        "report_to": "none",
        "disable_tqdm": False,
    }
    # TRL renamed max_seq_length to max_length; only pass fields supported by
    # the installed release and explicitly retain the intended sequence limit.
    sft_arg_values["max_seq_length"] = max_seq_length
    sft_args = SFTConfig(**_supported_kwargs(SFTConfig, sft_arg_values))

    # 7. Trainer setup
    trainer_values = {
        "model": model,
        "args": sft_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "peft_config": peft_config,
        "tokenizer": tokenizer,
        "processing_class": tokenizer,
    }
    trainer = SFTTrainer(**_supported_kwargs(SFTTrainer, trainer_values))

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

    logger.info("Validated schema-v2 train=%s eval=%s", train_stats, eval_stats)
    return train_result


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    if args.check_environment:
        environment = training_environment_report()
        print(json.dumps(environment, ensure_ascii=False, indent=2))
        if not environment["qlora_ready"]:
            sys.exit(1)
        return
    cfg = load_config(args.config)

    if args.model:
        cfg["model_name_or_path"] = args.model
    if args.output_dir:
        cfg["output_dir"] = args.output_dir

    if args.validate_only:
        manifest_path = cfg.get("split_manifest")
        if not manifest_path:
            raise ValueError("Config must provide split_manifest")
        frozen_manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8-sig"))
        source_dataset = frozen_manifest.get("source", {}).get("dataset")
        if not source_dataset:
            raise ValueError("Frozen split manifest does not identify its source dataset")
        frozen_manifest, _ = load_and_validate_manifest(source_dataset, manifest_path)
        result = {
            "train": validate_next_action_dataset(
                cfg["train_file"],
                expected_split="train",
                expected_game_ids=set(frozen_manifest["splits"]["train_game_ids"]["game_ids"]),
                expected_record_count=frozen_manifest["splits"]["train_game_ids"]["records"],
            ),
            "eval": validate_next_action_dataset(
                cfg["eval_file"],
                expected_split="validation",
                expected_game_ids=set(frozen_manifest["splits"]["validation_game_ids"]["game_ids"]),
                expected_record_count=frozen_manifest["splits"]["validation_game_ids"]["records"],
            ),
        }
        test_file = cfg.get("test_file")
        if test_file:
            result["test"] = validate_next_action_dataset(
                test_file,
                expected_split="test",
                expected_game_ids=set(frozen_manifest["splits"]["test_game_ids"]["game_ids"]),
                expected_record_count=frozen_manifest["splits"]["test_game_ids"]["records"],
            )
        temporal_file = cfg.get("temporal_holdout_file")
        if temporal_file:
            result["temporal_holdout"] = validate_next_action_dataset(
                temporal_file,
                expected_split="temporal_holdout",
                expected_game_ids=set(frozen_manifest["splits"]["temporal_holdout_game_ids"]["game_ids"]),
                expected_record_count=frozen_manifest["splits"]["temporal_holdout_game_ids"]["records"],
            )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    try:
        train_qlora(cfg, smoke_test=args.smoke_test)
    except Exception as e:
        logger.error("Training failed with error: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
