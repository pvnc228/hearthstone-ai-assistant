"""Safely extract frozen ModernCE features and train a small listwise ranker."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .evaluate_modernce import DEFAULT_HYPOTHESIS, DEFAULT_MODEL_DIR, state_premise
from .resource_guard import (
    ResourceBudget,
    ResourceGuard,
    ResourceLimitExceeded,
    configure_conservative_process,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN_FILE = PROJECT_ROOT / "data" / "processed" / "next_action_train_chatml.jsonl"
DEFAULT_VALIDATION_FILE = PROJECT_ROOT / "data" / "processed" / "next_action_validation_chatml.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "modernce_ranker"


def iter_records(path: Path | str, limit: int | None = None) -> Iterable[dict[str, Any]]:
    emitted = 0
    with Path(path).open("r", encoding="utf-8-sig") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            emitted += 1
            if limit is not None and emitted >= limit:
                return


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scan_dataset(path: Path | str, limit: int | None = None) -> dict[str, Any]:
    path = Path(path).resolve()
    groups: list[dict[str, Any]] = []
    pair_offset = 0
    for record in iter_records(path, limit):
        candidates = list(record.get("candidates", []))
        if not candidates:
            raise ValueError(f"{record.get('decision_id', '<unknown>')}: no candidates")
        chosen_id = record.get("chosen_candidate_id")
        gold_positions = [index for index, candidate in enumerate(candidates) if candidate.get("id") == chosen_id]
        if len(gold_positions) != 1:
            raise ValueError(
                f"{record.get('decision_id', '<unknown>')}: chosen candidate must occur exactly once"
            )
        groups.append(
            {
                "decision_id": str(record.get("decision_id", "")),
                "game_id": str(record.get("game_id", "")),
                "offset": pair_offset,
                "count": len(candidates),
                "gold_position": gold_positions[0],
                "action_type": str(record.get("gold_action", {}).get("type", "UNKNOWN")),
            }
        )
        pair_offset += len(candidates)
    if not groups:
        raise ValueError(f"No records found in {path}")
    return {
        "dataset": str(path),
        "dataset_sha256": sha256_file(path),
        "limit": limit,
        "records": len(groups),
        "pairs": pair_offset,
        "groups": groups,
    }


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def model_signature(model_dir: Path | str) -> dict[str, Any]:
    model_dir = Path(model_dir).resolve()
    weights = model_dir / "model.safetensors"
    stat = weights.stat()
    return {
        "model_dir": str(model_dir),
        "weights_size": stat.st_size,
        "weights_mtime_ns": stat.st_mtime_ns,
    }


def cache_paths(cache_dir: Path, split: str) -> dict[str, Path]:
    return {
        "features": cache_dir / f"{split}.features.f16",
        "metadata": cache_dir / f"{split}.metadata.json",
        "checkpoint": cache_dir / f"{split}.checkpoint.json",
    }


def prepare_cache(
    cache_dir: Path,
    split: str,
    metadata: dict[str, Any],
    hidden_size: int,
    signature: dict[str, Any],
):
    import numpy as np

    paths = cache_paths(cache_dir, split)
    expected_metadata = {
        **metadata,
        "hidden_size": hidden_size,
        "dtype": "float16",
        "model": signature,
    }
    if paths["metadata"].exists():
        existing = json.loads(paths["metadata"].read_text(encoding="utf-8-sig"))
        if existing != expected_metadata:
            raise ValueError(f"{split} cache metadata does not match current dataset/model")
    else:
        atomic_write_json(paths["metadata"], expected_metadata)

    expected_bytes = metadata["pairs"] * hidden_size * 2
    if paths["features"].exists():
        if paths["features"].stat().st_size != expected_bytes:
            raise ValueError(f"{split} feature cache has unexpected size")
        mode = "r+"
    else:
        mode = "w+"
    features = np.memmap(
        paths["features"], dtype=np.float16, mode=mode, shape=(metadata["pairs"], hidden_size)
    )
    if mode == "w+":
        features.flush()

    if paths["checkpoint"].exists():
        checkpoint = json.loads(paths["checkpoint"].read_text(encoding="utf-8-sig"))
    else:
        checkpoint = {
            "records_done": 0,
            "pairs_done": 0,
            "truncated_pairs": 0,
            "max_untruncated_tokens": 0,
            "complete": False,
        }
        atomic_write_json(paths["checkpoint"], checkpoint)

    records_done = int(checkpoint["records_done"])
    if not 0 <= records_done <= metadata["records"]:
        raise ValueError(f"{split} checkpoint record count is out of range")
    expected_pairs_done = (
        metadata["pairs"]
        if records_done == metadata["records"]
        else metadata["groups"][records_done]["offset"]
    )
    if checkpoint["pairs_done"] != expected_pairs_done:
        raise ValueError(f"{split} checkpoint is inconsistent with metadata")
    return features, checkpoint, paths


def extract_split(
    *,
    split: str,
    metadata: dict[str, Any],
    cache_dir: Path,
    hidden_size: int,
    signature: dict[str, Any],
    tokenizer,
    model,
    torch,
    guard: ResourceGuard,
    batch_size: int,
    max_length: int,
    flush_every_records: int,
) -> dict[str, Any]:
    features, checkpoint, paths = prepare_cache(cache_dir, split, metadata, hidden_size, signature)
    if checkpoint.get("complete"):
        print(f"[{split}] cache already complete", flush=True)
        return checkpoint

    records_done = int(checkpoint["records_done"])
    pairs_done = int(checkpoint["pairs_done"])
    truncated_pairs = int(checkpoint.get("truncated_pairs", 0))
    max_untruncated_tokens = int(checkpoint.get("max_untruncated_tokens", 0))
    started = time.monotonic()

    for record_index, record in enumerate(iter_records(metadata["dataset"], metadata["limit"])):
        if record_index < records_done:
            continue
        candidates = list(record["candidates"])
        premise = state_premise(record)
        hypotheses = [DEFAULT_HYPOTHESIS.format(candidate["description"]) for candidate in candidates]
        raw_lengths = tokenizer(
            [premise] * len(hypotheses), hypotheses, padding=False, truncation=False
        )["input_ids"]
        record_max_tokens = max(len(tokens) for tokens in raw_lengths)
        record_truncated = sum(len(tokens) > max_length for tokens in raw_lengths)
        write_offset = pairs_done

        try:
            for offset in range(0, len(hypotheses), batch_size):
                chunk = hypotheses[offset : offset + batch_size]
                encoded = tokenizer(
                    [premise] * len(chunk),
                    chunk,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                ).to("cuda")
                with torch.inference_mode():
                    hidden = model.model(**encoded).last_hidden_state
                    mask = encoded["attention_mask"].unsqueeze(-1)
                    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
                    pooled = model.head(pooled)
                array = pooled.to(dtype=torch.float16).cpu().numpy()
                features[write_offset : write_offset + len(chunk)] = array
                write_offset += len(chunk)
                guard.after_batch()
        except torch.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            raise ResourceLimitExceeded(f"CUDA OOM while extracting {split}") from exc

        records_done = record_index + 1
        pairs_done = write_offset
        truncated_pairs += record_truncated
        max_untruncated_tokens = max(max_untruncated_tokens, record_max_tokens)
        if records_done % flush_every_records == 0 or records_done == metadata["records"]:
            features.flush()
            checkpoint = {
                "records_done": records_done,
                "pairs_done": pairs_done,
                "truncated_pairs": truncated_pairs,
                "max_untruncated_tokens": max_untruncated_tokens,
                "complete": records_done == metadata["records"],
                "elapsed_seconds_this_run": round(time.monotonic() - started, 3),
                "resource_telemetry": guard.telemetry_summary(),
            }
            atomic_write_json(paths["checkpoint"], checkpoint)
        if records_done % 100 == 0 or records_done == metadata["records"]:
            print(
                f"[{split}] {records_done}/{metadata['records']} decisions, "
                f"{pairs_done}/{metadata['pairs']} pairs",
                flush=True,
            )
    return checkpoint


def summarize_predictions(
    groups: list[dict[str, Any]],
    predicted_positions: list[int],
    top3_positions: list[list[int]],
) -> dict[str, Any]:
    if not (len(groups) == len(predicted_positions) == len(top3_positions)):
        raise ValueError("Prediction lengths do not match groups")
    rows = []
    by_action: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for group, predicted, top3 in zip(groups, predicted_positions, top3_positions):
        row = {
            "count": int(group["count"]),
            "top1": predicted == group["gold_position"],
            "top3": group["gold_position"] in top3,
        }
        rows.append(row)
        by_action[str(group["action_type"])].append(row)

    def accuracy(items: list[dict[str, Any]], field: str) -> float:
        return sum(bool(item[field]) for item in items) / len(items) if items else 0.0

    multi = [row for row in rows if row["count"] > 1]
    expected_random_all = sum(1.0 / row["count"] for row in rows) / len(rows)
    expected_random_multi = (
        sum(1.0 / row["count"] for row in multi) / len(multi) if multi else 0.0
    )
    return {
        "decisions": len(rows),
        "top1_accuracy": accuracy(rows, "top1"),
        "top1_accuracy_n_gt_1": accuracy(multi, "top1"),
        "top3_accuracy": accuracy(rows, "top3"),
        "expected_random_top1": expected_random_all,
        "expected_random_top1_n_gt_1": expected_random_multi,
        "by_action_type": {
            action_type: {
                "total": len(action_rows),
                "top1_accuracy": accuracy(action_rows, "top1"),
            }
            for action_type, action_rows in sorted(by_action.items())
        },
    }


def evaluate_head(head, features, groups, torch) -> tuple[float, dict[str, Any]]:
    head.eval()
    losses: list[float] = []
    predicted: list[int] = []
    top3: list[list[int]] = []
    with torch.inference_mode():
        for group in groups:
            start = int(group["offset"])
            stop = start + int(group["count"])
            batch = torch.from_numpy(features[start:stop].astype("float32"))
            scores = head(batch).squeeze(-1)
            target = torch.tensor([int(group["gold_position"])])
            loss = torch.nn.functional.cross_entropy(scores.unsqueeze(0), target)
            ranking = torch.argsort(scores, descending=True).tolist()
            losses.append(float(loss))
            predicted.append(int(ranking[0]))
            top3.append([int(index) for index in ranking[:3]])
    return sum(losses) / len(losses), summarize_predictions(groups, predicted, top3)


def train_head(
    *,
    train_metadata: dict[str, Any],
    validation_metadata: dict[str, Any],
    cache_dir: Path,
    output_dir: Path,
    hidden_size: int,
    epochs: int,
    learning_rate: float,
    group_batch_size: int,
    seed: int,
) -> dict[str, Any]:
    import numpy as np
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    train_features = np.memmap(
        cache_paths(cache_dir, "train")["features"],
        dtype=np.float16,
        mode="r",
        shape=(train_metadata["pairs"], hidden_size),
    )
    validation_features = np.memmap(
        cache_paths(cache_dir, "validation")["features"],
        dtype=np.float16,
        mode="r",
        shape=(validation_metadata["pairs"], hidden_size),
    )
    train_groups = [group for group in train_metadata["groups"] if group["count"] > 1]
    head = torch.nn.Linear(hidden_size, 1)
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=0.01)
    best_accuracy = -1.0
    best_epoch = 0
    history = []
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "head.pt"

    for epoch in range(1, epochs + 1):
        head.train()
        order = list(range(len(train_groups)))
        random.Random(seed + epoch).shuffle(order)
        epoch_losses = []
        for batch_offset in range(0, len(order), group_batch_size):
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for group_index in order[batch_offset : batch_offset + group_batch_size]:
                group = train_groups[group_index]
                start = int(group["offset"])
                stop = start + int(group["count"])
                batch = torch.from_numpy(train_features[start:stop].astype("float32"))
                scores = head(batch).squeeze(-1)
                target = torch.tensor([int(group["gold_position"])])
                losses.append(torch.nn.functional.cross_entropy(scores.unsqueeze(0), target))
            loss = torch.stack(losses).mean()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach()))

        validation_loss, metrics = evaluate_head(head, validation_features, validation_metadata["groups"], torch)
        row = {
            "epoch": epoch,
            "train_loss": sum(epoch_losses) / len(epoch_losses),
            "validation_loss": validation_loss,
            "validation_metrics": metrics,
        }
        history.append(row)
        print(
            f"[head] epoch={epoch}/{epochs} train_loss={row['train_loss']:.4f} "
            f"val_top1_n_gt_1={metrics['top1_accuracy_n_gt_1']:.4f}",
            flush=True,
        )
        if metrics["top1_accuracy_n_gt_1"] > best_accuracy:
            best_accuracy = metrics["top1_accuracy_n_gt_1"]
            best_epoch = epoch
            temporary = best_path.with_suffix(".pt.tmp")
            torch.save(
                {
                    "state_dict": head.state_dict(),
                    "hidden_size": hidden_size,
                    "epoch": epoch,
                    "validation_metrics": metrics,
                },
                temporary,
            )
            os.replace(temporary, best_path)

    best = torch.load(best_path, map_location="cpu", weights_only=True)
    head.load_state_dict(best["state_dict"])
    validation_loss, metrics = evaluate_head(head, validation_features, validation_metadata["groups"], torch)
    return {
        "head": "linear",
        "hidden_size": hidden_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "group_batch_size": group_batch_size,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_validation_loss": validation_loss,
        "best_validation_metrics": metrics,
        "history": history,
        "artifact": str(best_path.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default=str(DEFAULT_TRAIN_FILE))
    parser.add_argument("--validation", default=str(DEFAULT_VALIDATION_FILE))
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--flush-every-records", type=int, default=25)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--group-batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scan-only", action="store_true")
    args = parser.parse_args()

    train_metadata = scan_dataset(args.train)
    validation_metadata = scan_dataset(args.validation)
    print(
        json.dumps(
            {
                "train_records": train_metadata["records"],
                "train_pairs": train_metadata["pairs"],
                "validation_records": validation_metadata["records"],
                "validation_pairs": validation_metadata["pairs"],
            },
            indent=2,
        ),
        flush=True,
    )
    if args.scan_only:
        return

    budget = ResourceBudget()
    budget.validate_batch_size(args.batch_size)
    guard = ResourceGuard(budget)
    guard.preflight()
    configure_conservative_process(budget)

    import torch
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    model_dir = Path(args.model_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    cache_dir = output_dir / "cache"
    signature = model_signature(model_dir)
    config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    config.reference_compile = False
    hidden_size = int(config.hidden_size)

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        config=config,
        dtype=torch.float16,
        local_files_only=True,
        attn_implementation="sdpa",
    ).to("cuda").eval()
    guard.wait_until_safe()

    extract_reports = {}
    for split, metadata in (("train", train_metadata), ("validation", validation_metadata)):
        extract_reports[split] = extract_split(
            split=split,
            metadata=metadata,
            cache_dir=cache_dir,
            hidden_size=hidden_size,
            signature=signature,
            tokenizer=tokenizer,
            model=model,
            torch=torch,
            guard=guard,
            batch_size=args.batch_size,
            max_length=args.max_length,
            flush_every_records=args.flush_every_records,
        )

    del model, tokenizer
    torch.cuda.empty_cache()
    guard.wait_until_safe()
    report = {
        "model": signature,
        "train": {key: value for key, value in train_metadata.items() if key != "groups"},
        "validation": {key: value for key, value in validation_metadata.items() if key != "groups"},
        "extraction": extract_reports,
        "resource_telemetry": guard.telemetry_summary(),
        "training": train_head(
            train_metadata=train_metadata,
            validation_metadata=validation_metadata,
            cache_dir=cache_dir,
            output_dir=output_dir,
            hidden_size=hidden_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            group_batch_size=args.group_batch_size,
            seed=args.seed,
        ),
    }
    atomic_write_json(output_dir / "report.json", report)
    print(json.dumps(report["training"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
