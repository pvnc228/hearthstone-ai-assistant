"""Benchmark a local ModernCE NLI model on frozen next-action records."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .resource_guard import (
    ResourceBudget,
    ResourceGuard,
    ResourceLimitExceeded,
    configure_conservative_process,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = Path(r"D:\models\ModernCE-large-nli")
DEFAULT_VALIDATION_FILE = PROJECT_ROOT / "data" / "processed" / "next_action_validation_chatml.jsonl"
DEFAULT_OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "next_action_baseline_modernce_validation.json"
DEFAULT_HYPOTHESIS = "Лучшее следующее действие: {}."
# This checkpoint's config.id2label is stale; its model card and semantic smoke use index 1.
DEFAULT_ENTAILMENT_INDEX = 1


def load_records(path: Path | str, limit: int | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8-sig") as source:
        for line in source:
            if not line.strip():
                continue
            records.append(json.loads(line))
            if limit is not None and len(records) >= limit:
                break
    if not records:
        raise ValueError(f"No records found in {path}")
    return records


def state_premise(record: dict[str, Any]) -> str:
    prompt = str(record["prompt"])
    marker = "\nДоступные действия:"
    if marker not in prompt:
        raise ValueError(f"Missing candidate marker in {record.get('decision_id', '<unknown>')}")
    return prompt.split(marker, 1)[0]


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def summarize_scores(scores: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(scores)
    successful = [row for row in rows if row["error"] is None]
    multi = [row for row in successful if row["candidate_count"] > 1]
    latencies = [float(row["latency_ms"]) for row in successful]
    by_action: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in successful:
        by_action[row["action_type"]].append(row)

    def rate(subset: list[dict[str, Any]], field: str) -> float:
        return sum(bool(row[field]) for row in subset) / len(subset) if subset else 0.0

    return {
        "attempted": len(rows),
        "successful": len(successful),
        "errors": len(rows) - len(successful),
        "top1_accuracy_all_attempts": sum(bool(row["top1_correct"]) for row in rows) / len(rows),
        "top1_accuracy_successful": rate(successful, "top1_correct"),
        "top1_accuracy_n_gt_1": rate(multi, "top1_correct"),
        "top3_accuracy_successful": rate(successful, "top3_correct"),
        "latency_ms": {
            "mean": round(statistics.mean(latencies), 3) if latencies else 0.0,
            "p50": round(statistics.median(latencies), 3) if latencies else 0.0,
            "p95": round(percentile(latencies, 0.95), 3),
            "max": round(max(latencies), 3) if latencies else 0.0,
        },
        "by_action_type": {
            action_type: {
                "total": len(action_rows),
                "top1_accuracy": rate(action_rows, "top1_correct"),
            }
            for action_type, action_rows in sorted(by_action.items())
        },
    }


def run_benchmark(
    records: Iterable[dict[str, Any]],
    *,
    model_dir: Path | str = DEFAULT_MODEL_DIR,
    hypothesis_template: str = DEFAULT_HYPOTHESIS,
    entailment_index: int = DEFAULT_ENTAILMENT_INDEX,
    batch_size: int = 2,
    max_length: int = 2048,
) -> dict[str, Any]:
    budget = ResourceBudget()
    budget.validate_batch_size(batch_size)
    guard = ResourceGuard(budget)
    guard.preflight()
    configure_conservative_process(budget)

    import torch
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this Python environment")
    if hypothesis_template.count("{}") != 1:
        raise ValueError("Hypothesis template must contain exactly one {} placeholder")

    model_path = Path(model_dir).resolve()
    started_loading = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    config.reference_compile = False
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        config=config,
        dtype=torch.float16,
        local_files_only=True,
        attn_implementation="sdpa",
    ).to("cuda").eval()
    torch.cuda.synchronize()
    load_ms = (time.perf_counter() - started_loading) * 1000

    warmup = tokenizer(
        ["A cat is a mammal."],
        ["A cat is an animal."],
        padding=True,
        return_tensors="pt",
    ).to("cuda")
    with torch.inference_mode():
        model(**warmup)
    torch.cuda.synchronize()
    guard.wait_until_safe()
    torch.cuda.reset_peak_memory_stats()

    scores: list[dict[str, Any]] = []
    total_pairs = 0
    truncated_pairs = 0
    max_untruncated_tokens = 0
    for record in records:
        candidates = list(record.get("candidates", []))
        row: dict[str, Any] = {
            "decision_id": str(record.get("decision_id", "")),
            "candidate_count": len(candidates),
            "chosen_candidate_id": record.get("chosen_candidate_id"),
            "predicted_candidate_id": None,
            "top1_correct": False,
            "top3_correct": False,
            "action_type": str(record.get("gold_action", {}).get("type", "UNKNOWN")),
            "latency_ms": 0.0,
            "error": None,
        }
        try:
            if not candidates:
                raise ValueError("No candidates")
            premise = state_premise(record)
            hypotheses = [hypothesis_template.format(candidate["description"]) for candidate in candidates]
            raw_lengths = tokenizer(
                [premise] * len(hypotheses),
                hypotheses,
                padding=False,
                truncation=False,
            )["input_ids"]
            pair_lengths = [len(tokens) for tokens in raw_lengths]
            max_untruncated_tokens = max(max_untruncated_tokens, max(pair_lengths))
            truncated_pairs += sum(length > max_length for length in pair_lengths)
            decision_scores: list[float] = []
            torch.cuda.synchronize()
            started = time.perf_counter()
            for offset in range(0, len(candidates), batch_size):
                chunk = hypotheses[offset : offset + batch_size]
                premises = [premise] * len(chunk)
                encoded = tokenizer(
                    premises,
                    chunk,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                ).to("cuda")
                with torch.inference_mode():
                    logits = model(**encoded).logits.float()
                    probabilities = torch.softmax(logits, dim=-1)[:, entailment_index]
                decision_scores.extend(probabilities.cpu().tolist())
                guard.after_batch()
            torch.cuda.synchronize()
            ranked = sorted(range(len(candidates)), key=decision_scores.__getitem__, reverse=True)
            predicted_id = candidates[ranked[0]]["id"]
            top3_ids = {candidates[index]["id"] for index in ranked[:3]}
            row["predicted_candidate_id"] = predicted_id
            row["top1_correct"] = predicted_id == record["chosen_candidate_id"]
            row["top3_correct"] = record["chosen_candidate_id"] in top3_ids
            total_pairs += len(candidates)
        except ResourceLimitExceeded:
            raise
        except torch.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            raise ResourceLimitExceeded("CUDA OOM: run aborted before continuing") from exc
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            started = time.perf_counter()
        row["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
        scores.append(row)

    return {
        "backend": "modernce_nli_cross_encoder",
        "model": str(model_path),
        "dtype": "float16",
        "device": torch.cuda.get_device_name(0),
        "config_id2label": model.config.id2label,
        "effective_entailment_index": entailment_index,
        "hypothesis_template": hypothesis_template,
        "batch_size": batch_size,
        "max_length": max_length,
        "load_ms": round(load_ms, 3),
        "pairs_scored": total_pairs,
        "truncated_pairs": truncated_pairs,
        "max_untruncated_tokens": max_untruncated_tokens,
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 1024 / 1024, 3),
        "resource_telemetry": guard.telemetry_summary(),
        "metrics": summarize_scores(scores),
        "scores": scores,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark local ModernCE on frozen next-action data")
    parser.add_argument("--input", default=str(DEFAULT_VALIDATION_FILE))
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_FILE))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--entailment-index", type=int, default=DEFAULT_ENTAILMENT_INDEX)
    parser.add_argument("--hypothesis-template", default=DEFAULT_HYPOTHESIS)
    args = parser.parse_args()

    records = load_records(args.input, args.limit)
    report = run_benchmark(
        records,
        model_dir=args.model_dir,
        hypothesis_template=args.hypothesis_template,
        entailment_index=args.entailment_index,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    report.update({"input": str(Path(args.input).resolve()), "records_requested": len(records)})
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "scores"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
