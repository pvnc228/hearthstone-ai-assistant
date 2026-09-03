"""Evaluate next-action responses against frozen schema-v2 records."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .next_action_contract import NEXT_ACTION_SYSTEM_PROMPT, parse_next_action_response
from .ollama_client import OllamaClient


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEST_FILE = PROJECT_ROOT / "data" / "processed" / "next_action_test_chatml.jsonl"
DEFAULT_OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "next_action_baseline_ollama.json"


def score_response(record: dict[str, Any], raw_response: str, latency_ms: float = 0.0) -> dict[str, Any]:
    candidate_ids = [candidate["id"] for candidate in record.get("candidates", [])]
    parsed = parse_next_action_response(raw_response, candidate_ids)
    chosen_id = record["chosen_candidate_id"]
    chosen = next(candidate for candidate in record["candidates"] if candidate["id"] == chosen_id)
    return {
        "format_valid": parsed.format_valid,
        "candidate_exists": parsed.candidate_exists,
        "top1_correct": parsed.candidate_id == chosen_id,
        "predicted_candidate_id": parsed.candidate_id,
        "chosen_candidate_id": chosen_id,
        "action_type": chosen.get("type", "UNKNOWN"),
        "latency_ms": round(latency_ms, 3),
    }


def summarize_scores(scores: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(scores)
    by_action: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_action[row["action_type"]].append(row)
    latencies = [float(row["latency_ms"]) for row in rows]

    def rate(field: str, subset: list[dict[str, Any]] | None = None) -> float:
        selected = subset if subset is not None else rows
        return sum(bool(row[field]) for row in selected) / len(selected) if selected else 0.0

    return {
        "total": len(rows),
        "top1_accuracy": rate("top1_correct"),
        "format_valid_rate": rate("format_valid"),
        "candidate_exists_rate": rate("candidate_exists"),
        "latency_ms": {
            "mean": round(statistics.mean(latencies), 3) if latencies else 0.0,
            "p50": round(statistics.median(latencies), 3) if latencies else 0.0,
            "max": round(max(latencies), 3) if latencies else 0.0,
        },
        "by_action_type": {
            action_type: {
                "total": len(action_rows),
                "top1_accuracy": rate("top1_correct", action_rows),
                "format_valid_rate": rate("format_valid", action_rows),
                "candidate_exists_rate": rate("candidate_exists", action_rows),
            }
            for action_type, action_rows in sorted(by_action.items())
        },
    }


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


def run_ollama_benchmark(
    records: Iterable[dict[str, Any]],
    *,
    model: str | None = None,
    timeout: float = 10.0,
) -> dict[str, Any]:
    """Runs the frozen prompts through the same Ollama runtime contract."""
    client = OllamaClient(model=model, timeout=timeout)
    scores: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for record in records:
        started = time.perf_counter()
        try:
            raw_response = client.generate(
                prompt=record["prompt"],
                system=NEXT_ACTION_SYSTEM_PROMPT,
                temperature=0.0,
                max_tokens=32,
            )
            scores.append(score_response(record, raw_response, (time.perf_counter() - started) * 1000))
        except Exception as exc:
            errors.append({"decision_id": str(record.get("decision_id", "")), "error": f"{type(exc).__name__}: {exc}"})
    return {
        "backend": "ollama",
        "model": client.model,
        "metrics": summarize_scores(scores),
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark a base Ollama model on frozen next-action test data")
    parser.add_argument("--input", default=str(DEFAULT_TEST_FILE))
    parser.add_argument("--model", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_FILE))
    args = parser.parse_args()
    records = load_records(args.input, args.limit)
    report = run_ollama_benchmark(records, model=args.model, timeout=args.timeout)
    report["status"] = "blocked" if report["errors"] else "complete"
    report.update({"input": str(Path(args.input).resolve()), "records_requested": len(records)})
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["errors"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
