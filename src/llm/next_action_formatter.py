"""Freeze game-level schema-v2 splits and format them for QLoRA/evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from .dataset_formatter import to_chatml
from .next_action_contract import (
    NEXT_ACTION_SYSTEM_PROMPT,
    build_next_action_prompt,
    format_next_action_completion,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "train_next_actions.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_MANIFEST_FILE = DEFAULT_OUTPUT_DIR / "next_action_split_manifest_v1.json"
_REPLAY_TIMESTAMP_RE = re.compile(r"""(?P<time>\d{4})-(?P<date>\d{6})\.hdtreplay$""", re.IGNORECASE)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _replay_datetime(replay_file: str) -> datetime:
    match = _REPLAY_TIMESTAMP_RE.search(Path(replay_file).name)
    if not match:
        return datetime.min
    try:
        return datetime.strptime(f"{match['time']}-{match['date']}", "%H%M-%d%m%y")
    except ValueError:
        return datetime.min


def _load_records(input_path: Path) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    records: list[dict[str, Any]] = []
    games: dict[str, list[dict[str, Any]]] = {}
    with input_path.open("r", encoding="utf-8-sig") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("schema_version") != 2:
                raise ValueError(f"{input_path}:{line_number}: expected schema_version=2")
            game_id = str(record.get("game_id") or "")
            if not game_id:
                raise ValueError(f"{input_path}:{line_number}: missing game_id")
            if "chosen_candidate_id" not in record:
                raise ValueError(f"{input_path}:{line_number}: missing chosen_candidate_id")
            records.append(record)
            games.setdefault(game_id, []).append(record)
    return records, games


def _game_sort_key(game_id: str, games: Mapping[str, list[dict[str, Any]]]) -> tuple[datetime, str, str]:
    replay_file = str(games[game_id][0].get("replay_file") or "")
    return (_replay_datetime(replay_file), replay_file.casefold(), game_id)


def _split_game_ids(
    games: Mapping[str, list[dict[str, Any]]],
    *,
    seed: int,
    validation_ratio: float,
    test_ratio: float,
    temporal_holdout_ratio: float,
) -> dict[str, list[str]]:
    if not games:
        raise ValueError("Cannot freeze splits for an empty dataset")
    if min(validation_ratio, test_ratio, temporal_holdout_ratio) <= 0:
        raise ValueError("All split ratios must be positive")
    ordered = sorted(games, key=lambda game_id: _game_sort_key(game_id, games))
    temporal_count = max(1, math.ceil(len(ordered) * temporal_holdout_ratio))
    temporal = ordered[-temporal_count:]
    remaining = ordered[:-temporal_count]

    rng = random.Random(seed)
    rng.shuffle(remaining)
    test_count = max(1, math.floor(len(remaining) * test_ratio))
    validation_count = max(1, math.floor(len(remaining) * validation_ratio))
    test = remaining[:test_count]
    validation = remaining[test_count : test_count + validation_count]
    train = remaining[test_count + validation_count :]
    if not train:
        raise ValueError("Split ratios leave no training games")
    return {
        "train_game_ids": sorted(train),
        "validation_game_ids": sorted(validation),
        "test_game_ids": sorted(test),
        "temporal_holdout_game_ids": sorted(temporal),
    }


def create_split_manifest(
    input_file: Path | str = DEFAULT_INPUT_FILE,
    manifest_file: Path | str = DEFAULT_MANIFEST_FILE,
    *,
    seed: int = 42,
    validation_ratio: float = 0.10,
    test_ratio: float = 0.10,
    temporal_holdout_ratio: float = 0.10,
    replace: bool = False,
) -> dict[str, Any]:
    """Creates an immutable-by-default manifest keyed by the source hash."""
    input_path = Path(input_file)
    manifest_path = Path(manifest_file)
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if manifest_path.exists() and not replace:
        raise FileExistsError(f"Split manifest already exists: {manifest_path}; use --replace explicitly")
    _, games = _load_records(input_path)
    splits = _split_game_ids(
        games,
        seed=seed,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
        temporal_holdout_ratio=temporal_holdout_ratio,
    )
    all_split_ids = [game_id for ids in splits.values() for game_id in ids]
    if len(all_split_ids) != len(set(all_split_ids)) or set(all_split_ids) != set(games):
        raise RuntimeError("Split construction produced overlapping or missing game IDs")

    manifest = {
        "manifest_version": 1,
        "schema_version": 2,
        "source": {
            "dataset": str(input_path.resolve()),
            "sha256": sha256_file(input_path),
            "records": sum(len(rows) for rows in games.values()),
            "games": len(games),
        },
        "policy": {
            "seed": seed,
            "validation_ratio": validation_ratio,
            "test_ratio": test_ratio,
            "temporal_holdout_ratio": temporal_holdout_ratio,
            "temporal_key": "replay_file HHMM-DDMMYY timestamp; game_id tie-breaker",
            "quarantine_policy": "excluded; accepted schema-v2 only",
        },
        "splits": {
            name: {
                "games": len(game_ids),
                "records": sum(len(games[game_id]) for game_id in game_ids),
                "game_ids": game_ids,
            }
            for name, game_ids in splits.items()
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_manifest = manifest_path.with_name(f"{manifest_path.name}.{uuid4().hex}.tmp")
    try:
        temporary_manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_manifest.replace(manifest_path)
    finally:
        if temporary_manifest.exists():
            temporary_manifest.unlink()
    return manifest


def load_and_validate_manifest(
    input_file: Path | str = DEFAULT_INPUT_FILE,
    manifest_file: Path | str = DEFAULT_MANIFEST_FILE,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    input_path = Path(input_file)
    manifest_path = Path(manifest_file)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Frozen split manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if manifest.get("schema_version") != 2 or manifest.get("manifest_version") != 1:
        raise ValueError("Unsupported next-action split manifest version")
    records, games = _load_records(input_path)
    source = manifest.get("source", {})
    if source.get("sha256") != sha256_file(input_path):
        raise ValueError("Dataset hash differs from frozen split manifest; regenerate explicitly")
    if source.get("games") != len(games) or source.get("records") != len(records):
        raise ValueError("Dataset counts differ from frozen split manifest")
    split_map = {
        name: list((manifest.get("splits", {}).get(name) or {}).get("game_ids", []))
        for name in ("train_game_ids", "validation_game_ids", "test_game_ids", "temporal_holdout_game_ids")
    }
    flattened = [game_id for values in split_map.values() for game_id in values]
    if len(flattened) != len(set(flattened)) or set(flattened) != set(games):
        raise ValueError("Frozen split manifest has overlapping or missing game IDs")
    return manifest, records


def _format_record(record: Mapping[str, Any], split: str) -> dict[str, Any]:
    prompt = build_next_action_prompt(record["state"], record["candidates"])
    completion = format_next_action_completion(record["chosen_candidate_id"])
    formatted = dict(record)
    formatted.update(
        {
            "dataset_contract": "next_action_v2",
            "split": split,
            "prompt": prompt,
            "completion": completion,
            "messages": to_chatml(prompt, completion, system_prompt=NEXT_ACTION_SYSTEM_PROMPT)["messages"],
        }
    )
    return formatted


def format_next_action_dataset(
    input_file: Path | str = DEFAULT_INPUT_FILE,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    manifest_file: Path | str = DEFAULT_MANIFEST_FILE,
) -> dict[str, int]:
    """Writes train/validation/test/temporal ChatML files from the frozen manifest."""
    manifest, records = load_and_validate_manifest(input_file, manifest_file)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    game_to_split: dict[str, str] = {}
    for manifest_name, split_name in (
        ("train_game_ids", "train"),
        ("validation_game_ids", "validation"),
        ("test_game_ids", "test"),
        ("temporal_holdout_game_ids", "temporal_holdout"),
    ):
        for game_id in manifest["splits"][manifest_name]["game_ids"]:
            game_to_split[game_id] = split_name

    counts: dict[str, int] = {name: 0 for name in ("train", "validation", "test", "temporal_holdout")}
    temporary_token = uuid4().hex
    temporary_paths = {
        split: output_path / f"next_action_{split}_chatml.jsonl.{temporary_token}.tmp"
        for split in counts
    }
    try:
        handles = {
            split: temporary_paths[split].open("w", encoding="utf-8", newline="\n")
            for split in counts
        }
        try:
            for record in records:
                split = game_to_split[record["game_id"]]
                handles[split].write(
                    json.dumps(_format_record(record, split), ensure_ascii=False, separators=(",", ":")) + "\n"
                )
                counts[split] += 1
        finally:
            for handle in handles.values():
                handle.close()
        for split, temporary_path in temporary_paths.items():
            temporary_path.replace(output_path / f"next_action_{split}_chatml.jsonl")
    finally:
        for temporary_path in temporary_paths.values():
            if temporary_path.exists():
                temporary_path.unlink()
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze and format schema-v2 next-action data")
    parser.add_argument("--input", default=str(DEFAULT_INPUT_FILE))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST_FILE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--replace-manifest", action="store_true")
    args = parser.parse_args()
    manifest_path = Path(args.manifest)
    if not manifest_path.exists() or args.replace_manifest:
        create_split_manifest(args.input, args.manifest, replace=args.replace_manifest)
    print(format_next_action_dataset(args.input, args.output_dir, args.manifest))


if __name__ == "__main__":
    main()
