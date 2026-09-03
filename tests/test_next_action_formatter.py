import json

import pytest

from src.llm.next_action_formatter import (
    create_split_manifest,
    format_next_action_dataset,
    load_and_validate_manifest,
)
from src.llm.train_qlora import validate_next_action_dataset


def _record(game_id: str, date: str, index: int) -> dict:
    return {
        "schema_version": 2,
        "game_id": game_id,
        "decision_id": f"{game_id}:option:0001",
        "replay_file": f"1200-{date}.hdtreplay",
        "turn_number": 1,
        "state": {
            "turn": 1,
            "active_player_id": 1,
            "mana": 1,
            "max_mana": 1,
            "friendly_hero": {"name": "Mage", "health": 30, "armor": 0},
            "opponent_hero": {"name": "Hunter", "health": 30, "armor": 0},
            "hand": [],
            "friendly_board": [],
            "opponent_board": [],
            "hero_power": {},
            "friendly_locations": [],
            "opponent_locations": [],
        },
        "candidates": [
            {
                "id": 1,
                "type": "END_TURN",
                "description": "Завершить ход",
            },
            {
                "id": 2,
                "type": "PLAY",
                "description": "Карта",
            },
        ],
        "gold_action": {"type": "END_TURN"},
        "chosen_candidate_id": 1,
    }


def test_freeze_and_format_splits_without_game_leakage(tmp_path):
    source = tmp_path / "accepted.jsonl"
    source.write_text(
        "".join(json.dumps(_record(f"game-{i}", f"{(i % 28) + 1:02d}0826", i), ensure_ascii=False) + "\n" for i in range(20)),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.json"
    output_dir = tmp_path / "formatted"

    manifest = create_split_manifest(source, manifest_path)
    counts = format_next_action_dataset(source, output_dir, manifest_path)

    split_sets = [set(manifest["splits"][name]["game_ids"]) for name in manifest["splits"]]
    assert sum(len(values) for values in split_sets) == 20
    assert sum(counts.values()) == 20
    assert all(not left & right for i, left in enumerate(split_sets) for right in split_sets[i + 1 :])
    assert validate_next_action_dataset(output_dir / "next_action_train_chatml.jsonl")["records"] == counts["train"]
    assert (output_dir / "next_action_temporal_holdout_chatml.jsonl").exists()


def test_manifest_is_frozen_against_source_drift(tmp_path):
    source = tmp_path / "accepted.jsonl"
    source.write_text(
        "".join(json.dumps(_record(f"game-{i}", f"0{i}0826", i)) + "\n" for i in range(4)),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.json"
    create_split_manifest(source, manifest_path)

    with pytest.raises(FileExistsError):
        create_split_manifest(source, manifest_path)

    source.write_text(json.dumps(_record("game-new", "020826", 2)) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash differs"):
        load_and_validate_manifest(source, manifest_path)
