import json

import pytest

from src.llm.train_modernce_ranker import scan_dataset, summarize_predictions


def _record(decision_id, candidate_ids, chosen_id, action_type="PLAY"):
    return {
        "decision_id": decision_id,
        "game_id": decision_id.split(":", 1)[0],
        "candidates": [
            {"id": candidate_id, "description": f"action {candidate_id}"}
            for candidate_id in candidate_ids
        ],
        "chosen_candidate_id": chosen_id,
        "gold_action": {"type": action_type},
        "prompt": "state\nДоступные действия:\n[1] action",
    }


def test_scan_dataset_builds_contiguous_groups(tmp_path):
    dataset = tmp_path / "records.jsonl"
    records = [
        _record("game-a:1", [10, 11], 11),
        _record("game-b:1", [20, 21, 22], 20, "END_TURN"),
    ]
    dataset.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    metadata = scan_dataset(dataset)

    assert metadata["records"] == 2
    assert metadata["pairs"] == 5
    assert metadata["groups"] == [
        {
            "decision_id": "game-a:1",
            "game_id": "game-a",
            "offset": 0,
            "count": 2,
            "gold_position": 1,
            "action_type": "PLAY",
        },
        {
            "decision_id": "game-b:1",
            "game_id": "game-b",
            "offset": 2,
            "count": 3,
            "gold_position": 0,
            "action_type": "END_TURN",
        },
    ]


def test_scan_dataset_rejects_missing_gold_candidate(tmp_path):
    dataset = tmp_path / "bad.jsonl"
    dataset.write_text(json.dumps(_record("game-a:1", [10, 11], 99)) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="chosen candidate"):
        scan_dataset(dataset)


def test_summarize_predictions_reports_multi_candidate_accuracy():
    groups = [
        {"count": 1, "gold_position": 0, "action_type": "END_TURN"},
        {"count": 2, "gold_position": 1, "action_type": "PLAY"},
        {"count": 4, "gold_position": 3, "action_type": "ATTACK"},
    ]

    summary = summarize_predictions(groups, predicted_positions=[0, 1, 0], top3_positions=[[0], [1, 0], [0, 1, 2]])

    assert summary["top1_accuracy"] == pytest.approx(2 / 3)
    assert summary["top1_accuracy_n_gt_1"] == pytest.approx(1 / 2)
    assert summary["top3_accuracy"] == pytest.approx(2 / 3)
    assert summary["by_action_type"]["PLAY"]["top1_accuracy"] == 1.0
