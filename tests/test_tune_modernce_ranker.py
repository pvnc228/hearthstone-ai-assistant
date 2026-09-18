import json

import pytest

from src.llm.tune_modernce_ranker import (
    ACTION_TYPES,
    assess_candidate_gates,
    candidate_type_ids,
    paired_game_bootstrap_ci,
    split_groups_by_game,
    sqrt_inverse_action_weights,
    tempered_action_weights,
)


def _group(game_id, action_type, offset):
    return {
        "decision_id": f"{game_id}:{offset}",
        "game_id": game_id,
        "offset": offset,
        "count": 2,
        "gold_position": 0,
        "action_type": action_type,
    }


def test_game_split_is_deterministic_and_disjoint():
    groups = [
        _group(f"game-{game_index}", "PLAY", game_index * 2 + decision)
        for game_index in range(10)
        for decision in range(2)
    ]

    fit_a, dev_a = split_groups_by_game(groups, dev_ratio=0.2, seed=42)
    fit_b, dev_b = split_groups_by_game(groups, dev_ratio=0.2, seed=42)

    assert fit_a == fit_b
    assert dev_a == dev_b
    assert {group["game_id"] for group in fit_a}.isdisjoint(
        {group["game_id"] for group in dev_a}
    )
    assert len({group["game_id"] for group in dev_a}) == 2


def test_sqrt_inverse_weights_upweight_rare_actions_and_average_to_one():
    groups = [
        *[_group(f"play-{index}", "PLAY", index * 2) for index in range(9)],
        _group("hero", "HERO_POWER", 100),
    ]

    weights = sqrt_inverse_action_weights(groups)

    assert weights["HERO_POWER"] > weights["PLAY"]
    weighted_mean = sum(weights[group["action_type"]] for group in groups) / len(groups)
    assert weighted_mean == pytest.approx(1.0)


def test_tempered_weights_stay_between_unweighted_and_full_balance():
    groups = [
        *[_group(f"play-{index}", "PLAY", index * 2) for index in range(9)],
        _group("hero", "HERO_POWER", 100),
    ]

    full = sqrt_inverse_action_weights(groups)
    tempered = tempered_action_weights(groups, exponent=0.5)

    assert 1.0 < tempered["HERO_POWER"] / tempered["PLAY"] < full["HERO_POWER"] / full["PLAY"]
    weighted_mean = sum(tempered[group["action_type"]] for group in groups) / len(groups)
    assert weighted_mean == pytest.approx(1.0)


def test_candidate_gate_rejects_rare_gain_that_collapses_play():
    baseline = {
        "by_action_type": {
            "PLAY": {"top1_accuracy_n_gt_1": 0.50},
            "END_TURN": {"top1_accuracy_n_gt_1": 0.02},
            "HERO_POWER": {"top1_accuracy_n_gt_1": 0.03},
        }
    }
    candidate = {
        "by_action_type": {
            "PLAY": {"top1_accuracy_n_gt_1": 0.39},
            "END_TURN": {"top1_accuracy_n_gt_1": 0.40},
            "HERO_POWER": {"top1_accuracy_n_gt_1": 0.50},
        }
    }

    rejected = assess_candidate_gates(candidate, baseline, ndcg_ci_lower=-0.005)
    candidate["by_action_type"]["PLAY"]["top1_accuracy_n_gt_1"] = 0.40
    accepted = assess_candidate_gates(candidate, baseline, ndcg_ci_lower=-0.005)

    assert rejected["eligible"] is False
    assert rejected["checks"]["play_retention"] is False
    assert accepted["eligible"] is True


def test_candidate_types_follow_feature_offsets(tmp_path):
    dataset = tmp_path / "records.jsonl"
    records = [
        {
            "decision_id": "game-a:1",
            "candidates": [{"type": "END_TURN"}, {"type": "HERO_POWER"}],
        },
        {
            "decision_id": "game-b:1",
            "candidates": [{"type": "PLAY"}, {"option_type": "ATTACK"}],
        },
    ]
    dataset.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    groups = [
        {"decision_id": "game-a:1", "offset": 0, "count": 2},
        {"decision_id": "game-b:1", "offset": 2, "count": 2},
    ]

    type_ids = candidate_type_ids(dataset, groups)

    assert type_ids == [
        ACTION_TYPES.index("END_TURN"),
        ACTION_TYPES.index("HERO_POWER"),
        ACTION_TYPES.index("PLAY"),
        ACTION_TYPES.index("ATTACK"),
    ]


def test_paired_bootstrap_detects_consistent_improvement():
    baseline = {f"game-{index}": 0.2 + index * 0.01 for index in range(20)}
    candidate = {game_id: score + 0.1 for game_id, score in baseline.items()}

    delta, lower, upper = paired_game_bootstrap_ci(
        baseline, candidate, seed=42, samples=500
    )

    assert delta == pytest.approx(0.1)
    assert lower > 0
    assert upper > 0
