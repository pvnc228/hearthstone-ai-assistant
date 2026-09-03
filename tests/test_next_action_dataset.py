import json

import pytest

from src.parser import OptionDecision, ReplayOptionCandidate, TurnSnapshot
from src.parser.next_action_dataset import (
    _audit_written_dataset,
    build_next_action_dataset,
    validate_option_decision,
)


def snapshot(*, mana=0):
    return TurnSnapshot(
        turn_number=3,
        active_player_id=1,
        active_player_name="HappyBread#21597",
        is_friendly_turn=True,
        friendly_mana=mana,
        friendly_max_mana=3,
        friendly_hero={"entity_id": 2, "health": 30, "armor": 0, "attack": 0, "can_attack": False},
        opponent_hero={"entity_id": 3, "health": 30, "armor": 0},
        friendly_hand=[],
        friendly_board=[],
        opponent_board=[],
        friendly_locations=[],
        opponent_locations=[],
        friendly_secrets=[],
        opponent_secrets_count=0,
        opponent_hand_count=0,
        hero_power={},
    )


def candidate(**overrides):
    values = {
        "candidate_id": 0,
        "option_id": 0,
        "option_type": "END_TURN",
        "action_type": "END_TURN",
        "entity_id": None,
        "entity_name": "End turn",
        "entity_card_id": "",
        "controller_id": 0,
        "description": "End turn",
    }
    values.update(overrides)
    return ReplayOptionCandidate(**values)


def decision(candidates, **overrides):
    values = {
        "sequence": 1,
        "options_id": 7,
        "snapshot": snapshot(),
        "candidates": candidates,
        "selected_option": 0,
        "selected_sub_option": -1,
        "selected_target": None,
        "selected_position": 0,
    }
    values.update(overrides)
    return OptionDecision(**values)


def test_validate_option_decision_accepts_exact_legal_selection():
    attack = candidate(
        candidate_id=1,
        option_id=4,
        option_type="POWER",
        action_type="ATTACK",
        entity_id=10,
        entity_name="River Crocolisk",
        entity_card_id="TEST_ATTACKER",
        controller_id=1,
        target_entity_id=3,
        target_name="Enemy hero",
        target_card_id="HERO_08",
        description="River Crocolisk -> Enemy hero",
    )
    result = validate_option_decision(
        decision(
            [candidate(), attack],
            selected_option=4,
            selected_target=3,
        )
    )

    assert result.accepted is True
    assert result.chosen_candidate_id == 1
    assert result.match_count == 1


def test_validate_option_decision_quarantines_unresolved_legal_alternative():
    unresolved = candidate(
        candidate_id=1,
        option_id=2,
        option_type="POWER",
        action_type="PLAY",
        entity_id=20,
        entity_name="Unknown Entity 20",
        entity_card_id="",
        controller_id=1,
    )

    result = validate_option_decision(decision([candidate(), unresolved]))

    assert result.accepted is False
    assert result.reason == "unresolved_legal_candidate"


def test_validate_option_decision_accepts_resolved_sub_option_without_target_cross_product():
    discover = candidate(
        candidate_id=1,
        option_id=2,
        option_type="POWER",
        action_type="PLAY",
        entity_id=20,
        entity_name="Discover card",
        entity_card_id="TEST_DISCOVER",
        controller_id=1,
        sub_option_id=0,
        sub_entity_id=21,
        sub_entity_name="Choice",
        sub_entity_card_id="TEST_CHOICE",
    )

    result = validate_option_decision(
        decision([candidate(), discover], selected_option=2, selected_sub_option=0)
    )

    assert result.accepted is True
    assert result.chosen_candidate_id == 1


def test_validate_option_decision_quarantines_sub_option_target_cross_product():
    discover = candidate(
        candidate_id=1,
        option_id=2,
        option_type="POWER",
        action_type="PLAY",
        entity_id=20,
        entity_name="Choose card",
        entity_card_id="TEST_CHOOSE",
        controller_id=1,
        target_entity_id=3,
        target_name="Enemy hero",
        target_card_id="HERO_08",
        sub_option_id=0,
        sub_entity_id=21,
        sub_entity_name="Choice",
        sub_entity_card_id="TEST_CHOICE",
    )

    result = validate_option_decision(
        decision(
            [candidate(), discover],
            selected_option=2,
            selected_sub_option=0,
            selected_target=3,
        )
    )

    assert result.accepted is False
    assert result.reason == "suboption_target_cross_product_unproven"


def test_validate_option_decision_matches_board_position():
    play = candidate(
        candidate_id=1,
        option_id=2,
        option_type="POWER",
        action_type="PLAY",
        entity_id=20,
        entity_name="River Crocolisk",
        entity_card_id="CS2_120",
        entity_card_type=4,
        controller_id=1,
        position=1,
    )

    accepted = validate_option_decision(
        decision([candidate(), play], selected_option=2, selected_position=1)
    )
    rejected = validate_option_decision(
        decision([candidate(), play], selected_option=2, selected_position=2)
    )

    assert accepted.accepted is True
    assert accepted.chosen_candidate_id == 1
    assert rejected.reason == "selected_position_outside_derived_range"


def test_validate_option_decision_quarantines_unreliable_mana_on_any_candidate():
    chosen = candidate(candidate_id=1)
    discounted_alternative = candidate(
        candidate_id=2,
        option_id=2,
        option_type="POWER",
        action_type="PLAY",
        entity_id=20,
        entity_name="Discounted card",
        entity_card_id="TEST_DISCOUNTED",
        controller_id=1,
        mana_cost=8,
    )

    result = validate_option_decision(decision([chosen, discounted_alternative], snapshot=snapshot(mana=3)))

    assert result.accepted is False
    assert result.reason == "candidate_mana_cost_mismatch"


def test_validate_option_decision_quarantines_unproven_tradeable_option():
    tradeable = candidate(
        candidate_id=1,
        option_id=1,
        option_type="POWER",
        action_type="PLAY",
        entity_id=20,
        entity_name="Tradeable minion",
        entity_card_id="TEST_TRADEABLE",
        controller_id=1,
        is_tradeable=True,
        position=1,
    )

    result = validate_option_decision(decision([candidate(), tradeable]))

    assert result.accepted is False
    assert result.reason == "tradeable_option_semantics_unproven"


def test_serialized_audit_checks_mana_on_every_candidate(tmp_path):
    output = tmp_path / "accepted.jsonl"
    output.write_text(
        json.dumps(
            {
                "decision_id": "game:option:0001",
                "chosen_candidate_id": 1,
                "state": {"active_player_id": 1, "mana": 3},
                "gold_action": {
                    "option_id": 0,
                    "sub_option_id": -1,
                    "target_id": None,
                    "position": 0,
                    "type": "END_TURN",
                },
                "candidates": [
                    {
                        "id": 1,
                        "option_id": 0,
                        "sub_option_id": -1,
                        "target_id": None,
                        "position": 0,
                        "type": "END_TURN",
                        "mana_cost": 0,
                        "legality": "power_log_end_turn",
                    },
                    {
                        "id": 2,
                        "option_id": 2,
                        "sub_option_id": -1,
                        "target_id": None,
                        "position": 0,
                        "type": "PLAY",
                        "mana_cost": 8,
                        "legality": "power_log_error_none",
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert _audit_written_dataset(output) == {"candidate_mana_violation": 1}


def test_build_refuses_a_shared_output_lock_even_with_another_report(tmp_path):
    output = tmp_path / "accepted.jsonl"
    quarantine = tmp_path / "quarantine.jsonl"
    report = tmp_path / "report.json"
    lock = output.with_suffix(output.suffix + ".lock")
    lock.write_text("123", encoding="ascii")

    with pytest.raises(RuntimeError, match="Another build is active"):
        build_next_action_dataset(
            output_path=output,
            quarantine_path=quarantine,
            report_path=report,
            max_replays=0,
        )


def test_build_rejects_an_artifact_path_that_collides_with_a_lock(tmp_path):
    output = tmp_path / "accepted.jsonl"

    with pytest.raises(ValueError, match="collides"):
        build_next_action_dataset(
            output_path=output,
            quarantine_path=output.with_suffix(output.suffix + ".lock"),
            report_path=tmp_path / "report.json",
            max_replays=0,
        )
