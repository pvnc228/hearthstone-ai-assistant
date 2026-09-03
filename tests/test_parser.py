"""
Unit and integration tests for Log Parser, State Tracker, Replay Reader, and Dataset Generator.
"""

import json
import os
import tempfile
from pathlib import Path
import pytest

from src.parser import (
    Entity,
    GameStateTracker,
    PowerEvent,
    format_turn_completion,
    format_turn_prompt,
    generate_dataset,
    load_deck_stats_index,
    parse_entity_ref,
    parse_power_log_lines,
    parse_replay_file,
)
from src.card_db import CardDatabase
from src.parser.replay_reader import _iter_decoded_log_lines


def test_parse_entity_ref():
    raw1 = "[entityName=Рыбалка id=19 zone=HAND zonePos=1 cardId=TSC_916 player=1]"
    res1 = parse_entity_ref(raw1)
    assert res1["id"] == 19
    assert res1["cardId"] == "TSC_916"
    assert res1["zone"] == "HAND"
    assert res1["player"] == 1
    assert res1["entityName"] == "Рыбалка"

    raw2 = "98"
    res2 = parse_entity_ref(raw2)
    assert res2["id"] == 98

    raw3 = "HappyBread#21597"
    res3 = parse_entity_ref(raw3)
    assert res3["name"] == "HappyBread#21597"


def test_replay_reader_expands_escaped_newline_export():
    raw = [b"D 00:00:00.0000000 FIRST\\nD 00:00:00.0000000 SECOND\\n"]

    assert list(_iter_decoded_log_lines(raw)) == [
        "D 00:00:00.0000000 FIRST\n",
        "D 00:00:00.0000000 SECOND\n",
    ]


def test_option_oracle_creates_lazy_snapshot_without_transition_marker():
    def ev(etype, **data):
        return type("E", (), {"event_type": etype, "data": data, "raw_line": ""})()

    tracker = GameStateTracker(card_db=CardDatabase(auto_load=True))
    tracker.current_turn = 1
    tracker.active_player_id = 1
    tracker.friendly_player_id = 1
    tracker.process_event(ev("OPTIONS_START", options_id=76))
    tracker.process_event(
        ev(
            "OPTION",
            option_id=0,
            option_type="END_TURN",
            main_entity={},
            error="INVALID",
            error_param="",
        )
    )
    tracker.process_event(
        ev(
            "SEND_OPTION",
            selected_option=0,
            selected_sub_option=-1,
            selected_target=0,
            selected_position=0,
        )
    )

    assert len(tracker.option_decisions) == 1
    assert tracker.option_decisions[0].snapshot.turn_number == 1


def test_parse_power_log_events():
    sample_lines = [
        "D 00:20:49.4982287 GameState.DebugPrintGame() - PlayerID=1, PlayerName=HappyBread#21597",
        "D 00:20:49.4982287 GameState.DebugPrintGame() - PlayerID=2, PlayerName=Enemy#1234",
        "D 00:20:49.4982287 GameState.DebugPrintPower() - CREATE_GAME",
        "D 00:20:49.4982287 GameState.DebugPrintPower() -     GameEntity EntityID=1",
        "D 00:20:49.4982287 GameState.DebugPrintPower() -     Player EntityID=2 PlayerID=1",
        "D 00:20:49.4982287 GameState.DebugPrintPower() -     Player EntityID=3 PlayerID=2",
        "D 00:20:49.4982287 GameState.DebugPrintPower() - TAG_CHANGE Entity=1 tag=TURN value=1",
        "D 00:20:49.4982287 GameState.DebugPrintPower() - TAG_CHANGE Entity=HappyBread#21597 tag=CURRENT_PLAYER value=1",
        "D 00:20:49.4982287 GameState.DebugPrintPower() - BLOCK_START BlockType=PLAY Entity=[entityName=Монетка id=71 zone=HAND cardId=ETC_COIN1 player=1]",
        "D 00:20:49.4982287 GameState.DebugPrintPower() - BLOCK_END",
    ]

    events = list(parse_power_log_lines(sample_lines))
    assert len(events) == 10
    assert events[0].event_type == "PLAYER_NAME"
    assert events[0].data["player_name"] == "HappyBread#21597"
    assert events[2].event_type == "CREATE_GAME"
    assert events[6].event_type == "TAG_CHANGE"
    assert events[8].event_type == "BLOCK_START"
    assert events[8].data["block_type"] == "PLAY"
    assert events[8].data["entity"]["id"] == 71


def test_block_start_real_world_formats():
    """Regression: real Power.log BLOCK_START lines (with Target, nested brackets, TAG_NOT_SET)."""
    lines = [
        # Target as bracketed entity + trailing TriggerKeyword=TAG_NOT_SET
        "D 13:40:12.1234567 GameState.DebugPrintPower() - BLOCK_START BlockType=PLAY Entity=[entityName=Монетка id=71 zone=HAND cardId=ETC_COIN1 player=1] EffectCardId= EffectIndex=-1 Target=[entityName=Герой id=64 zone=PLAY zonePos=0 cardId=HERO_08 player=2] SubOption=-1 TriggerKeyword=TAG_NOT_SET",
        # Target=0 (no target) — must parse entity, target resolves to id 0
        "D 13:40:12.1234567 GameState.DebugPrintPower() - BLOCK_START BlockType=PLAY Entity=[entityName=Огненный шар id=18 zone=HAND cardId=CS2_029 player=1] EffectCardId= EffectIndex=-1 Target=0 SubOption=-1 TriggerKeyword=TAG_NOT_SET",
        # Nested brackets inside entityName (UNKNOWN ENTITY [cardType=INVALID])
        "D 13:58:48.8491185 GameState.DebugPrintPower() - BLOCK_START BlockType=TRIGGER Entity=[entityName=UNKNOWN ENTITY [cardType=INVALID] id=10 zone=DECK zonePos=0 cardId= player=2] EffectCardId=System.Collections.Generic.List`1[System.String] EffectIndex=-1 Target=0 SubOption=-1 TriggerKeyword=TAG_NOT_SET",
        # Player name refs
        "D 13:40:12.1234567 GameState.DebugPrintPower() - BLOCK_START BlockType=PLAY Entity=HappyBread#21597 EffectCardId= EffectIndex=-1 Target=Enemy#1234 SubOption=-1 TriggerKeyword=TAG_NOT_SET",
    ]

    events = [e for e in parse_power_log_lines(lines) if e.event_type == "BLOCK_START"]
    assert len(events) == 4

    e0 = events[0].data
    assert e0["entity"]["id"] == 71
    assert e0["target"]["id"] == 64
    assert e0["sub_option"] == -1

    e1 = events[1].data
    assert e1["entity"]["id"] == 18
    assert e1["target"].get("id") == 0  # explicit no-target

    e2 = events[2].data
    assert e2["entity"]["id"] == 10  # nested brackets don't break id extraction
    assert e2["entity"]["entityName"] == "UNKNOWN ENTITY [cardType=INVALID]"

    e3 = events[3].data
    assert e3["entity"]["name"] == "HappyBread#21597"
    assert e3["target"]["name"] == "Enemy#1234"


def test_parse_options_and_selected_action():
    lines = [
        "D 15:57:31.3679283 GameState.DebugPrintOptions() - id=76",
        "D 15:57:31.3679283 GameState.DebugPrintOptions() -   option 0 type=END_TURN mainEntity= error=INVALID errorParam=",
        "D 15:57:31.3679283 GameState.DebugPrintOptions() -   option 4 type=POWER mainEntity=[entityName=Бандит id=146 zone=PLAY zonePos=3 cardId=WW_051t player=1] error=NONE errorParam=",
        "D 15:57:31.3679283 GameState.DebugPrintOptions() -     target 0 entity=[entityName=Андуин Пророк id=62 zone=PLAY zonePos=0 cardId=HERO_09d player=2] error=NONE errorParam=",
        "D 15:57:31.3679283 GameState.DebugPrintOptions() -     target 1 entity=[entityName=Бандит id=146 zone=PLAY zonePos=3 cardId=WW_051t player=1] error=REQ_ENEMY_TARGET errorParam=",
        "D 15:57:32.8908120 GameState.SendOption() - selectedOption=4 selectedSubOption=-1 selectedTarget=62 selectedPosition=0",
    ]

    events = list(parse_power_log_lines(lines))

    assert [event.event_type for event in events] == [
        "OPTIONS_START",
        "OPTION",
        "OPTION",
        "OPTION_TARGET",
        "OPTION_TARGET",
        "SEND_OPTION",
    ]
    assert events[2].data["option_id"] == 4
    assert events[2].data["main_entity"]["id"] == 146
    assert events[3].data["entity"]["id"] == 62
    assert events[4].data["error"] == "REQ_ENEMY_TARGET"
    assert events[5].data == {
        "selected_option": 4,
        "selected_sub_option": -1,
        "selected_target": 62,
        "selected_position": 0,
    }


def test_change_entity_replaces_stale_card_type_and_cost():
    line = (
        "D 15:04:13.2879032 GameState.DebugPrintPower() - "
        "CHANGE_ENTITY - Updating Entity=[entityName=Маэстро маскарада id=15 zone=HAND "
        "zonePos=3 cardId=SW_050 player=1] CardID=DREAM_05"
    )
    events = list(parse_power_log_lines([line]))
    assert len(events) == 1
    assert events[0].event_type == "CHANGE_ENTITY"
    assert events[0].data["entity"]["id"] == 15
    assert events[0].data["card_id"] == "DREAM_05"

    def ev(etype, **data):
        return type("E", (), {"event_type": etype, "data": data, "raw_line": ""})()

    tracker = GameStateTracker(card_db=CardDatabase(auto_load=True))
    tracker.process_event(ev("FULL_ENTITY", entity_id=15, card_id="SW_050"))
    tracker.process_event(ev("TAG", entity_id=15, tag="COST", value="2"))
    assert tracker.entities[15].card_type == 4
    assert tracker.entities[15].cost == 2

    tracker.process_event(
        ev("CHANGE_ENTITY", entity={"id": 15, "cardId": "SW_050"}, card_id="DREAM_05")
    )
    assert tracker.entities[15].card_id == "DREAM_05"
    assert tracker.entities[15].card_type == 5
    assert tracker.entities[15].cost == -1


def test_snapshot_dedup_and_actions_resolution():
    """Regression: TURN+CURRENT_PLAYER produced duplicate snapshots; opponent
    plays stayed 'UNKNOWN ENTITY' because SHOW_ENTITY resolves mid-block."""
    from src.card_db import CardDatabase
    from src.parser import GameStateTracker

    def ev(etype, **data):
        return type("E", (), {"event_type": etype, "data": data, "raw_line": ""})()

    db = CardDatabase(auto_load=True)
    tracker = GameStateTracker(card_db=db, friendly_player_name="HappyBread#21597")
    tracker.process_event(ev("CREATE_GAME"))
    tracker.process_event(ev("GAME_ENTITY", entity_id=1))
    tracker.process_event(ev("PLAYER_ENTITY", entity_id=2, player_id=1))
    tracker.process_event(ev("PLAYER_ENTITY", entity_id=3, player_id=2))
    tracker.process_event(ev("PLAYER_NAME", player_id=1, player_name="HappyBread#21597"))
    tracker.process_event(ev("PLAYER_NAME", player_id=2, player_name="Enemy#1234"))

    # Realistic ordering: TURN tag, then CURRENT_PLAYER, then STEP=MAIN_ACTION
    tracker.process_event(ev("TAG_CHANGE", entity={"id": 1}, tag="TURN", value="1"))
    tracker.process_event(ev("TAG_CHANGE", entity={"name": "HappyBread#21597"}, tag="CURRENT_PLAYER", value="1"))
    tracker.process_event(ev("TAG_CHANGE", entity={"id": 1}, tag="STEP", value="MAIN_ACTION"))

    # TURN alone must not create a snapshot; only STEP=MAIN_ACTION does,
    # and CURRENT_PLAYER + TURN must not produce duplicate snapshots.
    nums = [s.turn_number for s in tracker.turn_snapshots]
    assert nums.count(1) == 1

    # SHOW_ENTITY backfill: play recorded as UNKNOWN, card revealed mid-block
    tracker.process_event(
        ev("TAG", entity_id=43, tag="CONTROLLER", value="1")
    )
    tracker.process_event(
        ev("TAG", entity_id=43, tag="CARDTYPE", value="SPELL")
    )
    tracker.process_event(
        ev(
            "BLOCK_START",
            block_type="PLAY",
            entity={"id": 43},
            target={"id": 0},
            sub_option=-1,
        )
    )
    tracker.process_event(ev("SHOW_ENTITY", entity={"id": 43}, card_id="CS2_029"))

    tracker.finalize()
    acts = [a for s in tracker.turn_snapshots for a in s.actions]
    assert len(acts) == 1
    assert acts[0].entity_card_id == "CS2_029"
    assert acts[0].entity_name == "Огненный шар"


def test_named_current_player_and_action_controller_are_resolved():
    """Regression: named CURRENT_PLAYER refs must switch turns, and actions
    without a confirmed controller must not enter the active turn."""

    def ev(etype, **data):
        return type("E", (), {"event_type": etype, "data": data, "raw_line": ""})()

    tracker = GameStateTracker(
        card_db=CardDatabase(auto_load=True),
        friendly_player_name="HappyBread#21597",
    )
    tracker.process_event(ev("CREATE_GAME"))
    tracker.process_event(ev("GAME_ENTITY", entity_id=1))
    tracker.process_event(ev("PLAYER_ENTITY", entity_id=2, player_id=1))
    tracker.process_event(ev("PLAYER_ENTITY", entity_id=3, player_id=2))
    tracker.process_event(
        ev("PLAYER_NAME", player_id=2, player_name="UNKNOWN HUMAN PLAYER")
    )
    tracker.process_event(ev("TAG_CHANGE", entity={"id": 1}, tag="TURN", value="1"))
    tracker.process_event(ev("TAG_CHANGE", entity={"id": 2}, tag="CURRENT_PLAYER", value="1"))
    tracker.process_event(ev("TAG_CHANGE", entity={"id": 1}, tag="STEP", value="MAIN_ACTION"))

    # The replay does not always emit PLAYER_NAME before this alias appears.
    tracker.process_event(
        ev(
            "TAG_CHANGE",
            entity={"name": "WINES#21976"},
            tag="CURRENT_PLAYER",
            value="1",
        )
    )
    assert tracker.active_player_id == 2

    # A PLAY block with no entity/controller proof is unsafe and must be ignored.
    tracker.process_event(
        ev(
            "BLOCK_START",
            block_type="PLAY",
            entity={"id": 67},
            target={"id": 0},
            sub_option=-1,
        )
    )
    tracker.finalize()
    assert all(not snapshot.actions for snapshot in tracker.turn_snapshots)


def test_decision_points_capture_pre_action_state_and_end_turn():
    """Each label must be paired with state before the action, including pass."""

    def ev(etype, **data):
        return type("E", (), {"event_type": etype, "data": data, "raw_line": ""})()

    tracker = GameStateTracker(
        card_db=CardDatabase(auto_load=True),
        friendly_player_name="HappyBread#21597",
    )
    tracker.process_event(ev("CREATE_GAME"))
    tracker.process_event(ev("GAME_ENTITY", entity_id=1))
    tracker.process_event(ev("PLAYER_ENTITY", entity_id=2, player_id=1))
    tracker.process_event(ev("PLAYER_ENTITY", entity_id=3, player_id=2))
    tracker.process_event(ev("PLAYER_NAME", player_id=1, player_name="HappyBread#21597"))
    tracker.process_event(ev("PLAYER_NAME", player_id=2, player_name="Enemy#1234"))
    tracker.process_event(ev("TAG", entity_id=2, tag="RESOURCES", value="4"))
    tracker.process_event(ev("TAG_CHANGE", entity={"id": 1}, tag="TURN", value="1"))
    tracker.process_event(
        ev(
            "TAG_CHANGE",
            entity={"name": "HappyBread#21597"},
            tag="CURRENT_PLAYER",
            value="1",
        )
    )
    tracker.process_event(ev("TAG_CHANGE", entity={"id": 1}, tag="STEP", value="MAIN_ACTION"))

    tracker.process_event(ev("FULL_ENTITY", entity_id=43, card_id="CS2_029"))
    tracker.process_event(ev("TAG", entity_id=43, tag="CONTROLLER", value="1"))
    tracker.process_event(ev("TAG", entity_id=43, tag="CARDTYPE", value="SPELL"))
    tracker.process_event(ev("TAG", entity_id=43, tag="ZONE", value="HAND"))
    tracker.process_event(ev("TAG", entity_id=43, tag="COST", value="4"))
    tracker.process_event(
        ev(
            "BLOCK_START",
            block_type="PLAY",
            entity={"id": 43},
            target={"id": 0},
            sub_option=-1,
        )
    )

    assert len(tracker.decision_points) == 1
    play_decision = tracker.decision_points[0]
    assert play_decision.sequence == 1
    assert play_decision.action.details["controller_id"] == 1
    assert play_decision.action.details["_entity_id"] == 43
    assert any(card["entity_id"] == 43 for card in play_decision.snapshot.friendly_hand)

    tracker.process_event(ev("TAG_CHANGE", entity={"id": 43}, tag="ZONE", value="GRAVEYARD"))
    tracker.process_event(ev("TAG_CHANGE", entity={"id": 1}, tag="STEP", value="MAIN_END"))

    assert len(tracker.decision_points) == 2
    end_decision = tracker.decision_points[1]
    assert end_decision.sequence == 2
    assert end_decision.action.action_type == "END_TURN"
    assert all(card["entity_id"] != 43 for card in end_decision.snapshot.friendly_hand)


def test_selected_option_captures_complete_legal_candidate_set():
    def ev(etype, **data):
        return type("E", (), {"event_type": etype, "data": data, "raw_line": ""})()

    tracker = GameStateTracker(
        card_db=CardDatabase(auto_load=True),
        friendly_player_name="HappyBread#21597",
    )
    tracker.process_event(ev("CREATE_GAME"))
    tracker.process_event(ev("GAME_ENTITY", entity_id=1))
    tracker.process_event(ev("PLAYER_ENTITY", entity_id=2, player_id=1))
    tracker.process_event(ev("PLAYER_ENTITY", entity_id=3, player_id=2))
    tracker.process_event(ev("PLAYER_NAME", player_id=1, player_name="HappyBread#21597"))
    tracker.process_event(ev("PLAYER_NAME", player_id=2, player_name="Enemy#1234"))
    tracker.process_event(ev("TAG_CHANGE", entity={"id": 1}, tag="TURN", value="1"))
    tracker.process_event(
        ev("TAG_CHANGE", entity={"name": "HappyBread#21597"}, tag="CURRENT_PLAYER", value="1")
    )
    tracker.process_event(ev("TAG_CHANGE", entity={"id": 1}, tag="STEP", value="MAIN_ACTION"))

    for entity_id, card_id, controller, card_type in (
        (146, "WW_051t", 1, "MINION"),
        (62, "HERO_09d", 2, "HERO"),
    ):
        tracker.process_event(ev("FULL_ENTITY", entity_id=entity_id, card_id=card_id))
        tracker.process_event(ev("TAG", entity_id=entity_id, tag="CONTROLLER", value=str(controller)))
        tracker.process_event(ev("TAG", entity_id=entity_id, tag="CARDTYPE", value=card_type))
        tracker.process_event(ev("TAG", entity_id=entity_id, tag="ZONE", value="PLAY"))

    tracker.process_event(ev("OPTIONS_START", options_id=76))
    tracker.process_event(
        ev(
            "OPTION",
            option_id=0,
            option_type="END_TURN",
            main_entity={},
            error="INVALID",
            error_param="",
        )
    )
    tracker.process_event(
        ev(
            "OPTION",
            option_id=4,
            option_type="POWER",
            main_entity={"id": 146, "cardId": "WW_051t", "entityName": "Бандит", "zone": "PLAY"},
            error="NONE",
            error_param="",
        )
    )
    tracker.process_event(
        ev(
            "OPTION_TARGET",
            target_index=0,
            entity={"id": 62, "cardId": "HERO_09d", "entityName": "Андуин", "zone": "PLAY"},
            error="NONE",
            error_param="",
        )
    )
    tracker.process_event(
        ev(
            "OPTION_TARGET",
            target_index=1,
            entity={"id": 146, "cardId": "WW_051t", "entityName": "Бандит", "zone": "PLAY"},
            error="REQ_ENEMY_TARGET",
            error_param="",
        )
    )
    tracker.process_event(
        ev(
            "SEND_OPTION",
            selected_option=4,
            selected_sub_option=-1,
            selected_target=62,
            selected_position=0,
        )
    )

    assert len(tracker.option_decisions) == 1
    decision = tracker.option_decisions[0]
    assert decision.options_id == 76
    assert decision.selected_option == 4
    assert len(decision.candidates) == 2
    assert decision.candidates[0].action_type == "END_TURN"
    attack = decision.candidates[1]
    assert attack.action_type == "ATTACK"
    assert attack.entity_id == 146
    assert attack.target_entity_id == 62
    assert attack.option_id == 4
    assert attack.position == 0

    for entity_id, card_id, card_type, zone in (
        (150, "REV_990", "LOCATION", "PLAY"),
        (151, "CS2_120", "MINION", "HAND"),
    ):
        tracker.process_event(ev("FULL_ENTITY", entity_id=entity_id, card_id=card_id))
        tracker.process_event(ev("TAG", entity_id=entity_id, tag="CONTROLLER", value="1"))
        tracker.process_event(ev("TAG", entity_id=entity_id, tag="CARDTYPE", value=card_type))
        tracker.process_event(ev("TAG", entity_id=entity_id, tag="ZONE", value=zone))

    tracker.process_event(ev("OPTIONS_START", options_id=77))
    tracker.process_event(
        ev(
            "OPTION",
            option_id=0,
            option_type="END_TURN",
            main_entity={},
            error="INVALID",
            error_param="",
        )
    )
    tracker.process_event(
        ev(
            "OPTION",
            option_id=1,
            option_type="POWER",
            main_entity={"id": 151, "cardId": "CS2_120", "zone": "HAND"},
            error="NONE",
            error_param="",
        )
    )
    tracker.process_event(
        ev(
            "SEND_OPTION",
            selected_option=1,
            selected_sub_option=-1,
            selected_target=0,
            selected_position=3,
        )
    )

    play_decision = tracker.option_decisions[1]
    play_candidates = [candidate for candidate in play_decision.candidates if candidate.option_id == 1]
    assert [candidate.position for candidate in play_candidates] == [1, 2, 3]
    assert all(candidate.entity_card_type == 4 for candidate in play_candidates)


def test_option_candidate_resolves_card_db_name_over_unknown_ref_name():
    def ev(etype, **data):
        return type("E", (), {"event_type": etype, "data": data, "raw_line": ""})()

    tracker = GameStateTracker(card_db=CardDatabase(auto_load=True))
    tracker.process_event(ev("FULL_ENTITY", entity_id=151, card_id="CS2_029"))
    tracker.process_event(ev("TAG", entity_id=151, tag="ZONE", value="HAND"))
    tracker.process_event(ev("OPTIONS_START", options_id=1))
    tracker.process_event(
        ev(
            "OPTION",
            option_id=1,
            option_type="POWER",
            main_entity={
                "id": 151,
                "cardId": "CS2_029",
                "entityName": "  unknown entity [cardType=INVALID]  ",
                "zone": "HAND",
            },
            error="NONE",
            error_param="",
        )
    )

    candidates = tracker._build_option_candidates()

    assert len(candidates) == 1
    assert candidates[0].entity_name == "Огненный шар"


def test_option_candidate_marks_tradeable_card_semantics_unproven():
    def ev(etype, **data):
        return type("E", (), {"event_type": etype, "data": data, "raw_line": ""})()

    tracker = GameStateTracker(card_db=CardDatabase(auto_load=True))
    tracker.process_event(ev("FULL_ENTITY", entity_id=20, card_id="JAM_034"))
    tracker.process_event(ev("TAG", entity_id=20, tag="ZONE", value="HAND"))
    tracker.process_event(ev("OPTIONS_START", options_id=1))
    tracker.process_event(
        ev(
            "OPTION",
            option_id=1,
            option_type="POWER",
            main_entity={"id": 20, "cardId": "JAM_034", "zone": "HAND"},
            error="NONE",
            error_param="",
        )
    )

    candidates = tracker._build_option_candidates()

    assert candidates
    assert all(candidate.is_tradeable for candidate in candidates)


def test_entity_attack_budget_supports_windfury_and_mega_windfury():
    windfury = Entity(
        entity_id=10,
        tags={
            "ZONE": "PLAY",
            "ATK": 3,
            "WINDFURY": 1,
            "NUM_ATTACKS_THIS_TURN": 1,
        },
    )
    assert windfury.can_attack is True
    windfury.tags["NUM_ATTACKS_THIS_TURN"] = 2
    assert windfury.can_attack is False

    mega = Entity(
        entity_id=11,
        tags={
            "ZONE": "PLAY",
            "ATK": 1,
            "MEGA_WINDFURY": 1,
            "NUM_ATTACKS_THIS_TURN": 3,
        },
    )
    assert mega.can_attack is True
    mega.tags["NUM_ATTACKS_THIS_TURN"] = 4
    assert mega.can_attack is False

    rush = Entity(
        entity_id=12,
        tags={"ZONE": "PLAY", "ATK": 2, "RUSH": 1, "NUM_TURNS_IN_PLAY": 0},
    )
    assert rush.can_attack is True
    assert rush.can_attack_hero is False
    rush.tags["NUM_TURNS_IN_PLAY"] = 1
    assert rush.can_attack_hero is True


def test_deck_stats_index():
    stats = load_deck_stats_index()
    if not stats:
        pytest.skip("DeckStats.xml not available in HDT roaming directory")

    assert len(stats) >= 1000
    winning_ranked = [m for m in stats.values() if m.result == "Win" and m.game_mode == "Ranked"]
    assert len(winning_ranked) > 500


def test_replay_parsing_integration():
    replay_dir = os.path.expandvars(r"%APPDATA%\HearthstoneDeckTracker\Replays")
    if not os.path.exists(replay_dir):
        pytest.skip("Replay directory not found")

    files = [f for f in os.listdir(replay_dir) if f.endswith(".hdtreplay")]
    if not files:
        pytest.skip("No .hdtreplay files found")

    sample_path = os.path.join(replay_dir, files[0])
    stats_index = load_deck_stats_index()
    db = CardDatabase()

    replay = parse_replay_file(sample_path, card_db=db, deck_stats_index=stats_index)
    assert replay.metadata is not None
    assert len(replay.turn_snapshots) > 0

    # Test friendly turns
    friendly_turns = replay.friendly_turns
    assert len(friendly_turns) > 0

    first_turn = friendly_turns[0]
    assert first_turn.turn_number >= 1
    assert len(first_turn.actions) > 0

    # Format verification
    prompt = format_turn_prompt(first_turn, replay.metadata.player_hero, replay.metadata.opponent_hero)
    completion = format_turn_completion(first_turn.actions)

    assert "### Ход" in prompt
    assert "Ваш герой" in prompt
    assert "1. " in completion


def test_dataset_generation_batch(tmp_path):
    out_file = tmp_path / "test_dataset.jsonl"
    count = generate_dataset(output_path=out_file, filter_ranked_wins=True, max_replays=5)

    assert count > 0
    assert out_file.exists()

    # Verify JSONL lines
    with open(out_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        assert len(lines) == count
        first_obj = json.loads(lines[0])
        assert "prompt" in first_obj
        assert "completion" in first_obj
        assert "turn_number" in first_obj
        assert "game_id" in first_obj


def test_hsreplay_xml_parser():
    from src.parser.hsreplay_xml_parser import parse_hsreplay_xml_file

    db = CardDatabase()
    xml_files = list(Path("data/replays_hsreplay").glob("*.hsreplay.xml"))
    if xml_files:
        sample = xml_files[0]
        replay = parse_hsreplay_xml_file(sample, card_db=db)
        assert replay.metadata.game_id == sample.stem
        assert replay.metadata.result == "Win"
        assert len(replay.turn_snapshots) > 0
        assert "HappyBread" in replay.metadata.player_name
