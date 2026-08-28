"""
Unit and integration tests for Log Parser, State Tracker, Replay Reader, and Dataset Generator.
"""

import json
import os
import tempfile
from pathlib import Path
import pytest

from src.parser import (
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
