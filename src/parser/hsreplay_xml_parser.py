"""
Parser for standard HearthSim .hsreplay.xml files downloaded from HSReplay.net.
Transforms XML game event tree into a stream of PowerEvent objects with
integer-to-string tag mappings, matching GameStateTracker internal event loop.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from src.card_db import CardDatabase
from .log_parser import PowerEvent
from .replay_reader import GameMetadata, GameReplay
from .state_tracker import GameStateTracker

# HearthSim / Blizzard GameTag Integer to String Name Mapping
GAME_TAG_MAP: Dict[int, str] = {
    17: "PLAYSTATE",
    18: "HERO_ENTITY",
    19: "STEP",
    20: "TURN",
    24: "TEMP_RESOURCES",
    26: "RESOURCES",
    27: "RESOURCES_USED",
    30: "TEAM_ID",
    31: "DIVINE_SHIELD",
    32: "CHARGE",
    38: "JUST_PLAYED",
    44: "FATIGUE",
    45: "HEALTH",
    47: "ATK",
    48: "COST",
    49: "ZONE",
    50: "CONTROLLER",
    53: "CURRENT_PLAYER",
    187: "DAMAGE",
    189: "ARMOR",
    191: "STEALTH",
    192: "EXHAUSTED",
    197: "CHARGE",
    198: "WINDFURY",
    202: "CARDTYPE",
    203: "ZONE_POSITION",
    208: "FROZEN",
    268: "OVERLOAD_LOCKED",
    269: "OVERLOAD_OWED",
    338: "SILENCED",
    365: "TAUNT",
    385: "SECRET",
    794: "RUSH",
    937: "QUEST",
    1085: "REBORN",
    1520: "DORMANT",
    1765: "LOC_USE_REQ_MAX",
    1840: "LOCATION_ACTION_TARGET",
}

# Zone mapping
ZONE_MAP: Dict[int, str] = {
    1: "PLAY",
    2: "DECK",
    3: "HAND",
    4: "GRAVEYARD",
    5: "SECRET",
    6: "SETASIDE",
    7: "REMOVEDFROMGAME",
}

# CardType mapping
CARDTYPE_MAP: Dict[int, str] = {
    1: "GAME",
    2: "PLAYER",
    3: "HERO",
    4: "MINION",
    5: "SPELL",
    6: "ENCHANTMENT",
    7: "WEAPON",
    8: "ITEM",
    9: "TOKEN",
    10: "HERO_POWER",
    39: "LOCATION",
}

# Step mapping (HearthSim GameStep)
STEP_MAP: Dict[int, str] = {
    1: "INVALID",
    2: "BEGIN_FIRST",
    3: "BEGIN_SHUFFLE",
    4: "BEGIN_DRAW",
    5: "BEGIN_MULLIGAN",
    6: "MAIN_BEGIN",
    7: "MAIN_READY",
    8: "MAIN_RESOURCE",
    9: "MAIN_DRAW",
    10: "MAIN_START",
    11: "MAIN_ACTION",
    12: "MAIN_COMBAT",
    13: "MAIN_END",
    14: "MAIN_NEXT",
    15: "FINAL_WRAPUP",
    16: "FINAL_GAMEOVER",
    17: "MAIN_CLEANUP",
    18: "MAIN_START_TRIGGERS",
}


def _normalize_tag_name_and_val(raw_tag: str | int, raw_val: str | int) -> tuple[str, Any]:
    try:
        t_id = int(raw_tag)
        t_val = int(raw_val)
    except (ValueError, TypeError):
        return str(raw_tag), raw_val

    tag_name = GAME_TAG_MAP.get(t_id, str(t_id))
    tag_val: Any = t_val

    if tag_name == "ZONE":
        tag_val = ZONE_MAP.get(t_val, str(t_val))
    elif tag_name == "CARDTYPE":
        tag_val = CARDTYPE_MAP.get(t_val, str(t_val))
    elif tag_name == "STEP":
        if t_val in (6, 10, 11):
            tag_val = "MAIN_ACTION"
        elif t_val in (13, 14):
            tag_val = "MAIN_END"
        else:
            tag_val = STEP_MAP.get(t_val, str(t_val))

    return tag_name, str(tag_val)


def parse_hsreplay_xml_events(game_element: ET.Element) -> Iterator[PowerEvent]:
    """
    Recursively walks through <Game> element and yields sequential PowerEvents.
    """
    for elem in game_element:
        tag_name = elem.tag

        if tag_name == "GameEntity":
            eid = int(elem.attrib.get("entity") or elem.attrib.get("id", "1"))
            yield PowerEvent(
                event_type="GAME_ENTITY",
                data={"entity_id": eid},
            )
            for t in elem.findall("Tag"):
                t_name, t_val = _normalize_tag_name_and_val(t.attrib["tag"], t.attrib["value"])
                yield PowerEvent(
                    event_type="TAG_CHANGE",
                    data={"entity": {"id": eid}, "tag": t_name, "value": t_val},
                )

        elif tag_name == "Player":
            eid = int(elem.attrib.get("entity") or elem.attrib.get("id", "0"))
            pid = int(elem.attrib.get("playerID", "0"))
            pname = elem.attrib.get("name", f"Player{pid}")

            yield PowerEvent(
                event_type="PLAYER_ENTITY",
                data={"entity_id": eid, "player_id": pid},
            )
            yield PowerEvent(
                event_type="PLAYER_NAME",
                data={"player_id": pid, "player_name": pname},
            )
            for t in elem.findall("Tag"):
                t_name, t_val = _normalize_tag_name_and_val(t.attrib["tag"], t.attrib["value"])
                yield PowerEvent(
                    event_type="TAG_CHANGE",
                    data={"entity": {"id": eid, "name": pname}, "tag": t_name, "value": t_val},
                )

        elif tag_name == "FullEntity":
            eid = int(elem.attrib.get("entity") or elem.attrib.get("id", "0"))
            cid = elem.attrib.get("cardID", "")

            yield PowerEvent(
                event_type="FULL_ENTITY",
                data={"entity_id": eid, "card_id": cid},
            )
            for t in elem.findall("Tag"):
                t_name, t_val = _normalize_tag_name_and_val(t.attrib["tag"], t.attrib["value"])
                yield PowerEvent(
                    event_type="TAG",
                    data={"entity_id": eid, "tag": t_name, "value": t_val},
                )

        elif tag_name == "ShowEntity":
            eid = int(elem.attrib.get("entity") or elem.attrib.get("id", "0"))
            cid = elem.attrib.get("cardID", "")

            yield PowerEvent(
                event_type="SHOW_ENTITY",
                data={"entity": {"id": eid}, "card_id": cid},
            )
            for t in elem.findall("Tag"):
                t_name, t_val = _normalize_tag_name_and_val(t.attrib["tag"], t.attrib["value"])
                yield PowerEvent(
                    event_type="TAG_CHANGE",
                    data={"entity": {"id": eid}, "tag": t_name, "value": t_val},
                )

        elif tag_name == "TagChange":
            eid = int(elem.attrib.get("entity") or elem.attrib.get("id", "0"))
            t_name, t_val = _normalize_tag_name_and_val(elem.attrib["tag"], elem.attrib["value"])
            yield PowerEvent(
                event_type="TAG_CHANGE",
                data={"entity": {"id": eid}, "tag": t_name, "value": t_val},
            )

        elif tag_name == "Block":
            b_type = elem.attrib.get("type", "0")
            b_entity = int(elem.attrib.get("entity") or elem.attrib.get("id", "0"))
            b_target = int(elem.attrib.get("target")) if elem.attrib.get("target") else None

            type_str = "PLAY" if b_type == "7" else ("ATTACK" if b_type == "1" else ("POWER" if b_type == "6" else b_type))

            block_data: Dict[str, Any] = {
                "block_type": type_str,
                "entity": {"id": b_entity},
            }
            if b_target:
                block_data["target"] = {"id": b_target}

            yield PowerEvent(
                event_type="BLOCK_START",
                data=block_data,
            )

            # Recurse inside block children
            for sub_evt in parse_hsreplay_xml_events(elem):
                yield sub_evt

            yield PowerEvent(event_type="BLOCK_END", data={})


def parse_hsreplay_xml_file(xml_path: Path, card_db: CardDatabase) -> GameReplay:
    """
    Parses a single .hsreplay.xml file into a GameReplay object.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    game_elem = root.find("Game")
    if game_elem is None:
        raise ValueError(f"No <Game> element found in {xml_path}")

    # Extract Players
    players = game_elem.findall("Player")
    friendly_name = "HappyBread#21597"
    opp_name = "Opponent"

    for p in players:
        pname = p.attrib.get("name", "")
        if "HappyBread" in pname:
            friendly_name = pname
        else:
            opp_name = pname

    tracker = GameStateTracker(card_db=card_db, friendly_player_name=friendly_name)
    events = parse_hsreplay_xml_events(game_elem)
    for evt in events:
        tracker.process_event(evt)
    tracker.finalize()

    # Extract hero classes
    f_hero_name = "Герой"
    o_hero_name = "Герой противника"
    f_player = tracker.players.get(tracker.friendly_player_id)
    if f_player and f_player.hero_entity_id:
        h_ent = tracker.entities.get(f_player.hero_entity_id)
        if h_ent and h_ent.name:
            f_hero_name = h_ent.name

    o_pid = 2 if tracker.friendly_player_id == 1 else 1
    o_player = tracker.players.get(o_pid)
    if o_player and o_player.hero_entity_id:
        oh_ent = tracker.entities.get(o_player.hero_entity_id)
        if oh_ent and oh_ent.name:
            o_hero_name = oh_ent.name

    meta = GameMetadata(
        game_id=xml_path.stem,
        replay_file=xml_path.name,
        player_name=friendly_name,
        opponent_name=opp_name,
        player_hero=f_hero_name,
        opponent_hero=o_hero_name,
        result="Win",
        game_mode="Ranked",
        format="Standard" if game_elem.attrib.get("format") == "2" else "Wild",
        turns_count=len(tracker.turn_snapshots),
    )

    return GameReplay(metadata=meta, turn_snapshots=tracker.turn_snapshots)
