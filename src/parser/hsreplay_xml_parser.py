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
# Source of truth: HearthSim python-hearthstone hearthstone.enums.GameTag (verified 2026-08)
GAME_TAG_MAP: Dict[int, str] = {
    17: "PLAYSTATE",
    19: "STEP",
    20: "TURN",
    23: "CURRENT_PLAYER",
    25: "RESOURCES_USED",
    26: "RESOURCES",
    27: "HERO_ENTITY",
    43: "EXHAUSTED",
    44: "DAMAGE",
    45: "HEALTH",
    47: "ATK",
    48: "COST",
    49: "ZONE",
    50: "CONTROLLER",
    187: "DURABILITY",
    188: "SILENCED",
    189: "WINDFURY",
    190: "TAUNT",
    191: "STEALTH",
    194: "DIVINE_SHIELD",
    197: "CHARGE",
    198: "NEXT_STEP",
    202: "CARDTYPE",
    219: "SECRET",
    260: "FROZEN",
    263: "ZONE_POSITION",
    292: "ARMOR",
    295: "TEMP_RESOURCES",
    296: "OVERLOAD_OWED",
    297: "NUM_ATTACKS_THIS_TURN",
    393: "OVERLOAD_LOCKED",
    791: "RUSH",
    937: "QUEST",
    1085: "REBORN",
    1518: "DORMANT",
    1646: "HERO_POWER_ENTITY",
    2353: "LOCATION_COOLDOWN",
}

# CardType mapping (HearthSim CardType enum)
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

# Step mapping (HearthSim GameStep enum)
STEP_MAP: Dict[int, str] = {
    0: "INVALID",
    1: "BEGIN_FIRST",
    2: "BEGIN_SHUFFLE",
    3: "BEGIN_DRAW",
    4: "BEGIN_MULLIGAN",
    5: "MAIN_BEGIN",
    6: "MAIN_READY",
    7: "MAIN_RESOURCE",
    8: "MAIN_DRAW",
    9: "MAIN_START",
    10: "MAIN_ACTION",
    11: "MAIN_COMBAT",
    12: "MAIN_END",
    13: "MAIN_NEXT",
    14: "FINAL_WRAPUP",
    15: "FINAL_GAMEOVER",
    16: "MAIN_CLEANUP",
    17: "MAIN_START_TRIGGERS",
    18: "MAIN_SET_ACTION_STEP_TYPE",
    19: "MAIN_PRE_ACTION",
    20: "MAIN_POST_ACTION",
}

# State enum (PLAYSTATE values)
STATE_MAP: Dict[int, str] = {
    0: "INVALID",
    1: "LOADING",
    2: "RUNNING",
    3: "COMPLETE",
    4: "WON",
    5: "LOST",
    6: "TIED",
    7: "DISCONNECTED",
    8: "CONCEDED",
}

# Zone mapping (HearthSim ZONE enum)
ZONE_MAP: Dict[int, str] = {
    1: "PLAY",
    2: "DECK",
    3: "HAND",
    4: "GRAVEYARD",
    5: "SECRET",
    6: "SETASIDE",
    7: "REMOVEDFROMGAME",
}


def _to_entity_id(raw: Any, name_to_id: Optional[Dict[str, int]] = None) -> Optional[int]:
    """Converts an HSReplay entity attribute to an entity id.
    May be an int string, or a name ('GameEntity', player name) per spec."""
    if raw is None:
        return None
    s = str(raw).strip()
    if s.isdigit():
        return int(s)
    if name_to_id is not None and s in name_to_id:
        return name_to_id[s]
    if s == "GameEntity":
        return 1
    return None


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
    elif tag_name == "STEP" or tag_name == "NEXT_STEP":
        tag_val = STEP_MAP.get(t_val, str(t_val))
    elif tag_name == "PLAYSTATE":
        tag_val = STATE_MAP.get(t_val, str(t_val))

    return tag_name, str(tag_val)


def parse_hsreplay_xml_events(game_element: ET.Element, name_to_id: Optional[Dict[str, int]] = None) -> Iterator[PowerEvent]:
    """
    Recursively walks through <Game> element and yields sequential PowerEvents.
    name_to_id maps player names / 'GameEntity' to entity ids for name-based refs.
    """
    for elem in game_element:
        tag_name = elem.tag

        if tag_name == "GameEntity":
            eid = _to_entity_id(elem.attrib.get("entity") or elem.attrib.get("id", "1"))
            if eid is None:
                continue
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
            eid = _to_entity_id(elem.attrib.get("entity") or elem.attrib.get("id", "0"))
            pid = int(elem.attrib.get("playerID", "0"))
            pname = elem.attrib.get("name", f"Player{pid}")

            if eid is not None:
                yield PowerEvent(
                    event_type="PLAYER_ENTITY",
                    data={"entity_id": eid, "player_id": pid},
                )
                if name_to_id is not None:
                    name_to_id[pname] = eid
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
            eid = _to_entity_id(elem.attrib.get("entity") or elem.attrib.get("id", "0"), name_to_id)
            cid = elem.attrib.get("cardID", "")
            if eid is None:
                continue

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
            eid = _to_entity_id(elem.attrib.get("entity") or elem.attrib.get("id", "0"), name_to_id)
            cid = elem.attrib.get("cardID", "")
            if eid is None:
                continue

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
            eid = _to_entity_id(elem.attrib.get("entity") or elem.attrib.get("id", "0"), name_to_id)
            if eid is None:
                continue
            t_name, t_val = _normalize_tag_name_and_val(elem.attrib["tag"], elem.attrib["value"])
            yield PowerEvent(
                event_type="TAG_CHANGE",
                data={"entity": {"id": eid}, "tag": t_name, "value": t_val},
            )

        elif tag_name == "Block":
            b_type = elem.attrib.get("type", "0")
            b_entity = _to_entity_id(elem.attrib.get("entity") or elem.attrib.get("id", "0"), name_to_id)
            b_target = _to_entity_id(elem.attrib.get("target"), name_to_id) if elem.attrib.get("target") else None

            # HearthSim BlockType enum: ATTACK=1, POWER=3, TRIGGER=5, DEATHS=6, PLAY=7, FATIGUE=8
            type_str = {
                "1": "ATTACK",
                "3": "POWER",
                "5": "TRIGGER",
                "6": "DEATHS",
                "7": "PLAY",
                "8": "FATIGUE",
            }.get(b_type, b_type)

            block_data: Dict[str, Any] = {
                "block_type": type_str,
                "entity": {"id": b_entity if b_entity is not None else 0},
            }
            if b_target:
                block_data["target"] = {"id": b_target}

            yield PowerEvent(
                event_type="BLOCK_START",
                data=block_data,
            )

            # Recurse inside block children
            for sub_evt in parse_hsreplay_xml_events(elem, name_to_id):
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
    events = parse_hsreplay_xml_events(game_elem, name_to_id={"GameEntity": 1})
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

    # Real result/game mode from the PLAYSTATE tags on each player entity
    f_result = "Unknown"
    o_result = "Unknown"
    if f_player:
        f_result = f_player.playstate
    if o_player:
        o_result = o_player.playstate
    # Normalize: tracker stores WON/LOST/CONCEDED/TIED or "PLAYING" if tag was missing
    result_map = {"WON": "Win", "LOST": "Loss", "CONCEDED": "Loss", "TIED": "Tie"}
    my_result = result_map.get(f_result, "Unknown")
    if my_result == "Unknown" and o_result in ("LOST", "CONCEDED"):
        my_result = "Win"
    elif my_result == "Unknown" and o_result == "WON":
        my_result = "Loss"

    meta = GameMetadata(
        game_id=xml_path.stem,
        replay_file=xml_path.name,
        player_name=friendly_name,
        opponent_name=opp_name,
        player_hero=f_hero_name,
        opponent_hero=o_hero_name,
        result=my_result,
        game_mode="Ranked",
        format="Standard" if game_elem.attrib.get("format") == "2" else "Wild",
        turns_count=len(tracker.turn_snapshots),
    )

    return GameReplay(metadata=meta, turn_snapshots=tracker.turn_snapshots)
