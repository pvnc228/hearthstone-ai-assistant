"""
Streaming line-by-line parser for Hearthstone Power.log and HDT output_log.txt.
Converts raw log text into structured, typed PowerEvent streams.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, Iterable, Optional


@dataclass
class PowerEvent:
    event_type: str  # CREATE_GAME, TAG_CHANGE, SHOW_ENTITY, FULL_ENTITY, HIDE_ENTITY, BLOCK_START, BLOCK_END, TAG
    data: Dict[str, Any] = field(default_factory=dict)
    raw_line: str = ""


# Regex patterns
RE_LOG_PREFIX = re.compile(r"^(?:[DWEIT]\s+[\d:\.]+\s+)?(?:GameState\.(?:DebugPrintPower(?:List)?|DebugPrintGame)\(\)\s*-\s*)?(.*)$")
RE_CREATE_GAME = re.compile(r"^CREATE_GAME")
RE_GAME_ENTITY = re.compile(r"^GameEntity EntityID=(\d+)")
RE_PLAYER_ENTITY = re.compile(
    r"^Player EntityID=(\d+)\s+PlayerID=(\d+)(?:\s+GameAccountId=\[hi=(\d+)\s+lo=(\d+)\])?(?:\s+CardID=(\w*))?"
)
RE_PLAYER_NAME = re.compile(r"^PlayerID=(\d+),\s*PlayerName=(.*)$")
RE_TAG = re.compile(r"^tag=(\w+) value=(.*)$")
RE_TAG_CHANGE = re.compile(r"^TAG_CHANGE Entity=(.+?) tag=(\w+) value=(.*)$")
RE_FULL_ENTITY = re.compile(r"^FULL_ENTITY - (?:Creating|Updating) (?:ID=|Entity=)(\d+) CardID=(\w*)")
RE_SHOW_ENTITY = re.compile(r"^SHOW_ENTITY - Updating Entity=(.+?) CardID=(\w*)")
RE_HIDE_ENTITY = re.compile(r"^HIDE_ENTITY - Entity=(.+?) tag=(\w+) value=(.*)")
RE_BLOCK_START = re.compile(
    r"^BLOCK_START BlockType=(\w+) Entity=(.+?)(?:\s+EffectCardId=.*?)?(?:\s+EffectIndex=.*?)?(?:\s+Target=(.+?))?(?:\s+SubOption=(-?\d+))?(?:\s+TriggerKeyword=\d+)?$"
)
RE_BLOCK_END = re.compile(r"^BLOCK_END")


def parse_entity_ref(raw: str) -> Dict[str, Any]:
    """
    Parses entity representations from Hearthstone logs.
    Handles bracket notation: '[entityName=... id=19 zone=HAND cardId=TSC_916 player=1]'
    or pure numeric IDs: '98'
    or player names: 'HappyBread#21597'
    """
    if not raw:
        return {"raw": ""}
    raw = raw.strip()
    if raw.isdigit():
        return {"id": int(raw), "raw": raw}

    if raw.startswith("[") and raw.endswith("]"):
        content = raw[1:-1]
        res: Dict[str, Any] = {"raw": raw}
        m_id = re.search(r"\bid=(\d+)", content)
        if m_id:
            res["id"] = int(m_id.group(1))

        m_card = re.search(r"\bcardId=(\w*)", content)
        if m_card:
            res["cardId"] = m_card.group(1)

        m_zone = re.search(r"\bzone=(\w+)", content)
        if m_zone:
            res["zone"] = m_zone.group(1)

        m_zone_pos = re.search(r"\bzonePos=(\d+)", content)
        if m_zone_pos:
            res["zonePos"] = int(m_zone_pos.group(1))

        m_player = re.search(r"\bplayer=(\d+)", content)
        if m_player:
            res["player"] = int(m_player.group(1))

        m_name = re.search(r"entityName=(.*?)(?:\s+\w+=|$)", content)
        if m_name:
            res["entityName"] = m_name.group(1).strip()

        return res

    # Fallback to string identifier
    return {"name": raw, "raw": raw}


def parse_power_log_lines(lines: Iterable[str]) -> Generator[PowerEvent, None, None]:
    """
    Generator that parses log lines into high-level structured PowerEvent objects.
    """
    current_entity_id: Optional[int] = None

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Extract payload after prefix
        m_prefix = RE_LOG_PREFIX.match(stripped)
        payload = m_prefix.group(1).strip() if m_prefix else stripped

        if not payload or payload.startswith("Count="):
            continue

        # 1. TAG_CHANGE
        m_tc = RE_TAG_CHANGE.match(payload)
        if m_tc:
            ent_str, tag, val = m_tc.groups()
            yield PowerEvent(
                event_type="TAG_CHANGE",
                data={
                    "entity": parse_entity_ref(ent_str),
                    "tag": tag,
                    "value": val.strip(),
                },
                raw_line=stripped,
            )
            continue

        # 2. BLOCK_START
        m_bs = RE_BLOCK_START.match(payload)
        if m_bs:
            btype, ent_str, target_str, suboption = m_bs.groups()
            yield PowerEvent(
                event_type="BLOCK_START",
                data={
                    "block_type": btype,
                    "entity": parse_entity_ref(ent_str) if ent_str else {},
                    "target": parse_entity_ref(target_str) if target_str else {},
                    "sub_option": int(suboption) if suboption is not None else -1,
                },
                raw_line=stripped,
            )
            continue

        # 3. BLOCK_END
        if RE_BLOCK_END.match(payload):
            yield PowerEvent(event_type="BLOCK_END", raw_line=stripped)
            continue

        # 4. FULL_ENTITY
        m_fe = RE_FULL_ENTITY.match(payload)
        if m_fe:
            eid, cid = m_fe.groups()
            current_entity_id = int(eid)
            yield PowerEvent(
                event_type="FULL_ENTITY",
                data={"entity_id": current_entity_id, "card_id": cid or ""},
                raw_line=stripped,
            )
            continue

        # 5. SHOW_ENTITY
        m_se = RE_SHOW_ENTITY.match(payload)
        if m_se:
            ent_str, cid = m_se.groups()
            ent_ref = parse_entity_ref(ent_str)
            current_entity_id = ent_ref.get("id")
            yield PowerEvent(
                event_type="SHOW_ENTITY",
                data={"entity": ent_ref, "card_id": cid or ""},
                raw_line=stripped,
            )
            continue

        # 6. HIDE_ENTITY
        m_he = RE_HIDE_ENTITY.match(payload)
        if m_he:
            ent_str, tag, val = m_he.groups()
            yield PowerEvent(
                event_type="HIDE_ENTITY",
                data={
                    "entity": parse_entity_ref(ent_str),
                    "tag": tag,
                    "value": val.strip(),
                },
                raw_line=stripped,
            )
            continue

        # 7. CREATE_GAME
        if RE_CREATE_GAME.match(payload):
            yield PowerEvent(event_type="CREATE_GAME", raw_line=stripped)
            continue

        # 8. GameEntity
        m_ge = RE_GAME_ENTITY.match(payload)
        if m_ge:
            current_entity_id = int(m_ge.group(1))
            yield PowerEvent(
                event_type="GAME_ENTITY",
                data={"entity_id": current_entity_id},
                raw_line=stripped,
            )
            continue

        # 9. Player Entity
        m_pe = RE_PLAYER_ENTITY.match(payload)
        if m_pe:
            eid, pid, hi, lo, cid = m_pe.groups()
            current_entity_id = int(eid)
            yield PowerEvent(
                event_type="PLAYER_ENTITY",
                data={
                    "entity_id": current_entity_id,
                    "player_id": int(pid),
                    "hi": hi,
                    "lo": lo,
                    "card_id": cid or "",
                },
                raw_line=stripped,
            )
            continue

        # 10. Direct tag (within FULL_ENTITY, GameEntity, or Player)
        m_tag = RE_TAG.match(payload)
        if m_tag and current_entity_id is not None:
            tag, val = m_tag.groups()
            yield PowerEvent(
                event_type="TAG",
                data={
                    "entity_id": current_entity_id,
                    "tag": tag,
                    "value": val.strip(),
                },
                raw_line=stripped,
            )
            continue

        # 11. Player Name from DebugPrintGame
        m_pn = RE_PLAYER_NAME.match(payload)
        if m_pn:
            pid, pname = m_pn.groups()
            yield PowerEvent(
                event_type="PLAYER_NAME",
                data={
                    "player_id": int(pid),
                    "player_name": pname.strip(),
                },
                raw_line=stripped,
            )
            continue
