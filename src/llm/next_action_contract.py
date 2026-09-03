"""Shared prompt and response contract for schema-v2 next-action inference."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional


NEXT_ACTION_SYSTEM_PROMPT = (
    "Ты — тактический ассистент Hearthstone. "
    "Выбери ровно одно лучшее легальное следующее действие из списка кандидатов. "
    "Не придумывай кандидатов и не меняй их идентификаторы."
)

_SINGLE_PLAN_RE = re.compile(r"""\s*(?:ПЛАН|PLAN)\s*:\s*\[\s*(\d+)\s*\]\s*""", re.IGNORECASE)


@dataclass(frozen=True)
class NextActionParse:
    """Strict parse result used by evaluation and runtime guards."""

    candidate_id: Optional[int]
    format_valid: bool
    candidate_exists: bool


def _value(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(key, default)
    return getattr(item, key, default)


def candidate_to_prompt_dict(candidate: Any) -> dict[str, Any]:
    """Normalizes a schema-v2 or runtime candidate for the shared prompt."""
    candidate_id = _value(candidate, "id", _value(candidate, "index"))
    description = _value(candidate, "description", "") or ""
    return {"id": int(candidate_id), "description": str(description)}


def _entity_line(entity: Any, *, include_cost: bool = False) -> str:
    if not isinstance(entity, Mapping):
        return "—"
    name = entity.get("name") or entity.get("card_id") or "Неизвестная сущность"
    stats = ""
    attack = entity.get("attack")
    health = entity.get("health")
    if attack is not None or health is not None:
        stats = f" {attack if attack is not None else '?'} / {health if health is not None else '?'}"
    cost = f", {entity['cost']}м" if include_cost and entity.get("cost") is not None else ""
    return f"{name}{stats}{cost}"


def _zone_line(entities: Any, *, include_cost: bool = False) -> str:
    if not entities:
        return "—"
    return "; ".join(_entity_line(entity, include_cost=include_cost) for entity in entities)


def _hero_line(hero: Any) -> str:
    if not isinstance(hero, Mapping):
        return "—"
    name = hero.get("name") or hero.get("card_id") or "Герой"
    health = hero.get("health", "?")
    armor = hero.get("armor", 0)
    attack = hero.get("attack", 0)
    suffix = f"{health} HP"
    if armor:
        suffix += f", броня {armor}"
    if attack:
        suffix += f", атака {attack}"
    return f"{name} ({suffix})"


def snapshot_to_prompt_state(snapshot: Any) -> dict[str, Any]:
    """Extracts only the deterministic prompt fields from a runtime snapshot."""
    return {
        "turn": getattr(snapshot, "turn_number", 0),
        "mana": getattr(snapshot, "friendly_mana", 0),
        "max_mana": getattr(snapshot, "friendly_max_mana", 0),
        "friendly_hero": getattr(snapshot, "friendly_hero", {}),
        "opponent_hero": getattr(snapshot, "opponent_hero", {}),
        "hand": getattr(snapshot, "friendly_hand", []),
        "friendly_board": getattr(snapshot, "friendly_board", []),
        "opponent_board": getattr(snapshot, "opponent_board", []),
        "hero_power": getattr(snapshot, "hero_power", {}),
        "friendly_locations": getattr(snapshot, "friendly_locations", []),
        "opponent_locations": getattr(snapshot, "opponent_locations", []),
    }


def build_next_action_prompt(state: Mapping[str, Any], candidates: Iterable[Any]) -> str:
    """Builds the exact user prompt shared by training records and runtime."""
    candidate_dicts = [candidate_to_prompt_dict(candidate) for candidate in candidates]
    hand = state.get("hand", state.get("friendly_hand", []))
    friendly_board = state.get("friendly_board", [])
    opponent_board = state.get("opponent_board", [])
    hero_power = state.get("hero_power") or {}

    lines = [
        f"Ход {state.get('turn', 0)}. Доступно маны: {state.get('mana', 0)}/{state.get('max_mana', 0)}.",
        f"Твой герой: {_hero_line(state.get('friendly_hero'))}.",
        f"Герой противника: {_hero_line(state.get('opponent_hero'))}.",
        f"Рука: {_zone_line(hand, include_cost=True)}.",
        f"Твой стол: {_zone_line(friendly_board)}.",
        f"Стол противника: {_zone_line(opponent_board)}.",
    ]
    if hero_power:
        lines.append(f"Сила героя: {_entity_line(hero_power, include_cost=True)}.")
    if state.get("friendly_locations") or state.get("opponent_locations"):
        lines.append(f"Твои локации: {_zone_line(state.get('friendly_locations'))}.")
        lines.append(f"Локации противника: {_zone_line(state.get('opponent_locations'))}.")

    lines.append("Доступные действия:")
    lines.extend(f"[{candidate['id']}] {candidate['description']}" for candidate in candidate_dicts)
    lines.append("Ответь строго в формате: PLAN: [один идентификатор кандидата]")
    return "\n".join(lines)


def format_next_action_completion(chosen_candidate_id: int) -> str:
    """Formats the single-label completion used by schema-v2 SFT."""
    return f"PLAN: [{int(chosen_candidate_id)}]"


def parse_next_action_response(raw_text: str, candidate_ids: Iterable[int]) -> NextActionParse:
    """Parses only the single-candidate contract; free text is invalid."""
    match = _SINGLE_PLAN_RE.fullmatch(raw_text or "")
    if not match:
        return NextActionParse(None, format_valid=False, candidate_exists=False)
    candidate_id = int(match.group(1))
    candidate_set = {int(value) for value in candidate_ids}
    return NextActionParse(
        candidate_id=candidate_id,
        format_valid=True,
        candidate_exists=candidate_id in candidate_set,
    )
