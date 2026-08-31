"""
Semantic formatter for Hearthstone entities and game states.
Generates token-efficient, highly informative descriptions tailored for LLM reasoning.
"""

from typing import List, Optional
from .enums import CardType, CardClass, SpellSchool
from .models import Card


def format_card(card: Card, lang: str = "ru", token_graph=None, card_db=None) -> str:
    """
    Formats a card into a comprehensive, human-readable semantic string.
    Example:
      - 'Огненный шар [4 маны, Заклинание Огня]: Наносит 6 ед. урона.'
      - 'Лирой Дженкинс [5 маны, 6/2 Существо, Рывок]: Боевой клич: призывает двух дракончиков 1/1 для вашего оппонента.'
      - 'Рубиновое святилище [1 мана, Область, 3 прочности]: Дает существу +2 к атаке.'
    """
    name = card.name_ru if lang == "ru" and card.name_ru else card.name_en
    text = card.text_ru if lang == "ru" and card.text_ru else card.text_en
    type_name = card.type_name_ru if lang == "ru" else card.card_type.name

    details = [f"{card.cost} маны" if lang == "ru" else f"{card.cost} mana"]

    if card.is_minion:
        stats = f"{card.attack or 0}/{card.health or 0}"
        race_str = f" {card.race_name_ru}" if card.race_name_ru else ""
        details.append(f"{stats} {type_name}{race_str}")
    elif card.is_weapon:
        stats = f"{card.attack or 0}/{card.durability or card.health or 0}"
        details.append(f"{stats} {type_name}")
    elif card.is_location:
        dur = card.durability or card.health or 0
        details.append(f"{type_name}, {dur} прочности" if lang == "ru" else f"{type_name}, {dur} durability")
    elif card.is_spell:
        school = f" {card.spell_school_name_ru}" if card.spell_school_name_ru else ""
        details.append(f"{type_name}{school}")
    elif card.is_hero:
        details.append(f"{type_name} (броня: {card.health or 5})")
    elif card.is_hero_power:
        details.append(type_name)

    # Add runes if Death Knight
    if card.runes:
        runes_parts = []
        if card.runes.get("blood"):
            runes_parts.append(f"{card.runes['blood']} Кровь")
        if card.runes.get("frost"):
            runes_parts.append(f"{card.runes['frost']} Лед")
        if card.runes.get("unholy"):
            runes_parts.append(f"{card.runes['unholy']} Нечестивость")
        if runes_parts:
            details.append("Руны: " + ", ".join(runes_parts))

    # Add tourist info
    if card.tourist_class:
        details.append(f"Турист ({card.tourist_class})")

    # Add key mechanics if present
    mechanics_list = card.mechanics_ru if lang == "ru" else card.mechanics
    mech_str = ", ".join(m for m in mechanics_list if not m.startswith("Турист"))

    # Token summary if available
    token_str = ""
    if token_graph and card_db:
        token_str = token_graph.format_token_summary(card, card_db, lang=lang)

    header = f"{name} [{', '.join(details)}]"
    res = header
    if mech_str and text:
        res = f"{header}: {mech_str}. {text}"
    elif mech_str:
        res = f"{header}: {mech_str}"
    elif text:
        res = f"{header}: {text}"

    if token_str:
        res = f"{res} {token_str}"
    return res


def format_card_compact(card: Card, lang: str = "ru") -> str:
    """
    Ultra-compact string for listing cards in hand.
    Example:
      - '[4м] Огненный шар: 6 урона'
      - '[5м 6/2] Лирой Дженкинс (Рывок)'
    """
    name = card.name_ru if lang == "ru" and card.name_ru else card.name_en
    text = card.text_ru if lang == "ru" and card.text_ru else card.text_en

    if card.is_minion:
        stats = f"{card.attack or 0}/{card.health or 0}"
        mechs = ", ".join(card.mechanics_ru if lang == "ru" else card.mechanics)
        extra = f" ({mechs})" if mechs else ""
        return f"[{card.cost}м {stats}] {name}{extra}"

    elif card.is_weapon:
        stats = f"{card.attack or 0}/{card.durability or card.health or 0}"
        return f"[{card.cost}м Оружие {stats}] {name}"

    elif card.is_location:
        dur = card.durability or card.health or 0
        return f"[{card.cost}м Область {dur}пр] {name}"

    elif card.is_spell:
        short_text = text[:60] + "..." if len(text) > 60 else text
        return f"[{card.cost}м Заклинание] {name}: {short_text}" if short_text else f"[{card.cost}м Заклинание] {name}"

    elif card.is_hero_power:
        return f"[{card.cost}м Сила героя] {name}"

    return f"[{card.cost}м] {name}"


def format_board_minion(
    name: str,
    atk: int,
    hp: int,
    max_hp: int,
    can_attack: bool = False,
    is_taunt: bool = False,
    is_divine_shield: bool = False,
    is_stealthed: bool = False,
    is_frozen: bool = False,
    is_silenced: bool = False,
    is_reborn: bool = False,
    is_dormant: bool = False,
    is_titan: bool = False,
    is_starship: bool = False,
    additional_info: str = "",
) -> str:
    """
    Formats an active minion on the battlefield for the game board state prompt.
    """
    if is_dormant:
        return f"{name} [В спячке]"

    flags = []
    if is_taunt:
        flags.append("Провокация")
    if is_divine_shield:
        flags.append("Бож. щит")
    if is_stealthed:
        flags.append("Маскировка")
    if is_reborn:
        flags.append("Перерождение")
    if is_frozen:
        flags.append("Заморожен")
    if is_silenced:
        flags.append("Немота")
    if is_titan:
        flags.append("Титан")
    if is_starship:
        flags.append("Звездолет")

    status = "готов атаковать" if can_attack and not is_frozen else "не готов"
    flags_str = f" [{', '.join(flags)}]" if flags else ""
    extra = f" ({additional_info})" if additional_info else ""

    return f"{name} ({atk}/{hp}){flags_str} - {status}{extra}"
