"""
Card data model for Hearthstone entities.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import json

from .enums import (
    CardClass,
    CardType,
    Race,
    Rarity,
    SpellSchool,
    CARD_CLASS_NAMES_RU,
    CARD_TYPE_NAMES_RU,
    RACE_NAMES_RU,
    RARITY_NAMES_RU,
    SPELL_SCHOOL_NAMES_RU,
)


@dataclass
class Card:
    card_id: str
    dbf_id: int
    name_ru: str = ""
    name_en: str = ""
    cost: int = 0
    attack: Optional[int] = None
    health: Optional[int] = None
    durability: Optional[int] = None
    card_type: CardType = CardType.MINION
    card_class: CardClass = CardClass.NEUTRAL
    race: Optional[Race] = None
    spell_school: Optional[SpellSchool] = None
    rarity: Optional[Rarity] = None
    text_ru: str = ""
    text_en: str = ""
    mechanics: List[str] = field(default_factory=list)
    mechanics_ru: List[str] = field(default_factory=list)
    runes: Dict[str, int] = field(default_factory=dict)
    tourist_class: Optional[str] = None
    collectible: bool = False
    elite: bool = False
    card_set: int = 0
    artist: str = ""

    # Convenience properties
    @property
    def is_minion(self) -> bool:
        return self.card_type == CardType.MINION

    @property
    def is_spell(self) -> bool:
        return self.card_type == CardType.SPELL

    @property
    def is_weapon(self) -> bool:
        return self.card_type == CardType.WEAPON

    @property
    def is_hero(self) -> bool:
        return self.card_type == CardType.HERO

    @property
    def is_hero_power(self) -> bool:
        return self.card_type == CardType.HERO_POWER

    @property
    def is_location(self) -> bool:
        return self.card_type == CardType.LOCATION

    @property
    def is_titan(self) -> bool:
        return "Titan" in self.mechanics or "Титан" in self.mechanics_ru

    @property
    def is_colossal(self) -> bool:
        return "Colossal" in self.mechanics or "Гигант" in self.mechanics_ru

    @property
    def is_starship(self) -> bool:
        return (
            "Starship" in self.mechanics
            or "Starship Piece" in self.mechanics
            or "Звездолет" in self.mechanics_ru
            or "Деталь звездолета" in self.mechanics_ru
        )

    @property
    def is_tourist(self) -> bool:
        return self.tourist_class is not None

    @property
    def has_taunt(self) -> bool:
        return "Taunt" in self.mechanics or "Провокация" in self.mechanics_ru

    @property
    def has_charge(self) -> bool:
        return "Charge" in self.mechanics or "Рывок" in self.mechanics_ru

    @property
    def has_rush(self) -> bool:
        return "Rush" in self.mechanics or "Натиск" in self.mechanics_ru

    @property
    def has_divine_shield(self) -> bool:
        return "Divine Shield" in self.mechanics or "Божественный щит" in self.mechanics_ru

    @property
    def has_stealth(self) -> bool:
        return "Stealth" in self.mechanics or "Маскировка" in self.mechanics_ru

    @property
    def has_lifesteal(self) -> bool:
        return "Lifesteal" in self.mechanics or "Похищение жизни" in self.mechanics_ru

    @property
    def has_poisonous(self) -> bool:
        return (
            "Poisonous" in self.mechanics
            or "Venomous" in self.mechanics
            or "Яд" in self.mechanics_ru
            or "Ядовитость" in self.mechanics_ru
        )

    @property
    def has_reborn(self) -> bool:
        return "Reborn" in self.mechanics or "Перерождение" in self.mechanics_ru

    @property
    def has_battlecry(self) -> bool:
        return "Battlecry" in self.mechanics or "Боевой клич" in self.mechanics_ru

    @property
    def has_deathrattle(self) -> bool:
        return "Deathrattle" in self.mechanics or "Предсмертный хрип" in self.mechanics_ru

    @property
    def type_name_ru(self) -> str:
        return CARD_TYPE_NAMES_RU.get(self.card_type, "Неизвестно")

    @property
    def class_name_ru(self) -> str:
        return CARD_CLASS_NAMES_RU.get(self.card_class, "Нейтральная")

    @property
    def race_name_ru(self) -> Optional[str]:
        return RACE_NAMES_RU.get(self.race) if self.race else None

    @property
    def spell_school_name_ru(self) -> Optional[str]:
        return SPELL_SCHOOL_NAMES_RU.get(self.spell_school) if self.spell_school else None

    @property
    def rarity_name_ru(self) -> Optional[str]:
        return RARITY_NAMES_RU.get(self.rarity) if self.rarity else None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "card_id": self.card_id,
            "dbf_id": self.dbf_id,
            "name_ru": self.name_ru,
            "name_en": self.name_en,
            "cost": self.cost,
            "attack": self.attack,
            "health": self.health,
            "durability": self.durability,
            "card_type": int(self.card_type),
            "card_class": int(self.card_class),
            "race": int(self.race) if self.race is not None else None,
            "spell_school": int(self.spell_school) if self.spell_school is not None else None,
            "rarity": int(self.rarity) if self.rarity is not None else None,
            "text_ru": self.text_ru,
            "text_en": self.text_en,
            "mechanics": self.mechanics,
            "mechanics_ru": self.mechanics_ru,
            "runes": self.runes,
            "tourist_class": self.tourist_class,
            "collectible": self.collectible,
            "elite": self.elite,
            "card_set": self.card_set,
            "artist": self.artist,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Card":
        return cls(
            card_id=data.get("card_id", ""),
            dbf_id=data.get("dbf_id", 0),
            name_ru=data.get("name_ru", ""),
            name_en=data.get("name_en", ""),
            cost=data.get("cost", 0),
            attack=data.get("attack"),
            health=data.get("health"),
            durability=data.get("durability"),
            card_type=CardType(data.get("card_type", 4)),
            card_class=CardClass(data.get("card_class", 12)),
            race=Race(data["race"]) if data.get("race") is not None else None,
            spell_school=SpellSchool(data["spell_school"]) if data.get("spell_school") is not None else None,
            rarity=Rarity(data["rarity"]) if data.get("rarity") is not None else None,
            text_ru=data.get("text_ru", ""),
            text_en=data.get("text_en", ""),
            mechanics=data.get("mechanics", []),
            mechanics_ru=data.get("mechanics_ru", []),
            runes=data.get("runes", {}),
            tourist_class=data.get("tourist_class"),
            collectible=bool(data.get("collectible", False)),
            elite=bool(data.get("elite", False)),
            card_set=data.get("card_set", 0),
            artist=data.get("artist", ""),
        )
