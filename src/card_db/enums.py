"""
Hearthstone Enums and Game Tag definitions.
Updated with all modern mechanics (Titans, Locations, Tourists, Starships, Runes, Miniaturize, Gigantify, etc.).
"""

from enum import IntEnum


class CardType(IntEnum):
    GAME = 1
    PLAYER = 2
    HERO = 3
    MINION = 4
    SPELL = 5
    ENCHANTMENT = 6
    WEAPON = 7
    ITEM = 8
    TOKEN = 9
    HERO_POWER = 10
    LOCATION = 39
    BATTLEGROUND_HERO_BUDDY = 40
    BATTLEGROUND_QUEST_REWARD = 42
    BATTLEGROUND_ANOMALY = 43
    BATTLEGROUND_TRINKET = 44
    PET = 45


CARD_TYPE_NAMES_RU = {
    CardType.HERO: "Герой",
    CardType.MINION: "Существо",
    CardType.SPELL: "Заклинание",
    CardType.WEAPON: "Оружие",
    CardType.HERO_POWER: "Сила героя",
    CardType.LOCATION: "Область",
    CardType.ENCHANTMENT: "Эффект",
}

CARD_TYPE_NAMES_EN = {
    CardType.HERO: "Hero",
    CardType.MINION: "Minion",
    CardType.SPELL: "Spell",
    CardType.WEAPON: "Weapon",
    CardType.HERO_POWER: "Hero Power",
    CardType.LOCATION: "Location",
    CardType.ENCHANTMENT: "Enchantment",
}


class CardClass(IntEnum):
    DEATHKNIGHT = 1
    DRUID = 2
    HUNTER = 3
    MAGE = 4
    PALADIN = 5
    PRIEST = 6
    ROGUE = 7
    SHAMAN = 8
    WARLOCK = 9
    WARRIOR = 10
    DREAM = 11
    NEUTRAL = 12
    WHIZBANG = 13
    DEMONHUNTER = 14


CARD_CLASS_NAMES_RU = {
    CardClass.DEATHKNIGHT: "Рыцарь смерти",
    CardClass.DRUID: "Друид",
    CardClass.HUNTER: "Охотник",
    CardClass.MAGE: "Маг",
    CardClass.PALADIN: "Паладин",
    CardClass.PRIEST: "Жрец",
    CardClass.ROGUE: "Разбойник",
    CardClass.SHAMAN: "Шаман",
    CardClass.WARLOCK: "Чернокнижник",
    CardClass.WARRIOR: "Воин",
    CardClass.NEUTRAL: "Нейтральная",
    CardClass.DEMONHUNTER: "Охотник на демонов",
}


class Rarity(IntEnum):
    COMMON = 1
    FREE = 2
    RARE = 3
    EPIC = 4
    LEGENDARY = 5


RARITY_NAMES_RU = {
    Rarity.COMMON: "Обычная",
    Rarity.FREE: "Базовая",
    Rarity.RARE: "Редкая",
    Rarity.EPIC: "Эпическая",
    Rarity.LEGENDARY: "Легендарная",
}


class SpellSchool(IntEnum):
    NONE = 0
    ARCANE = 1
    FIRE = 2
    FROST = 3
    NATURE = 4
    HOLY = 5
    SHADOW = 6
    FEL = 7
    PHYSICAL = 8


SPELL_SCHOOL_NAMES_RU = {
    SpellSchool.ARCANE: "Тайная магия",
    SpellSchool.FIRE: "Огонь",
    SpellSchool.FROST: "Лед",
    SpellSchool.NATURE: "Природа",
    SpellSchool.HOLY: "Свет",
    SpellSchool.SHADOW: "Тьма",
    SpellSchool.FEL: "Скверна",
    SpellSchool.PHYSICAL: "Физический",
}


class Race(IntEnum):
    INVALID = 0
    MURLOC = 14
    DEMON = 15
    MECHANICAL = 17
    ELEMENTAL = 18
    OGRE = 19
    BEAST = 20
    TOTEM = 21
    PIRATE = 23
    DRAGON = 24
    ALL = 26
    UNDEAD = 38
    NAGA = 43
    QUILBOAR = 93


RACE_NAMES_RU = {
    Race.MURLOC: "Мурлок",
    Race.DEMON: "Демон",
    Race.MECHANICAL: "Механизм",
    Race.ELEMENTAL: "Элементаль",
    Race.OGRE: "Огр",
    Race.BEAST: "Зверь",
    Race.TOTEM: "Тотем",
    Race.PIRATE: "Пират",
    Race.DRAGON: "Дракон",
    Race.ALL: "Все расы",
    Race.UNDEAD: "Нежить",
    Race.NAGA: "Нага",
    Race.QUILBOAR: "Свинобраз",
}


# Boolean game mechanics mapping: Tag Name in CardDefs -> (English Name, Russian Name)
MECHANIC_TAGS = {
    "TAUNT": ("Taunt", "Провокация"),
    "CHARGE": ("Charge", "Рывок"),
    "RUSH": ("Rush", "Натиск"),
    "DIVINE_SHIELD": ("Divine Shield", "Божественный щит"),
    "STEALTH": ("Stealth", "Маскировка"),
    "POISONOUS": ("Poisonous", "Яд"),
    "VENOMOUS": ("Venomous", "Ядовитость"),
    "LIFESTEAL": ("Lifesteal", "Похищение жизни"),
    "REBORN": ("Reborn", "Перерождение"),
    "WINDFURY": ("Windfury", "Неистовство ветра"),
    "MEGA_WINDFURY": ("Mega-Windfury", "Меганеистовство ветра"),
    "FREEZE": ("Freeze", "Заморозка"),
    "SILENCE": ("Silence", "Немота"),
    "IMMUNE": ("Immune", "Неуязвимость"),
    "ELUSIVE": ("Elusive", "Неуловимость"),
    "BATTLECRY": ("Battlecry", "Боевой клич"),
    "DEATHRATTLE": ("Deathrattle", "Предсмертный хрип"),
    "COMBO": ("Combo", "Серия приемов"),
    "OVERLOAD": ("Overload", "Перегрузка"),
    "CHOOSE_ONE": ("Choose One", "Выберите один эффект"),
    "SECRET": ("Secret", "Секрет"),
    "QUEST": ("Quest", "Задание"),
    "QUESTLINE": ("Questline", "Цепочка заданий"),
    "SIDEQUEST": ("Sidequest", "Побочная задача"),
    "DISCOVER": ("Discover", "Раскопка"),
    "OUTCAST": ("Outcast", "Изгой"),
    "TRADEABLE": ("Tradeable", "Можно обменять"),
    "DREDGE": ("Dredge", "Улов"),
    "INFUSE": ("Infuse", "Насыщение"),
    "MANATHIRST": ("Manathirst", "Жажда маны"),
    "FORGE": ("Forge", "Ковка"),
    "TITAN": ("Titan", "Титан"),
    "QUICKDRAW": ("Quickdraw", "Быстрая стрельба"),
    "EXCAVATE": ("Excavate", "Добыча"),
    "MINIATURIZE": ("Miniaturize", "Миниатюризация"),
    "GIGANTIFY": ("Gigantify", "Гигантизация"),
    "TOURIST": ("Tourist", "Турист"),
    "STARSHIP": ("Starship", "Звездолет"),
    "STARSHIP_PIECE": ("Starship Piece", "Деталь звездолета"),
    "SPELLBURST": ("Spellburst", "Резонанс"),
    "FRENZY": ("Frenzy", "Бешенство"),
    "HONORABLE_KILL": ("Honorable Kill", "Почетная победа"),
    "OVERKILL": ("Overkill", "Сверхурон"),
    "CORRUPT": ("Corrupt", "Порча"),
    "ECHO": ("Echo", "Эхо"),
    "MAGNETIC": ("Magnetic", "Магнетизм"),
    "INSPIRE": ("Inspire", "Воодушевление"),
    "ADAPT": ("Adapt", "Адаптация"),
    "RECRUIT": ("Recruit", "Вербовка"),
    "TWINSPELL": ("Twinspell", "Дуплет"),
    "COLOSSAL": ("Colossal", "Гигант"),
    "CORPSE": ("Corpse", "Труп"),
    "CORPSE_SPENDER": ("Corpse Spender", "Тратит трупы"),
    "SPELLPOWER": ("Spell Damage", "Урон от заклинаний"),
}

# Tourist classes mapping
TOURIST_TAGS = {
    "DEATH_KNIGHT_TOURIST": "Рыцарь смерти",
    "DEMON_HUNTER_TOURIST": "Охотник на демонов",
    "DRUID_TOURIST": "Друид",
    "HUNTER_TOURIST": "Охотник",
    "MAGE_TOURIST": "Маг",
    "PALADIN_TOURIST": "Паладин",
    "PRIEST_TOURIST": "Жрец",
    "ROGUE_TOURIST": "Разбойник",
    "SHAMAN_TOURIST": "Шаман",
    "WARLOCK_TOURIST": "Чернокнижник",
    "WARRIOR_TOURIST": "Воин",
}
