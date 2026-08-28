"""
Script to test 20 diverse Hearthstone tactical scenarios on qwen2.5:1.5b-instruct-q8_0
and export the full, raw, untruncated model responses to data/benchmark_20_situations.md.
"""

import json
import time
from pathlib import Path
from src.card_db import CardDatabase
from src.llm import OllamaClient, ActionCandidate, generate_legal_candidates, parse_model_response
from src.parser import TurnSnapshot

OUTPUT_FILE = Path("data/benchmark_20_situations.md")
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# 20 Diverse Hearthstone Situations
SCENARIOS = [
    {
        "id": 1,
        "title": "Прямой летал со стола (6 HP у врага, два существа 3/3 готовы бить)",
        "snapshot": TurnSnapshot(
            turn_number=4, active_player_id=1, active_player_name="Player", is_friendly_turn=True,
            friendly_mana=4, friendly_max_mana=4,
            friendly_hero={"name": "Mage", "health": 25, "armor": 0},
            opponent_hero={"name": "Hunter", "health": 6, "armor": 0},
            friendly_hand=[{"card_id": "CS2_120", "name": "Речной кроколиск", "cost": 2, "card_type": 4}],
            friendly_board=[
                {"entity_id": 10, "name": "Вожак волков", "attack": 3, "health": 3, "can_attack": True},
                {"entity_id": 11, "name": "Железноклюв", "attack": 3, "health": 2, "can_attack": True}
            ],
            opponent_board=[], friendly_locations=[], opponent_locations=[], friendly_secrets=[],
            opponent_secrets_count=0, opponent_hand_count=3
        ),
        "expected": "Атака обоими существами в лицо (3+3 = 6 урона = Летал)."
    },
    {
        "id": 2,
        "title": "Летал заклинанием через Провокацию (Враг 6 HP за Провокацией 8/8, в руке Огненный шар)",
        "snapshot": TurnSnapshot(
            turn_number=5, active_player_id=1, active_player_name="Player", is_friendly_turn=True,
            friendly_mana=4, friendly_max_mana=4,
            friendly_hero={"name": "Mage", "health": 18, "armor": 0},
            opponent_hero={"name": "Warrior", "health": 6, "armor": 0},
            friendly_hand=[
                {"card_id": "CS2_029", "name": "Огненный шар", "cost": 4, "card_type": 5, "text": "Наносит 6 ед. урона."},
                {"card_id": "CS2_120", "name": "Речной кроколиск", "cost": 2, "card_type": 4}
            ],
            friendly_board=[
                {"entity_id": 10, "name": "Мурлок-налетчик", "attack": 2, "health": 1, "can_attack": True}
            ],
            opponent_board=[
                {"entity_id": 20, "name": "Горный великан", "attack": 8, "health": 8, "is_taunt": True}
            ],
            friendly_locations=[], opponent_locations=[], friendly_secrets=[],
            opponent_secrets_count=0, opponent_hand_count=4
        ),
        "expected": "Разыграть Огненный шар в лицо героя противника (6 урона в обход провокации = Летал)."
    },
    {
        "id": 3,
        "title": "Выгодный размен (Value Trade) против опасной угрозы 4/1",
        "snapshot": TurnSnapshot(
            turn_number=2, active_player_id=1, active_player_name="Player", is_friendly_turn=True,
            friendly_mana=2, friendly_max_mana=2,
            friendly_hero={"name": "Paladin", "health": 30, "armor": 0},
            opponent_hero={"name": "Rogue", "health": 28, "armor": 0},
            friendly_hand=[
                {"card_id": "CS2_120", "name": "Речной кроколиск", "cost": 2, "card_type": 4}
            ],
            friendly_board=[
                {"entity_id": 10, "name": "Ящер Кровавой Топи", "attack": 3, "health": 2, "can_attack": True}
            ],
            opponent_board=[
                {"entity_id": 20, "name": "Магмовый яростень", "attack": 4, "health": 1, "can_attack": False}
            ],
            friendly_locations=[], opponent_locations=[], friendly_secrets=[],
            opponent_secrets_count=0, opponent_hand_count=5
        ),
        "expected": "Разменять ящера в 4/1 либо разыграть 2-дроп и разменяться."
    },
    {
        "id": 4,
        "title": "Добивание пингом Силы героя (Маг: 2 маны, у врага существо 3/1)",
        "snapshot": TurnSnapshot(
            turn_number=2, active_player_id=1, active_player_name="Player", is_friendly_turn=True,
            friendly_mana=2, friendly_max_mana=2,
            friendly_hero={"name": "Mage", "health": 30, "armor": 0},
            opponent_hero={"name": "Priest", "health": 30, "armor": 0},
            friendly_hand=[
                {"card_id": "CS2_029", "name": "Огненный шар", "cost": 4, "card_type": 5}
            ],
            friendly_board=[],
            opponent_board=[
                {"entity_id": 20, "name": "Лепрогном", "attack": 3, "health": 1}
            ],
            friendly_locations=[], opponent_locations=[], friendly_secrets=[],
            opponent_secrets_count=0, opponent_hand_count=5
        ),
        "expected": "Использовать силу героя (Вспышка огня) на существо 3/1."
    },
    {
        "id": 5,
        "title": "Темповый розыгрыш по мане (Ход 3: выбор между 3-дропом и 1-дропом)",
        "snapshot": TurnSnapshot(
            turn_number=3, active_player_id=1, active_player_name="Player", is_friendly_turn=True,
            friendly_mana=3, friendly_max_mana=3,
            friendly_hero={"name": "Druid", "health": 30, "armor": 0},
            opponent_hero={"name": "Warlock", "health": 26, "armor": 0},
            friendly_hand=[
                {"card_id": "CS2_182", "name": "Ледяной йети", "cost": 4, "card_type": 4},
                {"card_id": "CS2_168", "name": "Лидер волчьих всадников", "cost": 3, "attack": 3, "health": 3, "card_type": 4},
                {"card_id": "CS2_189", "name": "Эльфийская лучница", "cost": 1, "card_type": 4}
            ],
            friendly_board=[], opponent_board=[], friendly_locations=[], opponent_locations=[],
            friendly_secrets=[], opponent_secrets_count=0, opponent_hand_count=5
        ),
        "expected": "Разыграть 3-дроп (3/3) за 3 маны в темп."
    },
    {
        "id": 6,
        "title": "Массовая зачистка стола против спама существ",
        "snapshot": TurnSnapshot(
            turn_number=4, active_player_id=1, active_player_name="Player", is_friendly_turn=True,
            friendly_mana=4, friendly_max_mana=4,
            friendly_hero={"name": "Priest", "health": 14, "armor": 0},
            opponent_hero={"name": "Paladin", "health": 28, "armor": 0},
            friendly_hand=[
                {"card_id": "CS2_236", "name": "Божественная кара", "cost": 1, "card_type": 5},
                {"card_id": "CS2_004", "name": "Кольцо света", "cost": 4, "card_type": 5, "text": "Наносит 2 ед. урона всем врагам."}
            ],
            friendly_board=[],
            opponent_board=[
                {"entity_id": 21, "name": "Паладин-рекрут", "attack": 1, "health": 1},
                {"entity_id": 22, "name": "Паладин-рекрут", "attack": 1, "health": 1},
                {"entity_id": 23, "name": "Мурлок-тидехантер", "attack": 2, "health": 1},
                {"entity_id": 24, "name": "Мурлок-скаут", "attack": 1, "health": 1}
            ],
            friendly_locations=[], opponent_locations=[], friendly_secrets=[],
            opponent_secrets_count=0, opponent_hand_count=4
        ),
        "expected": "Разыграть Кольцо света (4м) для полной зачистки стола."
    },
    {
        "id": 7,
        "title": "Атака сквозь Божественный щит (Снять щит слабым токеном 1/1 перед ударом 6/6)",
        "snapshot": TurnSnapshot(
            turn_number=5, active_player_id=1, active_player_name="Player", is_friendly_turn=True,
            friendly_mana=5, friendly_max_mana=5,
            friendly_hero={"name": "Paladin", "health": 22, "armor": 0},
            opponent_hero={"name": "Paladin", "health": 20, "armor": 0},
            friendly_hand=[],
            friendly_board=[
                {"entity_id": 10, "name": "Паладин-рекрут", "attack": 1, "health": 1, "can_attack": True},
                {"entity_id": 11, "name": "Огр Скалистой Пещеры", "attack": 6, "health": 7, "can_attack": True}
            ],
            opponent_board=[
                {"entity_id": 20, "name": "Тирион Фордринг", "attack": 6, "health": 6, "is_taunt": True, "is_divine_shield": True}
            ],
            friendly_locations=[], opponent_locations=[], friendly_secrets=[],
            opponent_secrets_count=0, opponent_hand_count=3
        ),
        "expected": "Ударить 1/1 в Тириона для снятия божественного щита, затем добить 6/7."
    },
    {
        "id": 8,
        "title": "Активация Области (Location) для баффа существа перед атакой",
        "snapshot": TurnSnapshot(
            turn_number=3, active_player_id=1, active_player_name="Player", is_friendly_turn=True,
            friendly_mana=3, friendly_max_mana=3,
            friendly_hero={"name": "Priest", "health": 28, "armor": 0},
            opponent_hero={"name": "Rogue", "health": 24, "armor": 0},
            friendly_hand=[
                {"card_id": "CS2_120", "name": "Речной кроколиск", "cost": 2, "card_type": 4}
            ],
            friendly_board=[
                {"entity_id": 10, "name": "Служительница солнца", "attack": 2, "health": 3, "can_attack": True}
            ],
            opponent_board=[
                {"entity_id": 20, "name": "Агент ШРУ", "attack": 3, "health": 3}
            ],
            friendly_locations=[
                {"entity_id": 30, "card_id": "REV_750", "name": "Собор Искупления", "durability": 3, "can_use": True}
            ],
            opponent_locations=[], friendly_secrets=[], opponent_secrets_count=0, opponent_hand_count=4
        ),
        "expected": "Активировать Собор Искупления на существо (+2/+1 и добор), затем уничтожить Агента ШРУ."
    },
    {
        "id": 9,
        "title": "Оружие + размен лицом для защиты стола",
        "snapshot": TurnSnapshot(
            turn_number=2, active_player_id=1, active_player_name="Player", is_friendly_turn=True,
            friendly_mana=2, friendly_max_mana=2,
            friendly_hero={"name": "Warrior", "health": 30, "armor": 0},
            opponent_hero={"name": "Hunter", "health": 28, "armor": 0},
            friendly_hand=[
                {"card_id": "CS2_106", "name": "Огненная секира", "cost": 2, "card_type": 7, "attack": 3, "durability": 2}
            ],
            friendly_board=[
                {"entity_id": 10, "name": "Трогг Железного рудника", "attack": 1, "health": 2, "can_attack": True}
            ],
            opponent_board=[
                {"entity_id": 20, "name": "Фазаний стрелок", "attack": 3, "health": 2}
            ],
            friendly_locations=[], opponent_locations=[], friendly_secrets=[],
            opponent_secrets_count=0, opponent_hand_count=4
        ),
        "expected": "Экипировать Огненную секиру и атаковать героем существо 3/2, сохранив Трогга."
    },
    {
        "id": 10,
        "title": "Срочная защита (Защитная провокация при низком здоровье героя 4 HP)",
        "snapshot": TurnSnapshot(
            turn_number=5, active_player_id=1, active_player_name="Player", is_friendly_turn=True,
            friendly_mana=5, friendly_max_mana=5,
            friendly_hero={"name": "Mage", "health": 4, "armor": 0},
            opponent_hero={"name": "Hunter", "health": 20, "armor": 0},
            friendly_hand=[
                {"card_id": "CS2_179", "name": "Щитоносец Сен'джин", "cost": 4, "card_type": 4, "attack": 3, "health": 5, "text": "Провокация"},
                {"card_id": "CS2_029", "name": "Огненный шар", "cost": 4, "card_type": 5}
            ],
            friendly_board=[],
            opponent_board=[
                {"entity_id": 20, "name": "Леокк", "attack": 2, "health": 4},
                {"entity_id": 21, "name": "Питомец", "attack": 3, "health": 2}
            ],
            friendly_locations=[], opponent_locations=[], friendly_secrets=[],
            opponent_secrets_count=0, opponent_hand_count=3
        ),
        "expected": "Разыграть Щитоносца Сен'джин с Провокацией, чтобы закрыть лицо от атаки на 5 урона."
    },
    {
        "id": 11,
        "title": "DK Руны и расход трупов (Рыцарь смерти: 3 трупа, заклинание Чумной удар)",
        "snapshot": TurnSnapshot(
            turn_number=3, active_player_id=1, active_player_name="Player", is_friendly_turn=True,
            friendly_mana=3, friendly_max_mana=3,
            friendly_hero={"name": "Deathknight", "health": 26, "armor": 0},
            opponent_hero={"name": "Paladin", "health": 26, "armor": 0},
            friendly_hand=[
                {"card_id": "RLK_012", "name": "Чумной удар", "cost": 2, "card_type": 5, "text": "Наносит 3 урона. Если погибает, призывает зомби 2/2 с Натиском."},
                {"card_id": "RLK_061", "name": "Лорд Ребрад", "cost": 8, "card_type": 4}
            ],
            friendly_board=[],
            opponent_board=[
                {"entity_id": 20, "name": "Рыцарь Серебряного Авангарда", "attack": 3, "health": 3}
            ],
            friendly_locations=[], opponent_locations=[], friendly_secrets=[],
            opponent_secrets_count=0, opponent_hand_count=5
        ),
        "expected": "Разыграть Чумной удар за 2 маны в существо 3/3."
    },
    {
        "id": 12,
        "title": "Титан на столе: активация Титана",
        "snapshot": TurnSnapshot(
            turn_number=6, active_player_id=1, active_player_name="Player", is_friendly_turn=True,
            friendly_mana=6, friendly_max_mana=6,
            friendly_hero={"name": "Mage", "health": 20, "armor": 0},
            opponent_hero={"name": "Warrior", "health": 18, "armor": 0},
            friendly_hand=[],
            friendly_board=[
                {"entity_id": 10, "card_id": "TTN_450", "name": "Норганнон", "attack": 3, "health": 8, "can_attack": True, "is_titan": True}
            ],
            opponent_board=[
                {"entity_id": 20, "name": "Тяжелый латник", "attack": 5, "health": 5}
            ],
            friendly_locations=[], opponent_locations=[], friendly_secrets=[],
            opponent_secrets_count=0, opponent_hand_count=4
        ),
        "expected": "Атаковать/активировать способность Титана Норганнона."
    },
    {
        "id": 13,
        "title": "Звездолет (Starship Piece) сборка и розыгрыш детали",
        "snapshot": TurnSnapshot(
            turn_number=4, active_player_id=1, active_player_name="Player", is_friendly_turn=True,
            friendly_mana=4, friendly_max_mana=4,
            friendly_hero={"name": "Neutral", "health": 28, "armor": 0},
            opponent_hero={"name": "Demonhunter", "health": 24, "armor": 0},
            friendly_hand=[
                {"card_id": "GDB_100", "name": "Каронитовый защитный кристалл", "cost": 4, "attack": 3, "health": 4, "card_type": 4, "text": "Провокация. Деталь звездолета."}
            ],
            friendly_board=[], opponent_board=[], friendly_locations=[], opponent_locations=[],
            friendly_secrets=[], opponent_secrets_count=0, opponent_hand_count=4
        ),
        "expected": "Разыграть деталь звездолета 3/4 с Провокацией."
    },
    {
        "id": 14,
        "title": "Серия приемов (Combo) Разбойника (Монетка -> Агент ШРУ)",
        "snapshot": TurnSnapshot(
            turn_number=2, active_player_id=1, active_player_name="Player", is_friendly_turn=True,
            friendly_mana=2, friendly_max_mana=2,
            friendly_hero={"name": "Rogue", "health": 30, "armor": 0},
            opponent_hero={"name": "Mage", "health": 30, "armor": 0},
            friendly_hand=[
                {"card_id": "GAME_005", "name": "Монетка", "cost": 0, "card_type": 5},
                {"card_id": "EX1_134", "name": "Агент ШРУ", "cost": 3, "attack": 3, "health": 3, "card_type": 4, "text": "Серия приемов: наносит 2 ед. урона."}
            ],
            friendly_board=[],
            opponent_board=[
                {"entity_id": 20, "name": "Ученица чародея", "attack": 3, "health": 2}
            ],
            friendly_locations=[], opponent_locations=[], friendly_secrets=[],
            opponent_secrets_count=0, opponent_hand_count=5
        ),
        "expected": "Монетка (0м) дает 3 маны -> Агент ШРУ (3м) наносит 2 урона и убивает 3/2."
    },
    {
        "id": 15,
        "title": "Миниатюризация (Miniaturize) и получение копии 1/1",
        "snapshot": TurnSnapshot(
            turn_number=3, active_player_id=1, active_player_name="Player", is_friendly_turn=True,
            friendly_mana=3, friendly_max_mana=3,
            friendly_hero={"name": "Paladin", "health": 30, "armor": 0},
            opponent_hero={"name": "Priest", "health": 30, "armor": 0},
            friendly_hand=[
                {"card_id": "MIS_025", "name": "Игрушечный капитан", "cost": 3, "attack": 2, "health": 3, "card_type": 4, "text": "Миниатюризация."}
            ],
            friendly_board=[], opponent_board=[], friendly_locations=[], opponent_locations=[],
            friendly_secrets=[], opponent_secrets_count=0, opponent_hand_count=5
        ),
        "expected": "Разыграть существо с Миниатюризацией для получения мини-копии 1/1 за 1 ману."
    },
    {
        "id": 16,
        "title": "Размен против Яда (Poisonous) токеном, а не тяжелым существом",
        "snapshot": TurnSnapshot(
            turn_number=4, active_player_id=1, active_player_name="Player", is_friendly_turn=True,
            friendly_mana=4, friendly_max_mana=4,
            friendly_hero={"name": "Druid", "health": 25, "armor": 0},
            opponent_hero={"name": "Hunter", "health": 20, "armor": 0},
            friendly_hand=[],
            friendly_board=[
                {"entity_id": 10, "name": "Огонек", "attack": 1, "health": 1, "can_attack": True},
                {"entity_id": 11, "name": "Железнодревень", "attack": 8, "health": 8, "can_attack": True}
            ],
            opponent_board=[
                {"entity_id": 20, "name": "Чумной нетопырь", "attack": 1, "health": 1, "is_taunt": True, "text": "Яд"}
            ],
            friendly_locations=[], opponent_locations=[], friendly_secrets=[],
            opponent_secrets_count=0, opponent_hand_count=3
        ),
        "expected": "Разменять Огонек 1/1 в ядовитого нетопыря, чтобы не потерять великана 8/8."
    },
    {
        "id": 17,
        "title": "Секрет Мага: проверка темпа против пустого стола",
        "snapshot": TurnSnapshot(
            turn_number=3, active_player_id=1, active_player_name="Player", is_friendly_turn=True,
            friendly_mana=3, friendly_max_mana=3,
            friendly_hero={"name": "Mage", "health": 30, "armor": 0},
            opponent_hero={"name": "Rogue", "health": 30, "armor": 0},
            friendly_hand=[
                {"card_id": "EX1_287", "name": "Антимагия", "cost": 3, "card_type": 5, "text": "Секрет: когда противник разыгрывает заклинание, отменяет его."}
            ],
            friendly_board=[], opponent_board=[], friendly_locations=[], opponent_locations=[],
            friendly_secrets=[], opponent_secrets_count=0, opponent_hand_count=5
        ),
        "expected": "Разыграть Секрет: Антимагия за 3 маны."
    },
    {
        "id": 18,
        "title": "Сложный расчет: размен + добивание в лицо оставшимся столом",
        "snapshot": TurnSnapshot(
            turn_number=5, active_player_id=1, active_player_name="Player", is_friendly_turn=True,
            friendly_mana=5, friendly_max_mana=5,
            friendly_hero={"name": "Hunter", "health": 20, "armor": 0},
            opponent_hero={"name": "Warlock", "health": 8, "armor": 0},
            friendly_hand=[
                {"card_id": "DS1_185", "name": "Чародейский выстрел", "cost": 1, "card_type": 5, "text": "Наносит 2 ед. урона."}
            ],
            friendly_board=[
                {"entity_id": 10, "name": "Вепрь-камнеклык", "attack": 1, "health": 1, "can_attack": True},
                {"entity_id": 11, "name": "Высокогрив саванны", "attack": 6, "health": 5, "can_attack": True}
            ],
            opponent_board=[
                {"entity_id": 20, "name": "Древодел", "attack": 1, "health": 1, "is_taunt": True}
            ],
            friendly_locations=[], opponent_locations=[], friendly_secrets=[],
            opponent_secrets_count=0, opponent_hand_count=4
        ),
        "expected": "Вепрь 1/1 бьет провокатора 1/1 -> Высокогрив 6 урона в лицо + Чародейский выстрел 2 урона = 8 урона (Летал)."
    },
    {
        "id": 19,
        "title": "Неистовство ветра (Windfury) двойная атака для летального урона",
        "snapshot": TurnSnapshot(
            turn_number=4, active_player_id=1, active_player_name="Player", is_friendly_turn=True,
            friendly_mana=4, friendly_max_mana=4,
            friendly_hero={"name": "Shaman", "health": 22, "armor": 0},
            opponent_hero={"name": "Priest", "health": 8, "armor": 0},
            friendly_hand=[],
            friendly_board=[
                {"entity_id": 10, "name": "Гарпия-ветрокрыл", "attack": 4, "health": 5, "can_attack": True, "text": "Неистовство ветра"}
            ],
            opponent_board=[], friendly_locations=[], opponent_locations=[],
            friendly_secrets=[], opponent_secrets_count=0, opponent_hand_count=4
        ),
        "expected": "Атака в лицо Гарпией дважды (4 + 4 = 8 урона = Летал)."
    },
    {
        "id": 20,
        "title": "Комплексный ход: зачистка стола + развитие своего темпа",
        "snapshot": TurnSnapshot(
            turn_number=6, active_player_id=1, active_player_name="Player", is_friendly_turn=True,
            friendly_mana=6, friendly_max_mana=6,
            friendly_hero={"name": "Mage", "health": 16, "armor": 0},
            opponent_hero={"name": "Warrior", "health": 22, "armor": 0},
            friendly_hand=[
                {"card_id": "CS2_024", "name": "Ледяная стрела", "cost": 2, "card_type": 5, "text": "Наносит 3 урона."},
                {"card_id": "CS2_182", "name": "Ледяной йети", "cost": 4, "attack": 4, "health": 5, "card_type": 4}
            ],
            friendly_board=[
                {"entity_id": 10, "name": "Мана-змей", "attack": 1, "health": 3, "can_attack": True}
            ],
            opponent_board=[
                {"entity_id": 20, "name": "Кор'кронский воин", "attack": 4, "health": 3}
            ],
            friendly_locations=[], opponent_locations=[], friendly_secrets=[],
            opponent_secrets_count=0, opponent_hand_count=3
        ),
        "expected": "Ледяная стрела (2м) убивает 4/3 -> Йети (4м) ставится на стол -> Мана-змей бьет в лицо."
    }
]

def run_benchmark():
    card_db = CardDatabase(auto_load=True)
    client = OllamaClient(model="qwen2.5:1.5b-instruct-q8_0")
    
    print(f"Running benchmark on 20 situations with {client.model}...")
    
    lines = []
    lines.append(f"# Benchmark: 20 тактических ситуаций Hearthstone на модели `{client.model}`\n")
    lines.append(f"Дата и время: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Оборудование: NVIDIA GeForce RTX 4060 8GB | Модель: `{client.model}` (Q8_0, 1.6 GB VRAM)\n")
    lines.append("---\n")
    
    for s in SCENARIOS:
        s_id = s["id"]
        title = s["title"]
        snap = s["snapshot"]
        expected = s["expected"]
        
        candidates = generate_legal_candidates(snap, card_db)
        
        # Build prompt
        opp_hp = snap.opponent_hero.get("health", 30) + snap.opponent_hero.get("armor", 0)
        p_lines = [
            "Ты — тактический ассистент Hearthstone. Выбери лучшую комбинацию действий из предложенного списка.",
            f"Ход {snap.turn_number}. Доступно маны: {snap.friendly_mana}/{snap.friendly_max_mana}.",
            f"Враг: {opp_hp} HP.",
            "\nДоступные действия:"
        ]
        for c in candidates:
            p_lines.append(f"[{c.index}] {c.description}")
        p_lines.append("\nОтветь строго в формате:\nПЛАН: [номера выбранных действий через запятую, например: 1, 3]\nОБОСНОВАНИЕ: [кратко 1 предложение]")
        prompt_str = "\n".join(p_lines)
        
        t0 = time.time()
        raw_resp = client.generate(prompt=prompt_str, temperature=0.1, max_tokens=200)
        latency = time.time() - t0
        
        parsed = parse_model_response(raw_resp, candidates, max_mana=snap.friendly_mana)
        
        print(f"Situation {s_id:02d}/20: {title[:40]}... ({latency:.2f}s)")
        
        lines.append(f"## Ситуация {s_id}: {title}\n")
        lines.append(f"**Ожидаемое оптимальное решение**: {expected}\n")
        lines.append("### 1. Промпт отправленный модели:")
        lines.append("```text")
        lines.append(prompt_str)
        lines.append("```\n")
        lines.append(f"### 2. Полный сырой ответ модели (Задержка: {latency:.2f}с):")
        lines.append("```text")
        lines.append(raw_resp)
        lines.append("```\n")
        lines.append("### 3. Распарсенный результат:")
        lines.append("```text")
        if parsed.actions:
            for a in parsed.actions:
                lines.append(f"• {a.description}")
        else:
            lines.append("• (Конец хода / нет действий)")
        lines.append(f"Обоснование: {parsed.reasoning}")
        lines.append(f"Потрачено маны: {parsed.total_mana_spent}/{snap.friendly_mana}")
        lines.append("```\n")
        lines.append("---\n")
        
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        
    print(f"\nAll 20 situations completed and saved to {OUTPUT_FILE}!")

if __name__ == "__main__":
    run_benchmark()
