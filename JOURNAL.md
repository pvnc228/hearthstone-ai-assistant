# JOURNAL.md — Hearthstone AI Assistant

## 2026-08-28 — Инициализация проекта и исследование системы

### 1. Аудит оборудования и среды
- **GPU**: NVIDIA GeForce RTX 4060 (8 188 MiB VRAM, драйвер 610.88, CUDA 13.3).
- **CPU & Память**: Intel Core i5-12400F (12 логических ядер), 32 GB RAM.
- **Python**: 3.13.3 (установлены `torch`, `transformers`, `peft`, `safetensors`).
- **Ollama**: 0.32.5 (модели `qwen2.5-coder:7b`, `qwen2.5:1.5b`, `deepseek-coder:6.7b`, `gemma4:12b`).

### 2. Аудит игровых данных на машине
- Обнаружена директория HDT: `C:\Users\mist8\AppData\Roaming\HearthstoneDeckTracker`.
- **Реплеи**: 1 041 файл `.hdtreplay` (zip архивы с `output_log.txt` / Power.log).
- **База матчей**: `DeckStats.xml` содержит 1 025 записанных игр (982 в рейтинге, 548 побед / 477 поражений, игрок `HappyBread#21597`).
- **Справочник карт**: `CardDefs\CardDefs.ruRU.xml` и `CardDefs.base.xml` содержат 35 807 карт с полными русскими и английскими названиями и описаниями.
- **Игра**: `D:\Hearthstone`, HDT активен в процессах.

### 3. Архитектурные решения
1. **Принцип Ponytail**: Использовать встроенные библиотеки Python (`zipfile`, `xml.etree.ElementTree`, `re`) для парсера реплеев без тяжелых внешних зависимостей.
2. **Локальный LLM инференс**: Использовать Ollama с `qwen2.5-coder:7b` и `qwen2.5:1.5b` для минимальной задержки (1.5-3 сек).
3. **Фильтрация данных**: Для обучающего набора и анализа брать в приоритете 548 победных Ranked-матчей.

### 4. Создана структура репозитория
- Создана директория: `C:\Users\mist8\.gemini\antigravity\scratch\hearthstone-ai-assistant`
- Созданы файлы управления проектом: `AGENTS.md`, `ROADMAP.md`, `JOURNAL.md`, `README.md`.

---

## 2026-08-28 — Завершение Фазы 1: База карт и семантический движок (Card DB)

### 1. Реализованные модули
- `src/card_db/enums.py`: Актуализированные перечисления `CardType` (включая `LOCATION`), `CardClass` (включая `DEATHKNIGHT`, `DEMONHUNTER`), `SpellSchool` (7 школ заклинаний), `Race`, `Rarity`, а также полный маппинг ключевых слов:
  - **Титаны** (`TITAN`)
  - **Области** (`LOCATION`)
  - **Туристы** (`TOURIST` + 11 классов)
  - **Звездолеты и детали** (`STARSHIP`, `STARSHIP_PIECE`)
  - **Миниатюризация / Гигантизация** (`MINIATURIZE`, `GIGANTIFY`)
  - **Руны Рыцаря Смерти** (`COST_BLOOD`, `COST_FROST`, `COST_UNHOLY`, `CORPSE`)
  - **Классические и вечнозеленые механики** (`TAUNT`, `DIVINE_SHIELD`, `RUSH`, `CHARGE`, `LIFESTEAL`, `POISONOUS`, `REBORN`, `WINDFURY`, `BATTLECRY`, `DEATHRATTLE`, `SECRET`, `QUEST`, `TRADEABLE`, `FORGE`, `EXCAVATE`, `QUICKDRAW` и др.).
- `src/card_db/cleaner.py`: Очистка игрового текста (удаление `[x]`, `$`, `@`, `_`, HTML-тегов и переносов строк).
- `src/card_db/models.py`: Датакласс `Card` с типизацией, свойствами и методами сериализации `to_dict()` / `from_dict()`.
- `src/card_db/indexer.py`: Парсер XML-словарей HDT, создание SQLite кэша `data/cache/cards.db` и мгновенное O(1) in-memory кэширование.
- `src/card_db/formatter.py`: Форматирование описаний карт и сущностей стола под токены LLM-промптов.

### 2. Бенчмарки и результаты тестов
- Распарсено сущностей HDT: **35 807 карт**.
- Размер SQLite базы: **13.99 MB**.
- Скорость O(1) выборки из памяти: **22+ млн lookups/sec** (100 000 поисков за 0.0045 с).
- Юнит-тесты: `7/7 passed` (`pytest tests/test_card_db.py`).

---

## 2026-08-28 — Завершение Фазы 2: Экстрактор реплеев и датасет (Replay Pipeline)

### 1. Реализованные модули
- `src/parser/log_parser.py`: Потоковый разбор `output_log.txt` / `Power.log`, парсинг событий `CREATE_GAME`, `TAG_CHANGE`, `SHOW_ENTITY`, `FULL_ENTITY`, `BLOCK_START`, `BLOCK_END`, `PLAYER_NAME` и bracket-нотации сущностей (`[entityName=... id=... cardId=...]`).
- `src/parser/state_tracker.py`: Детерминированный трекер сущностей и зон (`HAND`, `PLAY`, `SECRET`, `GRAVEYARD`), расчет доступной маны, здоровья/брони героев, характеристик существ и областей, а также генерация снимков `TurnSnapshot` и фиксация действий `PlayerAction`.
- `src/parser/replay_reader.py`: Потоковое чтение `.hdtreplay` zip-архивов и сопоставление с `DeckStats.xml` (метаданные матчей, победы/поражения, герои, колоды).
- `src/parser/dataset_generator.py`: Пакетная выгрузка структурированных обучающих пар `(Prompt: Состояние стола -> Completion: Цепочка победных действий)` в JSONL.

### 2. Результаты прогона и метрики
- Обработано реплеев: **1 041 файл** (все 1 025 матчей из `DeckStats.xml` найдены на диске).
- Отфильтровано победных рейтинговых матчей: **525 игр** с валидными ходами.
- Сгенерировано обучающих пар `[State -> Action]`: **5 174 записи**.
- Размер датасета `data/processed/train_actions.jsonl`: **7.41 MB**.
- Юнит и интеграционные тесты: `12/12 passed` (`pytest tests/ -v`).

---

## 2026-08-28 — Завершение Фазы 3: Ретроспективный тренер матчей (Post-Game Coach)

### 1. Архитектура для работы с малой моделью (Qwen-1.5B)
- Скачана модель высокого квантования: `qwen2.5:1.5b-instruct-q8_0` (1.6 GB VRAM).
- `src/llm/candidate_generator.py`: Детерминированный генератор легальных действий (расчет маны, строгая валидация провокаций Taunt, доступности атак существ и сил героя).
- `src/llm/ollama_client.py`: Клиент Ollama API с гранулярными таймаутами и автовыбором модели.
- `src/llm/response_parser.py`: Отказоустойчивый парсер ответов (извлечение индексов `PLAN: [1, 2]`, нечеткий поиск по сущностям, защита от перерасхода маны, безопасный эвристический fallback).
- `src/coach/analyzer.py`: Детектор летального урона (подсчет максимального взрывного урона), трекинг потерь темпа/маны, сравнение действий игрока с рекомендацией LLM.
- `src/coach/cli.py`: Консольный интерфейс `python -m src.coach.cli --latest`.

---

## 2026-08-28 — Интеграция HSReplay.net и сборка Master Dataset

### 1. Реализованные модули
- `src/parser/hsreplay_downloader.py`: Авторизованный загрузчик с обходом Cloudflare TLS fingerprinting через `curl_cffi` (`impersonate="chrome120"`). Автосканирование метаданных игр и скачивание `.hsreplay.xml` напрямую из AWS S3.
- `src/parser/hsreplay_xml_parser.py`: Полноценный парсер HearthSim XML-дерева (`<GameEntity>`, `<Player>`, `<FullEntity>`, `<ShowEntity>`, `<TagChange>`, `<Block>`) с маппингом целочисленных Blizzard GameTags в строковые события и передачей в `GameStateTracker`.
- `src/parser/hsreplay_dataset_builder.py`: Пакетная выгрузка тактических обучающих пар из сотен онлайн-матчей.
- `src/parser/build_master_dataset.py`: Единый пайплайн объединения локальных `.hdtreplay` и онлайн `.hsreplay.xml` с дедупликацией по хешу промпта.

### 2. Метрики и результаты
- Сканировано игр на аккаунте: **1 547 матчей** (830 побед).
- Скачано полных XML файлов: **412 матчей**.
- Извлечено обучающих пар из HSReplay: **2 337 ходов** (2 312 уникальных).
- Итоговый объединенный мастер-датасет: **7 486 ходов** (`data/processed/train_master_actions.jsonl`, 9.97 MB).
- Юнит-тесты: `16/16 passed` (`pytest tests/ -v`).

---

## 2026-08-31 — Исследование Hugging Face датасетов, интеграция TokenGraph и пайплайн QLoRA (Фаза 5)

### 1. Исследование датасетов через `ml-intern`
- Проведен технический аудит 5 ресурсов на Hugging Face через агент `ml-intern`:
  1. `TraceOnSnow/hearthstone-art-512`: Структурированный семантический граф токенов и карт (8 661 карт, 6 069 связей «родитель $\to$ токен»). Отобран для интеграции.
  2. `dvitel/hearthstone` + `dvitel/h1`: Кодогенерация симулятора Hearthbreaker (устарело, 665 карт).
  3. `FrancophonIA/Hearthstone`: Мультиязычные тексты карт (избыточно при наличии XML HDT на 35.8k карт).
  4. `Norod78/hearthstone-cards-512`: Text-to-Image карточки (для текущего пайплайна не требуется).

### 2. Реализованные модули
- `src/card_db/token_graph.py`: Семантический граф `TokenGraph` с микросекундным резолвингом дочерних сущностей и порождаемых токенов (`get_child_cards`, `get_parent_cards`, `format_token_summary`).
- `src/card_db/formatter.py`: Обогащение описания карт информацией о генерируемых токенах и тегах действий.
- `src/llm/dataset_formatter.py`: Модуль форматирования датасета тактических решений в стандарты ChatML / Alpaca (`sft_train_chatml.jsonl` — 4 677 пар, `sft_eval_chatml.jsonl` — 497 пар) с гарантией изоляции train/eval по ID матчей.
- `src/llm/train_qlora.py` & `configs/qlora_config.json`: Оптимизированный скрипт QLoRA обучения на базе `SFTTrainer`, `peft` и `bitsandbytes` 4-bit NF4 под 8GB VRAM (NVIDIA RTX 4060).

### 3. Результаты и верификация
- Тестовый набор: **28/28 passed** (`python -m pytest tests/ -v`).

---

## 2026-09-02 — Stage A: исправление владельца действия в replay parser

### 1. Найденная первопричина
- В части HDT-логов соперник сначала назывался `UNKNOWN HUMAN PLAYER`, а затем появлялся как реальный BattleTag (`WINES#21976`). Алиас не заменял placeholder в `player_id_by_name`, поэтому `CURRENT_PLAYER=1` не переключал `active_player_id` на игрока 2.
- `BLOCK_START` ранее принимал `PLAY`/`ATTACK` без подтвержденного `CONTROLLER`, что позволяло ошибочно приписывать чужие действия текущему игроку.

### 2. Исправление и доказательства
- `src/parser/state_tracker.py`: замена placeholder-алиаса на реальное имя игрока и guard по `CONTROLLER` перед записью действия.
- `tests/test_parser.py`: regression для именованного `CURRENT_PLAYER` и запрета действия без owner proof.
- До фикса на baseline было **149** cross-class hero-power mismatch из **1 338** действий силы героя; после полного прогона **0** из **670**.
- Проблемный replay `78a2ab60` после фикса содержит раздельный `active_player_id=2` для ходов Warlock; `Жизнеотвод` больше не попадает во friendly turns.
- Проверки: `29 passed`; полный ranked-win slice: **549** игр, **3 049** records.

Production `data/processed/train_actions.jsonl` пока не перезаписывался: текущий файл сохранен как baseline для отдельной регенерации и следующего аудита dataset ownership.
