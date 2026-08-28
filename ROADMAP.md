# ROADMAP.md — Hearthstone AI Assistant

## Легенда статусов
- ⚪ **Planned** (Запланировано)
- 🟡 **In Progress** (В процессе)
- 🟢 **Completed** (Завершено)
- 🔴 **Blocked** (Заблокировано)

---

## Фаза 1: Локальная база карт и семантический движок (Card DB)
- [x] 🟢 **1.1. XML Card Indexer**: Создан модуль `src/card_db/indexer.py` для парсинга `CardDefs.ruRU.xml` и `CardDefs.base.xml`.
- [x] 🟢 **1.2. Быстрый кэш SQLite / JSON**: Сгенерирован индекс на 35 807 карт с поддержкой современных механик (Титаны, Области, Туристы, Звездолеты, Руны DK, Миниатюризация, Гигантизация).
- [x] 🟢 **1.3. Форматтер описания сущностей**: Создан модуль `src/card_db/formatter.py` для генерации компактных семантических описаний карт и стола под LLM-промпты.

---

## Фаза 2: Экстрактор реплеев и датасет (Replay Pipeline)
- [x] 🟢 **2.1. Unzip & Log Streamer**: Модуль `src/parser/replay_reader.py` для потокового чтения `output_log.txt` из `.hdtreplay` без распаковки на диск.
- [x] 🟢 **2.2. GameState Tracker**: Детерминированный восстановитель состояния игры `src/parser/state_tracker.py` (ход, мана, здоровье героев, стол, рука игрока, сыгранные карты и цели атак).
- [x] 🟢 **2.3. Dataset Generator**: Пакетный генератор `src/parser/dataset_generator.py`: извлечено 5 174 обучающих пар `[State -> Action]` из 525 победных Ranked-матчей -> `data/processed/train_actions.jsonl` (7.41 MB).

---

## Фаза 3: Ретроспективный тренер матчей (Post-Game Coach)
- [x] 🟢 **3.1. Ollama LLM Client & Resilient Parser**: Модули `src/llm/ollama_client.py`, `src/llm/candidate_generator.py` и `src/llm/response_parser.py` с поддержкой `qwen2.5:1.5b-instruct-q8_0` / `7b` и защитой от галлюцинаций малых моделей.
- [x] 🟢 **3.2. Turn Analyzer & Lethal Detector**: Пошаговый разбор матча `src/coach/analyzer.py`: поиск упущенного летального урона, темповых потерь и сравнение с ходами игрока.
- [x] 🟢 **3.3. Coach CLI**: Консольный инструмент `python -m src.coach.cli --latest` / `--replay <file>` с выводом структурированного отчета на русском языке.

---

## Фаза 4: Live In-Game Watcher (Реальное время)
- [ ] ⚪ **4.1. Power.log Watcher**: Асинхронный tailer `D:\Hearthstone\Logs\Power.log`.
- [ ] ⚪ **4.2. Rule Validator (Compliance Engine)**: Быстрая проверка доступности маны и легальных целей.
- [ ] ⚪ **4.3. Real-Time Advisor CLI / Overlay**: Вывод 3-5 кандидатов действий за < 3 секунд на ход с приоритетами.

---

## Фаза 5: Персонализация и Fine-Tuning (QLoRA)
- [ ] ⚪ **5.1. Подготовка датасета под QLoRA**: Форматирование в ChatML / Alpaca формат.
- [ ] ⚪ **5.2. QLoRA обучение на RTX 4060**: Дообучение LoRA адаптера поверх Qwen-2.5-7B на 8 GB VRAM.
- [ ] ⚪ **5.3. Оценка качества**: Сравнение предсказаний базовой модели vs дообученной на тестовой выборке матчей.
