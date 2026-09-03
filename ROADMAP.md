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
- [x] 🟢 **1.4. Семантический граф токенов (TokenGraph)**: Интегрирован граф порождаемых карт и токенов (`src/card_db/token_graph.py`) на базе датасета `TraceOnSnow` (8 661 карт, 6 069 связей).

---

## Фаза 2: Экстрактор реплеев и датасет (Replay Pipeline)
- [x] 🟢 **2.1. Unzip & Log Streamer**: Модуль `src/parser/replay_reader.py` для потокового чтения `output_log.txt` из `.hdtreplay` без распаковки на диск.
- [x] 🟢 **2.2. GameState Tracker**: Детерминированный восстановитель состояния игры `src/parser/state_tracker.py` (ход, мана, здоровье героев, стол, рука игрока, сыгранные карты и цели атак).
- [x] 🟢 **2.3. Turn Dataset Generator**: Production `src/parser/dataset_generator.py` перегенерировал 3 049 turn records с 11 704 действиями из 549 победных Ranked replay -> `data/processed/train_actions.jsonl`.
- [x] 🟢 **2.4. Replay Option Oracle**: Парсятся `DebugPrintOptions`, `SendOption` и `CHANGE_ENTITY`; state tracker сохраняет pre-action snapshot, полный `error=NONE` candidate set и выбранный tuple с entity/target/sub-option/position IDs.
- [x] 🟢 **2.5. Next-action schema v2**: `src/parser/next_action_dataset.py` создал 12 840 accepted и 1 862 quarantine из 14 702 option decisions (87.3351% coverage); accepted gate violations — 0.

---

## Фаза 3: Ретроспективный тренер матчей (Post-Game Coach)
- [x] 🟢 **3.1. Ollama LLM Client & Resilient Parser**: Модули `src/llm/ollama_client.py`, `src/llm/candidate_generator.py` и `src/llm/response_parser.py` с поддержкой `qwen2.5:1.5b-instruct-q8_0` / `7b` и защитой от галлюцинаций малых моделей.
- [x] 🟢 **3.2. Turn Analyzer & Lethal Detector**: Пошаговый разбор матча `src/coach/analyzer.py`: поиск упущенного летального урона, темповых потерь и сравнение с ходами игрока.
- [x] 🟢 **3.3. Coach CLI**: Консольный инструмент `python -m src.coach.cli --latest` / `--replay <file>` с выводом структурированного отчета на русском языке.

---

## Фаза 4: Live In-Game Watcher (Реальное время — отложено)
- [ ] ⚪ **4.1. Power.log Watcher**: Асинхронный tailer `D:\Hearthstone\Logs\Power.log`.
- [ ] ⚪ **4.2. Rule Validator (Compliance Engine)**: Быстрая проверка доступности маны и легальных целей.
- [ ] ⚪ **4.3. Real-Time Advisor CLI / Overlay**: Вывод 3-5 кандидатов действий за < 3 секунд на ход с приоритетами.

---

## Фаза 5: Персонализация и Fine-Tuning (QLoRA)
- [x] 🟡 **5.1. Legacy SFT artifacts**: Старые free-text ChatML/Alpaca файлы созданы, но исключены из training-ready контура из-за несовпадения с production `PLAN: [индексы]` контрактом.
- [x] 🟢 **5.2. Production next-action artifact**: Schema v2 использует `state + legal candidates -> chosen candidate ID`; 1 862 decisions с непроверенной Tradeable/mana/sub-option семантикой остаются в quarantine.
- [x] 🟡 **5.3. QLoRA код**: `src/llm/train_qlora.py` и `configs/qlora_config.json` существуют, но training environment после аудита не ревалидирован.
- [x] 🟢 **5.4a. Frozen schema-v2 splits**: Создан `next_action_split_manifest_v1.json`; 540 игр распределены без пересечений на train/validation/test/temporal holdout, исходный accepted JSONL закреплен SHA-256.
- [x] 🟢 **5.4b. Schema-v2 formatter и trainer guard**: Общий prompt-контракт подключен к `MatchCoach`, ChatML formatter и QLoRA trainer; trainer отбрасывает legacy free-text config и проверяет manifest membership.
- [x] 🟢 **5.4c. Base-model evaluator**: Добавлен evaluator top-1/format/existence/latency с разбиением по типам действий; smoke подтвердил блокировку при недоступном Ollama.
- [ ] 🔴 **5.4d. Remaining readiness gates**: Проверить state transitions/dynamic costs, классифицировать 9 replay без option-событий, восстановить QLoRA-зависимости/CUDA/NF4 и выполнить полный baseline; quarantine для первого пилота исключен формальной политикой manifest.
- [ ] ⚪ **5.5. Запуск полного обучения и экспорт**: Только после readiness gates; обучение и экспорт сейчас не выполнялись.
