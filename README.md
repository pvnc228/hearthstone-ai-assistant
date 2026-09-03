# Hearthstone AI Assistant

Персональный тактический ИИ-ассистент и аналитический тренер для Hearthstone на базе локальных языковых моделей и реальных реплеев игрока (ранг Legend).

Проект разработан на основе научного исследования по имитационному обучению:
[Теоретическое обоснование и архитектура (RESEARCH.md)](docs/RESEARCH.md) — "От реплея к ходу: создание персонализированного AI-ассистента для Hearthstone с помощью имитационного обучения".

---

## Ключевые возможности

1. **Post-Game Coach (Ретроспективный тренер)**:
   - Полный разбор завершенных матчей из форматов `.hdtreplay` и `.hsreplay.xml`.
   - Детерминированный детектор летального урона (Lethal Detector) с расчетом максимального взрывного урона за ход.
   - Анализ эффективности использования маны и кривой темпа.
   - Сравнение фактических действий игрока с рекомендациями локальной LLM.

2. **Replay Dataset Pipeline (Пайплайн данных)**:
   - Парсинг локальной истории Hearthstone Deck Tracker (1 041 реплей).
   - Production next-action контракт: `state + replay-reported legal candidates -> chosen candidate ID`.
   - Schema v2 содержит 12 840 принятых решений из 14 702; 1 862 решения с непроверенной Tradeable/mana/sub-option семантикой изолированы в quarantine. HSReplay и старый free-text master dataset не входят в текущий training-ready контур.

3. **Deterministic Compliance Engine (Слой надежности)**:
   - Генератор легальных действий (Candidate Generator) с проверкой маны, провокаций (Taunt), заморозки, спячки (Dormant) и доступности сил героя.
   - Spell-кандидаты не создаются, пока для live/coach пути нет проверенного target contract; replay dataset получает цели из игрового option oracle.
   - Отказоустойчивый парсер ответов малых моделей (Response Parser) с защитой от перерасхода маны и невалидных целей.

4. **Локальный инференс без задержек**:
   - Работа через локальный Ollama API на GPU NVIDIA GeForce RTX 4060 (8 GB VRAM).
   - Время принятия решения: 0.8 - 1.2 секунды на ход (модель `qwen2.5:1.5b-instruct-q8_0`).

---

## Архитектура проекта

```text
hearthstone-ai-assistant/
├── README.md            # Документация проекта (без эмодзи)
├── AGENTS.md            # Системные инварианты, профиль оборудования и стандарты разработки
├── ROADMAP.md           # Дорожная карта задач и статусы реализации
├── JOURNAL.md           # Инженерный журнал, замеры скорости и бенчмарки
├── docs/
│   └── RESEARCH.md      # Теоретическая основа и обзор литературы (Behavioral Cloning, QLoRA)
├── src/
│   ├── card_db/         # База карт (35 807 сущностей, SQLite + in-memory O(1) кэш)
│   │   ├── enums.py     # Ключевые слова: Титаны, Области, Туристы, Звездолеты, Руны
│   │   ├── indexer.py   # Индексатор CardDefs.ruRU.xml
│   │   ├── models.py    # Датаклассы карт и механизмы сериализации
│   │   └── formatter.py # Форматирование карт для LLM-промптов
│   ├── parser/          # Парсеры логов и реплеев
│   │   ├── log_parser.py             # Потоковый разбор Power.log
│   │   ├── state_tracker.py          # Трекер игрового состояния и TurnSnapshot
│   │   ├── replay_reader.py          # Чтение архивов .hdtreplay и DeckStats.xml
│   │   ├── next_action_dataset.py     # Schema-v2 state + candidates -> chosen ID
│   │   ├── hsreplay_downloader.py    # Авторизованный загрузчик с HSReplay.net
│   │   ├── hsreplay_xml_parser.py    # Парсер HearthSim .hsreplay.xml дерева
│   │   ├── dataset_generator.py      # Генератор локального датасета
│   │   └── build_master_dataset.py   # Сборка единого мастер-датасета
│   ├── coach/           # Ретроспективный тренер
│   │   ├── analyzer.py  # Тактический анализ матча, летальный урон, темп
│   │   └── cli.py       # Консольный интерфейс тренера
│   └── llm/             # Интеграция с языковыми моделями
│       ├── candidate_generator.py # Генератор легальных кандидатов
│       ├── response_parser.py     # Отказоустойчивый парсер ответов модели
│       ├── next_action_contract.py # Общий prompt/response contract schema v2
│       ├── next_action_formatter.py # Frozen game-level splits и ChatML
│       ├── evaluate_next_action.py # Base-model evaluator
│       ├── train_qlora.py          # QLoRA trainer с schema-v2 guard
│       └── ollama_client.py        # Клиент локального Ollama API
├── data/
│   ├── processed/       # Turn baseline, schema-v2 next-action и quarantine
│   └── cache/           # Кэши карт и индексы матчей
└── tests/               # Юнит- и интеграционные тесты
```

---

## Требования к окружению

- **ОС**: Windows 10/11 x64
- **Python**: 3.11 - 3.13
- **GPU**: NVIDIA с поддержкой CUDA (рекомендуется от 6 GB VRAM)
- **Локальный LLM-сервер**: Ollama (`ollama serve`, порт 11434)
- **Модели**: `qwen2.5:1.5b-instruct-q8_0`, `qwen2.5-coder:7b`

---

## Быстрый старт

### 1. Установка зависимостей
```powershell
pip install -r requirements.txt
```

Для QLoRA устанавливайте дополнительные зависимости только после проверки CUDA:
```powershell
pip install -r requirements-qlora.txt
python -m src.llm.train_qlora --check-environment
```

### 2. Запуск тестов
```powershell
python -m pytest tests/ -v
```

### 3. Индексация базы карт
```powershell
python -m src.card_db.indexer
```

### 4. Разбор последнего сыгранного матча
```powershell
python -m src.coach.cli --latest
```

### 5. Сборка production next-action датасета
```powershell
python -m src.parser.next_action_dataset
```

Команда перезаписывает `train_next_actions.jsonl`, `train_next_actions_quarantine.jsonl` и `next_action_validation_report.json`. Наличие production-артефакта не означает готовность к QLoRA: актуальный verdict хранится в отчёте и сейчас равен `false`.

### 6. Подготовка frozen QLoRA splits
```powershell
python -m src.llm.next_action_formatter
python -m src.llm.train_qlora --validate-only
python -m src.llm.train_qlora --check-environment
```

Formatter создаёт `next_action_split_manifest_v1.json` и четыре ChatML-файла: train, validation, test и temporal holdout. Manifest привязан к SHA-256 accepted schema-v2 датасета; его нельзя заменить без явного `--replace-manifest`.

### 7. Base-model benchmark
```powershell
python -m src.llm.evaluate_next_action --input data/processed/next_action_test_chatml.jsonl --model qwen2.5:1.5b
```

Evaluator считает top-1 accuracy по `chosen_candidate_id`, корректность формата ответа, существование выбранного кандидата, latency и разбивку по типам действий. При недоступном Ollama команда завершится с ненулевым exit code и сохранит report со статусом `blocked`.

---

## Результаты бенчмарков

- **Индексация карт**: 35 807 карт распарсено за 1.4 секунды, скорость поиска из памяти: > 20 млн lookups/sec.
- **Turn baseline**: 3 049 записей, 11 704 действия из 549 ranked-win replay; SHA-256 `2d9a0679e667510fe9eedfb062d226d574deef3b251f3f6d023f0ceb734a7736`.
- **Next-action schema v2**: 12 840 accepted / 1 862 quarantine из 14 702 option decisions, coverage 87.3351%; accepted gate violations — 0.
- **Frozen next-action splits**: 9 320 train / 1 198 validation / 1 117 test / 1 205 temporal holdout; 540 игр распределены без пересечений по `game_id`.
- **Readiness status (2026-09-03)**: formatted schema-v2 artifacts and evaluator are ready; QLoRA remains blocked by missing `datasets`, `trl`, `bitsandbytes`, CPU-only PyTorch, and unavailable Ollama baseline.
