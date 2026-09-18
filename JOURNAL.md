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

Production `data/processed/train_actions.jsonl` был перезаписан следующим Stage B/C прогоном; актуальные метрики приведены ниже.

---

## 2026-09-03 — Stage B/C: replay option oracle и next-action schema v2

### 1. Реализация

- `log_parser.py` разбирает `DebugPrintOptions`, `SendOption` и `CHANGE_ENTITY`.
- `state_tracker.py` сохраняет pre-action snapshot, полный legal option set, выбранный `option/sub-option/target/position`, стабильные entity IDs и board position.
- Исправлены board limit 7, hero attacks, immune/can't-be-attacked, Windfury/Mega-Windfury, Rush face restriction, dynamic hero-power cost и `END_TURN`.
- `UNKNOWN ENTITY` больше не перекрывает имя CardDB при известном свежем `cardId`.
- `next_action_dataset.py` потоково и атомарно пишет schema-v2 accepted/quarantine/report и повторно аудирует accepted-файл.

### 2. Production-метрики

- `train_actions.jsonl`: 3 049 turn records, 538 непустых уникальных `game_id`, 11 704 действия, SHA-256 `2d9a0679e667510fe9eedfb062d226d574deef3b251f3f6d023f0ceb734a7736`.
- Источник schema v2: 549 ranked-win replay; option decisions присутствуют в 539 играх.
- Всего option selections: 14 691; accepted: 12 829; quarantine: 1 862; coverage: 87.3256%.
- Accepted action counts: `PLAY=5766`, `ATTACK=3400`, `END_TURN=2760`, `HERO_POWER=597`, `LOCATION=294`, `POWER=12`.
- Quarantine: `tradeable_option_semantics_unproven=1725`, `candidate_mana_cost_mismatch=105`, `suboption_target_cross_product_unproven=32`.
- Независимый JSONL-аудит: 12 829 уникальных accepted decisions, 539 games, violations `{}`; SHA-256 `2fa1e2c70c57f22847b0f18e9ff2d7914aef63bb15b0eb152a4a5b0dd0a4df0e`.

### 3. Readiness

- QLoRA readiness: `false`.
- Блокеры: три quarantine-класса, 10 ranked replay без option decisions, незамороженные validation/test splits, неревалидированное training environment и отсутствующий base-model benchmark.
- Обучение, установка ML-зависимостей, `ml-intern`, commit и push не выполнялись.

### 4. Финальные проверки

- Focused Stage B/C: `34 passed in 10.38s`, exit code 0.
- Полный suite: `49 passed in 12.54s`, exit code 0.
- `py -m compileall -q src`: exit code 0.
- `git diff --check`: exit code 0; выведены только предупреждения о будущем преобразовании LF в CRLF.
- Отдельная проверка modified/untracked source и docs: trailing whitespace не найден.
- Независимый повторный review закрыл все четыре P1 и P2 finding после regression fixes; новых подтверждённых дефектов в проверенном scope не осталось.
- Все `.pytest-tmp-*` разрешены внутри корня репозитория, но два вызова `Remove-Item -Recurse -Force` отклонены execution policy; каталоги остались как неотслеживаемые временные файлы.

---

## 2026-09-03 — Readiness Stage D: frozen splits, shared contract и evaluator

### 1. Реализовано

- `src/llm/next_action_contract.py` вынес общий state/candidates prompt и строгий `PLAN: [candidate_id]` response contract.
- `src/coach/analyzer.py` использует тот же prompt builder, что и schema-v2 formatter.
- `src/llm/next_action_formatter.py` создает immutable-by-default manifest с hash-lock исходного accepted JSONL и game-level train/validation/test/temporal holdout.
- `src/llm/train_qlora.py` больше не принимает legacy free-text конфигурацию; `--validate-only` проверяет schema-v2, prompt/completion и принадлежность frozen split.
- `src/llm/evaluate_next_action.py` добавляет Ollama base-model benchmark с top-1, форматом, candidate existence, latency и action breakdown.

### 2. Артефакты

- `next_action_split_manifest_v1.json`: 540 игр, 12 840 accepted records, SHA-256 исходного accepted JSONL `49d5a8163d77d7aef8a158ba69d9c4850dce84911d5b8a44d28c4e8235e6e3df`.
- Formatted files: train `9320`, validation `1198`, test `1117`, temporal holdout `1205` records.
- Quarantine в первой версии формально исключен политикой manifest: `accepted schema-v2 only`.
- После lazy-snapshot correction один replay без marker перехода хода восстановлен: текущая coverage — 12 840 accepted из 14 702 option selections (`87.3351%`); девять ranked-win replay по-прежнему не содержат `DebugPrintOptions`/`SendOption`.

### 3. Проверки и фактические блокеры

- Full suite: `57 passed in 13.29s`, exit code 0.
- `py -m compileall -q src`: exit code 0.
- `git diff --check`: exit code 0; только LF/CRLF warnings.
- `py -m src.llm.train_qlora --validate-only`: train `9320/390 games`, eval `1198/48 games`, test `1117/48 games`, temporal `1205/54 games`.
- `py -m src.llm.train_qlora --check-environment`: `datasets`, `trl`, `bitsandbytes` отсутствуют; `torch 2.9.0+cpu`, CUDA unavailable; `qlora_ready=false`.
- Base-model smoke против `qwen2.5:1.5b`: Ollama `127.0.0.1:11434` timeout; report status `blocked`, exit code non-zero.
Обучение, установка зависимостей, commit и push не выполнялись.

---

## 2026-09-12 — Архитектурная заметка: Post-Training через NSD / DPO (Negative Feedback Loop)

### 1. Суть концепции (исследование arXiv:2609.11699, Negative Self-Distillation)
В стандартном имитационном обучении (Behavioral Cloning / SFT) модель обучается исключительно на положительных примерах — действиях легендарного игрока из `train_master_actions.jsonl` / `schema-v2`. При этом модель не понимает *почему* альтернативные ходы плохи или нелегальны.
Метод Negative Self-Distillation (NSD) предлагает целенаправленно штрафовать модель за самосгенерированные ошибки через динамический вентильный механизм (dynamic gating), предотвращающий языковой коллапс.

### 2. Применение к Hearthstone AI Assistant
Hearthstone обладает идеальной детерминированной средой для генерации пар предпочтений (Preference Pairs):
1. **Нелегальные ходы**: попытка разыграть карту без маны, атаковать существо сквозь провокацию (`Taunt`) или скрытность (`Stealth`), применить заклинание по невалидной цели (`Elusive`).
2. **Тактические зевки**: размен существ не в том порядке (потеря урона), пропуск очевидного летального урона (`Lethal Detector`), разыгрывание карт с `Battlecry` без учёта переполнения стола.

### 3. Регламент и последовательность (YAGNI & Staging)
- **Блокер текущего этапа**: Внедрять NSD/DPO прямо сейчас **категорически запрещено**. Модель ещё не прошла базовую SFT/QLoRA калибровку. Обучение на ошибках до освоения синтаксиса контракта `PLAN: [candidate_id]` разрушит базовые веса.
- **Фаза 1 (Текущая)**: Закрыть блокеры данных (quarantine-классы, CUDA-рантайм) и обучить чистый базовый SFT QLoRA на `schema-v2` (модель `Qwen3-4B-Instruct` / `Qwen2.5-Coder-7B`).
- **Фаза 2 (Post-Training / DPO через `ml-intern`)**:
  - Прогнать базовую модель через `evaluate_next_action.py` и симулятор стола.
  - Собрать датасет пар предпочтений: `(Prompt: State + Legal Candidates, Chosen: Реплейное действие, Rejected: Самосгенерированная ошибка модели)`.
  - Запустить DPO (Direct Preference Optimization) в `ml-intern` (`TRL / DPOTrainer`) с сохранением формата контракта.

---

## 2026-09-17 — Архитектурная ревизия: NLI Cross-Encoder / Jev-архитектура (openjev) как кандидат на селектор действий

### 1. Контекст и предпосылки исследования
- 16 сентября 2026 года TypeSafe AI представили парадигму System One Models (Jev) — one-pass scoring дискретных опций без авторегрессионной генерации токенов. В тот же день AlexWortega опубликовал `AlexWortega/openjev` — открытое воспроизведение этой концепции на базе декодера `Qwen3.5-4B`, дообученного как 3-классовый NLI Cross-Encoder (`contradiction`, `entailment`, `neutral`) по методологии Lee Miller (`dleemiller`).
- Аудит текущего контракта проекта `schema-v2` (`state + replay-reported legal candidates -> chosen candidate ID`) показал 100% соответствие задаче Jev:
  - Выбор ровно одного действия из дискретного набора легальных кандидатов (среднее 9.9 кандидатов на шаг, min 1, max 84).
  - Текущая авторегрессионная схема с генерацией строки `PLAN: [id]` через Ollama создает оверхед по задержке (0.8–1.2 с на шаг) и подвержена ошибкам парсинга (`format_valid` / `candidate_exists`).

### 2. Сравнительный анализ архитектур: Авторегрессионный SFT vs. Jev / openjev
| Критерий | Текущий контур (Ollama `qwen2.5:1.5b-instruct` / SFT QLoRA) | Альтернативный контур (Jev / openjev NLI Cross-Encoder) |
| :--- | :--- | :--- |
| **Парадигма инференса** | Генеративная авторегрессия токенов строки `PLAN: [id]`. | One-pass прямой проход, ранжирование опций по $P(\text{entailment})$. |
| **Задержка (Latency)** | 800–1 200 мс на действие (5–6 с на ход). | **40–70 мс** на весь батч кандидатов на GPU. |
| **Надежность схемы** | Риск `format_valid=False` (лишние токены) и `candidate_exists=False` (галлюцинация несуществующего ID). | **100% валидность формата**, 0% галлюцинаций ID (выбор строго по списку легальных кандидатов). |
| **Сложность обучения** | Блокирован: требует `bitsandbytes`, `trl`, CUDA SFT пайплайна под генерацию токенов. | Минимальная: обучение лёгкого зонда `LatentMLPHead` на латентах за 2–3 минуты без тяжелых зависимостей. |
| **VRAM и совместимость** | 1.6 ГБ (`qwen2.5:1.5b-instruct-q8_0`) до 8 ГБ (7B). | 1.6 ГБ (`ModernCE-large-nli`), ~4.5 ГБ (`openjev-2B`), ~9 ГБ (`openjev-4B`). |

### 3. План валидации и интеграции
1. **Сбор весовых артефактов**:
   - Скачивание `AlexWortega/openjev` (4B NLI, 9.07 GB) и `dleemiller/ModernCE-large-nli` (ModernBERT-large 395M, 1.58 GB) в `D:\models\`.
2. **Offline Zero-shot Benchmark**:
   - Создать экспериментальный селектор `src/llm/openjev_client.py`, совместимый с интерфейсом `evaluate_next_action.py`.
   - Запустить оценку zero-shot ранжирования на `next_action_test_chatml.jsonl` (1 117 записей) по метрике `top1_accuracy`.
   - Проверить влияние формулировки гипотезы (прямое действие vs тактическое утверждение State-Statement).
3. **Обучение LatentMLPHead на реплеях Легенды**:
   - Прогнать 9 320 обучающих примеров через замороженный backbone, извлечь скрытые векторы последнего токена (`last-token pooling`).
   - Обучить `LatentMLPHead` (архитектура `Linear(d, 512) -> GELU -> Dropout -> Linear(512, 1)`) с функцией потерь `soft_bce` на `chosen_candidate_id`.
   - Замерить `top1_accuracy` на валидационном (1 198) и тестовом (1 117) срезах.
4. **Решение о целевой архитектуре**:
   - Сопоставить latency, точность попадания в ход игрока и стабильность работы.
   - При подтверждении задержки (<100 мс) и сопоставимой точности — утвердить NLI Cross-Encoder / Jev как основной рантайм вместо медленного авторегрессионного контура.

### 4. Руководство по запуску и инференсу моделей (Runbook)

#### Почему НЕ `llama-server.exe`:
- Сервер `llama-server.exe` спроектирован для авторегрессионных генеративных моделей в формате GGUF с протоколом генерации текста `/v1/chat/completions`.
- `openjev` и `ModernCE` — это **NLI Cross-Encoders** (архитектура `ForSequenceClassification`), сохраненные в формате PyTorch Safetensors.
- Они не генерируют токены, а вычисляют логиты классификации над парой текстов (`Premise` + `Hypothesis`). Напрямую через `llama-server` без специального GGUF-преобразования головы классификации они не запускаются.

#### Вариант A: In-Process в Python (Рекомендуемый для минимальной задержки)
Прямой вызов внутри проекта исключает оверхед HTTP-сокетов и обеспечивает чистый latency в ~40–60 мс на GPU.

1. **Запуск `openjev-4b` (`D:\models\openjev-4b`)**:
```python
import sys
import torch
sys.path.append(r"D:\models\openjev-4b")
from modeling_openjev import OpenJevCrossEncoder

# Загрузка локальных весов (рекомендуется bfloat16 на GPU)
jev = OpenJevCrossEncoder(
    path=r"D:\models\openjev-4b",
    subfolder="qwen3.5-4b-nli",
    device="cuda",
    dtype=torch.bfloat16
)

# Ранжирование кандидатов за один прямой проход:
best_opt_idx = jev.rerank(
    question=state_prompt,
    options=[c["description"] for c in candidates],
    hyp_fmt="Следующее действие: {}"
)
chosen_candidate_id = candidates[best_opt_idx]["id"]
```

2. **Запуск `ModernCE-large-nli` (`D:\models\ModernCE-large-nli`)**:
Легковесный бейзлайн (395M параметров, ~1.5 ГБ VRAM, инференс ~15 мс):
```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

path = r"D:\models\ModernCE-large-nli"
tok = AutoTokenizer.from_pretrained(path)
model = AutoModelForSequenceClassification.from_pretrained(
    path, dtype=torch.float16
).cuda().eval()

# Пакетное кодирование пар (State, Candidate)
pairs = [(state_prompt, f"Следующее действие: {c['description']}") for c in candidates]
enc = tok(
    [f"Premise: {p}\nHypothesis: {h}" for p, h in pairs],
    padding=True, truncation=True, return_tensors="pt"
).to("cuda")

with torch.no_grad():
    logits = model(**enc).logits
    # Метки dleemiller: 0 = contradiction, 1 = entailment, 2 = neutral
    entailment_scores = logits[:, 1]
    best_opt_idx = int(entailment_scores.argmax().cpu())
chosen_candidate_id = candidates[best_opt_idx]["id"]
```

#### Вариант B: Автономный локальный микросервис (FastAPI)
Если необходимо сохранить изоляцию процессов между трекером Hearthstone и ML-рантаймом (по аналогии с Ollama):
- Создается легковесный сервис `scripts/serve_openjev.py` с эндпоинтом `POST /rerank`.
- Запуск: `uvicorn scripts.serve_openjev:app --host 127.0.0.1 --port 8000`.
- Ассистент отправляет `{"state": state_prompt, "candidates": candidate_list}` и за 50–60 мс получает `{"chosen_id": 7}`.

#### Аппаратный профиль и VRAM на RTX 4060 (8 GB):
- `ModernCE-large-nli` (395M): занимает всего ~1.5 ГБ VRAM, оставляя 6.5 ГБ под игру Hearthstone и фоновые процессы.
- `openjev-4b` (4B): в `bfloat16` занимает ~8.6 ГБ VRAM. При параллельно запущенной игре может вызывать частичный spillover в Shared RAM. При необходимости экономии памяти доступна загрузка в 4-битном режиме через `bitsandbytes` (`load_in_4bit=True`), снижающая потребление до ~3.5 ГБ VRAM.




## 2026-09-17 — Проверка кандидата NLI: уточнение статуса и runbook

Подробный разбор и TODO: [plans/2026-09-17-nli-selector-assessment.md](plans/2026-09-17-nli-selector-assessment.md).

- openjev/ModernCE остаются экспериментальными кандидатами на селектор. Цифры 40–70 мс и 2–3 минуты из предыдущей записи не подтверждены локальным Hearthstone benchmark.
- ModernCE требует pair-tokenization; config и model card расходятся по порядку меток. Нужен sanity-check, а для P(entailment) — softmax вместо сравнения сырых logits.
- openjev BF16 не укладывается в RTX 4060 8 GB с резервом. Wrapper не поддерживает прямой флаг 4-bit; режим квантизации отдельно не проверен.
- CUDA Torch уже есть в D:/AI: voodoo-dyn-quant, exllamav3, ComfyUI embedded и kohya_ss. В voodoo подтверждены CUDA-операция, transformers 5.17.0 и импорты обоих классификаторов (exit 0). CPU-only 2.9.0 относится только к системному py.
- На текущем test expected random top1 = 29.22%, first-candidate = 21.49%. Train содержит 91 831 пары state/candidate. Заявления о качестве и скорости требуют отдельного model forward и frozen benchmark.
- Исходный код и model config в этой сессии не менялись; обучение и инференс весов не запускались.

## 2026-09-17 — ModernCE frozen validation: zero-shot кандидат отклонен как base runtime

- Добавлен `src/llm/evaluate_modernce.py`; полный отчет: `data/processed/next_action_baseline_modernce_validation.json`.
- Среда: `D:\AI\voodoo-dyn-quant\.venv`, torch `2.14.0+cu132`, transformers `5.17.0`, RTX 4060, FP16.
- Sanity-check подтвердил реальный порядок logits `[contradiction, entailment, neutral]`; локальный `config.json` подписывает их неверно. Использован entailment index 1 и pair-tokenization.
- Frozen validation: 1198 решений, 10 638 пар, errors/truncations 0, peak allocated 1197.875 MiB.
- Top-1 all 31.55%; при N>1 — 23.08% против expected random 18.98% и first-candidate 11.91%. Delta к random +4.10 п.п.; game-cluster bootstrap 95% CI +1.00...+7.75 п.п.
- Top-3 all 60.10%; при N>1 — 55.16%.
- Latency: p50 74.213 ms, p95 398.193 ms, max 1746.969 ms. Заявление 40–70 ms на любой набор кандидатов опровергнуто.
- Критический class breakdown при N>1: `END_TURN=0/127`, `HERO_POWER=0/64`. Из 1066 предсказаний модель выбрала `PLAY=792`, `ATTACK=258`, `LOCATION=12`, `END_TURN=3`, `POWER=1`, `HERO_POWER=0`; все три END_TURN были ошибочны.
- Английский hypothesis template на первых 100 validation решений не изменил top-1: 22%.
- Решение: zero-shot ModernCE не продвигать как базовую модель/greedy selector. Test и temporal не запускались. Допустимое продолжение ветки — supervised listwise head на frozen embeddings с game-level validation и обязательным контролем END_TURN/HERO_POWER.
- Верификация: benchmark exit code 0; `py_compile` и `git diff --check` для runner exit code 0. Commit/push не выполнялись.

## 2026-09-17 — NVIDIA driver fault и обязательный resource guard

- Во время пилотного извлечения 2056 ModernCE embeddings пользователь наблюдал около 11 GB общей GPU memory, 100% CPU и OOM/сбой. Полный train extraction не запускался.
- Windows System log подтвердил `nvlddmkm`, Event ID 153 в 19:51:37: `Error occurred on GPUID: 100`. Централизованного Resource-Exhaustion Event 2004 не найдено.
- Ранее опубликованный `torch.cuda.max_memory_allocated=1221.41 MiB` отражал только allocator PyTorch и не описывал WDDM, driver workspace и shared GPU memory. Использовать его как доказательство безопасности было ошибкой.
- Добавлен `src/llm/resource_guard.py`; `evaluate_modernce.py` переведен на безопасный профиль: batch max 2, CPU threads 2, inter-op 1, tokenizer parallelism off, process priority Below Normal, PyTorch allocator cap 40% VRAM, SDPA, `reference_compile=false`, cooldown и telemetry watchdog.
- Run блокируется по живой телеметрии при GPU used >4096 MiB, GPU free <4096 MiB, available RAM <8192 MiB или GPU temperature >75 C. После 60 секунд небезопасного состояния процесс завершается вместо продолжения к OOM. Временные окна и отложенные запуски не используются: момент запуска определяет пользователь.
- Любой будущий extractor обязан писать FP16 features небольшими атомарными чанками/memmap с flush/checkpoint, не держать полный корпус в RAM и не запускаться параллельно с игрой или другими GPU workloads.
- Выполнен ручной безопасный pilot на 200 validation решений / 1683 парах: batch 1, errors 0, truncations 0. По 212 watchdog-замерам max total GPU used 2303 MiB, min GPU free 5654 MiB, min RAM available 14484 MiB, max GPU temperature 43 C. PyTorch peak allocated 775.678 MiB.
- Pilot quality: top-1 all 28.0%, top-1 при N>1 24.61%, top-3 all 64.0%; latency p50 502.231 ms, p95 2145.554 ms. Это проверка безопасного режима, не основание менять прежнее решение по zero-shot модели.
- Проверки: `7 passed in 0.24s`; `py_compile`, `git diff --check` и pilot exit code 0. Автоматизация отложенного запуска удалена; новых расписаний не создано.

## 2026-09-18 — Автономный full supervised ModernCE run

- Добавлен `src/llm/train_modernce_ranker.py`: frozen ModernCE mean-pooled features, локальный FP16 memmap, атомарный checkpoint каждые 25 решений, batch 2, существующий GPU/RAM/temperature watchdog и простая linear listwise head на CPU.
- Full scope: frozen train 9320 решений / 91831 пар и validation 1198 / 10638. Test и temporal не используются.
- Минимальный CUDA smoke 4+4 решения завершён полностью с exit code 0; CPU checks: 10 passed, `py_compile` и `git diff --check` exit code 0.
- По прямой команде пользователя full run запущен вручную без расписания как скрытый автономный процесс PID 35760. Логи: `data/processed/modernce_ranker/full_run.stdout.log` и `full_run.stderr.log`; прогресс сохраняется в `data/processed/modernce_ranker/cache/` и может быть продолжен после остановки.
- Codex не опрашивает процесс в фоне. Фактический результат будет считаться подтверждённым только после завершения процесса и чтения `data/processed/modernce_ranker/report.json` по следующему прямому запросу пользователя.

## 2026-09-18 — Full supervised ModernCE run завершён

- Извлечены все frozen features: train 9320 решений / 91831 пар и validation 1198 / 10638 пар. Обрезанных входов 0. GPU OOM и ошибок процесса не было; PID 35760 завершился штатно.
- ResourceGuard: 6759 samples, max total GPU used 3659 MiB, min GPU free 4298 MiB, min RAM available 11219 MiB, max GPU temperature 65 C.
- Linear head: best epoch 7/10. Validation top-1 all 35.893%; top-1 при N>1 27.955% против корректного expected random N>1 18.978% (+8.977 п.п.); top-3 all 64.692%.
- В сравнении с прежним zero-shot ModernCE (31.55% all; 23.08% N>1) голова дала +4.34 п.п. all и +4.87 п.п. N>1 на validation. Это ещё не test/temporal evidence.
- Breakdown best epoch: ATTACK 52.76%, END_TURN 50.97%, PLAY 26.06%, LOCATION 31.25%, HERO_POWER 0/64, POWER 50% на 2 случаях. Нулевая HERO_POWER остаётся блокером для базового runtime.
- Артефакты: `data/processed/modernce_ranker/head.pt`, `report.json`, FP16 memmap cache и stdout/stderr logs. JSON baseline для N>1 исправлен и повторно проверен; tests 10 passed, `py_compile`, `git diff --check` exit code 0.

## 2026-09-18 — CPU-only tuning редких action types запущен

- Добавлен `src/llm/tune_modernce_ranker.py`, который использует готовые frozen features без повторного ModernCE/GPU forward.
- Внутри train создан deterministic game-level split: 332 fit games / 58 dev games. Validation не участвует в выборе конфигурации и оценивается один раз после refit на полном train.
- Абляции: unweighted linear, sqrt-inverse weighted linear, weighted linear + candidate-type one-hot, weighted 128-hidden MLP + type one-hot. Веса считаются только на fit и clipped около 5x; `POWER` не получает экстремальный вес.
- Primary selector: mean NDCG@5 на internal dev. Небазовый вариант принимается только при положительной нижней границе paired game-bootstrap 95% CI против unweighted linear. HERO_POWER остаётся diagnostic, а не единственным selector. Для выбранного варианта проверяются seeds 41/42/43, затем выполняется refit на полном train.
- ML Intern подтвердил необходимость game-level isolation, fold-local weighting, type one-hot ablation и rare-class diagnostic. Его оценка random как reciprocal mean list length отвергнута: корректный baseline усредняет `1/N` на том же decision set.
- Focused verification: 14 passed; `py_compile`, `git diff --check` exit code 0. CPU-only job запущен автономно PID 20848, BelowNormal, 2 threads; логи `data/processed/modernce_ranker/tuning/tuning.stdout.log` и `tuning.stderr.log`. Codex не опрашивает процесс без нового запроса пользователя.

## 2026-09-18 — Tempered class-weight experiment завершён

- `tune_modernce_ranker.py` расширен CPU-only сеткой `w^alpha`, где `alpha=0.25/0.5/0.75`, с/без candidate action-type one-hot. Добавлен `--cpu-threads`; по прямой команде пользователя эксперимент выполнен на 8 потоках по существующему frozen feature cache без GPU forward и без чтения test/temporal.
- Защитный selector допускает небазовый вариант только при `PLAY >= 80%` от unweighted baseline, `END_TURN >= 10%`, `HERO_POWER >= 10%` и paired game-bootstrap NDCG@5 lower CI `>= -0.01`. Среди прошедших вариантов primary metric — multi-candidate top-1.
- Ни один tempered-вариант не прошёл все гейты. Лучший по общей точности `linear_tempered_type_025` дал dev N>1 top-1 28.40% против 25.08%, NDCG@5 0.5869 против 0.5642 и CI дельты `[+0.0029, +0.0330]`, но `PLAY` упал до 15.70% и `END_TURN` остался 0%; вариант отклонён.
- Лучший компромисс без type-feature, `linear_tempered_025`, сохранил `PLAY=19.19%`, поднял `HERO_POWER=22.54%` и N>1 top-1 до 26.05%, но `END_TURN=0.77%`; вариант отклонён.
- Варианты `alpha=0.75` подняли `END_TURN` до 11.54–16.92% и `HERO_POWER` до 63.38–67.61%, но обрушили `PLAY` до 8.14–11.05% и не прошли NDCG non-inferiority. Корень проблемы не сводится к силе class weights.
- Selector оставил `linear_unweighted`; его повторная validation совпала с прошлым результатом: top-1 all 34.14%, N>1 25.98%, top-3 63.02%, NDCG@5 0.5642. Test и temporal по-прежнему не использованы.
- Артефакты эксперимента: `data/processed/modernce_ranker/tuning_tempered/head.pt`, `report.json`, `run.log`. Время 101.768 s, CPU threads 8, OOM/ошибок процесса не было. Проверки: 16 passed; `py_compile` и `git diff --check` exit code 0.
- Обязательный ML Intern review был запрошен дважды: первая попытка искала проект из неверного cwd, вторая зависла без содержательного ответа и была остановлена; результаты эксперимента от него не заявляются.

## 2026-09-18 — Закреплённый план следующего этапа: factorized action-type ranker

Статус: **planned, не запущен**. Продолжать сетку class weights запрещено: validation показала устойчивый конфликт между редкими типами и `PLAY`.

1. На существующем train feature cache реализовать двухчастное распределение без нового GPU extraction:
   - `P(type | state, candidate set)` — state-conditioned type gate по агрегату candidate embeddings с маской только реально доступных типов;
   - `P(candidate | type, state)` — listwise ranking только среди кандидатов одного типа;
   - итоговый score: `log P(type) + log P(candidate | type)`, без глобального one-hot bias, который вызвал текущий коллапс.
2. Все архитектурные решения и коэффициенты выбирать только на прежнем game-level internal dev. Validation использовать как вторичный regression check; она уже наблюдалась в текущих экспериментах и больше не считается полностью нетронутой финальной оценкой.
3. Обязательные internal-dev гейты перед refit: `PLAY >= 80%` unweighted baseline; `END_TURN >= 10%`; `HERO_POWER >= 10%`; multi-candidate top-1 выше baseline; paired game-bootstrap NDCG@5 lower CI `>= -0.01`; стабильность на seeds 41/42/43.
4. Только после прохождения гейтов выполнить refit на полном train и один validation regression check. При провале любого гейта ветка ModernCE не продвигается в runtime.
5. При успешной фиксации архитектуры безопасно извлечь test/temporal embeddings через существующий ResourceGuard и один раз оценить замороженную модель. До этой точки test и temporal не открывать.
6. Runtime-интеграция разрешена только при сохранении выигрыша над decision-level random baseline на test и temporal, отсутствии class collapse и соблюдении локального latency/resource бюджета RTX 4060. Иначе ModernCE-ветка документируется как исследовательская и закрывается.
