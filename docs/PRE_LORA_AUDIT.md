# Pre-LoRA readiness audit

Дата аудита: 2026-09-01
Статус: **полное обучение заблокировано до исправления данных и тренировочного контура**

## 1. Решение

Текущий проект не готов к полному QLoRA-обучению. Основной риск создаёт датасет: он содержит неизвестные действия, противоречия между героем и силой героя, действия при недостаточной мане и карты, которых нет во входном состоянии. Тренировочный формат также расходится с production-инференсом.

Запуск LoRA на текущих файлах закрепит ошибки парсера и научит модель отвечать в формате, который приложение не использует. Смена базовой модели не устранит эти дефекты.

Рекомендуемый порядок работ:

1. Исправить восстановление состояния и владельца действий.
2. Сделать пошаговый датасет `state + legal candidates -> next action`.
3. Добавить автоматический data-quality gate.
4. Заморозить отдельный test set и автоматизировать benchmark.
5. Починить и зафиксировать ML-окружение.
6. Запустить короткий QLoRA smoke-test на `Qwen/Qwen3-4B-Instruct-2507`.
7. Переходить к полному обучению только после сравнения base и LoRA на одном тесте.

## 2. Объём аудита

Аудит включал:

- исходные и отформатированные JSONL-датасеты;
- parser и state tracker;
- генератор легальных кандидатов и parser ответа модели;
- QLoRA-скрипт, конфигурацию и установленные зависимости;
- существующие тесты и benchmark-файлы;
- модели размером 3B-7B по официальным Hugging Face model cards;
- состояние GPU, Ollama и Git.

Аудит не включал полное обучение, загрузку весов новой модели и изменение production-кода.

## 3. Фактически выполненные проверки

### 3.1. Тесты

Команда:

```powershell
py -m pytest tests -q
```

Результат:

```text
28 passed in 8.07s
exit code: 0
```

Тесты подтверждают работу существующих unit-сценариев. Они не подтверждают семантическую корректность production-датасета. Например, `tests/test_dataset_formatter.py` проверяет структуру ChatML и наличие выходных файлов, но не валидирует владельца действия, ману, цели или соответствие карты руке.

### 3.2. QLoRA smoke-test

Команда:

```powershell
py -m src.llm.train_qlora --smoke-test
```

Результат:

```text
ModuleNotFoundError: No module named 'datasets'
exit code: 1
```

Скрипт остановился до загрузки модели и до первого training step.

### 3.3. GPU

`nvidia-smi` во время аудита показал:

```text
NVIDIA GeForce RTX 4060
total VRAM: 8188 MiB
free VRAM: 6011 MiB
driver: 610.88
```

Перед QLoRA нужен отдельный pre-flight. Проект должен учитывать занятое место, активации, CUDA runtime и резерв 15-20%, а не только размер NF4-весов.

### 3.4. Ollama

Запросы к `http://127.0.0.1:11434/api/tags` и `/api/ps` получили отказ в подключении. Ollama во время аудита не работал, поэтому список локальных моделей и фактическое размещение в VRAM не подтверждены.

### 3.5. Git

До создания этого документа рабочее дерево было чистым. `git diff --check` завершился с exit code 0.

## 4. Состояние датасетов

### 4.1. Сводные результаты

| Файл | Записей | Игр | `Unknown` в completion | Очевидно проблемные записи | Разыгранная карта отсутствует во входе |
|---|---:|---:|---:|---:|---:|
| `train_actions.jsonl` | 5 174 | 515 | 391 | 986, 19,1% | 5 758 из 10 738 действий, 53,6% |
| `train_hsreplay_actions.jsonl` | 2 337 | 219 | 1 345 | 1 393, 59,6% | 3 004 из 4 734 действий, 63,5% |
| `train_master_actions.jsonl` | 7 486 | 734 | 1 715 | 2 358, 31,5% | 8 715 из 15 425 действий, 56,5% |

Колонка «очевидно проблемные записи» считает строки, в которых найден `Unknown`/`Неизвестно` или использование силы героя при доступной мане меньше 2. Это нижняя оценка: проверка не симулировала полную последовательность действий и не проверяла все требования карт.

Отдельно найдено 462 примера, где completion использует силу героя при `0` доступной маны.

Отсутствие разыгранной карты во входном prompt имеет два возможных объяснения:

- parser потерял карту или перепутал состояние;
- игрок получил карту после добора, раскопки, случайной генерации или другого действия внутри хода.

Оба варианта делают текущую пару непригодной для схемы «состояние в начале хода -> вся последовательность». Модель не может вывести скрытый будущий добор из начального состояния.

### 4.2. HSReplay subset

HSReplay subset нельзя добавлять в обучение в текущем виде:

- 1 345 из 2 337 completion содержат неизвестную карту или сущность;
- в 1 965 prompt герой или герой противника указан общим либо ошибочным именем;
- найдено 25 повторяющихся prompt с конфликтующими completion;
- в `player_hero` встречаются названия существ и заклинаний;
- названия hero power не совпадают с указанным классом героя.

Сначала нужно исправить `hsreplay_xml_parser.py` и проверить выбор friendly player на реальных XML. До этого лучше обучаться на меньшем локальном наборе, который прошёл строгую валидацию.

### 4.3. Локальный subset

Локальные replay тоже содержат противоречия. В одном из первых примеров prompt указывает Death Knight, а completion использует шаманскую силу героя «Призыв тотема» при `0/1` маны. В выборке также встречаются пары наподобие:

- `Warlock -> Поднять щит!`;
- `Warlock -> Призыв тотема`;
- `Mage -> Смена облика`;
- `Rogue -> Жизнеотвод`.

Это указывает на смешивание действий между игроками или снимками хода. Точную первопричину нужно подтвердить трассировкой конкретного replay через `GameStateTracker`.

### 4.4. Master не используется при SFT

`src/llm/dataset_formatter.py` по умолчанию читает:

```text
data/processed/train_actions.jsonl
```

Он не читает `train_master_actions.jsonl`. Поэтому текущие SFT-файлы содержат 5 174 локальных примера, хотя README и ROADMAP описывают master на 7 486 примеров как источник QLoRA.

Фактический SFT split:

| Split | Записей | Игр | `Unknown` | Hero power при 0 маны |
|---|---:|---:|---:|---:|
| train | 4 677 | 463 | 355 | 428 |
| eval | 497 | 52 | 36 | 34 |

Положительный результат: train и eval не пересекаются по `game_id`; совпадающих prompt между ними также не найдено.

### 4.5. Дедупликация

`src/parser/build_master_dataset.py` дедуплицирует prompt через встроенный `hash()` Python. Хеш меняется между процессами и теоретически допускает коллизии. Для этой задачи достаточно хранить сам prompt в set или использовать стабильный SHA-256, если нужен компактный идентификатор.

Проверенный master не содержит повторяющихся prompt, но текущий способ остаётся лишним источником недетерминизма.

## 5. Неверный обучающий контракт

### 5.1. Training и inference используют разные задачи

Текущий SFT prompt содержит состояние стола и вопрос:

```text
Каковы наилучшие действия на этом ходу?
```

Completion содержит свободный текст:

```text
1. Разыграть карту: ...
2. Атаковать: ...
3. Завершить ход
```

Production inference формирует другой prompt: он передаёт пронумерованный список кандидатов и требует:

```text
ПЛАН: [1, 3]
ОБОСНОВАНИЕ: ...
```

LoRA будет оптимизировать формат, который production parser не ожидает. Response parser попытается вытащить числа из свободного текста, но такой fallback не заменяет согласованный контракт.

### 5.2. Full-turn imitation использует неизвестное будущее

Один snapshot фиксирует начало хода, после чего completion включает все действия до конца хода. После первой карты состояние может измениться:

- игрок добирает или создаёт карту;
- стоимость карт меняется;
- провокатор умирает и открывает новые цели;
- освобождается или заполняется место на столе;
- игрок получает временную ману;
- секрет, location или hero power меняет доступные действия.

Один статичный список кандидатов не описывает такую последовательность. Нужен новый snapshot после каждого действия.

### 5.3. Победный ход не равен оптимальному

Dataset generator оставляет действия игрока из победных Ranked-матчей и системный prompt называет их оптимальными. Победивший игрок мог ошибиться на отдельном ходу.

Проекту нужно выбрать цель:

- персональный behavioral clone повторяет стиль игрока;
- tactical advisor выбирает лучший ход по проверяемой метрике.

Для tactical advisor нужны экспертные accept sets, детерминированные tactical labels или отдельная система оценки. Win-only фильтр не создаёт такие метки.

### 5.4. Пропущенные решения `END_TURN`

Dataset generator исключает friendly turns без действий. Completion всегда добавляет «Завершить ход», но candidate generator не создаёт `END_TURN` как действие.

Модель не получает примеры, где пас или раннее завершение хода является правильным решением. Production-контракт также не позволяет выбрать конец хода как явный кандидат.

### 5.5. Недостаточный контекст карт

Prompt содержит имя, стоимость и базовые характеристики. Он не передаёт полный текст карт и часть важных данных:

- текст battlecry, deathrattle, aura и spell;
- weapon и его durability;
- актуальные requirements цели;
- историю созданных карт и известную информацию о руке;
- graveyard, deck state и часть счётчиков механик;
- card ID для однозначного сопоставления сущностей.

Новые карты и патчевые изменения нельзя надёжно восстановить из имени. `TokenGraph` и компактный formatter карт пока не входят в обучающий prompt.

## 6. Проблемы deterministic compliance engine

### 6.1. Размер стола

`src/llm/candidate_generator.py` считает стол заполненным при 10 существах:

```python
board_full = len(snapshot.friendly_board) >= 10
```

Стандартный лимит Hearthstone равен 7 существам. Генератор предлагает нелегальный розыгрыш восьмого, девятого и десятого существа.

### 6.2. Требования целей

Генератор не индексирует `RequiresTarget` и связанные PlayRequirements. Вместо этого он использует поиск нескольких русских слов для AoE, а остальным заклинаниям добавляет героя противника и каждое вражеское существо.

Из-за этого benchmark содержит невозможные варианты:

- Монетка получает цель;
- Антимагия получает цель;
- массовые заклинания получают варианты по одной цели;
- заклинания на союзников не получают союзные цели;
- заклинания без цели могут получить цель.

До исправления этих требований слой нельзя считать compliance engine.

### 6.3. Hero power

Candidate generator создаёт один общий вариант hero power без целей. Mage ping, Priest heal и другие targeted powers требуют отдельные варианты для каждой легальной цели.

Существующий benchmark ожидает, что Mage ping добьёт существо 3/1, но список кандидатов не кодирует эту цель. Такой тест невозможно пройти корректно.

### 6.4. Windfury

`Entity.can_attack` учитывает две атаки с Windfury. Candidate generator создаёт один кандидат атаки на цель, а response parser запрещает повторное использование attacker entity. Поэтому двойная атака Windfury недоступна.

### 6.5. Последовательная легальность

Response parser суммирует стоимость выбранных действий, но не применяет их к состоянию. Он не умеет:

- добавить ману после Монетки;
- пересчитать стоимость после discount;
- удалить погибшую цель;
- снять Taunt и открыть лицо;
- учесть добор и созданные карты;
- обновить board capacity;
- запретить повторный розыгрыш одной entity через разные target-кандидаты.

Для полного плана нужен пошаговый transition validator. Более простой вариант для первой версии: модель выбирает одно следующее действие, приложение применяет событие и запрашивает следующий выбор по обновлённому состоянию.

### 6.6. Fallback скрывает качество модели

Response parser извлекает любые числа из текста, если модель не вернула `PLAN`. Числа из HP, маны или рассуждения могут случайно совпасть с candidate ID. При полном провале parser выбирает эвристический план.

Benchmark должен считать raw parse failure и fallback rate отдельно. Нельзя засчитывать эвристический fallback как успешный ответ модели.

## 7. Рекомендуемый формат датасета

Для первого рабочего LoRA нужен next-action контракт:

```json
{
  "game_id": "...",
  "decision_id": "...",
  "state": {
    "turn": 5,
    "mana": 4,
    "friendly_hero": {},
    "opponent_hero": {},
    "hand": [],
    "friendly_board": [],
    "opponent_board": []
  },
  "candidates": [
    {"id": 1, "type": "PLAY", "entity_id": 42, "target_id": null},
    {"id": 2, "type": "ATTACK", "entity_id": 11, "target_id": 90},
    {"id": 3, "type": "END_TURN", "entity_id": null, "target_id": null}
  ],
  "chosen_candidate_id": 2
}
```

Пайплайн должен делать следующее для каждого фактического действия:

1. Восстановить состояние непосредственно перед действием.
2. Сгенерировать легальные кандидаты.
3. Сопоставить фактическое действие ровно с одним кандидатом.
4. Записать пример только при успешном сопоставлении.
5. Применить действие или перейти к следующему событию replay.
6. Снять новый snapshot перед следующим решением.

Если фактическое действие не сопоставилось, validator должен отклонить запись и сохранить причину в отчёте. Он не должен заменять имя на `Unknown Card`.

## 8. Data-quality gate

Полная сборка датасета должна завершаться ошибкой, если нарушен любой обязательный порог:

| Проверка | Требование до pilot training |
|---|---:|
| `Unknown` action/card/entity | 0 |
| Действие чужого controller | 0 |
| Gold action отсутствует среди legal candidates | 0 |
| Gold action сопоставился с несколькими кандидатами | 0 |
| Mana violation после пошаговой симуляции | 0 |
| Illegal target | 0 |
| Пересечение game ID между split | 0 |
| Пересечение точных prompt между split | 0 |
| Запись без card ID для сыгранной карты | 0 либо явный quarantine |

Validator должен выводить:

- количество принятых и отклонённых решений;
- причины отклонения;
- распределение по классам, колодам и turn number;
- количество решений `PLAY`, `ATTACK`, `HERO_POWER`, `LOCATION`, `END_TURN`;
- долю примеров с добором или генерацией между действиями;
- p50, p95, p99 и maximum длины после токенизации выбранным tokenizer.

## 9. Split и evaluation set

Текущий split по целому `game_id` нужно сохранить. Для итоговой оценки нужны три набора:

- train;
- validation для выбора checkpoint и параметров;
- test, который не используется до финального сравнения.

Random game split недостаточен для проверки новых патчей и колод. Test set лучше собрать из последних матчей или выделить по времени. Дополнительный challenge set должен включать:

- прямой и комбинированный lethal;
- Taunt, Divine Shield, Reborn, Rush и Windfury;
- targeted и untargeted spells;
- hero powers с целями;
- Coin, cost reduction и generated cards;
- полный стол;
- location, secrets и современные механики;
- ситуацию, где лучший ответ равен `END_TURN`.

Для позиций с несколькими равноценными действиями нужен набор допустимых ответов, а не один exact label.

## 10. Benchmark base vs LoRA

Существующий benchmark из 20 ситуаций сохраняет сырые ответы, но не считает агрегированные метрики. Часть его candidate lists содержит нелегальные или неполные действия. Его нужно исправить до использования как quality gate.

Минимальный автоматический benchmark:

| Метрика | Что измеряет |
|---|---|
| Raw format success | Модель вернула корректную схему без fallback |
| Legal next action | Выбранный action разрешён validator |
| Top-1 agreement | Совпадение с gold или accept set |
| Full-plan exact match | Только для детерминированных коротких линий |
| Lethal recall | Доля найденных гарантированных lethal |
| Illegal target rate | Ошибки выбора цели |
| Mana violation rate | Ошибки последовательного бюджета маны |
| Fallback rate | Доля ответов, заменённых эвристикой |
| Latency p50/p95 | Скорость ответа на целевой машине |
| Peak VRAM | Максимум памяти при одинаковом контексте |

Сравнивать нужно при одинаковых:

- prompt и chat template;
- quantization;
- generation parameters;
- context length;
- test positions;
- validator version.

В таблице результатов должны присутствовать deterministic heuristic, base model и base + LoRA. LoRA имеет смысл выпускать, если она улучшает тактические метрики без роста illegal rate и p95 latency за допустимую границу.

## 11. Состояние тренировочного контура

### 11.1. Зависимости

Текущий `requirements.txt` содержит:

- `torch>=2.2.0`;
- `transformers>=4.40.0`;
- `peft>=0.10.0`;
- `safetensors>=0.4.0`.

В нём отсутствуют прямые runtime-зависимости trainer:

- `datasets`;
- `trl`;
- `bitsandbytes`;
- `accelerate`.

Фактическое окружение во время аудита:

| Пакет | Состояние |
|---|---|
| Python | 3.13.3 |
| torch | 2.9.0 |
| transformers | 4.57.6 |
| peft | 0.17.1 |
| accelerate | 1.10.1 |
| datasets | не установлен |
| trl | не установлен |
| bitsandbytes | не установлен |

Версии не образуют воспроизводимый training environment. Нужен отдельный `requirements-train.txt` или lockfile с комбинацией, которая прошла GPU smoke-test на этой машине.

### 11.2. Текущий TRL API

Официальный PyPI на дату аудита публиковал:

| Пакет | Последняя версия |
|---|---:|
| trl | 1.12.0 |
| datasets | 5.0.1 |
| bitsandbytes | 0.50.2 |
| peft | 0.20.0 |
| transformers | 5.16.1 |
| accelerate | 1.14.0 |

Текущий `SFTConfig` использует `max_length`, а `SFTTrainer` принимает `processing_class`. Проект передаёт `max_seq_length` и `tokenizer`. После установки актуального TRL скрипт потребует обновления или согласованного старого pin.

Обновление на текущий API предпочтительнее неограниченных зависимостей `>=`. После обновления нужно проверить:

- chat template выбранной модели;
- EOS и PAD tokens;
- loss только на assistant/completion tokens;
- truncation policy;
- сохранение и повторную загрузку adapter;
- resume from checkpoint.

### 11.3. Loss masking

ChatML dataset содержит system, user и assistant messages. Trainer должен считать loss по assistant/completion tokens. Иначе значительная часть градиента уйдёт на воспроизведение постоянного system prompt и входного состояния.

Для актуального TRL можно использовать `assistant_only_loss=True`, если chat template возвращает assistant mask. Альтернатива состоит в явной токенизации и установке `-100` для system/user labels. Выбранный способ нужно подтвердить unit-тестом по token labels.

## 12. Выбор базовой модели

### 12.1. Shortlist

| Hugging Face ID | Параметры | Лицензия | Доступ | Оценка для проекта |
|---|---:|---|---|---|
| `Qwen/Qwen3-4B-Instruct-2507` | 4,02B | Apache 2.0 | открытый | Основной кандидат |
| `microsoft/Phi-4-mini-instruct` | 3,84B | MIT | открытый | A/B-кандидат |
| `HuggingFaceTB/SmolLM3-3B` | 3,08B | Apache 2.0 | открытый | Быстрый дополнительный baseline |
| `Qwen/Qwen2.5-7B-Instruct` | 7,62B | Apache 2.0 | открытый | Quality ceiling, тесный QLoRA на 8 GB |
| `Qwen/Qwen2.5-3B-Instruct` | 3,09B | Qwen Research | открытый | Русский baseline, проверить условия лицензии |
| `google/gemma-3-4b-it` | 4,30B | Gemma | gated | Низкий приоритет из-за доступа и лицензии |

### 12.2. Основная рекомендация

Первый pilot стоит проводить на `Qwen/Qwen3-4B-Instruct-2507`:

- модель помещается между слишком слабым 1.5B baseline и тесным 7B;
- model card заявляет улучшенное instruction following и multilingual coverage;
- модель поддерживает структурированный выбор;
- Apache 2.0 упрощает использование;
- Qwen3 поддерживается Transformers версии 4.51 и новее.

Поддержка контекста до 262 144 токенов не нужна этой задаче. Training context должен соответствовать фактическому p99 чистых примеров. Длинный контекст увеличивает VRAM и время без пользы для короткого state prompt.

### 12.3. Phi-4-mini-instruct

`microsoft/Phi-4-mini-instruct` подходит для A/B-теста:

- 3,84B параметров;
- MIT;
- 128K context;
- русский входит в заявленный multilingual набор;
- model card описывает instruction following и function calling.

Microsoft указывает, что модель обучалась преимущественно на английском и качество других языков может быть ниже. Русский benchmark проекта должен решить, подходит ли Phi лучше Qwen.

### 12.4. SmolLM3-3B

SmolLM3 удобен как быстрый открытый baseline. Model card указывает Apache 2.0, 3B параметров и наличие русского training data. Русский не относится к шести основным поддерживаемым языкам, поэтому модель не должна заменять Qwen без локального сравнения.

### 12.5. Qwen2.5-7B-Instruct

Текущая 7B-модель остаётся полезным quality ceiling. NF4-веса занимают около половины сырого FP16-размера, но обучение требует место для quantization metadata, LoRA, gradients, optimizer, activations и CUDA runtime. Batch size 2 при sequence length 1024 может упереться в 8 GB.

7B следует тестировать после 4B, с batch size 1, коротким контекстом и измерением `torch.cuda.max_memory_allocated()`. Нельзя считать inference-fit доказательством training-fit.

## 13. Начальная конфигурация pilot QLoRA

Параметры ниже задают безопасную стартовую точку. Их нужно подтвердить smoke-test и замером VRAM.

```json
{
  "model_name_or_path": "Qwen/Qwen3-4B-Instruct-2507",
  "max_length": 512,
  "lora_r": 8,
  "lora_alpha": 16,
  "lora_dropout": 0.05,
  "per_device_train_batch_size": 1,
  "gradient_accumulation_steps": 16,
  "learning_rate": 0.0001,
  "num_train_epochs": 1,
  "warmup_ratio": 0.03,
  "gradient_checkpointing": true,
  "assistant_only_loss": true,
  "seed": 42,
  "data_seed": 42
}
```

`max_length=512` служит начальным значением. Если p99 выбранного tokenizer превышает 512, нужно поднять лимит до ближайшего достаточного значения, например 768. Нельзя выбирать длину по максимальному context window модели.

Этапы запуска:

1. Tokenization audit без загрузки модели.
2. Overfit на 32 проверенных примерах.
3. Два training step на GPU.
4. 300-500 чистых примеров, один epoch.
5. Полный pilot, один epoch.
6. Второй epoch только при улучшении validation tactical metrics.

Loss сам по себе не определяет качество советника. Решение о продолжении принимает benchmark.

## 14. План исправлений

### Этап A. Parser и state ownership

- Воспроизвести несколько примеров с несовпадающей hero power.
- Логировать `game_id`, global turn, active player, friendly player, controller, entity ID и action owner.
- Исправить переходы `TURN`, `CURRENT_PLAYER` и `STEP=MAIN_ACTION` по production replay.
- Добавить regression tests с реальными последовательностями событий.
- Запретить запись action, если controller не подтверждён.

Критерий завершения: validator не находит смешивания действий между игроками на полном локальном наборе.

### Этап B. Legal candidate engine

- Исправить board limit на 7.
- Индексировать PlayRequirements из CardDefs.
- Добавить friendly, enemy, hero и no-target варианты по требованиям карты.
- Добавить targeted hero powers.
- Учесть оставшиеся атаки Windfury.
- Добавить `END_TURN`.
- Использовать entity ID для запрета повторного розыгрыша одной карты.

Критерий завершения: gold action каждого принятого примера входит ровно в один candidate.

### Этап C. Next-action dataset

- Снимать state перед каждым решением.
- Записывать legal candidates и выбранный candidate ID.
- Добавить компактный текст актуальных карт.
- Отправлять ошибки в quarantine JSONL с reason code.
- Отключить HSReplay subset до прохождения тех же gates.

Критерий завершения: все обязательные проверки из раздела 8 дают ноль ошибок.

### Этап D. Evaluation

- Заморозить validation и test по game ID.
- Добавить временной holdout.
- Исправить 20 synthetic situations.
- Добавить машинно проверяемые expected candidate sets.
- Считать метрики без fallback и после fallback отдельно.

Критерий завершения: один скрипт выдаёт сравнимую таблицу качества, latency и VRAM.

### Этап E. Training environment

- Создать отдельное воспроизводимое окружение.
- Зафиксировать версии после успешного smoke-test.
- Обновить trainer под выбранный TRL API.
- Проверить assistant-only loss.
- Проверить save, reload и inference adapter.

Критерий завершения: smoke-test проходит с exit code 0 и сохраняет загружаемый adapter.

### Этап F. Model sweep и LoRA

- Прогнать base benchmark для Qwen3-4B и Phi-4-mini.
- Оставить текущую Qwen2.5-1.5B как latency baseline.
- Использовать Qwen2.5-7B как quality ceiling, если VRAM позволяет.
- Обучить один pilot adapter на победителе base benchmark.
- Сравнить base и LoRA на замороженном test set.

Критерий завершения: LoRA улучшает tactical metrics и не повышает illegal/fallback rate.

## 15. Definition of ready for full training

Полный LoRA-run разрешён после выполнения всех условий:

- production replay regression tests проходят;
- `Unknown` и owner mismatch отсутствуют в train, validation и test;
- каждое gold action проходит candidate validator;
- training и inference используют один prompt/output contract;
- отдельный test set заморожен;
- benchmark считает legality, tactical quality, fallback, latency и VRAM;
- training dependencies зафиксированы;
- 32-example overfit проходит;
- GPU smoke-test проходит;
- adapter сохраняется и загружается;
- pilot LoRA улучшает base model на test set.

До выполнения этих условий увеличение датасета, epochs или размера модели повысит стоимость эксперимента, но не качество доказательств.

## 16. Остаточные неопределённости

- Аудит выявил симптомы смешивания игроков, но не трассировал первопричину на конкретном replay до события Power.log.
- Проверка карт, отсутствующих во входной руке, не различала parser error и легальный добор внутри хода. Оба случая требуют next-action snapshots.
- VRAM-параметры pilot конфигурации не проверены training runner.
- Качество моделей на русском Hearthstone нельзя вывести из общих model cards. Его подтвердит локальный benchmark.
- `ml-intern` не вернул shortlist: приватный запрос заблокировала защита, обезличенный запуск завис без session-log и был остановлен. Данные моделей проверены прямыми read-only запросами к официальным Hugging Face model cards и API.

## 17. Источники

### Репозиторий

- [`src/llm/dataset_formatter.py`](../src/llm/dataset_formatter.py)
- [`src/llm/train_qlora.py`](../src/llm/train_qlora.py)
- [`src/llm/candidate_generator.py`](../src/llm/candidate_generator.py)
- [`src/llm/response_parser.py`](../src/llm/response_parser.py)
- [`src/parser/dataset_generator.py`](../src/parser/dataset_generator.py)
- [`src/parser/build_master_dataset.py`](../src/parser/build_master_dataset.py)
- [`src/parser/state_tracker.py`](../src/parser/state_tracker.py)
- [`src/parser/hsreplay_xml_parser.py`](../src/parser/hsreplay_xml_parser.py)
- [`tests/test_dataset_formatter.py`](../tests/test_dataset_formatter.py)
- [`data/benchmark_20_situations.md`](../data/benchmark_20_situations.md)

### Официальные model cards

- [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
- [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
- [microsoft/Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct)
- [HuggingFaceTB/SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B)
- [google/gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it)

### Training stack

- [TRL SFTConfig source](https://github.com/huggingface/trl/blob/main/trl/trainer/sft_config.py)
- [TRL SFTTrainer source](https://github.com/huggingface/trl/blob/main/trl/trainer/sft_trainer.py)
- [TRL on PyPI](https://pypi.org/project/trl/)
- [Datasets on PyPI](https://pypi.org/project/datasets/)
- [bitsandbytes on PyPI](https://pypi.org/project/bitsandbytes/)
- [PEFT on PyPI](https://pypi.org/project/peft/)
- [Transformers on PyPI](https://pypi.org/project/transformers/)
- [Accelerate on PyPI](https://pypi.org/project/accelerate/)
