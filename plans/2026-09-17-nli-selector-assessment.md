# NLI / openjev: оценка кандидата на селектор действий

Дата: 2026-09-17. Статус: исследование; выбор новой базовой модели не утвержден.

## TODO и границы

- [x] Прочитать JOURNAL, паспорт и предыдущую сессию.
- [x] Проверить текущий контракт, локальные модели и аппаратный runtime.
- [x] Сверить model card, API и риски предложенного runbook.
- [x] Пересчитать размеры split и простые baseline без инференса.
- [x] Описать минимальный эксперимент и критерии решения.
- [x] Найти существующие CUDA environments в D:/AI (см. уточнение ниже).
- [x] Smoke реального ModernCE model forward.
- [x] Frozen validation benchmark ModernCE.
- [x] Принять решение по zero-shot ModernCE: не продвигать в базовый runtime.
- [ ] Test/temporal не вскрывать до выбора и обучения следующего варианта.

Код приложения, конфигурация обучения и веса в этой сессии не меняются. Существующие изменения JOURNAL.md, configs/qlora_config.json и src/llm/train_qlora.py принадлежат предыдущей работе.

## Вывод

Ранжирование заданных кандидатов соответствует контракту schema-v2 и заслуживает проверки. Это кандидат на компонент выбора следующего действия, а не готовая замена объясняющего Coach. Для первого эксперимента разумнее ModernCE-large-nli: меньше весов и проще проверка. openjev-4B оставить второй веткой после проверки model forward и режима памяти. Его BF16 runbook не подходит имеющейся RTX 4060 8 GB.

NLI entailment не является оценкой полезности хода. Два действия могут быть легальны и согласованы с состоянием, но одно пропускает летал. Совпадение с действием игрока измеряет имитацию, а не оптимальность или win rate. Отсутствие выбора действия в реплее не доказывает, что оно плохое.

## Проверенные факты и исправления к записи журнала

1. Локально присутствуют обе папки моделей. openjev checkpoint: D:/models/openjev-4b/qwen3.5-4b-nli/model.safetensors, 9 078 635 984 байта. Наличие файла не подтверждает целостность или успешный forward. В текущем py: torch 2.9.0+cpu, cuda.is_available() == False. nvidia-smi: RTX 4060, всего 8188 MiB, занято 2265 MiB на момент проверки.
2. Размер файла не равен peak VRAM. BF16 openjev требует больше бюджета этой GPU уже под веса; дополнительно нужны активации, рабочие буферы и резерв 15–20%. Shared RAM нельзя считать выполнением целевого latency. Wrapper OpenJevCrossEncoder не принимает quantization_config/load_in_4bit: в нем from_pretrained(dtype=...) и затем model.to(device). Квантизация требует отдельной проверки загрузчика, а не добавления флага в существующий вызов.
3. OpenJevCrossEncoder.rerank возвращает индекс опции. predict возвращает softmax [contradiction, entailment, neutral]. Он обрабатывает пары state/candidate батчами bs=32, повторяя state для каждого кандидата. Это не постоянная стоимость одного состояния независимо от N.
4. ModernCE в upstream code/eval.py получает tokenizer(premises, hypotheses, ...). Журнальный пример с одной строкой Premise/Hypothesis переносит шаблон Qwen на ModernBERT и требует исправления перед benchmark.
5. ModernCE model card указывает [contradiction, entailment, neutral], но локальный config.json: entailment=0, neutral=1, contradiction=2. Upstream openjev содержит явный override и комментарий о неверных config labels. Этот override привязан к HF repo ID и не сработает автоматически для локального пути. Перед оценкой требуется NLI sanity на известных парах и явное закрепление mapping для проверенной ревизии; одной проверки config недостаточно.
6. Для заявленного P(entailment) нужен softmax по трем классам каждого примера. Argmax сырых logits[:, 1] между кандидатами может дать другой выбор: [100,101,100] имеет P_ent около 0.576, [0,2,0] — около 0.787, хотя сырой ent-logit выше у первого.
7. 40–70 мс, 15 мс и обучение за 2–3 минуты не измерены здесь. Model card ModernCE приводит throughput на Blackwell PRO 6000 Max-Q, bs=32; это не задержка Hearthstone на RTX 4060. Извлечение латентов включает все пары train: 91 831, а не только 9320 forward-примеров.
8. Встроенный LatentMLPHead.fit делает собственный grouped_split(qid). Прямая передача decision_id не сохраняет game-level split проекта. Нужны отдельные frozen train/validation; нормализация fit только на train. Last-token pooling применим к Qwen wrapper; нельзя механически переносить его на ModernBERT.
9. Русскоязычный prompt проекта расходится с English model cards обоих кандидатов. Качество на нем неизвестно; перевод и шаблоны выбирать только на validation, фиксируя версии. Не считать перевод названий карт автоматически корректным.
10. Выбор по индексу устраняет генерацию несуществующего ID только при непустом списке, правильном отображении index->id и конечных scores. Он не доказывает полноту/легальность live-кандидатов. Replay-reported options не подтверждают качество live generator. Нужны guards для empty, NaN, исключений и устаревшего состояния.

## Пересчет данных без модели

| Split | Решений | Игр | Пар state/candidate | Max кандидатов | Expected random top1 | First candidate top1 | N=1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 9320 | 390 | 91831 | 101 | 28.16% | 21.30% | 1129 |
| validation | 1198 | 48 | 10638 | 85 | 27.91% | 21.62% | 132 |
| test | 1117 | 48 | 11031 | 84 | 29.22% | 21.49% | 149 |
| temporal | 1205 | 54 | 10130 | 68 | 33.32% | 23.07% | 202 |

Expected random = mean(1/N), не 1/mean(N). Это вычисление по текущим JSONL, не модельный benchmark. Проверка hash-lock manifest повторно не выполнялась.

Существующий неотслеживаемый next_action_baseline_ollama.json: qwen2.5-coder:7b, 100 записей, top1=0, format_valid=1%, p50=527.127 мс, mean=577.058 мс. Артефакт прочитан, запуск не воспроизводился, provenance/raw responses не проверены. По нему нельзя заключать, что NLI сильнее генеративной модели: сначала разобраться с провалом формата.

## Минимальный эксперимент

1. Проверить frozen manifest, game disjointness, ревизии моделей и runtime. Использовать проверенный существующий CUDA environment из D:/AI; не изменять текущую среду обучения. Без silent truncation: считать длины пар, отказы/обрезку отражать отдельно.
2. Smoke ModernCE: несколько очевидных NLI пар на английском и русском, контроль pair-tokenization и label mapping. Затем небольшая фиксированная выборка validation с разным N и длиной; начать с малого batch.
3. Получать premise из состояния без команды PLAN и без gold. Кандидат передавать отдельной hypothesis; candidate ID использовать только для обратного отображения. Существующий build_next_action_prompt содержит весь список и генеративную инструкцию, поэтому не годится для прямой передачи без адаптации. Сохранять тот же набор фактов состояния для честного сравнения.
4. На validation сравнить ограниченные заранее шаблоны, NLI probability baseline и first/random baseline. Отдельно оценить N>1, action type и размер набора; проверить перестановку кандидатов. Не подбирать шаблон на test.
5. Метрики: attempted/success/errors, top1 на всех попытках с ошибками как промахами, conditional top1, top-k для advisory UI, p50/p95/max end-to-end latency, peak VRAM, число обрезок. CUDA синхронизировать при замерах; warmup отделить. Проверить задержку также с запущенной игрой. Текущий evaluator агрегирует только успешные ответы и не дает p95: нельзя напрямую выдавать это за полную метрику нового benchmark.
6. Если zero-shot слаб, отдельно оценить supervised frozen-backbone ranker на train с frozen validation. Начать с простой головы; listwise CE по кандидатам состояния — естественный вариант для one-choice supervision, upstream soft BCE — сравниваемая альтернатива. Не называть остальные легальные варианты доказанно ошибочными. Feature extraction и fit измерять отдельно.
7. После фиксации решений один раз оценить test и temporal. Сравнение качества сопровождать неопределенностью по играм; состояния одной игры зависимы. Порог <100 мс — цель эксперимента, не факт. До продвижения нужны превосходство над простыми baseline, приемлемое сравнение с исправленным генеративным baseline, отсутствие OOM и пригодный p95.

## Источники

- https://huggingface.co/AlexWortega/openjev — локально прочитаны README.md, modeling_openjev.py и code/eval.py из D:/models/openjev-4b; локальная ревизия не установлена.
- https://huggingface.co/dleemiller/ModernCE-large-nli — model card просмотрена online 2026-09-17; config прочитан локально.
- src/llm/next_action_contract.py, src/llm/evaluate_next_action.py, configs/qlora_config.json и текущие data/processed/next_action_*_chatml.jsonl.

## Проверки этой сессии

Read-only команды inspection, nvidia-smi, torch probe и JSONL statistics завершились с exit code 0. Модели не загружались в память; GPU inference, обучение и тесты приложения не запускались. ML исследование дополнительно делегировано ml-intern с лимитом 3 итерации, без скачивания весов и обучения.

## Уточнение после проверки D:/AI

По просьбе пользователя проверены существующие интерпретаторы. Отсутствие CUDA в системном py НЕ является блокером машины:

| Environment | torch | CUDA build | cuda.is_available |
|---|---|---|---|
| D:/AI/voodoo-dyn-quant/.venv | 2.14.0+cu132 | 13.2 | True |
| D:/AI/exllamav3/.venv | 2.10.0+cu128 | 12.8 | True |
| D:/AI/ComfyUI_portable/python_embeded | 2.13.0+cu130 | 13.0 | True |
| D:/AI/ComfyUI_portable/kohya_ss/.venv | 2.7.0+cu128 | 12.8 | True |

Все четыре сообщили NVIDIA GeForce RTX 4060, exit code 0. В voodoo-dyn-quant дополнительно: transformers 5.17.0; CUDA tensor square/sum = 14.0; импорты ModernBertForSequenceClassification и Qwen3_5ForSequenceClassification успешны, exit code 0. Это проверка CUDA вычисления и импортов, не model forward. ML Intern venv не содержит torch, что не мешает делегированному remote research.

ML Intern завершился exit code 0 после трех итераций чтения HF card/config/code, но итогового аналитического ответа не выдал. Его запуск не считается независимым подтверждением выводов. Выводы выше опираются на непосредственно прочитанные исходники и выполненные проверки.

## Фактический ModernCE benchmark

Добавлен воспроизводимый runner `src/llm/evaluate_modernce.py`. Он использует отдельную pair-tokenization, `softmax(logits)[:, 1]`, разбивает большие наборы кандидатов на пакеты и измеряет путь tokenizer -> CUDA forward -> ranking с CUDA synchronization. Диагностический подсчет исходной длины вынесен из latency timer.

Sanity-check на RTX 4060 подтвердил фактический порядок logits `[contradiction, entailment, neutral]`, несмотря на несовместимую разметку `config.json`. Пакет из шести коротких контрольных пар: 22.43 ms после warmup; peak allocated 767.98 MiB. Английские entailment/contradiction пары разделены правильно; русские пары имеют заметно более слабую уверенность.

Frozen validation, русский шаблон `Лучшее следующее действие: {}.`:

- 1198/1198 решений, 10 638 state/candidate пар, errors 0, truncations 0, max 461 tokens;
- FP16, RTX 4060, model load 2029.943 ms, peak allocated 1197.875 MiB;
- top-1 all 31.55%, но 132 записи имеют N=1;
- честный top-1 при N>1: 23.08%; expected random 18.98%; first-candidate 11.91%;
- delta к expected random: +4.10 п.п.; game-cluster bootstrap 95% CI: +1.00...+7.75 п.п. по 48 играм;
- top-3 all 60.10%; top-3 при N>1 55.16%;
- latency p50 74.213 ms, p95 398.193 ms, max 1746.969 ms; при росте N задержка почти линейна;
- при N>1: END_TURN 0/127 и HERO_POWER 0/64. Модель предсказала PLAY 792 раза, ATTACK 258, LOCATION 12, END_TURN 3, POWER 1 и ни разу HERO_POWER. Все три предсказания END_TURN были ошибочны.

Английский шаблон `The best next action is: {}.` на тех же первых 100 validation записей сохранил top-1 22%; улучшения над русским шаблоном нет. Полный test и temporal не запускались.

Решение: zero-shot ModernCE не является приемлемой базовой моделью или greedy-селектором. Положительный средний сигнал над random не компенсирует нулевую точность двух критических классов и хвост latency. Модель можно оставить исследовательским backbone для supervised frozen-encoder ranker. Следующий ML эксперимент, если продолжать ветку: обучить на frozen train embeddings простую listwise голову с game-level validation и отдельным контролем END_TURN/HERO_POWER; только затем принимать решение о test/temporal.

Артефакты:

- `data/processed/next_action_baseline_modernce_validation.json` — полный отчет и per-decision scores; SHA-256 `64463B00DD159C8AB1F54E13D212B226744A1ACD8825C2B974D7F1B7B57E7BA0`.
- validation JSONL SHA-256 `20B2A17BA7F8CBE2BA0D74D69E374A778BF6841B2C6439557C76DBA98502F9CD`.
- model.safetensors SHA-256 `5A950FDDF7E45E27A9C585097CE3F1C767D8C59500FB775AE5507D8EE432B3AB`.

Проверки: benchmark exit code 0; `py_compile` exit code 0; `git diff --check` exit code 0 для нового runner. ML Intern отдельно просмотрел итоговые числа и согласился, что продвижение в greedy base runtime не обосновано; его утверждение о top-3 60.10% относилось к метрике all, поэтому в решении используется пересчитанное 55.16% при N>1.

## Safety incident и новый gate

Пилот extraction на 2056 парах привел к наблюдаемым 11 GB общей GPU memory, 100% CPU и OOM/driver fault. Windows подтвердил `nvlddmkm` Event 153. Значение `torch.cuda.max_memory_allocated` не учитывало WDDM/shared memory и не может использоваться отдельно как safety metric.

Supervised experiment приостановлен. До продолжения обязательны:

- `ResourceGuard.preflight()` по текущей GPU/RAM/temperature telemetry; время запуска задаёт пользователь;
- batch <=2, CPU threads <=2, inter-op=1, Below Normal process priority;
- allocator cap 40%, минимум 4096 MiB свободной dedicated VRAM и 8192 MiB RAM;
- `SDPA`, `reference_compile=false`, tokenizer parallelism disabled;
- watchdog каждые 8 батчей и завершение после 60 секунд небезопасного состояния;
- FP16 memmap/chunked cache с checkpoint/flush; отсутствие полного корпуса или повторенных prompt strings в RAM;
- отдельный короткий запуск только после явного решения пользователя продолжить.

Реализован и проверен общий `src/llm/resource_guard.py`; текущий evaluator использует его. Безмодельные tests: 7 passed. Ручной safe pilot на 200 validation решений завершён с exit code 0: 1683 пары, errors 0; max total GPU used 2303 MiB, min GPU free 5654 MiB, min RAM available 14484 MiB, max temperature 43 C. Результат сохранён в `data/processed/next_action_baseline_modernce_validation_safe_200.json`. Отложенная автоматизация удалена; расписания не используются.
