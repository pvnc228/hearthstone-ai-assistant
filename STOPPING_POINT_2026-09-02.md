# Точка остановки — 2026-09-02

## Цель текущей работы

1. Перегенерировать production `data/processed/train_actions.jsonl` исправленным parser.
2. Провести Stage B: детерминированная проверка легальных действий.
3. Собрать next-action dataset в контракте `state + legal candidates -> chosen candidate ID`.
4. Только после этого повторно оценить QLoRA readiness.

Ограничения пользователя: обучение не запускать; `D:\AI\ml-intern` не использовать; commit/push не делать без отдельной просьбы.

## Состояние репозитория на входе

- `master == origin/master`, HEAD `fe6eabcf61b86fc61bbc6c7a8bba5b1c0f0def77` (`fix: preserve replay action ownership`).
- Рабочее дерево было чистым.
- Все перечисленные ниже изменения сейчас находятся только в рабочем дереве; commit/push не выполнялись.

## Что уже сделано

### Production turn dataset

- Выполнено `py -m src.parser.dataset_generator`.
- Фактический вывод: `Generated 3049 turn action records from 549 games`.
- Текущий `data/processed/train_actions.jsonl`: 3 049 строк, 538 непустых уникальных `game_id`, 11 704 действий.
- SHA-256: `2d9a0679e667510fe9eedfb062d226d574deef3b251f3f6d023f0ceb734a7736`.
- Осталось предупреждение `runpy` из-за eager import `dataset_generator` в `src/parser/__init__.py`; генерация завершилась с exit code 0.

### Parser и состояние

- `src/parser/log_parser.py` разбирает `DebugPrintOptions` и `SendOption`:
  - `OPTIONS_START`, `OPTION`, `OPTION_TARGET`, `OPTION_SUB_OPTION`, `SEND_OPTION`.
- В `src/parser/state_tracker.py` добавлены `ReplayOptionCandidate` и `OptionDecision`; перед `SendOption` сохраняется snapshot, полный набор legal options (`error=NONE`) и выбранный tuple.
- Добавлены стабильные entity IDs, `END_TURN`, позиции на доске, target/sub-option поля.
- Исправлены board limit 7, герой-атакующий, immune/can't-be-attacked, Windfury/Mega-Windfury, Rush face restriction и живая стоимость hero power.
- Итерация replay сделана детерминированной по имени файла.
- Добавлен parser `CHANGE_ENTITY` и общий `_set_entity_card`, который сбрасывает старые `CARDTYPE`/`COST` при превращении карты.
- Option entity теперь предпочитает свежие поля из `DebugPrintOptions`; это исправляет устаревшие card ID/type, но текущая реализация слишком буквально принимает имя `UNKNOWN ENTITY` — см. главный незавершённый дефект ниже.

### Next-action builder (WIP)

- Добавлен `src/parser/next_action_dataset.py`.
- Формат schema v2 содержит state, полный candidate set, `chosen_candidate_id`, `gold_action`, `option/sub/target/position`, источник легальности и отдельный quarantine JSONL.
- Запись выполняется потоково через временный файл с atomic replace.
- Встроен независимый повторный аудит записанного accepted-файла.
- Легальность целей берётся из игрового oracle replay: `DebugPrintOptions error=NONE`; выбранный ход — из `SendOption`.
- Для minion/location play позиции разворачиваются в candidate IDs; решение с несогласованной позицией карантинится.
- Комбинации, где одновременно есть sub-option и target, пока карантинятся как непроверенный cross-product.
- Если сохранённая стоимость выше доступной маны, запись карантинится как `state_mana_cost_mismatch`, а не объявляется нелегальным ходом: игровой oracle уже подтвердил ход, проблема в реконструкции state/cost.

### Тесты

- Последний актуальный focused run после `CHANGE_ENTITY`:
  - команда: `py -m pytest tests/test_next_action_dataset.py tests/test_parser.py tests/test_coach.py -q --tb=short -p no:cacheprovider --basetemp .pytest-tmp-focused-options-4`
  - результат: `26 passed in 8.68s`, exit code 0.
- Более ранний полный suite был `34 passed`, но он выполнен до последних правок `CHANGE_ENTITY`/position и не является финальным доказательством текущего дерева.
- `py_compile` проходил до последних `CHANGE_ENTITY`-правок; focused pytest после них успешно импортировал и исполнил изменённый код.

## Последний smoke и точная точка остановки

Последняя команда:

```powershell
py -m src.parser.next_action_dataset --limit 5 `
  --output "$env:TEMP\hs-next-smoke-v3.jsonl" `
  --quarantine "$env:TEMP\hs-next-smoke-v3-quarantine.jsonl" `
  --report "$env:TEMP\hs-next-smoke-v3-report.json"
```

Результат: exit code 0, 227 option decisions из 5 игр:

- accepted: 190;
- quarantined: 37;
- coverage: 83.7004%;
- причины: `unresolved_legal_candidate=36`, `state_mana_cost_mismatch=1`;
- accepted gate violations: `{}`.

Этот результат — диагностический, не production. Резкое появление 36 unresolved после исправления `CHANGE_ENTITY` имеет подтверждённую причину: `_option_entity()` предпочитает `ref.entityName="UNKNOWN ENTITY [cardType=INVALID]"` известному имени из CardDB/tracker. Пример:

- card ID `EX1_144` и target полностью разрешены;
- description остаётся `UNKNOWN ENTITY [cardType=INVALID] -> Изворотливый рептилоид`;
- из-за этого вся decision корректно уходит в quarantine.

## С чего продолжить после перезапуска

1. Исправить `GameStateTracker._option_entity()` в `src/parser/state_tracker.py`:
   - свежий непустой `ref.cardId` по-прежнему должен иметь приоритет;
   - имя `ref.entityName`, начинающееся с `UNKNOWN ENTITY`, не должно перекрывать имя из CardDB/tracker;
   - при известном выбранном card ID сначала разрешать имя через CardDB, затем fallback к нормальному ref/tracker name.
2. Добавить regression test: option ref имеет `UNKNOWN ENTITY`, но card ID известен — кандидат обязан получить реальное имя.
3. Повторить focused tests и smoke `--limit 5`. Ожидаемо 36 ложных unresolved должны исчезнуть; ожидание нужно подтвердить запуском.
4. Прогнать corpus diagnostic/full build. До изменения position/transform было подтверждено:
   - 549 ranked-win replay;
   - 14 691 `SendOption` decisions;
   - все 14 691 выбранных tuple имели ровно одно совпадение в option set;
   - option data присутствовали в 539/549 играх;
   - 116 decisions содержали sub-options, 32 — одновременно sub-option и target;
   - 81 выбранный ход имел reconstructed mana-cost hint выше snapshot mana;
   - 2 decisions содержали unresolved candidates.
   Эти цифры получены до финального `CHANGE_ENTITY` fix и должны быть перемерены.
5. Только после стабильного smoke выполнить production:

```powershell
py -m src.parser.next_action_dataset
```

6. Перезаписать старые `data/processed/train_next_actions.jsonl`, `train_next_actions_quarantine.jsonl` и `next_action_validation_report.json`. Текущие файлы в `data/processed` созданы более ранней консервативной версией builder и не являются финальными schema-v2 production artifacts.
7. Обновить `thoughts/research/2026-09-02-next-action-dataset.md`, `specs/2026-09-02-next-action-dataset.md` и `plans/2026-09-02-next-action-dataset.md`: сейчас они ещё описывают ранний fallback без replay option oracle.
8. Добавить итоговый Stage B/C report и обновить `README.md`, `ROADMAP.md`, `JOURNAL.md` только фактическими финальными метриками.
9. Финальные проверки:
   - полный `py -m pytest tests -q --tb=short -p no:cacheprovider --basetemp <новая папка>`;
   - `py -m compileall -q src`;
   - независимый JSONL audit/line counts/hash;
   - `git diff --check`;
   - `git status --short --branch`.
10. Удалить только созданные `.pytest-tmp-*` после проверки абсолютных путей внутри repo.

## QLoRA readiness

Сейчас `false`; обучение не запускалось. Даже после production dataset остаются обязательные проверки/работы:

- устранить или формально принять quarantine причины;
- заморозить train/validation/test split по `game_id`;
- отдельно проверить fidelity динамических mana costs и state transitions;
- повторно валидировать training environment (`datasets`, `trl`, `bitsandbytes`, `accelerate` ранее не были готовы);
- выполнить base-model benchmark до QLoRA.

## Текущее рабочее дерево

Изменены tracked:

- `data/processed/train_actions.jsonl`
- `src/llm/candidate_generator.py`
- `src/parser/__init__.py`
- `src/parser/log_parser.py`
- `src/parser/replay_reader.py`
- `src/parser/state_tracker.py`
- `tests/test_coach.py`
- `tests/test_parser.py`

Новые/untracked:

- `src/parser/next_action_dataset.py`
- `tests/test_next_action_dataset.py`
- `data/processed/train_next_actions*.jsonl`
- `data/processed/next_action_validation_report.json`
- `thoughts/`, `specs/`, `plans/`
- `.pytest-tmp-*`
- этот stopping-point файл.

На момент сохранения связанных Python-процессов не запущено. Ветка всё ещё `master...origin/master`, но рабочее дерево намеренно грязное из-за незавершённой локальной работы.
