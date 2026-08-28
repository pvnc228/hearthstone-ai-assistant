# AGENTS.md — Hearthstone AI Assistant

## 1. Project Overview & Mission
Разработка персонального ИИ-ассистента и тактического тренера для Hearthstone на базе локальных LLM и данных пользователя (`HappyBread#21597`).
Система решает две ключевые задачи:
1. **Ретроспективный анализ (Coach)**: Детальный разбор завершённых матчей, поиск мисплеев, упущенного летального урона и ошибок распределения маны/темпа.
2. **Ассистент реального времени (Live Advisor)**: Считывание живого `Power.log` во время матча и выдача 3-5 легальных вариантов действий за 1.5–3 секунды через локальный Ollama (`qwen2.5-coder:7b` / `qwen2.5:1.5b`).

---

## 2. Hardware Profile & Execution Bounds
- **GPU**: NVIDIA GeForce RTX 4060 (8 188 MiB VRAM). Доступно под инференс ~6.5 GB.
- **CPU**: 12th Gen Intel Core i5-12400F (6C / 12T).
- **RAM**: 32 GB DDR4.
- **OS**: Windows 11 x64, Shell: PowerShell.
- **LLM Engine**: Ollama (порт 11434). Доступные локальные модели: `qwen2.5-coder:7b`, `qwen2.5:1.5b`, `deepseek-coder:6.7b`, `gemma4:12b`.
- **Python**: Python 3.13 (`torch`, `transformers`, `peft`, `safetensors`).

---

## 3. Data Sources & File Locations
- **HDT Roaming Directory**: `C:\Users\mist8\AppData\Roaming\HearthstoneDeckTracker`
  - **Replays**: `%APPDATA%\HearthstoneDeckTracker\Replays\` (1 041 `.hdtreplay` zip-файлов).
  - **Stats**: `%APPDATA%\HearthstoneDeckTracker\DeckStats.xml` (1 025 игр, 548 побед).
  - **Cards XML**: `%APPDATA%\HearthstoneDeckTracker\CardDefs\CardDefs.ruRU.xml` и `CardDefs.base.xml` (35 807 карт).
- **Hearthstone Game Directory**: `D:\Hearthstone`
- **Hearthstone Live Logs**: `D:\Hearthstone\Logs\Power.log` (или буфер HDT).

---

## 4. Engineering Invariants & Code Standards (Ponytail Rules)
1. **Stdlib First & YAGNI**:
   - Парсинг XML через `xml.etree.ElementTree`.
   - Распаковка `.hdtreplay` через `zipfile`.
   - Потоковое чтение `Power.log` без загрузки гигабайтных логов целиком в память.
2. **Robust Parser Resilience**:
   - Неизвестные теги игры или новые ID карт в `Power.log` НЕ должны ронять процесс. Использовать `try/except` и безопасные `.get()` дефолты.
   - Имена сущностей сопоставлять через `CardDefs.ruRU.xml`.
3. **Cross-Platform & Path Handling**:
   - Использовать `os.path` или `pathlib.Path` с обязательным раскрытием переменных окружения (`os.path.expandvars`).
   - Файлы читать/писать строго в `encoding="utf-8"`.
4. **Fast Local Inference**:
   - Никаких блокирующих 60-секундных сетевых вызовов в live-режиме.
   - Запросы к Ollama оформлять с тайм-аутом, валидацией схемы JSON/формата ответа и минимальным промптом (только существенные факты стола).
5. **Deterministic Rule Filter**:
   - Перед выводом рекомендаций LLM фильтровать действия по мане, легальным таргетам и картам в руке детерминированным Python-кодом (compliance check).

---

## 5. Directory Layout
```text
hearthstone-ai-assistant/
├── AGENTS.md            # Данный файл: профиль, инварианты, правила
├── ROADMAP.md           # Дорожная карта и статусы задач
├── JOURNAL.md           # Инженерный журнал, замеры, бенчмарки
├── README.md            # Быстрый старт
├── src/
│   ├── parser/          # Парсер .hdtreplay и живого Power.log
│   ├── card_db/         # Индексатор CardDefs.ruRU.xml
│   ├── coach/           # Ретроспективный разбор матчей
│   ├── live/            # Live Watcher и Overlay/CLI
│   └── llm/             # Клиент Ollama и сборщик промптов
├── data/
│   ├── processed/       # Сформированные JSONL датасеты
│   └── cache/           # Быстрый кэш карт и индексов
└── tests/               # Юнит-тесты компонентов
```
