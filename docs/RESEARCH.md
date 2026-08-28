# От реплея к ходу: создание персонализированного AI-ассистента для Hearthstone с помощью имитационного обучения

> Исходный теоретический фундамент и научное обоснование архитектуры проекта (исследование на базе Qwen).

---

## 1. Фундаментальные подходы к обучению модели

### 1.1. Имитационное обучение (Imitation Learning) и Behavioral Cloning (BC)
Цель разработки языковой модели для Hearthstone — воспроизведение экспертных решений пользователя и генерация стратегически обоснованных ходов. Основой выступает парадигма Imitation Learning (IL), где в качестве эксперта выступает сам пользователь и история его игр за несколько лет.

Прямолинейная реализация IL — **Behavioral Cloning (BC)** — сводит задачу к обучению с учителем (Supervised Fine-Tuning) на парах `[Состояние игры -> Действие]`:
- **Состояние игры (Prompt)**: здоровье и броня героев, текущая и максимальная мана, существа на столе (с атакой, здоровьем, баффами, божественными щитами, неистовством ветра, перерождением, титанами), области (Locations), карты в руке со стоимостью, активные секреты, история сыгранных карт.
- **Действие (Completion)**: последовательность действий за ход ("сыграть Карту X на Юнита Y", "использовать способность Героя", "атаковать Героя противника существом Z", "завершить ход").

### 1.2. Проблемы стандартного Behavioral Cloning и методы их решения
1. **Деградация распределения (Covariate Shift)**: при столкновении с новыми ситуациями вне обучающей выборки точность чистой BC-модели может резко падать.
2. **Переобучение на шум**: пользовательские партии содержат эмоциональные или неоптимальные ходы.
3. **Решение (Data Filtering)**: стратегический отбор демонстраций только из победных рейтинговых матчей (Ranked Wins) и состояний с положительным темпом.
4. **Решение (Thought Cloning & Explainability)**: обучение модели генерации цепочки мыслей (Chain-of-Thought) перед выводом действия для повышения обобщаемости и интерпретируемости.
5. **Решение (Runtime Compliance & Rule-Based Machine)**: детерминированный фильтр легальных действий (Candidate Generator + Response Parser), гарантирующий 100% соблюдение правил игры (мана, провокации Taunt, цели).

---

## 2. Сравнение архитектурных методологий

| Метод | Описание | Преимущества | Ограничения |
| :--- | :--- | :--- | :--- |
| **Behavioral Cloning (BC)** | Прямое обучение модели предсказывать действие по состоянию стола. | Простота реализации, не требует сложной среды симуляции. | Чувствителен к шуму, требует фильтрации побед. |
| **Data Filtering** | Обучение только на победных рейтинговых матчах с высоким рангом. | Снижение шума, фокусировка на успешных стратегиях. | Требует достаточного объема истории игр. |
| **Decision Transformer** | Обработка траекторий состояний, действий и функции награды (победа/поражение). | Учет долгосрочных последствий ходов. | Повышенные требования к длине контекста. |
| **Hybrid (BC + RL)** | Начальное приближение через BC с последующей оптимизацией политик через RL. | Быстрый старт + глубокая оптимизация выигрыша. | Сложность настройки функции награды в ККИ. |
| **Thought Cloning** | Обучение генерации рассуждений и обоснования тактики перед выдачей хода. | Высокая интерпретируемость и доверие игрока. | Требует аннотированных объяснений ходов. |

---

## 3. Практическая реализация в проекте

Архитектура Hearthstone AI Assistant разделена на три ключевых контура:

1. **База знаний и семантика (Card DB)**:
   - Парсинг словарей `CardDefs.ruRU.xml` (35 800+ карт) в SQLite и быстрый in-memory индекс O(1).
   - Поддержка актуальных механик (Титаны, Области, Туристы, Звездолеты, Миниатюризация, Руны).

2. **Ретроспективный тренер (Post-Game Coach)**:
   - Анализ завершенных матчей (.hdtreplay и .hsreplay.xml).
   - Расчет упущенного летального урона (Lethal Detector) и просадок темпа (Mana Inefficiency).
   - Сравнение решений игрока с рекомендациями локальной LLM.

3. **Слой надежности и соблюдения правил (Compliance Engine)**:
   - Детерминированный генератор легальных кандидатов (учет маны, провокаций, спящих существ).
   - Отказоустойчивый парсер ответов малых моделей (Qwen 1.5B/7B).

---

## 4. Список литературы и источников

1. Reading numbers from HS Replay and understanding the meta. Reddit Hearthstone.
2. A Neural Network Approach to Hearthstone Win Rate Prediction. ResearchGate.
3. Predicting Hearthstone game outcome with machine learning. Elie Bursztein.
4. Large Language Models as Game-Playing Agents in Slay the Spire. ACM DL.
5. From LLM-Driven Trading Card Generation to Procedural Deck Building. arXiv:2604.27972.
6. Chaos Cards: Creating Novel Digital Card Games through LLMs. AAAI AIIDE.
7. Mastering Strategy Card Game (Hearthstone) with Improved Techniques. ResearchGate.
8. Tracing LLM Reasoning Processes with Strategic Games. OpenReview.
9. UrzaGPT: LoRA-Tuned Large Language Models for Card Games. arXiv:2508.08382.
10. Behavior Transformers: Cloning k modes with one stone. arXiv:2206.11251.
11. Multi-Game Decision Transformers. arXiv:2205.15241.
12. Learning to Beat ByteRL: Exploitability of Collectible Card Games. arXiv:2404.16689.
13. AgentBench: Evaluating LLMs as Agents. arXiv:2308.03688.
14. Runtime Verification for AI Agents: Policies and Compliance Engines. Substack.
15. A Rule-Based Policy Engine Approach for ReAct Agents. Medium Tech.
