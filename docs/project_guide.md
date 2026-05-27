# Наукограды / дискурс-граф: что сделано и куда смотреть

Краткий навигатор по репозиторию после фазы сбора данных, EDA и конструктора (май 2026).

---

## Карта проекта

```
sna/
├── data/
│   ├── raw/naukogrady/documents.json      # корпус 221 док
│   ├── graphs/naukogrady/                 # граф из load_data (legacy)
│   └── graphs/naukogrady/constructor/     # фаза 1: дискурс + baseline + стресс-тест
├── output/eda/                            # EDA: фигуры, JSON, проверки
├── discourse_graph/                       # пакет конструктора
├── scripts/run_eda.py                     # генерация EDA одной командой
├── load_data.ipynb                        # сбор корпуса
├── eda_naukogrady.ipynb                   # интерактивный EDA
├── constructor_demo.ipynb                 # демо конструктора
├── run_constructor.py                     # CLI конструктора
└── docs/
    ├── project_summary.md                 # план проекта
    ├── discourse.md                       # ресерч методов
    ├── naukogrady_data_collection.md      # как собран корпус
    ├── eda_report.md                      # мини-отчёт EDA ← читать первым после цифр
    └── project_guide.md                   # этот файл
```

---

## Что уже сделано

| Этап | Статус | Где результат |
|------|--------|----------------|
| Сбор корпуса «Наукограды» | ✓ | `data/raw/naukogrady/documents.json`, `docs/naukogrady_data_collection.md` |
| Ресерч методов | ✓ | `docs/discourse.md` |
| EDA + проверки адекватности | ✓ | `output/eda/`, `docs/eda_report.md`, `eda_naukogrady.ipynb` |
| Визуализация графов | ✓ | `output/eda/figures/07_network_*.png`, `08_communities_*.png` |
| Конструктор фаза 1 | ✓ | `discourse_graph/`, `run_constructor.py` |
| GraphRAG baseline | ✓ | `constructor/graphrag_baseline/` |
| Стресс-тест (dropout + seeds) | ✓ | `constructor/invariant_core/`, `stress_test.json` |
| HF-модель rubert-tiny2 | ✓ | эмбеддинги, опциональный `--summarize` |

**Не сделано (фаза 2+):** reasoning по графу, экспертная оценка, полноценный Microsoft GraphRAG CLI.

---

## Куда смотреть по задаче

| Вопрос | Файл / папка |
|--------|----------------|
| Сколько документов и откуда? | `docs/naukogrady_data_collection.md`, `output/eda/corpus_stats.json` |
| Корпус адекватен? | `output/eda/sanity_checks.json`, `docs/eda_report.md` |
| Графики EDA | `output/eda/figures/` |
| Основной дискурс-граф | `data/graphs/naukogrady/constructor/discourse/` |
| Устойчивое ядро | `data/graphs/naukogrady/constructor/invariant_core/` |
| Baseline для сравнения | `data/graphs/naukogrady/constructor/graphrag_baseline/` |
| Сообщества baseline | `graphrag_baseline/communities.json` |
| Параметры последнего прогона | `constructor/run_meta.json` |
| «Жареные» рёбра | `output/eda/top_surprisal_edges.csv` |
| Как пересобрать граф | `run_constructor.py`, `constructor_demo.ipynb` |

---

## Команды

```bash
# EDA + все картинки
.venv/bin/python scripts/run_eda.py

# Конструктор (полный)
.venv/bin/python run_constructor.py --summarize

# Быстрее: без стресс-теста и baseline
.venv/bin/python run_constructor.py --no-stress --no-baseline
```

---

## Как связаны EDA и конструктор

1. **EDA** проверяет корпус и сравнивает **три графа** (legacy / constructor / baseline).
2. Метрики плотности (**~2 рёбра на узел**) обосновывают пороги в `ConstructorConfig` (`min_pmi`, `max_vertices`).
3. Топ сущностей и городов из EDA — sanity check, что NER не «уехал» в нерелевантный домен.
4. **surprisal** из конструктора экспортируется в EDA для приоритизации связей в фазе 2.
5. При изменении корпуса или методов: снова `load_data.ipynb` → `run_constructor.py` → `scripts/run_eda.py` → обновить `eda_report.md` при необходимости.

---

## Зависимости

```bash
pip install -r requirements.txt
```

Подробности сбора: `load_data.ipynb` (ячейка 0 — pip).

---

## Контакты по смыслу проекта

План и метрики успеха: `docs/project_summary.md` (опрос ~20 экспертов, сравнение с GraphRAG и long-context).
