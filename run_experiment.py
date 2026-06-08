#!/usr/bin/env python3
"""
Запуск полного эксперимента: данные → граф → ответы LLM → метрики → отчёт.

Перед запуском:
    cp .env.example .env
    # вставь ключ в .env

Запуск:
    .venv/bin/python run_experiment.py          # если есть .venv в проекте
    python3 run_experiment.py                   # если активирован нужный venv
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# --- Читаем .env без лишних зависимостей ---

def load_dotenv(path: str = ".env") -> None:
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip())

load_dotenv()

OPENROUTER_KEY   = os.environ.get("OPENROUTER_KEY", "")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "deepseek/deepseek-chat")
OPENROUTER_URL   = "https://openrouter.ai/api/v1"

CORPUS_XLSX = Path("Сайт_Русская_Истина_Статьи_по_тегам_истина_правда_ложь.xlsx")
CORPUS_JSON = Path("data/raw/russkaya_istina/documents.json")
GRAPH_DIR   = Path("data/graphs/russkaya_istina/constructor")
GRAPH_PATH  = GRAPH_DIR / "discourse" / "discourse_graph.graphml"
BASE_PATH   = GRAPH_DIR / "graphrag_baseline" / "discourse_graph.graphml"
QUESTIONS   = Path("data/eval/questions_russkaya_istina.json")
REPORT_OUT  = Path("output/expert_report_russkaya_istina.md")
HOT_EDGES   = GRAPH_DIR / "discourse" / "edges.csv"
HOT_REPORT  = Path("output/hot_edges_report.md")
METRICS_OUT = Path("output/rag_metrics_report.md")
FIGURES_DIR = Path("output/rag_eval_figures")


def step(msg: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print('='*60)


# ── Шаг 1: Загрузка данных ──────────────────────────────────────

step("Шаг 1/5: Загрузка xlsx → documents.json")

if CORPUS_JSON.exists():
    print(f"Уже есть: {CORPUS_JSON} — пропускаем загрузку")
else:
    if not CORPUS_XLSX.exists():
        print(f"ОШИБКА: не найден файл {CORPUS_XLSX}")
        sys.exit(1)
    from scripts.load_russkaya_istina import main as load_data
    load_data()

with open(CORPUS_JSON, encoding="utf-8") as f:
    docs = json.load(f)
print(f"Корпус: {len(docs)} документов")


# ── Шаг 2: Построение графа ─────────────────────────────────────

step("Шаг 2/5: Построение дискурс-графа (конструктор)")

if GRAPH_PATH.exists():
    print(f"Уже есть: {GRAPH_PATH} — пропускаем построение")
    import networkx as nx
    G = nx.read_graphml(str(GRAPH_PATH))
else:
    from discourse_graph.config import ConstructorConfig
    from discourse_graph.pipeline import DiscourseGraphConstructor

    cfg = ConstructorConfig.from_preset("russkaya_istina")
    result = DiscourseGraphConstructor(cfg).run(
        str(CORPUS_JSON),
        run_stress=True,
        run_baseline=True,
    )
    G = result["graph"]

import networkx as nx
if not GRAPH_PATH.exists():
    G = nx.read_graphml(str(GRAPH_PATH))
print(f"Граф: {G.number_of_nodes()} узлов, {G.number_of_edges()} рёбер")


# ── Шаг 3: Ответы LLM + экспертный отчёт ───────────────────────

step("Шаг 3/5: Генерация ответов LLM → отчёт для экспертов")

DEFAULT_QUESTIONS = [
    "Какие концепты авторы противопоставляют истине в данном корпусе?",
    "Как описывается связь между истиной и властью в текстах?",
    "Какие авторы наиболее часто обращаются к теме лжи и обмана?",
    "Что такое «люди истины» и как они противопоставляются «людям жизни»?",
    "Как в дискурсе соотносятся понятия «правда», «истина» и «справедливость»?",
    "Какова роль пропаганды в дискурсе об истине и лжи?",
    "Что авторы считают главными источниками лжи или заблуждений в обществе?",
    "Как описывается связь национальной идеи с концептами истины и правды?",
    "Какие риторические конструкции (противопоставления) используются чаще всего?",
    "Как тема «поиска истины» соотносится с религиозными и философскими концептами?",
]

if not QUESTIONS.exists():
    QUESTIONS.parent.mkdir(parents=True, exist_ok=True)
    with open(QUESTIONS, "w", encoding="utf-8") as f:
        json.dump(DEFAULT_QUESTIONS, f, ensure_ascii=False, indent=2)
    print(f"Создан файл вопросов: {QUESTIONS}")

with open(QUESTIONS, encoding="utf-8") as f:
    questions = json.load(f)
print(f"Вопросов: {len(questions)}")

if not OPENROUTER_KEY:
    print("\nWARNING: OPENROUTER_KEY не задан в .env")
    print("Генерируем stub-отчёт. Для реальных ответов:")
    print("  echo 'OPENROUTER_KEY=sk-or-...' >> .env\n")

G_base = nx.read_graphml(str(BASE_PATH)) if BASE_PATH.exists() else nx.Graph()

answers_our: list[str]      = []
answers_baseline: list[str] = []
answers_longctx: list[str]  = []

if OPENROUTER_KEY:
    try:
        from openai import OpenAI
        from discourse_graph.qa import DiscourseQA, flat_context_answer

        client = OpenAI(api_key=OPENROUTER_KEY, base_url=OPENROUTER_URL)
        qa_our  = DiscourseQA(G, client, model=OPENROUTER_MODEL, strategy="lightrag", docs=docs)
        qa_base = DiscourseQA(G_base, client, model=OPENROUTER_MODEL, strategy="community", docs=docs)

        for i, q in enumerate(questions, 1):
            print(f"  [{i}/{len(questions)}] {q[:70]}")
            answers_our.append(qa_our.answer(q))
            if G_base.number_of_nodes() > 0:
                answers_baseline.append(qa_base.answer(q))
            else:
                answers_baseline.append("(baseline граф пустой — пересоберите: удали GRAPH_PATH и перезапусти)")
            answers_longctx.append(flat_context_answer(docs, q, client, model=OPENROUTER_MODEL))

    except ImportError:
        print("Установи openai: pip install openai")
        sys.exit(1)
else:
    for q in questions:
        answers_our.append(f"[stub — добавь OPENROUTER_KEY в .env]")
        answers_baseline.append(f"[stub]")
        answers_longctx.append(f"[stub]")

# Собираем отчёт
import random

def build_report(questions, our, base, lc):
    rng = random.Random(42)
    lines = [
        "# Экспертная оценка: дискурс-граф vs baseline",
        "",
        "Домен: **истина, правда и ложь**  ",
        "Инструкция: для каждого вопроса выберите лучший ответ (A, B или C).  ",
        "",
        "---", "",
    ]
    mapping = []
    for i, (q, o, b, l) in enumerate(zip(questions, our, base, lc), 1):
        labeled = [("our", o), ("baseline", b), ("longctx", l)]
        rng.shuffle(labeled)
        letter_map = {letter: lbl for (lbl, _), letter in zip(labeled, ["A", "B", "C"])}
        mapping.append({"q": i, **letter_map})

        lines += [f"## Вопрос {i}", f"**{q}**", ""]
        for (lbl, ans), letter in zip(labeled, ["A", "B", "C"]):
            lines += [f"### Ответ {letter}", ans.strip() or "_пусто_", ""]
        lines += ["**Ваш выбор (A/B/C):** ___", "", "---", ""]

    lines.append("<!-- DEANON KEY")
    for m in mapping:
        row = {k: v for k, v in m.items() if k != "q"}
        lines.append(f"Q{m['q']}: " + ", ".join(f"{l}={s}" for l, s in sorted(row.items())))
    lines.append("-->")
    return "\n".join(lines)

report = build_report(questions, answers_our, answers_baseline, answers_longctx)
REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
REPORT_OUT.write_text(report, encoding="utf-8")
print(f"\nОтчёт: {REPORT_OUT}")


# ── Шаг 4: Интерпретатор «жареных» связей ──────────────────────

step("Шаг 4/5: Интерпретатор «жареных» связей")

from discourse_graph.interpret import interpret_hot_edges

llm_client = None
if OPENROUTER_KEY:
    from openai import OpenAI
    llm_client = OpenAI(api_key=OPENROUTER_KEY, base_url=OPENROUTER_URL)

interpret_hot_edges(
    G,
    surprisal_csv=HOT_EDGES,
    domain="истина, правда и ложь",
    top_n=10,
    llm_client=llm_client,
    model=OPENROUTER_MODEL,
    out_path=HOT_REPORT,
)
print(f"Hot-edges отчёт: {HOT_REPORT}")


# ── Шаг 5: Автоматические метрики + визуализация ────────────────

step("Шаг 5/5: Автоматические метрики сравнения RAG-методов")

from discourse_graph.eval_metrics import (
    compare_methods,
    plot_metrics_comparison,
    plot_radar_comparison,
    plot_per_question_heatmap,
    plot_composite_ranking,
    save_metrics_report,
)

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Пропускаем метрики для stub-ответов (нет ключа)
if not OPENROUTER_KEY:
    print("OPENROUTER_KEY не задан — метрики считаются по stub-ответам (нулевые).")
    print("Для реальных метрик добавьте ключ в .env и перезапустите.")

answers_by_method = {
    "discourse_graph": answers_our,
    "graphrag_baseline": answers_baseline,
    "long_context": answers_longctx,
}

print("Вычисляем метрики...")
comparison = compare_methods(
    questions=questions,
    answers_by_method=answers_by_method,
    corpus_docs=docs,
)

# Вывод сводной таблицы в консоль
print("\n── Сводная таблица метрик ──────────────────────────────────")
metric_fields = ["rouge1", "rouge2", "rougeL", "lexical_diversity",
                 "context_coverage", "specificity", "length_score", "composite"]
header = f"{'Метод':<22}" + "".join(f"{m:>12}" for m in metric_fields)
print(header)
print("-" * len(header))
for method, data in comparison.items():
    row = f"{method:<22}" + "".join(f"{data.get(m, 0.0):>12.3f}" for m in metric_fields)
    print(row)

# Определяем победителя по composite
best_method = max(comparison, key=lambda m: comparison[m].get("composite", 0.0))
best_score = comparison[best_method].get("composite", 0.0)
print(f"\n🏆 Лучший метод по итоговому баллу: {best_method} ({best_score:.3f})")

# Сохраняем markdown-отчёт
save_metrics_report(comparison, questions, out_path=str(METRICS_OUT))
print(f"\nМетрики-отчёт: {METRICS_OUT}")

# ── Визуализации ─────────────────────────────────────────────────

print("\nСтроим визуализации...")

# 1. Bar-chart по всем метрикам
fig1 = plot_metrics_comparison(
    comparison,
    out_path=str(FIGURES_DIR / "01_metrics_comparison.png"),
    title="Сравнение RAG-методов: автоматические метрики",
)
print(f"  → {FIGURES_DIR}/01_metrics_comparison.png")

# 2. Radar-диаграмма профилей методов
fig2 = plot_radar_comparison(
    comparison,
    out_path=str(FIGURES_DIR / "02_radar_comparison.png"),
    title="Радар-диаграмма: профиль методов",
)
print(f"  → {FIGURES_DIR}/02_radar_comparison.png")

# 3. Тепловая карта composite по вопросам
fig3 = plot_per_question_heatmap(
    comparison,
    questions=questions,
    metric="composite",
    out_path=str(FIGURES_DIR / "03_heatmap_composite.png"),
    title="Итоговый балл по вопросам (методы × вопросы)",
)
print(f"  → {FIGURES_DIR}/03_heatmap_composite.png")

# 4. Тепловая карта context_coverage по вопросам
fig4 = plot_per_question_heatmap(
    comparison,
    questions=questions,
    metric="context_coverage",
    out_path=str(FIGURES_DIR / "04_heatmap_coverage.png"),
    title="Покрытие запроса по вопросам",
)
print(f"  → {FIGURES_DIR}/04_heatmap_coverage.png")

# 5. Stacked ranking chart
fig5 = plot_composite_ranking(
    comparison,
    out_path=str(FIGURES_DIR / "05_composite_ranking.png"),
    title="Итоговый рейтинг методов (вклад каждой метрики)",
)
print(f"  → {FIGURES_DIR}/05_composite_ranking.png")

import matplotlib.pyplot as plt
plt.close("all")


# ── Итог ────────────────────────────────────────────────────────

print(f"""
{'='*60}
Готово. Файлы:
  {CORPUS_JSON}
  {GRAPH_PATH}
  {REPORT_OUT}
  {HOT_REPORT}
  {METRICS_OUT}
  {FIGURES_DIR}/01_metrics_comparison.png
  {FIGURES_DIR}/02_radar_comparison.png
  {FIGURES_DIR}/03_heatmap_composite.png
  {FIGURES_DIR}/04_heatmap_coverage.png
  {FIGURES_DIR}/05_composite_ranking.png
{'='*60}
""")
