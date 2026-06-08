"""
Автоматические метрики для сравнения RAG-методов.

Метрики (без LLM, полностью детерминированные):
  - ROUGE-1/2/L (лексическое перекрытие)
  - Lexical Diversity (TTR — type-token ratio)
  - Answer Length (нормированная длина ответа)
  - Context Coverage (доля слов запроса, покрытых ответом)
  - Specificity Score (доля редких слов — обратная TF по корпусу)

Все метрики нормированы в [0, 1], выше = лучше.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import NamedTuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ── Токенизация ──────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    return re.findall(r"[а-яёa-z]{2,}", text.lower())


# ── ROUGE ────────────────────────────────────────────────────────

def _ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def rouge_n(hypothesis: str, reference: str, n: int = 1) -> float:
    """ROUGE-N: recall-oriented overlap of n-grams."""
    hyp = _ngrams(_tokenize(hypothesis), n)
    ref = _ngrams(_tokenize(reference), n)
    if not ref:
        return 0.0
    overlap = sum(min(hyp[k], ref[k]) for k in ref)
    return overlap / sum(ref.values())


def rouge_l(hypothesis: str, reference: str) -> float:
    """ROUGE-L: longest common subsequence (token-level)."""
    hyp = _tokenize(hypothesis)
    ref = _tokenize(reference)
    if not hyp or not ref:
        return 0.0
    m, n = len(ref), len(hyp)
    # DP LCS
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    precision = lcs / n if n else 0.0
    recall = lcs / m if m else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ── Дополнительные метрики ───────────────────────────────────────

def lexical_diversity(text: str) -> float:
    """TTR: type-token ratio (разнообразие лексики)."""
    tokens = _tokenize(text)
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def context_coverage(answer: str, query: str) -> float:
    """Доля слов запроса (≥3 символа), встречающихся в ответе."""
    q_words = {w for w in _tokenize(query) if len(w) >= 3}
    if not q_words:
        return 0.0
    a_words = set(_tokenize(answer))
    return len(q_words & a_words) / len(q_words)


def specificity_score(answer: str, corpus_docs: list[dict]) -> float:
    """
    Специфичность: доля слов ответа, редких в корпусе (IDF > медианы).
    Высокая специфичность → ответ использует доменную лексику, а не общие слова.
    """
    # Строим IDF по корпусу
    doc_freq: Counter[str] = Counter()
    n_docs = len(corpus_docs)
    for doc in corpus_docs:
        words = set(_tokenize(doc.get("text", "")))
        doc_freq.update(words)
    if n_docs == 0:
        return 0.0

    idf = {w: math.log((n_docs + 1) / (df + 1)) for w, df in doc_freq.items()}
    median_idf = float(np.median(list(idf.values()))) if idf else 0.0

    ans_tokens = _tokenize(answer)
    if not ans_tokens:
        return 0.0
    specific = sum(1 for t in ans_tokens if idf.get(t, 0) > median_idf)
    return specific / len(ans_tokens)


def answer_length_score(answer: str, target_len: int = 300) -> float:
    """
    Нормированная длина ответа: штрафуем слишком короткие и слишком длинные.
    Оптимум ~300 токенов (развёрнутый, но не раздутый ответ).
    """
    n = len(_tokenize(answer))
    if n == 0:
        return 0.0
    if n <= target_len:
        return n / target_len
    # Мягкий штраф за избыточную длину
    return target_len / n


# ── Агрегация ────────────────────────────────────────────────────

class AnswerMetrics(NamedTuple):
    rouge1: float
    rouge2: float
    rougeL: float
    lexical_diversity: float
    context_coverage: float
    specificity: float
    length_score: float

    def mean(self) -> float:
        return float(np.mean(list(self)))


def compute_metrics(
    answer: str,
    query: str,
    reference: str | None = None,
    corpus_docs: list[dict] | None = None,
) -> AnswerMetrics:
    """
    Вычисляет все метрики для одного ответа.

    reference: эталонный ответ (если есть). Если None — используем query как reference
               (измеряем, насколько ответ «отвечает» на вопрос).
    corpus_docs: корпус для specificity. Если None — specificity = 0.
    """
    ref = reference if reference else query
    return AnswerMetrics(
        rouge1=rouge_n(answer, ref, n=1),
        rouge2=rouge_n(answer, ref, n=2),
        rougeL=rouge_l(answer, ref),
        lexical_diversity=lexical_diversity(answer),
        context_coverage=context_coverage(answer, query),
        specificity=specificity_score(answer, corpus_docs or []),
        length_score=answer_length_score(answer),
    )


def compare_methods(
    questions: list[str],
    answers_by_method: dict[str, list[str]],
    corpus_docs: list[dict] | None = None,
    references: list[str] | None = None,
) -> dict[str, dict]:
    """
    Сравнивает несколько методов по всем вопросам.

    Возвращает:
        {method_name: {metric_name: mean_value, "per_question": [AnswerMetrics, ...]}}
    """
    results: dict[str, dict] = {}
    for method, answers in answers_by_method.items():
        per_q: list[AnswerMetrics] = []
        for i, (q, a) in enumerate(zip(questions, answers)):
            ref = references[i] if references else None
            m = compute_metrics(a, q, reference=ref, corpus_docs=corpus_docs)
            per_q.append(m)

        # Средние по всем вопросам
        fields = AnswerMetrics._fields
        means = {
            f: float(np.mean([getattr(m, f) for m in per_q]))
            for f in fields
        }
        means["composite"] = float(np.mean(list(means.values())))
        results[method] = {**means, "per_question": per_q}

    return results


# ── Визуализация ─────────────────────────────────────────────────

METRIC_LABELS = {
    "rouge1": "ROUGE-1",
    "rouge2": "ROUGE-2",
    "rougeL": "ROUGE-L",
    "lexical_diversity": "Лекс. разнообразие",
    "context_coverage": "Покрытие запроса",
    "specificity": "Специфичность",
    "length_score": "Длина ответа",
    "composite": "Итоговый балл",
}

METHOD_COLORS = {
    "discourse_graph": "#2196F3",   # синий — наш метод
    "graphrag_baseline": "#FF9800", # оранжевый — GraphRAG baseline
    "long_context": "#9C27B0",      # фиолетовый — длинный контекст
}

DEFAULT_COLORS = ["#2196F3", "#FF9800", "#9C27B0", "#4CAF50", "#F44336"]


def plot_metrics_comparison(
    comparison: dict[str, dict],
    out_path: str | None = None,
    *,
    title: str = "Сравнение RAG-методов: автоматические метрики",
    figsize: tuple[float, float] = (14, 8),
) -> plt.Figure:
    """
    Строит сравнительный bar-chart по всем метрикам для каждого метода.
    Сохраняет в out_path если указан.
    """
    metrics = [f for f in AnswerMetrics._fields] + ["composite"]
    methods = list(comparison.keys())
    x = np.arange(len(metrics))
    width = 0.8 / max(len(methods), 1)

    fig, ax = plt.subplots(figsize=figsize)

    for i, method in enumerate(methods):
        vals = [comparison[method].get(m, 0.0) for m in metrics]
        color = METHOD_COLORS.get(method, DEFAULT_COLORS[i % len(DEFAULT_COLORS)])
        offset = (i - len(methods) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width * 0.9, label=method, color=color, alpha=0.85)
        # Подписи значений
        for bar, val in zip(bars, vals):
            if val > 0.02:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.2f}",
                    ha="center", va="bottom", fontsize=7, rotation=45,
                )

    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS.get(m, m) for m in metrics], rotation=30, ha="right", fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Значение метрики (0–1)", fontsize=10)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    if out_path:
        from pathlib import Path
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")

    return fig


def plot_radar_comparison(
    comparison: dict[str, dict],
    out_path: str | None = None,
    *,
    title: str = "Радар-диаграмма: профиль методов",
    figsize: tuple[float, float] = (8, 8),
) -> plt.Figure:
    """
    Radar/spider chart для сравнения методов по ключевым метрикам.
    """
    # Только основные метрики (без composite)
    metrics = list(AnswerMetrics._fields)
    labels = [METRIC_LABELS.get(m, m) for m in metrics]
    n = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]  # замкнуть

    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"polar": True})

    methods = list(comparison.keys())
    for i, method in enumerate(methods):
        vals = [comparison[method].get(m, 0.0) for m in metrics]
        vals += vals[:1]
        color = METHOD_COLORS.get(method, DEFAULT_COLORS[i % len(DEFAULT_COLORS)])
        ax.plot(angles, vals, "o-", linewidth=2, color=color, label=method)
        ax.fill(angles, vals, alpha=0.12, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=7)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
    fig.tight_layout()

    if out_path:
        from pathlib import Path
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")

    return fig


def plot_per_question_heatmap(
    comparison: dict[str, dict],
    questions: list[str],
    metric: str = "composite",
    out_path: str | None = None,
    *,
    title: str | None = None,
    figsize: tuple[float, float] = (12, 6),
) -> plt.Figure:
    """
    Тепловая карта: методы × вопросы, цвет = значение метрики.
    Позволяет увидеть, на каких вопросах каждый метод выигрывает.
    """
    methods = list(comparison.keys())
    n_q = len(questions)

    # Собираем матрицу значений
    matrix = np.zeros((len(methods), n_q))
    for i, method in enumerate(methods):
        per_q = comparison[method].get("per_question", [])
        for j in range(min(n_q, len(per_q))):
            m = per_q[j]
            if metric == "composite":
                matrix[i, j] = m.mean()
            else:
                matrix[i, j] = getattr(m, metric, 0.0)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label=METRIC_LABELS.get(metric, metric))

    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=9)
    ax.set_xticks(range(n_q))
    short_q = [f"Q{j+1}" for j in range(n_q)]
    ax.set_xticklabels(short_q, fontsize=9)

    # Подписи значений в ячейках
    for i in range(len(methods)):
        for j in range(n_q):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="black")

    _title = title or f"Метрика «{METRIC_LABELS.get(metric, metric)}» по вопросам"
    ax.set_title(_title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Вопрос", fontsize=10)

    # Легенда вопросов
    legend_lines = [f"Q{j+1}: {q[:60]}{'…' if len(q) > 60 else ''}" for j, q in enumerate(questions)]
    fig.text(0.01, -0.02, "\n".join(legend_lines), fontsize=6.5,
             verticalalignment="top", family="monospace")

    fig.tight_layout()

    if out_path:
        from pathlib import Path
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")

    return fig


def plot_composite_ranking(
    comparison: dict[str, dict],
    out_path: str | None = None,
    *,
    title: str = "Итоговый рейтинг методов",
    figsize: tuple[float, float] = (8, 5),
) -> plt.Figure:
    """
    Горизонтальный bar-chart итогового балла (composite) с разбивкой по метрикам.
    """
    methods = list(comparison.keys())
    metric_fields = list(AnswerMetrics._fields)
    n_metrics = len(metric_fields)

    # Stacked bar: каждая метрика — отдельный сегмент
    cmap = plt.cm.get_cmap("tab10", n_metrics)
    bottoms = np.zeros(len(methods))
    fig, ax = plt.subplots(figsize=figsize)

    for k, mf in enumerate(metric_fields):
        vals = np.array([comparison[m].get(mf, 0.0) / n_metrics for m in methods])
        ax.barh(methods, vals, left=bottoms, color=cmap(k),
                label=METRIC_LABELS.get(mf, mf), alpha=0.85)
        bottoms += vals

    # Итоговый балл как текст
    for i, method in enumerate(methods):
        composite = comparison[method].get("composite", 0.0)
        ax.text(bottoms[i] + 0.005, i, f"{composite:.3f}",
                va="center", fontsize=9, fontweight="bold")

    ax.set_xlim(0, max(bottoms) * 1.15)
    ax.set_xlabel("Нормированный вклад метрик", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()

    if out_path:
        from pathlib import Path
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")

    return fig


def save_metrics_report(
    comparison: dict[str, dict],
    questions: list[str],
    out_path: str = "output/rag_metrics_report.md",
) -> str:
    """Генерирует markdown-отчёт с таблицей метрик."""
    from pathlib import Path

    metric_fields = list(AnswerMetrics._fields) + ["composite"]
    lines = [
        "# Автоматические метрики: сравнение RAG-методов",
        "",
        "## Средние значения по всем вопросам",
        "",
    ]

    # Заголовок таблицы
    header = "| Метод | " + " | ".join(METRIC_LABELS.get(m, m) for m in metric_fields) + " |"
    sep = "|" + "|".join(["---"] * (len(metric_fields) + 1)) + "|"
    lines += [header, sep]

    for method, data in comparison.items():
        row = f"| **{method}** | "
        row += " | ".join(f"{data.get(m, 0.0):.3f}" for m in metric_fields)
        row += " |"
        lines.append(row)

    lines += ["", "---", "", "## Описание метрик", ""]
    descriptions = {
        "rouge1": "**ROUGE-1** — доля унiграмм ответа, совпадающих с запросом (лексическое перекрытие).",
        "rouge2": "**ROUGE-2** — доля биграмм ответа, совпадающих с запросом.",
        "rougeL": "**ROUGE-L** — наибольшая общая подпоследовательность (структурное сходство).",
        "lexical_diversity": "**Лекс. разнообразие** — TTR (type-token ratio): разнообразие словаря ответа.",
        "context_coverage": "**Покрытие запроса** — доля слов вопроса, встречающихся в ответе.",
        "specificity": "**Специфичность** — доля редких (доменных) слов в ответе (высокий IDF).",
        "length_score": "**Длина ответа** — нормированная длина (оптимум ~300 токенов).",
        "composite": "**Итоговый балл** — среднее по всем метрикам.",
    }
    for m in metric_fields:
        if m in descriptions:
            lines.append(f"- {descriptions[m]}")

    lines += ["", "---", "", "## Результаты по вопросам", ""]
    for j, q in enumerate(questions):
        lines.append(f"### Q{j+1}: {q}")
        lines.append("")
        q_header = "| Метод | " + " | ".join(METRIC_LABELS.get(m, m) for m in AnswerMetrics._fields) + " |"
        q_sep = "|" + "|".join(["---"] * (len(AnswerMetrics._fields) + 1)) + "|"
        lines += [q_header, q_sep]
        for method, data in comparison.items():
            per_q = data.get("per_question", [])
            if j < len(per_q):
                m_obj = per_q[j]
                row = f"| {method} | " + " | ".join(f"{getattr(m_obj, f):.3f}" for f in AnswerMetrics._fields) + " |"
            else:
                row = f"| {method} | " + " | ".join(["—"] * len(AnswerMetrics._fields)) + " |"
            lines.append(row)
        lines.append("")

    report = "\n".join(lines)
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(report, encoding="utf-8")
    return report
