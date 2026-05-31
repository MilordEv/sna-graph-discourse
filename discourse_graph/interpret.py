"""
Интерпретатор «жареных» связей: подсвечивает рёбра с высоким surprisal
(неожиданные совместные появления) и объясняет их через LLM.

Вход: дискурс-граф + (опционально) OpenAI-совместимый клиент.
Выход: markdown-отчёт output/hot_edges_report.md.
"""
from __future__ import annotations

import csv
from pathlib import Path

import networkx as nx

PROMPT_TEMPLATE = """Ты — аналитик дискурса. В текстовом корпусе на тему «{domain}»
была обнаружена неожиданная смысловая связь между двумя концептами:

  «{node_a}» и «{node_b}»

Эта пара имеет высокий показатель surprisal (= {surprisal:.2f}), что означает:
базовая языковая модель не ожидает их совместного появления, но в данном корпусе
они встречаются вместе значимо часто.

Объясни в 2–3 предложениях, почему в данном дискурсе эти концепты оказываются связаны.
Что их объединяет с точки зрения темы «{domain}»?"""


def load_surprisal_edges(
    path: str | Path,
    top_n: int = 20,
) -> list[dict]:
    """Загружает top_n рёбер с наибольшим surprisal из CSV."""
    p = Path(path)
    if not p.exists():
        return []
    rows = []
    with open(p, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                row["surprisal"] = float(row.get("surprisal", 0))
            except ValueError:
                row["surprisal"] = 0.0
            rows.append(row)
    rows.sort(key=lambda r: r["surprisal"], reverse=True)
    return rows[:top_n]


def interpret_hot_edges(
    G: nx.Graph,
    surprisal_csv: str | Path,
    domain: str = "истина, правда и ложь",
    top_n: int = 15,
    llm_client=None,
    model: str = "deepseek/deepseek-chat",
    out_path: str | Path = "output/hot_edges_report.md",
) -> str:
    """
    Генерирует markdown-отчёт с объяснениями «жареных» связей.

    Если llm_client=None — создаёт stub-отчёт без LLM (полезно для проверки pipeline).
    """
    edges = load_surprisal_edges(surprisal_csv, top_n=top_n)
    if not edges:
        return "(Файл с surprisal-рёбрами не найден или пустой)"

    lines = [
        f"# Интерпретатор «жареных» связей",
        f"",
        f"Домен: **{domain}**  ",
        f"Топ-{len(edges)} рёбер по показателю surprisal  ",
        f"(surprisal = насколько базовая модель удивлена совместным появлением концептов)",
        f"",
    ]

    for i, row in enumerate(edges, 1):
        u = row.get("source", row.get("u", "?"))
        v = row.get("target", row.get("v", "?"))
        surprisal = row["surprisal"]

        # Дополнительные атрибуты из графа
        edge_data = G.get_edge_data(u, v, {})
        methods = edge_data.get("methods", "")
        if isinstance(methods, list):
            methods = ", ".join(methods)
        relation = edge_data.get("relation", "")
        weight = edge_data.get("weight", "")

        lines.append(f"## {i}. «{u}» — «{v}»")
        lines.append(f"- **Surprisal:** {surprisal:.2f}")
        if weight:
            lines.append(f"- **Вес (частота):** {weight}")
        if methods:
            lines.append(f"- **Методы построения:** {methods}")
        if relation:
            lines.append(f"- **Тип отношения:** {relation}")
        lines.append("")

        if llm_client:
            prompt = PROMPT_TEMPLATE.format(
                domain=domain,
                node_a=u,
                node_b=v,
                surprisal=surprisal,
            )
            try:
                resp = llm_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.4,
                    max_tokens=300,
                )
                explanation = resp.choices[0].message.content.strip()
            except Exception as e:
                explanation = f"(Ошибка LLM: {e})"
        else:
            explanation = (
                f"_[LLM не подключён — укажите llm_client для автоматической интерпретации]_  \n"
                f"Подсказка: почему «{u}» и «{v}» появляются вместе в контексте «{domain}»?"
            )

        lines.append(f"**Интерпретация:**  \n{explanation}")
        lines.append("")
        lines.append("---")
        lines.append("")

    report = "\n".join(lines)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(report)

    return report
