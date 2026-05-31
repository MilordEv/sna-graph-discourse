"""
Генератор отчёта для экспертной оценки (blind review).

Для каждого вопроса из списка получает ответы от трёх систем:
  1. Дискурс-граф + reasoning (наш подход)
  2. GraphRAG baseline
  3. Длинный контекст (без графа)

Сохраняет markdown с анонимизированными колонками A/B/C для слепого сравнения.

Запуск:
    python scripts/export_expert_report.py \
        --graph data/graphs/russkaya_istina/constructor/discourse/discourse_graph.graphml \
        --baseline data/graphs/russkaya_istina/constructor/graphrag_baseline/discourse_graph.graphml \
        --corpus data/raw/russkaya_istina/documents.json \
        --questions questions.json \
        --api-key $DEEPSEEK_KEY \
        --out output/expert_report.md
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Экспортировать отчёт для экспертной оценки")
    p.add_argument("--graph", required=True, help="Путь к discourse_graph.graphml (наш)")
    p.add_argument("--baseline", required=True, help="Путь к graphml GraphRAG baseline")
    p.add_argument("--corpus", required=True, help="Путь к documents.json")
    p.add_argument("--questions", required=True, help="JSON-файл со списком вопросов")
    p.add_argument("--api-key", default="", help="API-ключ DeepSeek/OpenAI")
    p.add_argument("--base-url", default="https://openrouter.ai/api/v1", help="Base URL API")
    p.add_argument("--model", default="deepseek/deepseek-chat")
    p.add_argument("--strategy", default="lightrag", choices=["walk", "community", "lightrag"])
    p.add_argument("--out", default="output/expert_report.md")
    return p.parse_args()


def load_graph(path: str):
    import networkx as nx
    return nx.read_graphml(path)


def build_report(
    questions: list[str],
    answers_our: list[str],
    answers_baseline: list[str],
    answers_longctx: list[str],
    domain: str,
) -> str:
    """
    Собирает слепой отчёт: системы перемешаны как A/B/C случайно,
    маппинг сохраняется в конце для деанонимизации.
    """
    seed = 42
    rng = random.Random(seed)
    mapping: list[dict] = []

    lines = [
        f"# Экспертная оценка: дискурс-граф vs baseline",
        f"",
        f"Домен: **{domain}**  ",
        f"Инструкция: для каждого вопроса выберите лучший ответ (A, B или C).  ",
        f"Оцените по критериям: точность, полнота, связность с темой.  ",
        f"",
        "---",
        "",
    ]

    for i, (q, our, base, lc) in enumerate(
        zip(questions, answers_our, answers_baseline, answers_longctx), 1
    ):
        # Перемешиваем ответы
        labeled = [("our", our), ("baseline", base), ("longctx", lc)]
        rng.shuffle(labeled)
        # letter → system_name (для деанонимизации)
        letter_map = {letter: lbl for (lbl, _), letter in zip(labeled, ["A", "B", "C"])}
        mapping.append({"question_id": i, **letter_map})

        lines.append(f"## Вопрос {i}")
        lines.append(f"**{q}**")
        lines.append("")
        for (lbl, ans), letter in zip(labeled, ["A", "B", "C"]):
            lines.append(f"### Ответ {letter}")
            lines.append(ans.strip() if ans.strip() else "_Ответ не получен_")
            lines.append("")
        lines.append("**Ваш выбор (A/B/C):** ___")
        lines.append("")
        lines.append("---")
        lines.append("")

    # Ключ деанонимизации (скрыт в конце)
    lines.append("<!-- DEANON KEY (не показывать экспертам до завершения оценки)")
    for m in mapping:
        a_sys = {v: k for k, v in m.items() if k != "question_id"}
        lines.append(
            f"Q{m['question_id']}: "
            + ", ".join(f"{letter}={sys_name}" for letter, sys_name in sorted(a_sys.items()))
        )
    lines.append("-->")

    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    # Загружаем вопросы
    with open(args.questions, encoding="utf-8") as f:
        questions: list[str] = json.load(f)

    if not questions:
        print("Список вопросов пустой")
        sys.exit(1)

    # Загружаем корпус
    with open(args.corpus, encoding="utf-8") as f:
        docs: list[dict] = json.load(f)

    # Загружаем графы
    G_our = load_graph(args.graph)
    G_base = load_graph(args.baseline)

    domain = "истина, правда и ложь"

    answers_our: list[str] = []
    answers_baseline: list[str] = []
    answers_longctx: list[str] = []

    if args.api_key:
        try:
            from openai import OpenAI
            from discourse_graph.qa import DiscourseQA, flat_context_answer

            client = OpenAI(api_key=args.api_key, base_url=args.base_url)
            qa_our = DiscourseQA(G_our, client, model=args.model, strategy=args.strategy)
            qa_base = DiscourseQA(G_base, client, model=args.model, strategy="community")

            for q in questions:
                print(f"[Наш граф] {q[:60]}...")
                answers_our.append(qa_our.answer(q))
                print(f"[Baseline] {q[:60]}...")
                answers_baseline.append(qa_base.answer(q))
                print(f"[LongCtx]  {q[:60]}...")
                answers_longctx.append(flat_context_answer(docs, q, client, model=args.model))

        except ImportError:
            print("Установите openai: pip install openai")
            sys.exit(1)
    else:
        print("API-ключ не передан — генерируем stub-отчёт (заглушки для ответов)")
        for q in questions:
            answers_our.append(f"[Наш дискурс-граф — ответ будет получен с API-ключом]\nВопрос: {q}")
            answers_baseline.append(f"[GraphRAG baseline — ответ будет получен с API-ключом]\nВопрос: {q}")
            answers_longctx.append(f"[Длинный контекст — ответ будет получен с API-ключом]\nВопрос: {q}")

    report = build_report(questions, answers_our, answers_baseline, answers_longctx, domain)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\nОтчёт сохранён: {out}")
    print(f"Вопросов: {len(questions)} | Документов в корпусе: {len(docs)}")


if __name__ == "__main__":
    main()
