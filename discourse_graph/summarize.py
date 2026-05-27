from __future__ import annotations

from typing import TYPE_CHECKING

from discourse_graph.models import DEFAULT_HF_MODEL, summarize_extractive

if TYPE_CHECKING:
    import networkx as nx


def summarize_vertices_hf(
    G: "nx.Graph",
    docs: list[dict],
    model_name: str = DEFAULT_HF_MODEL,
) -> None:
    """
    Саморайз вершин через лёгкий энкодер Hugging Face (по умолчанию rubert-tiny2).
    Для каждого узла выбирается наиболее релевантный фрагмент из корпуса.
    """
    snippets: dict[str, list[str]] = {}
    nodes = list(G.nodes())
    for doc in docs:
        text_l = doc.get("text", "").lower()
        for n in nodes:
            if n not in text_l:
                continue
            idx = text_l.find(n)
            snip = doc["text"][max(0, idx - 100) : idx + len(n) + 150]
            snippets.setdefault(n, []).append(snip[:400])
            if len(snippets[n]) >= 5:
                break

    for n in nodes:
        summary = summarize_extractive(n, snippets.get(n, []), model_name=model_name)
        if summary:
            G.nodes[n]["summary"] = summary


# обратная совместимость
summarize_vertices_deepseek = summarize_vertices_hf
