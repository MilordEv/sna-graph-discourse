"""
Walk-by-nodes: BFS по графу от seed-вершин, близких к запросу.
Возвращает контекст для LLM-промпта: seed-концепты, типизированные связи
(контраст в приоритете) и короткие сниппеты из корпуса.

Узлы графа — ЛЕММЫ, поэтому и сопоставление запроса с вершинами, и поиск
сниппетов идут по леммам (иначе «истине» не нашло бы узел «истина»).
"""
from __future__ import annotations

from collections import deque

import networkx as nx

from discourse_graph.utils import lemma_grams, lemmatize_tokens

_CORE = ("истина", "правда", "ложь", "обман", "заблуждение")


def _seed_nodes(G: nx.Graph, query: str, top_k: int = 6) -> list[str]:
    """Сопоставление запрос → вершины по ЛЕММАМ (узлы графа — леммы)."""
    qlem = set(lemmatize_tokens(query))
    scored: list[tuple[float, str]] = []
    for node in G.nodes():
        ntoks = set(node.split())
        overlap = len(ntoks & qlem)
        if overlap:
            score = overlap / max(len(ntoks), 1)
            if ntoks <= qlem:  # узел целиком в запросе — точнее
                score += 1.0
            scored.append((score, node))
    scored.sort(reverse=True)
    seeds = [n for _, n in scored[:top_k]]
    if not seeds:
        seeds = [n for n in _CORE if n in G][:top_k]
    return seeds


def _format_edge(G: nx.Graph, u: str, v: str) -> str:
    data = G[u][v]
    rel = data.get("relation", "")
    methods = data.get("methods", "")
    m = methods if isinstance(methods, str) else ",".join(methods)
    tag = "контраст" if rel == "contrast" else ("эмоц." if "emotional" in str(m) else "со-вст.")
    return f'«{u}» — «{v}» [{tag}, вес {data.get("weight", 1)}]'


def _snippets_for_seeds(
    seeds: list[str], docs: list[dict], max_total: int = 4, max_chars: int = 220
) -> list[str]:
    """Короткие сниппеты из корпуса, где встречается лемма seed-узла."""
    seedset = set(seeds)
    out: list[str] = []
    for doc in docs:
        if len(out) >= max_total:
            break
        grams = lemma_grams(doc.get("text", ""))
        hit = seedset & grams
        if hit:
            title = doc.get("title", "")[:45]
            out.append(f"[{title}] {doc.get('text','')[:max_chars].strip()}…")
    return out


def retrieve_walk(
    G: nx.Graph,
    query: str,
    top_k: int = 6,
    depth: int = 2,
    docs: list[dict] | None = None,
) -> str:
    seeds = _seed_nodes(G, query, top_k)
    if not seeds:
        return "(граф не содержит вершин, связанных с запросом)"

    visited_nodes: set[str] = set()
    visited_edges: set[tuple[str, str]] = set()
    queue: deque[tuple[str, int]] = deque((s, 0) for s in seeds)
    while queue:
        node, dist = queue.popleft()
        if node in visited_nodes or dist > depth:
            continue
        visited_nodes.add(node)
        for nbr in G.neighbors(node):
            visited_edges.add((min(node, nbr), max(node, nbr)))
            if nbr not in visited_nodes and dist + 1 <= depth:
                queue.append((nbr, dist + 1))

    # Связи: контраст в приоритете, затем по весу. Только инцидентные seed/ядру.
    seedset = set(seeds)
    edges = []
    for u, v in visited_edges:
        if not G.has_edge(u, v):
            continue
        d = G[u][v]
        prio_seed = 1 if (u in seedset or v in seedset) else 0
        contrast = 1 if d.get("relation") == "contrast" else 0
        edges.append((prio_seed, contrast, d.get("weight", 1), _format_edge(G, u, v)))
    edges.sort(key=lambda x: (-x[0], -x[1], -x[2]))

    concepts = sorted(visited_nodes, key=lambda n: G.degree(n, weight="weight"), reverse=True)[:18]

    lines = ["## Контекст дискурс-графа (walk-by-nodes, локальный)"]
    lines.append(f"Концепты запроса: {', '.join(seeds)}")
    lines.append("")
    lines.append("Типизированные связи (контраст в приоритете):")
    for _, _, _, e in edges[:22]:
        lines.append("  • " + e)
    lines.append("")
    lines.append("Концепты в окрестности: " + ", ".join(concepts))

    if docs:
        snips = _snippets_for_seeds(seeds, docs)
        if snips:
            lines.append("")
            lines.append("Фрагменты корпуса:")
            for s in snips:
                lines.append("  • " + s)
    return "\n".join(lines)
