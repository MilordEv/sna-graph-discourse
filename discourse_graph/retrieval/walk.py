"""
Walk-by-nodes: BFS по графу от seed-вершин, близких к запросу.
Возвращает контекст как список текстовых фрагментов для LLM-промпта.
"""
from __future__ import annotations

from collections import deque

import networkx as nx


def _seed_nodes(G: nx.Graph, query: str, top_k: int = 5) -> list[str]:
    """Простое лексическое совпадение запрос → вершины графа."""
    ql = query.lower()
    scored: list[tuple[float, str]] = []
    for node in G.nodes():
        nl = node.lower()
        # Точное вхождение лучше частичного
        if nl in ql or ql in nl:
            scored.append((1.0, node))
        else:
            # Мягкий скор по количеству общих слов
            q_words = set(ql.split())
            n_words = set(nl.split())
            overlap = len(q_words & n_words)
            if overlap:
                scored.append((overlap / max(len(q_words), 1), node))
    scored.sort(reverse=True)
    return [n for _, n in scored[:top_k]]


def _format_edge(G: nx.Graph, u: str, v: str) -> str:
    data = G[u][v]
    methods = data.get("methods", "")
    relation = data.get("relation", "")
    parts = [f'"{u}" — "{v}"']
    if relation:
        parts.append(f"тип: {relation}")
    if methods:
        m = methods if isinstance(methods, str) else ",".join(methods)
        parts.append(f"методы: {m}")
    w = data.get("weight", "")
    if w:
        parts.append(f"вес: {w}")
    return " | ".join(parts)


def _snippets_for_seeds(
    seeds: list[str], docs: list[dict], max_per_seed: int = 2, max_chars: int = 200
) -> dict[str, list[str]]:
    """Находит короткие сниппеты из корпуса для каждого seed-узла."""
    result: dict[str, list[str]] = {s: [] for s in seeds}
    for doc in docs:
        text = doc.get("text", "")
        title = doc.get("title", "")
        tl = text.lower()
        for seed in seeds:
            if len(result[seed]) >= max_per_seed:
                continue
            if seed in tl:
                # Ищем ближайший фрагмент вокруг вхождения
                idx = tl.find(seed)
                start = max(0, idx - 60)
                end = min(len(text), idx + len(seed) + 100)
                snippet = ("…" if start > 0 else "") + text[start:end].strip() + "…"
                label = f"[{title[:40]}]" if title else "[б/н]"
                result[seed].append(f"{label} «{snippet[:max_chars]}»")
    return result


def retrieve_walk(
    G: nx.Graph,
    query: str,
    top_k: int = 5,
    depth: int = 2,
    docs: list[dict] | None = None,
) -> str:
    """
    BFS от seed-вершин на глубину `depth`.
    Возвращает форматированный текст с узлами, рёбрами и сниппетами для LLM-промпта.
    """
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
            edge_key = (min(node, nbr), max(node, nbr))
            visited_edges.add(edge_key)
            if nbr not in visited_nodes and dist + 1 <= depth:
                queue.append((nbr, dist + 1))

    lines = ["## Контекст из дискурс-графа (стратегия: walk-by-nodes)"]
    lines.append(f"Seed-вершины (близкие к запросу «{query}»): {', '.join(seeds)}")
    lines.append("")
    lines.append(f"Вершины ({len(visited_nodes)}):")
    for n in sorted(visited_nodes):
        attr = G.nodes[n]
        et = attr.get("entity_type", "")
        score = attr.get("score", "")
        parts = [f"  • {n}"]
        if et:
            parts.append(f"[{et}]")
        if score:
            parts.append(f"score={float(score):.2f}")
        lines.append(" ".join(parts))

    lines.append("")
    lines.append(f"Связи ({len(visited_edges)}):")
    for u, v in sorted(visited_edges):
        if G.has_edge(u, v):
            lines.append("  • " + _format_edge(G, u, v))

    # Сниппеты из корпуса для seed-узлов
    if docs:
        snippets = _snippets_for_seeds(seeds, docs)
        any_snippets = any(v for v in snippets.values())
        if any_snippets:
            lines.append("")
            lines.append("Фрагменты из корпуса (для seed-вершин):")
            for seed, snips in snippets.items():
                if snips:
                    lines.append(f"  [{seed}]")
                    for s in snips:
                        lines.append(f"    {s}")

    return "\n".join(lines)
