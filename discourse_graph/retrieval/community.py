"""
Community-based retrieval: детектируем сообщества в графе, для каждого
генерируем резюме (через LLM или stub), при запросе выбираем топ-k сообществ
по лексическому/семантическому сходству с запросом.
"""
from __future__ import annotations

import re
from collections import Counter

import networkx as nx


def _detect_communities(G: nx.Graph) -> list[frozenset[str]]:
    if G.number_of_nodes() == 0:
        return []
    try:
        return list(nx.community.louvain_communities(G, seed=42))
    except Exception:
        # Запасной вариант: greedy modularity
        return list(nx.community.greedy_modularity_communities(G))


def _community_summary_stub(G: nx.Graph, comm: frozenset[str]) -> str:
    """Stub-резюме без LLM: топ-узлы по степени + типы связей."""
    sub = G.subgraph(comm)
    deg = Counter(dict(sub.degree(weight="weight")))
    top = [n for n, _ in deg.most_common(8)]
    edge_types: set[str] = set()
    for _, _, data in sub.edges(data=True):
        m = data.get("methods", "")
        if isinstance(m, list):
            edge_types.update(m)
        elif m:
            edge_types.update(m.split(","))
    return (
        f"Концепты: {', '.join(top)}. "
        f"Типы связей: {', '.join(sorted(edge_types))}. "
        f"Размер сообщества: {len(comm)}."
    )


def _query_score(summary: str, query: str) -> float:
    """Простой скор: число общих слов (≥3 символа) между резюме и запросом."""
    words = lambda s: set(re.findall(r"[а-яёa-z]{3,}", s.lower()))
    overlap = words(summary) & words(query)
    return len(overlap)


def build_community_index(
    G: nx.Graph,
    llm_fn: "Callable[[str], str] | None" = None,
) -> list[dict]:
    """
    Строит индекс сообществ со stub-резюме (или через LLM если передана функция).
    Возвращает список записей {community_id, nodes, summary}.
    """
    comms = _detect_communities(G)
    index = []
    for i, comm in enumerate(comms):
        if llm_fn:
            sub = G.subgraph(comm)
            top_terms = [n for n, _ in Counter(dict(sub.degree(weight="weight"))).most_common(8)]
            prompt = (
                "Опиши тематику следующей группы концептов из дискурс-графа одним предложением: "
                + ", ".join(top_terms)
            )
            summary = llm_fn(prompt)
        else:
            summary = _community_summary_stub(G, comm)
        index.append({
            "community_id": i,
            "nodes": sorted(comm),
            "summary": summary,
        })
    return index


def retrieve_community(
    G: nx.Graph,
    query: str,
    top_k: int = 3,
    llm_fn: "Callable[[str], str] | None" = None,
) -> str:
    """
    Выбирает топ-k сообществ по близости к запросу.
    Возвращает форматированный контекст для LLM-промпта.
    """
    index = build_community_index(G, llm_fn)
    if not index:
        return "(граф пустой — сообщества не обнаружены)"

    scored = sorted(index, key=lambda c: _query_score(c["summary"], query), reverse=True)
    top = scored[:top_k]

    lines = ["## Контекст из дискурс-графа (стратегия: community-based)"]
    lines.append(f"Запрос: «{query}»")
    lines.append(f"Выбрано сообществ: {len(top)} из {len(index)}")
    lines.append("")

    for entry in top:
        lines.append(f"### Сообщество #{entry['community_id']} ({len(entry['nodes'])} узлов)")
        lines.append(f"Резюме: {entry['summary']}")
        lines.append(f"Узлы: {', '.join(entry['nodes'][:15])}")
        # Внутренние рёбра сообщества
        comm_set = set(entry["nodes"])
        edges_desc: list[str] = []
        for u, v, data in G.edges(data=True):
            if u in comm_set and v in comm_set:
                rel = data.get("relation", "")
                m = data.get("methods", "")
                if isinstance(m, list):
                    m = ",".join(m)
                lbl = f'"{u}" — "{v}"'
                if rel:
                    lbl += f" [{rel}]"
                if m:
                    lbl += f" ({m})"
                edges_desc.append(lbl)
        if edges_desc:
            lines.append(f"Связи ({len(edges_desc)}): " + "; ".join(edges_desc[:10]))
        lines.append("")

    return "\n".join(lines)
