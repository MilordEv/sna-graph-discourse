"""
LightRAG-style: двухуровневый поиск — локальный (walk) + глобальный (community).
Объединяет результаты, дедуплицирует, формирует единый контекст для LLM.
"""
from __future__ import annotations

import networkx as nx

from discourse_graph.retrieval.community import retrieve_community
from discourse_graph.retrieval.walk import retrieve_walk


def retrieve_lightrag(
    G: nx.Graph,
    query: str,
    walk_top_k: int = 5,
    walk_depth: int = 2,
    community_top_k: int = 2,
    llm_fn: "Callable[[str], str] | None" = None,
    docs: list[dict] | None = None,
) -> str:
    """
    Двухуровневый поиск:
    - Local: walk-by-nodes (узкий контекст вокруг seed-вершин)
    - Global: community-based (широкий тематический контекст)

    Возвращает объединённый форматированный текст для LLM-промпта.
    """
    local_ctx = retrieve_walk(G, query, top_k=walk_top_k, depth=walk_depth, docs=docs)
    global_ctx = retrieve_community(G, query, top_k=community_top_k, llm_fn=llm_fn)

    lines = [
        "## Контекст из дискурс-графа (стратегия: LightRAG — local + global)",
        "",
        "### Локальный контекст (walk-by-nodes)",
        local_ctx,
        "",
        "### Глобальный контекст (community-based)",
        global_ctx,
    ]
    return "\n".join(lines)
