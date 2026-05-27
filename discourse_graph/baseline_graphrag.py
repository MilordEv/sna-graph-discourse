from __future__ import annotations

import json
from collections import Counter
from itertools import combinations
from pathlib import Path

import networkx as nx

from discourse_graph.utils import apply_pmi, filter_graph, normalize_label, split_sentences


def build_vanilla_graphrag_graph(
    docs: list[dict],
    top_nodes: int = 300,
    min_weight: int = 2,
    min_pmi: float = 4.0,
) -> nx.Graph:
    """
    Упрощённый GraphRAG-baseline: NER-сущности + co-occurrence по документу,
    без риторики/эмоций. Близко к entity graph + частотная фильтрация.
    """
    G = nx.Graph()
    for doc in docs:
        entities = list(
            {
                normalize_label(e["text"])
                for e in doc.get("entities", [])
                if len(normalize_label(e["text"])) >= 2
            }
        )
        # уровень документа (типично для KG/GraphRAG extraction pipelines)
        if len(entities) >= 2:
            for a, b in combinations(entities, 2):
                if G.has_edge(a, b):
                    G[a][b]["weight"] += 1
                else:
                    G.add_edge(a, b, weight=1, methods=["graphrag_baseline"])
        # дополнительно предложения
        for sent in split_sentences(doc.get("text", "")):
            sl = sent.lower()
            present = [e for e in entities if e in sl]
            if len(present) >= 2:
                for a, b in combinations(present, 2):
                    if G.has_edge(a, b):
                        G[a][b]["weight"] += 1
                    else:
                        G.add_edge(a, b, weight=1, methods=["graphrag_baseline"])

    return filter_graph(G, top_nodes=top_nodes, min_weight=min_weight, min_pmi=min_pmi)


def export_graphrag_communities(
    G: nx.Graph,
    out_path: str | Path,
) -> list[dict]:
    """Community summaries (заглушка без LLM): топ-узлы по степени в каждой компоненте."""
    import pandas as pd

    communities = list(nx.community.greedy_modularity_communities(G)) if G.number_of_nodes() else []
    records = []
    for i, comm in enumerate(communities):
        sub = G.subgraph(comm)
        deg = Counter(dict(sub.degree(weight="weight")))
        top_terms = [n for n, _ in deg.most_common(8)]
        records.append(
            {
                "community_id": i,
                "size": len(comm),
                "top_entities": top_terms,
                "summary_stub": "Ключевые сущности: " + ", ".join(top_terms),
            }
        )
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    pd.DataFrame(records).to_csv(path.with_suffix(".csv"), index=False)
    return records
