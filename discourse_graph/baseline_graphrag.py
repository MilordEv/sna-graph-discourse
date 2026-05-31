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
    без риторики/эмоций. Если NER пустой — фолбэк на TF-IDF термины.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    G = nx.Graph()

    # Собираем сущности из NER
    all_entities: list[list[str]] = []
    for doc in docs:
        ents = list(
            {
                normalize_label(e["text"])
                for e in doc.get("entities", [])
                if len(normalize_label(e["text"])) >= 2
            }
        )
        all_entities.append(ents)

    # Если NER пустой везде — фолбэк на TF-IDF топ-термины
    if not any(all_entities):
        texts = [d.get("text", "") for d in docs]
        try:
            vec = TfidfVectorizer(
                max_features=top_nodes,
                ngram_range=(1, 2),
                min_df=2,
                token_pattern=r"(?u)\b[а-яёa-z][а-яёa-z\-]{2,}\b",
            )
            X = vec.fit_transform(texts)
            terms = vec.get_feature_names_out()
            for doc_idx, doc in enumerate(docs):
                tl = doc.get("text", "").lower()
                ents = [t for t in terms if t in tl]
                all_entities[doc_idx] = ents
        except Exception:
            pass

    for entities in all_entities:
        if len(entities) >= 2:
            for a, b in combinations(entities, 2):
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
