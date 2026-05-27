from __future__ import annotations

import json
import math
import re
from collections import Counter
from itertools import combinations
from pathlib import Path

import networkx as nx

SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+")
WORD_RE = re.compile(r"[а-яёa-z0-9][а-яёa-z0-9\-]*", re.I)


def load_documents(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENT_SPLIT.split(text) if s.strip()]


def split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def normalize_label(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def tokenize(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def apply_pmi(G: nx.Graph) -> nx.Graph:
    total = sum(d.get("weight", 1) for _, _, d in G.edges(data=True))
    if total == 0:
        return G
    node_freq: Counter[str] = Counter()
    for u, v, data in G.edges(data=True):
        w = data.get("weight", 1)
        node_freq[u] += w
        node_freq[v] += w
    for u, v, data in G.edges(data=True):
        w = data["weight"]
        pxy = w / total
        px = node_freq[u] / total
        py = node_freq[v] / total
        if pxy > 0 and px > 0 and py > 0:
            data["pmi"] = round(math.log2(pxy / (px * py)), 4)
        else:
            data["pmi"] = 0.0
    return G


def filter_graph(
    G: nx.Graph,
    top_nodes: int = 400,
    min_weight: int = 2,
    min_pmi: float = 5.0,
) -> nx.Graph:
    if G.number_of_edges() == 0:
        return G
    deg: Counter[str] = Counter()
    for u, v, data in G.edges(data=True):
        w = data.get("weight", 1)
        deg[u] += w
        deg[v] += w
    top = {n for n, _ in deg.most_common(top_nodes)}
    G2 = nx.Graph()
    for u, v, data in G.edges(data=True):
        if u not in top or v not in top:
            continue
        if data.get("weight", 0) < min_weight:
            continue
        G2.add_edge(u, v, **data)
    G2 = apply_pmi(G2)
    remove = [(u, v) for u, v, d in G2.edges(data=True) if d.get("pmi", 0) < min_pmi]
    G2.remove_edges_from(remove)
    G2.remove_nodes_from(list(nx.isolates(G2)))
    return G2


def merge_edge_attrs(G: nx.Graph, u: str, v: str, **attrs) -> None:
    if u == v:
        return
    if G.has_edge(u, v):
        for k, val in attrs.items():
            if k == "weight":
                G[u][v][k] = G[u][v].get(k, 0) + val
            elif k == "methods":
                existing = set(G[u][v].get(k, []))
                existing.update(val if isinstance(val, list) else [val])
                G[u][v][k] = sorted(existing)
            else:
                G[u][v][k] = val
    else:
        G.add_edge(u, v, **attrs)


def add_cooccurrence_edges(
    G: nx.Graph,
    nodes_in_unit: list[str],
    weight: int = 1,
    method: str = "cooccurrence",
) -> None:
    unique = list(dict.fromkeys(nodes_in_unit))
    if len(unique) < 2:
        return
    for a, b in combinations(unique, 2):
        merge_edge_attrs(
            G,
            a,
            b,
            weight=weight,
            methods=[method],
        )


def _graph_for_export(G: nx.Graph) -> nx.Graph:
    """GraphML не поддерживает list — сериализуем в строки."""
    H = G.copy()
    for _, _, data in H.edges(data=True):
        if "methods" in data and isinstance(data["methods"], list):
            data["methods"] = ",".join(data["methods"])
    for _, data in H.nodes(data=True):
        for k, v in list(data.items()):
            if isinstance(v, list):
                data[k] = ",".join(str(x) for x in v)
    return H


def save_graph(G: nx.Graph, out_dir: str | Path) -> None:
    import pandas as pd

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(_graph_for_export(G), out / "discourse_graph.graphml")
    edges = [
        {"source": u, "target": v, **{k: d[k] for k in d}}
        for u, v, d in G.edges(data=True)
    ]
    pd.DataFrame(edges).to_csv(out / "edges.csv", index=False)
    nodes = [{"id": n, **G.nodes[n]} for n in G.nodes()]
    pd.DataFrame(nodes).to_csv(out / "nodes.csv", index=False)
