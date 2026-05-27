from __future__ import annotations

import random
from collections import Counter

import networkx as nx

from discourse_graph.config import ConstructorConfig


def dropout_by_actor(docs: list[dict], fraction: float, seed: int) -> list[dict]:
    """Имитация dropout «экспертов»: убираем документы с известным actor."""
    rng = random.Random(seed)
    with_actor = [d for d in docs if d.get("actor")]
    without = [d for d in docs if not d.get("actor")]
    if not with_actor:
        # fallback: dropout по source
        return dropout_by_source(docs, fraction, seed)
    n_remove = max(1, int(len(with_actor) * fraction))
    remove_ids = set(rng.sample([d["doc_id"] for d in with_actor], min(n_remove, len(with_actor))))
    kept = [d for d in docs if d["doc_id"] not in remove_ids]
    return kept if len(kept) >= 10 else docs


def dropout_by_source(docs: list[dict], fraction: float, seed: int) -> list[dict]:
    rng = random.Random(seed)
    sources = list({d.get("source", "unknown") for d in docs})
    if len(sources) < 2:
        shuffled = docs.copy()
        rng.shuffle(shuffled)
        cut = max(10, int(len(shuffled) * (1 - fraction)))
        return shuffled[:cut]
    n_remove = max(1, int(len(sources) * fraction))
    drop = set(rng.sample(sources, min(n_remove, len(sources) - 1)))
    return [d for d in docs if d.get("source") not in drop]


def invariant_core(
    graphs: list[nx.Graph],
    threshold: float = 0.6,
) -> tuple[set[str], set[tuple[str, str]]]:
    """Узлы и рёбра, стабильные в доле >= threshold прогонов."""
    if not graphs:
        return set(), set()
    n = len(graphs)
    node_votes: Counter[str] = Counter()
    edge_votes: Counter[tuple[str, str]] = Counter()
    for G in graphs:
        for node in G.nodes():
            node_votes[node] += 1
        for u, v in G.edges():
            edge_votes[tuple(sorted((u, v)))] += 1
    min_votes = max(1, int(n * threshold))
    core_nodes = {k for k, v in node_votes.items() if v >= min_votes}
    core_edges = {k for k, v in edge_votes.items() if v >= min_votes}
    return core_nodes, core_edges


def run_stress_test(
    docs: list[dict],
    config: ConstructorConfig,
) -> dict:
    from discourse_graph.pipeline import DiscourseGraphConstructor

    constructor = DiscourseGraphConstructor(config)
    graphs: list[nx.Graph] = []
    seeds = [config.random_seed + i * 17 for i in range(config.stress_n_seeds)]

    for seed in seeds:
        subset = dropout_by_actor(docs, config.stress_dropout_fraction, seed)
        # лёгкая стохастика: subsample 90% документов
        rng = random.Random(seed)
        if len(subset) > 20:
            idx = rng.sample(range(len(subset)), int(len(subset) * 0.9))
            subset = [subset[i] for i in sorted(idx)]
        G = constructor.build(subset, run_summarize=False)
        graphs.append(G)

    core_nodes, core_edges = invariant_core(graphs, config.stress_core_threshold)
    G_core = nx.Graph()
    for n in core_nodes:
        G_core.add_node(n)
    for u, v in core_edges:
        G_core.add_edge(u, v)

    return {
        "n_runs": len(graphs),
        "core_nodes": len(core_nodes),
        "core_edges": len(core_edges),
        "graph_core": G_core,
        "graphs": graphs,
        "node_stability": {
            n: sum(1 for G in graphs if G.has_node(n)) / len(graphs)
            for n in set().union(*[set(G.nodes()) for G in graphs])
        },
    }
