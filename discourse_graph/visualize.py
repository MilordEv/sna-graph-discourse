from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


def _parse_methods(val) -> str:
    if pd.isna(val):
        return ""
    s = str(val).replace("[", "").replace("]", "").replace("'", "")
    return s


def load_graph_from_csv(nodes_path: Path, edges_path: Path) -> nx.Graph:
    nodes = pd.read_csv(nodes_path)
    edges = pd.read_csv(edges_path)
    G = nx.Graph()
    for _, row in nodes.iterrows():
        G.add_node(row["id"], **{k: row[k] for k in nodes.columns if k != "id" and pd.notna(row[k])})
    for _, row in edges.iterrows():
        attrs = {k: row[k] for k in edges.columns if k not in ("source", "target") and pd.notna(row[k])}
        if "methods" in attrs:
            attrs["methods"] = _parse_methods(attrs["methods"])
        G.add_edge(row["source"], row["target"], **attrs)
    return G


def _top_subgraph(G: nx.Graph, max_nodes: int) -> nx.Graph:
    deg = dict(G.degree(weight="weight"))
    top = sorted(deg, key=deg.get, reverse=True)[:max_nodes]
    return G.subgraph(top)


def show_graph_network(
    G: nx.Graph,
    *,
    title: str = "Дискурс-граф",
    max_nodes: int = 50,
    seed: int = 42,
    figsize: tuple[float, float] = (14, 11),
) -> plt.Figure | None:
    """Интерактивная отрисовка подграфа топ-узлов (для ноутбуков)."""
    if G.number_of_nodes() == 0:
        return None
    sub = _top_subgraph(G, max_nodes)
    pos = nx.spring_layout(sub, seed=seed, k=1.1, iterations=50)
    sizes = [200 + 60 * sub.degree(n, weight="weight") for n in sub.nodes()]
    weights = [sub[u][v].get("weight", 1) for u, v in sub.edges()]
    w_max = max(weights) if weights else 1

    fig, ax = plt.subplots(figsize=figsize)
    nx.draw_networkx_edges(
        sub, pos, alpha=0.35, width=[1 + 3 * w / w_max for w in weights], edge_color="#888888", ax=ax
    )
    nx.draw_networkx_nodes(sub, pos, node_size=sizes, node_color="#7eb8da", edgecolors="#333", ax=ax)
    nx.draw_networkx_labels(sub, pos, font_size=7, ax=ax)
    ax.set_title(f"{title} (топ-{len(sub)} узлов)", fontsize=14)
    ax.axis("off")
    fig.tight_layout()
    return fig


def plot_graph_network(
    G: nx.Graph,
    out_path: Path,
    *,
    title: str = "Дискурс-граф",
    max_nodes: int = 50,
    seed: int = 42,
) -> None:
    fig = show_graph_network(G, title=title, max_nodes=max_nodes, seed=seed)
    if fig is not None:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


def show_communities(
    G: nx.Graph,
    *,
    title: str = "Сообщества (greedy modularity)",
    max_nodes: int = 60,
    seed: int = 42,
    figsize: tuple[float, float] = (14, 11),
) -> plt.Figure | None:
    if G.number_of_nodes() < 3:
        return None
    sub = _top_subgraph(G, max_nodes)
    communities = list(nx.community.greedy_modularity_communities(sub, weight="weight"))
    color_map = {}
    palette = plt.cm.tab20(np.linspace(0, 1, max(len(communities), 1)))
    for i, comm in enumerate(communities):
        for n in comm:
            color_map[n] = palette[i % len(palette)]

    pos = nx.spring_layout(sub, seed=seed, k=1.1)
    fig, ax = plt.subplots(figsize=figsize)
    colors = [color_map.get(n, "#cccccc") for n in sub.nodes()]
    nx.draw_networkx_edges(sub, pos, alpha=0.3, ax=ax)
    nx.draw_networkx_nodes(sub, pos, node_color=colors, node_size=280, edgecolors="#333", ax=ax)
    nx.draw_networkx_labels(sub, pos, font_size=7, ax=ax)
    ax.set_title(f"{title} ({len(communities)} сообществ)", fontsize=14)
    ax.axis("off")
    fig.tight_layout()
    return fig


def plot_communities(
    G: nx.Graph,
    out_path: Path,
    *,
    title: str = "Сообщества (greedy modularity)",
    max_nodes: int = 60,
    seed: int = 42,
) -> None:
    fig = show_communities(G, title=title, max_nodes=max_nodes, seed=seed)
    if fig is not None:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


def show_degree_distribution(G: nx.Graph, *, title: str = "Распределение степеней") -> plt.Figure | None:
    degrees = np.array([d for _, d in G.degree()])
    if len(degrees) == 0:
        return None
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(degrees, bins=min(30, max(5, len(set(degrees)))), color="steelblue", edgecolor="white")
    ax.set_xlabel("Степень узла")
    ax.set_ylabel("Число узлов")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def show_graphs_comparison(
    graphs: dict[str, nx.Graph],
    *,
    title: str = "Сравнение графов",
) -> plt.Figure:
    labels = list(graphs.keys())
    nodes = [graphs[k].number_of_nodes() for k in labels]
    edges = [graphs[k].number_of_edges() for k in labels]
    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - w / 2, nodes, w, label="узлы", color="#4c72b0")
    ax.bar(x + w / 2, edges, w, label="рёбра", color="#dd8452")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_degree_distribution(degrees: np.ndarray, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(degrees, bins=min(30, max(5, len(set(degrees)))), color="steelblue", edgecolor="white")
    ax.set_xlabel("Степень узла")
    ax.set_ylabel("Число узлов")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_edge_threshold_curve(weights: np.ndarray, n_nodes: int, out_path: Path) -> None:
    thresholds = np.arange(1, int(max(weights)) + 1) if len(weights) else [1]
    n_edges = [(weights >= t).sum() for t in thresholds]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(thresholds, n_edges, "b-", lw=2)
    ax.axhline(n_nodes * 2, color="gray", ls="--", label="~2n")
    ax.axhline(n_nodes * 3, color="gray", ls=":", label="~3n")
    ax.set_xlabel("Минимальный вес ребра")
    ax.set_ylabel("Число рёбер")
    ax.set_title("Чувствительность плотности к порогу")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_comparison_bars(stats: dict, out_path: Path) -> None:
    """Сравнение метрик: корпусный граф vs конструктор vs baseline."""
    labels = list(stats.keys())
    nodes = [stats[k]["nodes"] for k in labels]
    edges = [stats[k]["edges"] for k in labels]
    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - w / 2, nodes, w, label="узлы", color="#4c72b0")
    ax.bar(x + w / 2, edges, w, label="рёбра", color="#dd8452")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_title("Сравнение графов")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
