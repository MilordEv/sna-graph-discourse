#!/usr/bin/env python3
"""
Полный EDA корпуса и графов. Сохраняет фигуры и JSON-статистику в output/eda/.
Запуск: python scripts/run_eda.py
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from discourse_graph.visualize import (
    load_graph_from_csv,
    plot_communities,
    plot_comparison_bars,
    plot_degree_distribution,
    plot_edge_threshold_curve,
    plot_graph_network,
)

CORPUS = ROOT / "data/raw/naukogrady/documents.json"
GRAPH_LEGACY = ROOT / "data/graphs/naukogrady"
GRAPH_CONSTRUCTOR = ROOT / "data/graphs/naukogrady/constructor/discourse"
GRAPH_BASELINE = ROOT / "data/graphs/naukogrady/constructor/graphrag_baseline"
OUT = ROOT / "output/eda"
FIG = OUT / "figures"

NAUKOGRAD_MARKERS = re.compile(
    r"наукоград|наукоёмк|нии|технопарк|исследовател|сколково|дубна|обнинск",
    re.I,
)
CITIES = {
    "долгопрудный", "дубна", "обнинск", "королёв", "королев", "фрязино",
    "пущино", "жуковский", "реутов", "троицк", "черноголовка", "сколково", "иннополис",
}


def load_corpus() -> pd.DataFrame:
    with open(CORPUS, encoding="utf-8") as f:
        docs = json.load(f)
    df = pd.DataFrame(docs)
    df["text_len"] = df["text"].str.len()
    df["n_entities"] = df["entities"].apply(len)
    df["n_keywords"] = df["keywords"].apply(len)
    return df


def graph_metrics(nodes_path: Path, edges_path: Path) -> dict:
    if not nodes_path.exists():
        return {}
    G = load_graph_from_csv(nodes_path, edges_path)
    degrees = np.array([d for _, d in G.degree()])
    w_deg = np.array([d for _, d in G.degree(weight="weight")]) if G.number_of_edges() else degrees
    edges_df = pd.read_csv(edges_path)
    weights = edges_df["weight"].values if "weight" in edges_df.columns else np.ones(len(edges_df))
    pmi = edges_df["pmi"].dropna().values if "pmi" in edges_df.columns else []

    return {
        "nodes": int(G.number_of_nodes()),
        "edges": int(G.number_of_edges()),
        "avg_degree": float(degrees.mean()) if len(degrees) else 0,
        "max_degree": int(degrees.max()) if len(degrees) else 0,
        "components": int(nx.number_connected_components(G)),
        "density": float(nx.density(G)),
        "pmi_median": float(np.median(pmi)) if len(pmi) else None,
        "edge_weight_median": float(np.median(weights)) if len(weights) else None,
        "edges_per_node": round(G.number_of_edges() * 2 / max(G.number_of_nodes(), 1), 2),
    }


def run_sanity_checks(df: pd.DataFrame, graphs: dict[str, dict]) -> dict:
    checks = []

    def add(name: str, ok: bool, detail: str):
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    add("corpus_size", 200 <= len(df) <= 250, f"документов: {len(df)}")
    med_len = df["text_len"].median()
    add("median_text_len", 800 <= med_len <= 5000, f"медиана длины: {med_len:.0f}")

    title_hit = df["title"].str.contains("наукоград", case=False, na=False).sum()
    add("title_relevance", title_hit >= 50, f"«наукоград» в заголовке: {title_hit} док.")

    city_docs = df["naukograd_city"].notna().sum()
    add("city_metadata", city_docs >= 100, f"документов с городом: {city_docs}")

    ent_med = df["n_entities"].median()
    add("ner_present", ent_med >= 10, f"медиана NER/док: {ent_med:.0f}")

    # топ-NER: ожидаем города и научные орг
    ent_c = Counter()
    for ents in df["entities"]:
        for e in ents:
            ent_c[e["text"].lower()] += 1
    top_ent = [t for t, _ in ent_c.most_common(20)]
    domain_hits = sum(
        1 for t in top_ent
        if any(c in t for c in CITIES) or "наук" in t or "ран" in t or "нии" in t
    )
    add("top_ner_domain", domain_hits >= 5, f"доменных сущностей в топ-20: {domain_hits}")

    kw_c = Counter()
    for kws in df["keywords"]:
        for k in kws:
            kw_c[k.lower()] += 1
    top_kw_list = [t for t, _ in kw_c.most_common(50)]
    top_kw = " ".join(top_kw_list[:15])
    kw_domain = sum(
        1 for t in top_kw_list
        if "наукоград" in t or "науко" in t or "нии" in t or any(c in t for c in CITIES)
    )
    add(
        "keywords_domain",
        kw_domain >= 3,
        f"доменных kw в топ-50: {kw_domain}; примеры: {top_kw[:80]}…",
    )

    for gname, gm in graphs.items():
        if not gm:
            continue
        ratio = gm.get("edges_per_node", 0)
        add(f"{gname}_not_dense", ratio <= 6, f"{gname}: {ratio} рёбер/узел (цель ≤3–4)")
        add(f"{gname}_has_edges", gm["edges"] >= 100, f"{gname}: {gm['edges']} рёбер")

    cons = graphs.get("constructor", {})
    if cons:
        add("constructor_pmi", (cons.get("pmi_median") or 0) >= 4, f"медиана PMI: {cons.get('pmi_median')}")

    passed = sum(1 for c in checks if c["ok"])
    return {
        "passed": passed,
        "total": len(checks),
        "all_ok": passed == len(checks),
        "checks": checks,
    }


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    df = load_corpus()

    # --- корпус: фигуры ---
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].hist(df["text_len"], bins=40, color="steelblue", edgecolor="white")
    axes[0].set_title("Длина текста")
    axes[0].set_xlabel("символы")
    axes[1].hist(df["n_entities"], bins=30, color="coral", edgecolor="white")
    axes[1].set_title("NER на документ")
    axes[2].hist(df["n_keywords"], bins=12, color="seagreen", edgecolor="white")
    axes[2].set_title("YAKE keywords")
    fig.tight_layout()
    fig.savefig(FIG / "01_corpus_distributions.png", dpi=120)
    plt.close(fig)

    # метаданные
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    df["voice_type"].value_counts().plot(kind="bar", ax=axes[0], color="teal")
    axes[0].set_title("voice_type")
    axes[0].tick_params(axis="x", rotation=30)
    cities = df["naukograd_city"].dropna().value_counts().head(10)
    cities.plot(kind="barh", ax=axes[1], color="purple")
    axes[1].set_title("Топ городов")
    axes[1].invert_yaxis()
    fig.tight_layout()
    fig.savefig(FIG / "02_metadata.png", dpi=120)
    plt.close(fig)

    # relevance_reason
    if "relevance_reason" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        df["relevance_reason"].value_counts().head(8).plot(kind="barh", ax=ax)
        ax.set_title("Причина попадания в корпус")
        fig.tight_layout()
        fig.savefig(FIG / "03_relevance_reason.png", dpi=120)
        plt.close(fig)

    corpus_stats = {
        "n_docs": int(len(df)),
        "total_chars": int(df["text_len"].sum()),
        "median_text_len": float(df["text_len"].median()),
        "voice_type": df["voice_type"].value_counts().to_dict(),
        "n_with_city": int(df["naukograd_city"].notna().sum()),
        "top_cities": df["naukograd_city"].value_counts().head(10).to_dict(),
    }

    # --- графы ---
    graph_paths = {
        "legacy_cooccurrence": (GRAPH_LEGACY / "nodes.csv", GRAPH_LEGACY / "edges.csv"),
        "constructor": (GRAPH_CONSTRUCTOR / "nodes.csv", GRAPH_CONSTRUCTOR / "edges.csv"),
        "graphrag_baseline": (GRAPH_BASELINE / "nodes.csv", GRAPH_BASELINE / "edges.csv"),
    }
    graphs_metrics = {}
    for name, (npth, epth) in graph_paths.items():
        graphs_metrics[name] = graph_metrics(npth, epth)

    plot_comparison_bars(
        {k: v for k, v in graphs_metrics.items() if v},
        FIG / "04_graph_comparison.png",
    )

    for name, (npth, epth) in graph_paths.items():
        if not npth.exists():
            continue
        G = load_graph_from_csv(npth, epth)
        edges_df = pd.read_csv(epth)
        degrees = np.array([d for _, d in G.degree()])
        plot_degree_distribution(degrees, FIG / f"05_degree_{name}.png", f"Степени: {name}")
        if "weight" in edges_df.columns:
            plot_edge_threshold_curve(
                edges_df["weight"].values,
                G.number_of_nodes(),
                FIG / f"06_threshold_{name}.png",
            )
        plot_graph_network(G, FIG / f"07_network_{name}.png", title=f"Сеть: {name}", max_nodes=45)
        if G.number_of_nodes() >= 5:
            plot_communities(G, FIG / f"08_communities_{name}.png", title=f"Сообщества: {name}")

    # surprisal top (constructor)
    ce = GRAPH_CONSTRUCTOR / "edges.csv"
    if ce.exists():
        edf = pd.read_csv(ce)
        if "surprisal" in edf.columns:
            top = edf.nlargest(12, "surprisal")[["source", "target", "surprisal", "weight"]]
            top.to_csv(OUT / "top_surprisal_edges.csv", index=False)

    sanity = run_sanity_checks(df, graphs_metrics)
    with open(OUT / "corpus_stats.json", "w", encoding="utf-8") as f:
        json.dump(corpus_stats, f, ensure_ascii=False, indent=2)
    with open(OUT / "graph_metrics.json", "w", encoding="utf-8") as f:
        json.dump(graphs_metrics, f, ensure_ascii=False, indent=2)
    with open(OUT / "sanity_checks.json", "w", encoding="utf-8") as f:
        json.dump(sanity, f, ensure_ascii=False, indent=2)

    print(f"EDA сохранён в {OUT}")
    print(f"Проверки: {sanity['passed']}/{sanity['total']} OK")
    for c in sanity["checks"]:
        mark = "✓" if c["ok"] else "✗"
        print(f"  {mark} {c['check']}: {c['detail']}")
    if not sanity["all_ok"]:
        print("(часть проверок — мягкие эвристики; смотрите detail)")


if __name__ == "__main__":
    main()
