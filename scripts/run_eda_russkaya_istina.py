#!/usr/bin/env python3
"""
EDA корпуса «Русская Истина» (теги: истина, правда, ложь).
Источник данных: Сайт_Русская_Истина_Статьи_по_тегам_истина_правда_ложь.xlsx
Сохраняет фигуры и JSON-статистику в output/eda_russkaya_istina/.
Запуск: python scripts/run_eda_russkaya_istina.py
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

XLSX = ROOT / "Сайт_Русская_Истина_Статьи_по_тегам_истина_правда_ложь.xlsx"
GRAPH_CONSTRUCTOR = ROOT / "data/graphs/russkaya_istina/constructor/discourse"
GRAPH_BASELINE = ROOT / "data/graphs/russkaya_istina/constructor/graphrag_baseline"
GRAPH_INVARIANT = ROOT / "data/graphs/russkaya_istina/constructor/invariant_core"
OUT = ROOT / "output/eda_russkaya_istina"
FIG = OUT / "figures"

# Маркеры домена «истина/правда/ложь»
DOMAIN_MARKERS = re.compile(
    r"истин|правд|ложь|лжи|обман|постправд|релятивизм|честност|заблужден",
    re.I,
)

DOMAIN_CONCEPTS = {
    "истина", "правда", "ложь", "обман", "постправда",
    "релятивизм", "честность", "заблуждение",
}


def load_xlsx() -> pd.DataFrame:
    """Загружает данные из xlsx и нормализует поля."""
    df = pd.read_excel(XLSX, sheet_name="Лист1", engine="openpyxl")
    # Нормализуем имена колонок
    df.columns = ["date", "author", "title", "url", "summary", "tags"]

    # Очищаем теги: убираем кавычки, разбиваем по запятой
    def parse_tags(val):
        if pd.isna(val) or val is True or val is False:
            return []
        s = str(val).strip().strip("'\"")
        return [t.strip() for t in s.split(",") if t.strip()]

    df["tags_list"] = df["tags"].apply(parse_tags)
    df["n_tags"] = df["tags_list"].apply(len)

    # Длина аннотации
    df["summary_len"] = df["summary"].astype(str).str.len()

    # Дата
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["year"] = df["date"].dt.year

    # Убираем строки без названия
    df = df[df["title"].notna() & (df["title"].astype(str).str.strip() != "nan")].copy()
    df = df.reset_index(drop=True)

    return df


def graph_metrics(nodes_path: Path, edges_path: Path) -> dict:
    if not nodes_path.exists():
        return {}
    G = load_graph_from_csv(nodes_path, edges_path)
    degrees = np.array([d for _, d in G.degree()])
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

    # Корпус
    add("corpus_size", 30 <= len(df) <= 100, f"статей: {len(df)}")

    med_len = df["summary_len"].median()
    add("median_summary_len", 50 <= med_len <= 500, f"медиана длины аннотации: {med_len:.0f}")

    # Доменная релевантность по заголовкам
    title_hits = df["title"].astype(str).str.contains(
        r"истин|правд|ложь|лжи|обман|честност", case=False, na=False
    ).sum()
    add("title_relevance", title_hits >= 10, f"доменных слов в заголовках: {title_hits} статей")

    # Авторы
    n_authors = df["author"].nunique()
    add("author_diversity", n_authors >= 5, f"уникальных авторов: {n_authors}")

    # Теги
    all_tags = [t for tl in df["tags_list"] for t in tl]
    tag_c = Counter(all_tags)
    domain_tags = sum(1 for t in tag_c if any(c in t.lower() for c in DOMAIN_CONCEPTS))
    add("tags_domain", domain_tags >= 2, f"доменных тегов: {domain_tags}")

    # Временной охват
    years = df["year"].dropna()
    year_span = int(years.max() - years.min()) if len(years) >= 2 else 0
    add("temporal_span", year_span >= 2, f"лет охвата: {year_span} ({int(years.min()) if len(years) else '?'}–{int(years.max()) if len(years) else '?'})")

    # Аннотации
    has_summary = df["summary"].notna().sum()
    add("summaries_present", has_summary >= len(df) * 0.8, f"статей с аннотацией: {has_summary}/{len(df)}")

    # Графы
    for gname, gm in graphs.items():
        if not gm:
            continue
        ratio = gm.get("edges_per_node", 0)
        add(f"{gname}_not_dense", ratio <= 8, f"{gname}: {ratio} рёбер/узел (цель ≤6)")
        add(f"{gname}_has_edges", gm["edges"] >= 10, f"{gname}: {gm['edges']} рёбер")

    cons = graphs.get("constructor", {})
    if cons:
        add("constructor_pmi", (cons.get("pmi_median") or 0) >= 1, f"медиана PMI: {cons.get('pmi_median')}")

    passed = sum(1 for c in checks if c["ok"])
    return {
        "passed": passed,
        "total": len(checks),
        "all_ok": passed == len(checks),
        "checks": checks,
    }


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    df = load_xlsx()

    print(f"Загружено статей: {len(df)}")

    # ── 1. Корпус: распределения ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Длина аннотации
    axes[0].hist(df["summary_len"].dropna(), bins=20, color="steelblue", edgecolor="white")
    axes[0].set_title("Длина аннотации (символы)")
    axes[0].set_xlabel("символы")

    # Год публикации
    year_counts = df["year"].value_counts().sort_index()
    axes[1].bar(year_counts.index.astype(int), year_counts.values, color="coral", edgecolor="white")
    axes[1].set_title("Публикации по годам")
    axes[1].set_xlabel("год")
    axes[1].tick_params(axis="x", rotation=45)

    # Число тегов на статью
    axes[2].hist(df["n_tags"], bins=range(0, df["n_tags"].max() + 2), color="seagreen", edgecolor="white", align="left")
    axes[2].set_title("Тегов на статью")
    axes[2].set_xlabel("кол-во тегов")

    fig.tight_layout()
    fig.savefig(FIG / "01_corpus_distributions.png", dpi=120)
    plt.close(fig)

    # ── 2. Авторы и теги ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    top_authors = df["author"].value_counts().head(12)
    top_authors.plot(kind="barh", ax=axes[0], color="teal")
    axes[0].set_title("Топ авторов")
    axes[0].invert_yaxis()
    axes[0].set_xlabel("статей")

    all_tags = [t for tl in df["tags_list"] for t in tl]
    tag_c = Counter(all_tags)
    if tag_c:
        top_tags = pd.Series(dict(tag_c.most_common(12)))
        top_tags.plot(kind="barh", ax=axes[1], color="purple")
        axes[1].set_title("Топ тегов")
        axes[1].invert_yaxis()
        axes[1].set_xlabel("упоминаний")
    else:
        axes[1].text(0.5, 0.5, "Нет тегов", ha="center", va="center")
        axes[1].set_title("Теги")

    fig.tight_layout()
    fig.savefig(FIG / "02_authors_tags.png", dpi=120)
    plt.close(fig)

    # ── 3. Доменная релевантность по аннотациям ───────────────────────────────
    domain_words = ["истина", "правда", "ложь", "обман", "постправда",
                    "релятивизм", "честность", "заблуждение"]
    word_counts = {}
    for w in domain_words:
        cnt = df["summary"].astype(str).str.lower().str.count(w).sum()
        if cnt > 0:
            word_counts[w] = int(cnt)

    if word_counts:
        fig, ax = plt.subplots(figsize=(9, 4))
        wc_series = pd.Series(word_counts).sort_values(ascending=True)
        wc_series.plot(kind="barh", ax=ax, color="darkorange")
        ax.set_title("Частота доменных слов в аннотациях")
        ax.set_xlabel("упоминаний")
        fig.tight_layout()
        fig.savefig(FIG / "03_domain_words.png", dpi=120)
        plt.close(fig)

    # ── 4. Временная динамика ─────────────────────────────────────────────────
    if df["year"].notna().sum() >= 3:
        fig, ax = plt.subplots(figsize=(10, 4))
        year_counts.plot(kind="bar", ax=ax, color="steelblue", edgecolor="white")
        ax.set_title("Динамика публикаций по годам")
        ax.set_xlabel("год")
        ax.set_ylabel("статей")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(FIG / "04_temporal_dynamics.png", dpi=120)
        plt.close(fig)

    # ── corpus_stats JSON ─────────────────────────────────────────────────────
    corpus_stats = {
        "n_articles": int(len(df)),
        "n_authors": int(df["author"].nunique()),
        "median_summary_len": float(df["summary_len"].median()),
        "year_min": int(df["year"].min()) if df["year"].notna().any() else None,
        "year_max": int(df["year"].max()) if df["year"].notna().any() else None,
        "top_authors": df["author"].value_counts().head(10).to_dict(),
        "top_tags": dict(Counter(all_tags).most_common(10)),
        "domain_word_counts": word_counts,
    }

    # ── 5. Графы ──────────────────────────────────────────────────────────────
    graph_paths = {
        "constructor": (GRAPH_CONSTRUCTOR / "nodes.csv", GRAPH_CONSTRUCTOR / "edges.csv"),
        "graphrag_baseline": (GRAPH_BASELINE / "nodes.csv", GRAPH_BASELINE / "edges.csv"),
        "invariant_core": (GRAPH_INVARIANT / "nodes.csv", GRAPH_INVARIANT / "edges.csv"),
    }
    graphs_metrics = {}
    for name, (npth, epth) in graph_paths.items():
        graphs_metrics[name] = graph_metrics(npth, epth)

    available = {k: v for k, v in graphs_metrics.items() if v}
    if available:
        plot_comparison_bars(available, FIG / "05_graph_comparison.png")

    for name, (npth, epth) in graph_paths.items():
        if not npth.exists():
            continue
        G = load_graph_from_csv(npth, epth)
        edges_df = pd.read_csv(epth)
        degrees = np.array([d for _, d in G.degree()])
        plot_degree_distribution(degrees, FIG / f"06_degree_{name}.png", f"Степени: {name}")
        if "weight" in edges_df.columns:
            plot_edge_threshold_curve(
                edges_df["weight"].values,
                G.number_of_nodes(),
                FIG / f"07_threshold_{name}.png",
            )
        plot_graph_network(G, FIG / f"08_network_{name}.png", title=f"Сеть: {name}", max_nodes=45)
        if G.number_of_nodes() >= 5:
            plot_communities(G, FIG / f"09_communities_{name}.png", title=f"Сообщества: {name}")

    # ── 6. Surprisal top (constructor) ───────────────────────────────────────
    ce = GRAPH_CONSTRUCTOR / "edges.csv"
    if ce.exists():
        edf = pd.read_csv(ce)
        if "surprisal" in edf.columns:
            top = edf.nlargest(12, "surprisal")[["source", "target", "surprisal", "weight"]]
            top.to_csv(OUT / "top_surprisal_edges.csv", index=False)

    # ── 7. Sanity checks ─────────────────────────────────────────────────────
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
