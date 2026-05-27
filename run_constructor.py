#!/usr/bin/env python3
"""CLI для конструктора дискурс-графа (фаза 1)."""

from __future__ import annotations

import argparse
from pathlib import Path

from discourse_graph.config import ConstructorConfig
from discourse_graph.pipeline import DiscourseGraphConstructor

DEFAULT_CORPUS = Path("data/raw/naukogrady/documents.json")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Конструктор дискурс-графа — фаза 1")
    p.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    p.add_argument("--output", type=Path, default=Path("data/graphs/naukogrady/constructor"))
    p.add_argument(
        "--vertices",
        nargs="+",
        default=["ner", "yake", "tfidf", "textrank", "paraphrase_cluster"],
    )
    p.add_argument(
        "--edges",
        nargs="+",
        default=["cooccurrence", "anaphora", "rhetorical", "emotional", "perplexity"],
    )
    p.add_argument("--no-stress", action="store_true")
    p.add_argument("--no-baseline", action="store_true")
    p.add_argument("--summarize", action="store_true", help="Саморайз вершин (HF rubert-tiny2)")
    p.add_argument(
        "--hf-model",
        default="cointegrated/rubert-tiny2",
        help="Модель Hugging Face для эмбеддингов и саморайза",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ConstructorConfig(
        vertex_methods=args.vertices,
        edge_methods=args.edges,
        output_dir=str(args.output),
        random_seed=args.seed,
        summarize_vertices=args.summarize,
        hf_model=args.hf_model,
    )
    constructor = DiscourseGraphConstructor(cfg)
    constructor.run(
        args.corpus,
        run_stress=not args.no_stress,
        run_baseline=not args.no_baseline,
    )


if __name__ == "__main__":
    main()
