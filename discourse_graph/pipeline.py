from __future__ import annotations

import json
from pathlib import Path

import networkx as nx

from discourse_graph.baseline_graphrag import build_vanilla_graphrag_graph, export_graphrag_communities
from discourse_graph.config import ConstructorConfig
from discourse_graph.edges.builders import build_edges
from discourse_graph.stress_test import run_stress_test
from discourse_graph.summarize import summarize_vertices_hf
from discourse_graph.utils import filter_graph, load_documents, save_graph
from discourse_graph.vertices.extractors import extract_vertices


class DiscourseGraphConstructor:
    """Конструктор: комбинирует методы отбора вершин и построения рёбер."""

    def __init__(self, config: ConstructorConfig | None = None):
        self.config = config or ConstructorConfig()

    def build(
        self,
        docs: list[dict],
        *,
        run_summarize: bool | None = None,
    ) -> nx.Graph:
        cfg = self.config
        vertices = extract_vertices(docs, cfg.vertex_methods, cfg)
        G = build_edges(docs, vertices, cfg.edge_methods, cfg)
        for n, v in vertices.items():
            if n not in G:
                G.add_node(n)
            G.nodes[n]["score"] = v.score
            G.nodes[n]["methods"] = ",".join(v.sources)
            if v.entity_type:
                G.nodes[n]["entity_type"] = v.entity_type
            if v.cluster_id is not None:
                G.nodes[n]["cluster_id"] = v.cluster_id

        G = filter_graph(
            G,
            top_nodes=cfg.max_vertices,
            min_weight=cfg.min_edge_weight,
            min_pmi=cfg.min_pmi,
            backbone_k=cfg.backbone_k,
        )

        do_sum = run_summarize if run_summarize is not None else cfg.summarize_vertices
        if do_sum:
            summarize_vertices_hf(G, docs, model_name=cfg.hf_model)

        return G

    def run(
        self,
        corpus_path: str | Path,
        *,
        run_stress: bool = True,
        run_baseline: bool = True,
    ) -> dict:
        docs = load_documents(corpus_path)
        out = Path(self.config.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        G = self.build(docs)
        save_graph(G, out / "discourse")
        meta = {
            "n_docs": len(docs),
            "vertex_methods": self.config.vertex_methods,
            "edge_methods": self.config.edge_methods,
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
        }

        with open(out / "run_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        result = {"graph": G, "meta": meta}

        if run_stress:
            stress = run_stress_test(docs, self.config)
            save_graph(stress["graph_core"], out / "invariant_core")
            stress_report = {
                k: v
                for k, v in stress.items()
                if k not in ("graphs", "graph_core")
            }
            stress_report["core_nodes"] = stress["core_nodes"]
            stress_report["core_edges"] = stress["core_edges"]
            with open(out / "stress_test.json", "w", encoding="utf-8") as f:
                json.dump(stress_report, f, ensure_ascii=False, indent=2)
            result["stress"] = stress

        if run_baseline:
            G_base = build_vanilla_graphrag_graph(docs)
            save_graph(G_base, out / "graphrag_baseline")
            export_graphrag_communities(
                G_base, out / "graphrag_baseline" / "communities.json"
            )
            result["baseline"] = G_base

        print(
            f"Готово: {meta['nodes']} узлов, {meta['edges']} рёбер -> {out}"
        )
        return result
