from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

VertexMethod = Literal[
    "ner",
    "tfidf",
    "bertscore",
    "yake",
    "textrank",
    "keybert",
    "paraphrase_cluster",
]

EdgeMethod = Literal[
    "cooccurrence",
    "anaphora",
    "rhetorical",
    "perplexity",
    "emotional",
]


@dataclass
class ConstructorConfig:
    """Параметры конструктора — переключение методов без правки кода."""

    vertex_methods: list[VertexMethod] = field(
        default_factory=lambda: ["ner", "yake", "tfidf"]
    )
    edge_methods: list[EdgeMethod] = field(
        default_factory=lambda: ["cooccurrence", "rhetorical", "emotional"]
    )

    # Отбор вершин
    max_vertices: int = 400
    min_vertex_score: float = 0.0
    paraphrase_similarity: float = 0.85

    # Построение рёбер
    cooccurrence_level: Literal["sentence", "paragraph", "window"] = "sentence"
    window_size: int = 3
    min_edge_weight: int = 2
    min_pmi: float = 5.0
    target_edge_ratio: float = 2.5  # ~2n–3n рёбер

    # Стресс-тест
    stress_dropout_fraction: float = 0.2
    stress_n_seeds: int = 5
    stress_core_threshold: float = 0.6

    # Саморайз (Hugging Face, по умолчанию cointegrated/rubert-tiny2)
    summarize_vertices: bool = False
    hf_model: str = "cointegrated/rubert-tiny2"

    # Вывод
    output_dir: str = "data/graphs/naukogrady/constructor"
    random_seed: int = 42
