from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

VertexMethod = Literal[
    "ner",
    "tfidf",
    "docfreq",
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
    backbone_k: int = 0  # k-next-neighbourhood (Drieger): топ-k рёбер на узел; 0 = выкл.
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

    @classmethod
    def from_preset(cls, name: str) -> "ConstructorConfig":
        """Готовые наборы параметров для конкретных доменов."""
        if name == "russkaya_istina":
            # Полные тексты статей (44 doc, медиана ~3.7 тыс. слов).
            # Риторика и эмоции — ядро дискурса об истине/лжи.
            return cls(
                vertex_methods=["docfreq", "yake"],
                edge_methods=["cooccurrence", "rhetorical", "emotional", "perplexity"],
                max_vertices=110,
                min_vertex_score=0.0,
                cooccurrence_level="sentence",
                min_edge_weight=2,
                min_pmi=-10.0,  # PMI оставляем как атрибут, но не фильтруем: иначе
                                # отсекаются связи между частыми концептами (истина↔пропаганда)
                backbone_k=10,  # оставляем у каждого концепта ~10 сильнейших связей
                stress_dropout_fraction=0.2,
                stress_n_seeds=5,
                stress_core_threshold=0.5,
                output_dir="data/graphs/russkaya_istina/constructor",
            )
        if name == "naukogrady":
            return cls(
                vertex_methods=["ner", "yake", "tfidf", "textrank", "paraphrase_cluster"],
                edge_methods=["cooccurrence", "anaphora", "rhetorical", "emotional", "perplexity"],
                output_dir="data/graphs/naukogrady/constructor",
            )
        raise ValueError(f"Неизвестный preset: {name!r}. Доступны: 'russkaya_istina', 'naukogrady'")
