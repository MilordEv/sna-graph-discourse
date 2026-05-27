from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class VertexCandidate:
    label: str
    score: float
    method: str
    entity_type: str | None = None
    cluster_id: int | None = None
    sources: list[str] = field(default_factory=list)


def merge_vertex_candidates(
    candidates: list[VertexCandidate],
    max_vertices: int = 400,
    min_score: float = 0.0,
) -> dict[str, VertexCandidate]:
    merged: dict[str, VertexCandidate] = {}
    for c in candidates:
        key = c.label
        if key in merged:
            prev = merged[key]
            prev.score = max(prev.score, c.score)
            if c.method not in prev.sources:
                prev.sources.append(c.method)
        else:
            merged[key] = VertexCandidate(
                label=key,
                score=c.score,
                method=c.method,
                entity_type=c.entity_type,
                cluster_id=c.cluster_id,
                sources=[c.method],
            )
    filtered = [v for v in merged.values() if v.score >= min_score]
    filtered.sort(key=lambda x: x.score, reverse=True)
    return {v.label: v for v in filtered[:max_vertices]}
