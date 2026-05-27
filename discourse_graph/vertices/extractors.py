from __future__ import annotations

import re
from collections import Counter, defaultdict

from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from discourse_graph.config import ConstructorConfig, VertexMethod
from discourse_graph.models import (
    DEFAULT_HF_MODEL,
    cosine_sim_matrix,
    encode_texts,
    extract_keywords_embed,
    rank_labels_vs_reference,
)
from discourse_graph.utils import normalize_label, tokenize
from discourse_graph.vertices.base import VertexCandidate, merge_vertex_candidates

# --- NER (из корпуса) ---


def vertices_ner(docs: list[dict]) -> list[VertexCandidate]:
    freq: Counter[str] = Counter()
    types: dict[str, str] = {}
    for doc in docs:
        for ent in doc.get("entities", []):
            label = normalize_label(ent["text"])
            if len(label) < 2:
                continue
            freq[label] += 1
            types.setdefault(label, ent.get("type", ""))
    if not freq:
        return []
    max_f = max(freq.values())
    return [
        VertexCandidate(label=k, score=v / max_f, method="ner", entity_type=types.get(k))
        for k, v in freq.most_common()
    ]


# --- TF-IDF ---


def vertices_tfidf(docs: list[dict], top_k: int = 300) -> list[VertexCandidate]:
    texts = [d.get("text", "") for d in docs]
    if not texts:
        return []
    vec = TfidfVectorizer(
        max_features=top_k * 3,
        ngram_range=(1, 3),
        min_df=2,
        token_pattern=r"(?u)\b[а-яёa-z][а-яёa-z\-]{1,}\b",
    )
    try:
        X = vec.fit_transform(texts)
    except ValueError:
        return []
    scores = X.sum(axis=0).A1
    terms = vec.get_feature_names_out()
    order = scores.argsort()[::-1][:top_k]
    max_s = float(scores[order[0]]) if len(order) else 1.0
    return [
        VertexCandidate(label=terms[i], score=float(scores[i]) / max_s, method="tfidf")
        for i in order
        if scores[i] > 0
    ]


# --- YAKE (из метаданных документов) ---


def vertices_yake(docs: list[dict]) -> list[VertexCandidate]:
    freq: Counter[str] = Counter()
    for doc in docs:
        for kw in doc.get("keywords", []):
            label = normalize_label(kw)
            if len(label) >= 2:
                freq[label] += 1
    if not freq:
        return []
    max_f = max(freq.values())
    return [
        VertexCandidate(label=k, score=v / max_f, method="yake")
        for k, v in freq.most_common()
    ]


# --- TextRank (через summa / fallback) ---


def vertices_textrank(docs: list[dict], top_k: int = 150) -> list[VertexCandidate]:
    try:
        from summa import keywords as summa_kw
    except ImportError:
        return _textrank_nx(docs, top_k)
    corpus = "\n".join(d.get("text", "")[:8000] for d in docs)
    if not corpus.strip():
        return []
    try:
        kws = summa_kw.keywords(corpus, language="russian", scores=True)
    except Exception:
        return _textrank_nx(docs, top_k)
    out: list[VertexCandidate] = []
    for item in kws.split("\n")[:top_k]:
        if not item.strip():
            continue
        if "\t" in item:
            phrase, score = item.rsplit("\t", 1)
            try:
                sc = float(score)
            except ValueError:
                sc = 1.0
        else:
            phrase, sc = item, 1.0
        out.append(
            VertexCandidate(label=normalize_label(phrase), score=sc, method="textrank")
        )
    if out:
        mx = max(c.score for c in out)
        for c in out:
            c.score /= mx
    return out


def _textrank_nx(docs: list[dict], top_k: int) -> list[VertexCandidate]:
    """Упрощённый TextRank: co-occurrence в окне + PageRank."""
    import networkx as nx

    window = 4
    G = nx.Graph()
    for doc in docs:
        toks = tokenize(doc.get("text", ""))
        for i in range(len(toks)):
            w = toks[i : i + window]
            if len(w) < 2:
                continue
            for a, b in combinations_pairs(w):
                if G.has_edge(a, b):
                    G[a][b]["weight"] += 1
                else:
                    G.add_edge(a, b, weight=1)
    if G.number_of_nodes() == 0:
        return []
    pr = nx.pagerank(G, weight="weight")
    ranked = sorted(pr.items(), key=lambda x: x[1], reverse=True)[:top_k]
    mx = ranked[0][1] if ranked else 1.0
    return [
        VertexCandidate(label=k, score=v / mx, method="textrank")
        for k, v in ranked
        if len(k) > 2
    ]


def combinations_pairs(words: list[str]) -> list[tuple[str, str]]:
    from itertools import combinations

    uniq = list(dict.fromkeys(words))
    return list(combinations(uniq, 2))


# --- KeyBERT (опционально) ---


def vertices_keybert(
    docs: list[dict],
    top_k: int = 100,
    model_name: str = DEFAULT_HF_MODEL,
) -> list[VertexCandidate]:
    """KeyBERT-логика на rubert-tiny2 (без отдельной зависимости keybert)."""
    sample = "\n\n".join(d.get("text", "")[:4000] for d in docs[:30])
    if not sample.strip():
        return []
    try:
        extracted = extract_keywords_embed(sample, model_name=model_name, top_k=top_k)
    except ImportError:
        return []
    if not extracted:
        return []
    mx = max(s for _, s in extracted)
    return [
        VertexCandidate(label=normalize_label(k), score=s / mx, method="keybert")
        for k, s in extracted
    ]


# --- BERTScore: ранжирование NER/ключевиков относительно корпуса ---


def vertices_bertscore(
    docs: list[dict],
    seed_labels: list[str],
    top_k: int = 200,
    model_name: str = DEFAULT_HF_MODEL,
) -> list[VertexCandidate]:
    """BERTScore-логика: косинус эмбеддинга метки к заголовкам корпуса."""
    if not seed_labels:
        return []
    ref = " ".join(d.get("title", "") for d in docs[:50])
    if not ref.strip():
        ref = docs[0].get("text", "")[:2000]
    labels = seed_labels[:top_k]
    try:
        f1_list = rank_labels_vs_reference(labels, ref, model_name=model_name)
    except ImportError:
        return []
    mx = max(f1_list) if f1_list else 1.0
    return [
        VertexCandidate(label=labels[i], score=f1_list[i] / mx, method="bertscore")
        for i in range(len(labels))
    ]


# --- Парафраз-кластеризация ---


def vertices_paraphrase_cluster(
    candidates: list[VertexCandidate],
    similarity: float = 0.85,
    model_name: str = DEFAULT_HF_MODEL,
) -> list[VertexCandidate]:
    if len(candidates) < 2:
        return candidates
    labels = [c.label for c in candidates]
    try:
        emb = encode_texts(labels, model_name=model_name)
        sim = cosine_sim_matrix(emb)
        dist = 1 - sim
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=1 - similarity,
            metric="precomputed",
            linkage="average",
        )
        cluster_ids = clustering.fit_predict(dist)
    except ImportError:
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
        X = vec.fit_transform(labels)
        sim = cosine_similarity(X)
        dist = 1 - sim
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=1 - similarity,
            metric="precomputed",
            linkage="average",
        )
        cluster_ids = clustering.fit_predict(dist)

    clusters: dict[int, list[VertexCandidate]] = defaultdict(list)
    for cid, c in zip(cluster_ids, candidates):
        clusters[int(cid)].append(c)

    merged: list[VertexCandidate] = []
    for cid, group in clusters.items():
        group.sort(key=lambda x: x.score, reverse=True)
        rep = group[0]
        rep.score = max(c.score for c in group)
        rep.cluster_id = cid
        rep.method = "paraphrase_cluster"
        merged.append(rep)
    return merged


# --- Реестр ---

_EXTRACTORS = {
    "ner": lambda docs, cfg: vertices_ner(docs),
    "tfidf": lambda docs, cfg: vertices_tfidf(docs, top_k=cfg.max_vertices),
    "yake": lambda docs, cfg: vertices_yake(docs),
    "textrank": lambda docs, cfg: vertices_textrank(docs),
    "keybert": lambda docs, cfg: vertices_keybert(docs),
}


def extract_vertices(
    docs: list[dict],
    methods: list[VertexMethod],
    config: ConstructorConfig,
) -> dict[str, VertexCandidate]:
    model_name = config.hf_model
    all_candidates: list[VertexCandidate] = []
    for m in methods:
        if m == "bertscore":
            continue
        if m == "paraphrase_cluster":
            continue
        fn = _EXTRACTORS.get(m)
        if fn:
            if m == "keybert":
                all_candidates.extend(vertices_keybert(docs, model_name=model_name))
            else:
                all_candidates.extend(fn(docs, config))

    if "bertscore" in methods:
        seeds = list({c.label for c in all_candidates})[:300]
        all_candidates.extend(
            vertices_bertscore(docs, seeds, model_name=model_name)
        )

    if "paraphrase_cluster" in methods:
        all_candidates = vertices_paraphrase_cluster(
            all_candidates,
            similarity=config.paraphrase_similarity,
            model_name=model_name,
        )

    return merge_vertex_candidates(
        all_candidates,
        max_vertices=config.max_vertices,
        min_score=config.min_vertex_score,
    )
