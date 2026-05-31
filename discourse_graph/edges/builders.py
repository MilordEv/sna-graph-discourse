from __future__ import annotations

import math
import re
from collections import Counter
from itertools import combinations

import networkx as nx

from discourse_graph.config import ConstructorConfig, EdgeMethod
from discourse_graph.utils import (
    add_cooccurrence_edges,
    merge_edge_attrs,
    normalize_label,
    split_paragraphs,
    split_sentences,
    tokenize,
)
from discourse_graph.vertices.base import VertexCandidate

# Риторические маркеры (Mann / RST — контраст, уточнение)
CONTRAST_PATTERNS = [
    re.compile(r"\bне\s+.{1,40}\s*,?\s*а\s+", re.I),
    re.compile(r"\bв\s+отличие\s+от\b", re.I),
    re.compile(r"\bв\s+противоположность\b", re.I),
    re.compile(r"\bоднако\b", re.I),
    re.compile(r"\bзато\b", re.I),
    re.compile(r"\b—\s*это\s+вам\s+не\b", re.I),
    # Паттерны для дискурса «истина / ложь»
    re.compile(r"\bвместо\s+(истины|правды|факта)\b", re.I),
    re.compile(r"\bподмен(ить|яет|яют|яя)\s+понятие\b", re.I),
    re.compile(r"\bне\s+(правда|истина|факт)\s*,?\s*а\s+(ложь|обман|фальш|иллюзи)", re.I),
    re.compile(r"\b(ложь|обман|фальш)\s+вместо\s+(правды|истины)\b", re.I),
    re.compile(r"\bпротивопоставл(ение|яет|яют)\b", re.I),
    re.compile(r"\bв\s+то\s+время\s+как\b", re.I),
]
ANAPHORA_PRONOUNS = re.compile(
    r"\b(он|она|оно|они|его|её|их|этот|эта|это|эти|такой|такая|такое|данный|данная)\b",
    re.I,
)
EMOTION_LEXICON = {
    # Общие (наукограды и др.)
    "важн",
    "критич",
    "успех",
    "прорыв",
    "кризис",
    "проблем",
    "поддерж",
    "автоном",
    "риск",
    "лидер",
    "отстав",
    "провал",
    "достижен",
    "перспектив",
    "угроз",
    # Домен «истина / правда / ложь»
    "обман",
    "манипул",
    "фальш",
    "лжив",
    "достоверн",
    "честн",
    "скрыт",
    "иллюзи",
    "мифологи",
    "заблужден",
    "релятивизм",
    "постправд",
    "искажен",
    "дезинформ",
    "пропаганд",
    "абсолютн",
    "справедлив",
}


def _nodes_in_text(text: str, vertex_set: set[str]) -> list[str]:
    tl = text.lower()
    found = [v for v in vertex_set if v in tl]
    return list(dict.fromkeys(found))


def _units(doc: dict, level: str, window_size: int) -> list[str]:
    text = doc.get("text", "")
    if level == "paragraph":
        return split_paragraphs(text)
    if level == "window":
        sents = split_sentences(text)
        return [
            " ".join(sents[i : i + window_size])
            for i in range(max(1, len(sents) - window_size + 1))
        ]
    return split_sentences(text)


def build_cooccurrence(
    G: nx.Graph,
    docs: list[dict],
    vertices: dict[str, VertexCandidate],
    config: ConstructorConfig,
) -> None:
    vset = set(vertices.keys())
    for doc in docs:
        for unit in _units(doc, config.cooccurrence_level, config.window_size):
            nodes = _nodes_in_text(unit, vset)
            add_cooccurrence_edges(G, nodes, weight=1, method="cooccurrence")


def build_anaphora(
    G: nx.Graph,
    docs: list[dict],
    vertices: dict[str, VertexCandidate],
) -> None:
    """Связь местоимения / анафоры с недавно упомянутой сущностью в окне 2 предложений."""
    vset = set(vertices.keys())
    for doc in docs:
        sents = split_sentences(doc.get("text", ""))
        recent: list[str] = []
        for sent in sents:
            ents = _nodes_in_text(sent, vset)
            if ANAPHORA_PRONOUNS.search(sent) and recent and ents:
                for r in recent[-3:]:
                    for e in ents:
                        merge_edge_attrs(G, r, e, weight=1, methods=["anaphora"])
            if ents:
                recent = (recent + ents)[-5:]


def build_rhetorical(
    G: nx.Graph,
    docs: list[dict],
    vertices: dict[str, VertexCandidate],
) -> None:
    vset = set(vertices.keys())
    for doc in docs:
        for sent in split_sentences(doc.get("text", "")):
            if not any(p.search(sent) for p in CONTRAST_PATTERNS):
                continue
            nodes = _nodes_in_text(sent, vset)
            if len(nodes) >= 2:
                for a, b in combinations(nodes, 2):
                    merge_edge_attrs(
                        G,
                        a,
                        b,
                        weight=2,
                        methods=["rhetorical"],
                        relation="contrast",
                    )


def build_emotional(
    G: nx.Graph,
    docs: list[dict],
    vertices: dict[str, VertexCandidate],
) -> None:
    vset = set(vertices.keys())
    for doc in docs:
        for sent in split_sentences(doc.get("text", "")):
            sl = sent.lower()
            if not any(m in sl for m in EMOTION_LEXICON):
                continue
            nodes = _nodes_in_text(sent, vset)
            if len(nodes) >= 2:
                for a, b in combinations(nodes, 2):
                    merge_edge_attrs(
                        G,
                        a,
                        b,
                        weight=1,
                        methods=["emotional"],
                        emotional=True,
                    )


def build_perplexity_scores(
    G: nx.Graph,
    docs: list[dict],
) -> None:
    """Surprisal по корпусу: -log P(u,v) — для интерпретации «жареных» связей."""
    uni: Counter[str] = Counter()
    bi: Counter[tuple[str, str]] = Counter()
    for doc in docs:
        toks = tokenize(doc.get("text", ""))
        uni.update(toks)
        for i in range(len(toks) - 1):
            bi[(toks[i], toks[i + 1])] += 1
    total_uni = sum(uni.values()) or 1
    total_bi = sum(bi.values()) or 1

    for u, v, data in list(G.edges(data=True)):
        tu = u.split() if " " in u else [u]
        tv = v.split() if " " in v else [v]
        surprisal = 0.0
        n = 0
        for a in tu:
            for b in tv:
                p_uni = (uni.get(a, 0) + 1) / (total_uni + len(uni))
                p_bi = (bi.get((a, b), 0) + 1) / (total_bi + len(bi))
                surprisal += -math.log2(max(p_bi, p_uni * 0.01, 1e-9))
                n += 1
        data["surprisal"] = round(surprisal / max(n, 1), 4)
        data.setdefault("methods", [])
        if "perplexity" not in data["methods"]:
            data["methods"].append("perplexity")


def build_edges(
    docs: list[dict],
    vertices: dict[str, VertexCandidate],
    methods: list[EdgeMethod],
    config: ConstructorConfig,
) -> nx.Graph:
    G = nx.Graph()
    for m in methods:
        if m == "cooccurrence":
            build_cooccurrence(G, docs, vertices, config)
        elif m == "anaphora":
            build_anaphora(G, docs, vertices)
        elif m == "rhetorical":
            build_rhetorical(G, docs, vertices)
        elif m == "emotional":
            build_emotional(G, docs, vertices)

    if "perplexity" in methods:
        build_perplexity_scores(G, docs)

    return G
