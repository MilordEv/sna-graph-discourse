from __future__ import annotations

from functools import lru_cache

import numpy as np

# ~29M параметров, хорошо для русского; https://huggingface.co/cointegrated/rubert-tiny2
DEFAULT_HF_MODEL = "cointegrated/rubert-tiny2"


@lru_cache(maxsize=1)
def _load_encoder(model_name: str = DEFAULT_HF_MODEL):
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model, torch


def encode_texts(
    texts: list[str],
    model_name: str = DEFAULT_HF_MODEL,
    batch_size: int = 32,
) -> np.ndarray:
    """Mean-pooling эмбеддинги [CLS-слой последнего hidden state]."""
    if not texts:
        return np.zeros((0, 312), dtype=np.float32)

    tokenizer, model, torch = _load_encoder(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    vectors: list[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            out = model(**enc)
        # mean over tokens (mask padding)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        summed = (out.last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        emb = (summed / counts).cpu().numpy()
        vectors.append(emb)

    result = np.vstack(vectors).astype(np.float32)
    # L2 normalize для косинусной близости через dot product
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    return result / np.clip(norms, 1e-9, None)


def cosine_sim_matrix(a: np.ndarray, b: np.ndarray | None = None) -> np.ndarray:
    if b is None:
        b = a
    return a @ b.T


def rank_labels_vs_reference(
    labels: list[str],
    reference: str,
    model_name: str = DEFAULT_HF_MODEL,
) -> list[float]:
    """Замена BERTScore: косинус эмбеддинга метки к референсу корпуса."""
    if not labels:
        return []
    embs = encode_texts([reference] + labels, model_name=model_name)
    ref = embs[0:1]
    scores = (embs[1:] @ ref.T).ravel()
    return scores.tolist()


def extract_keywords_embed(
    document: str,
    model_name: str = DEFAULT_HF_MODEL,
    top_k: int = 100,
    diversity: float = 0.5,
) -> list[tuple[str, float]]:
    """Замена KeyBERT: кандидаты из YAKE-подобных n-грамм + MMR по эмбеддингам."""
    import re

    doc = document[:12000]
    words = re.findall(r"[а-яёa-z][а-яёa-z\-]{2,}", doc.lower())
    candidates: set[str] = set()
    for n in (1, 2, 3):
        for i in range(len(words) - n + 1):
            phrase = " ".join(words[i : i + n])
            if len(phrase) >= 4:
                candidates.add(phrase)
    if not candidates:
        return []
    cand_list = list(candidates)[:800]
    doc_emb = encode_texts([doc], model_name=model_name)[0]
    cand_embs = encode_texts(cand_list, model_name=model_name)

    sim_to_doc = cand_embs @ doc_emb
    order = np.argsort(-sim_to_doc)
    selected: list[int] = []
    selected_embs: list[np.ndarray] = []
    for idx in order:
        if len(selected) >= top_k:
            break
        if not selected:
            selected.append(int(idx))
            selected_embs.append(cand_embs[idx])
            continue
        max_redundancy = max(float(cand_embs[idx] @ e) for e in selected_embs)
        mmr = (1 - diversity) * float(sim_to_doc[idx]) - diversity * max_redundancy
        if mmr > 0.05 or len(selected) < 5:
            selected.append(int(idx))
            selected_embs.append(cand_embs[idx])

    return [(cand_list[i], float(sim_to_doc[i])) for i in selected]


def summarize_extractive(
    concept: str,
    snippets: list[str],
    model_name: str = DEFAULT_HF_MODEL,
    max_words: int = 15,
) -> str:
    """Краткое описание вершины: самое релевантное предложение из корпуса."""
    if not snippets:
        return ""
    texts = [s.strip() for s in snippets if s.strip()]
    if not texts:
        return ""
    embs = encode_texts([concept] + texts, model_name=model_name)
    best_i = int(np.argmax(embs[1:] @ embs[0]))
    sent = texts[best_i]
    words = sent.split()
    if len(words) > max_words:
        sent = " ".join(words[:max_words]) + "…"
    return sent
