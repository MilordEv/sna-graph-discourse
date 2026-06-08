"""
Исправленная оценка RAG-методов (v2).

Главное отличие от v1: метрики считаются ОТНОСИТЕЛЬНО ЭТАЛОННОГО ОТВЕТА (gold),
а не относительно текста вопроса. ROUGE-против-вопроса в v1 награждал ответы,
повторяющие формулировку вопроса, и штрафовал содержательные ответы.

Метрики (все в [0,1], выше = лучше):
  - rouge1 / rouge2 / rougeL : лексическое перекрытие с эталоном
  - semantic                 : TF-IDF косинус с эталоном (char+word n-grams)
  - keypoint_coverage        : доля ключевых тезисов эталона, покрытых ответом
  - faithfulness             : оценка LLM-судьи (или подаётся извне)
  - length_score             : мягкая нормировка длины

Композит — взвешенное среднее с акцентом на содержательность.
LLM-судья (llm_judge) использует OpenAI-совместимый клиент (DeepSeek и др.).
"""
from __future__ import annotations

import json
import re
from collections import Counter
from typing import NamedTuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Токенизация ──────────────────────────────────────────────────

def _tok(text: str) -> list[str]:
    return re.findall(r"[а-яёa-z]{2,}", (text or "").lower())

def _ngrams(toks: list[str], n: int) -> Counter:
    return Counter(tuple(toks[i:i+n]) for i in range(len(toks)-n+1))

# ── ROUGE vs ЭТАЛОН ──────────────────────────────────────────────

def rouge_n(hyp: str, ref: str, n: int = 1) -> float:
    H, R = _ngrams(_tok(hyp), n), _ngrams(_tok(ref), n)
    if not R:
        return 0.0
    overlap = sum(min(H[k], R[k]) for k in R)
    return overlap / sum(R.values())

def rouge_l(hyp: str, ref: str) -> float:
    h, r = _tok(hyp), _tok(ref)
    if not h or not r:
        return 0.0
    m, nn = len(r), len(h)
    dp = [[0]*(nn+1) for _ in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, nn+1):
            dp[i][j] = dp[i-1][j-1]+1 if r[i-1] == h[j-1] else max(dp[i-1][j], dp[i][j-1])
    lcs = dp[m][nn]
    p, rec = lcs/nn, lcs/m
    return 0.0 if p+rec == 0 else 2*p*rec/(p+rec)

# ── Семантическое сходство (TF-IDF косинус) ──────────────────────

def semantic_sim(hyp: str, ref: str) -> float:
    if not _tok(hyp) or not _tok(ref):
        return 0.0
    try:
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
        X = vec.fit_transform([hyp, ref])
        return float(cosine_similarity(X[0], X[1])[0, 0])
    except ValueError:
        return 0.0

# ── Покрытие ключевых тезисов эталона ────────────────────────────

def keypoint_coverage(answer: str, keypoints: list[str]) -> float:
    """Доля ключевых концептов (gold key-points), упомянутых в ответе.
    Концепт засчитан, если любой его токен-стем (>=4 симв.) найден в ответе."""
    if not keypoints:
        return 0.0
    al = (answer or "").lower()
    hit = 0
    for kp in keypoints:
        stems = [t[:6] for t in _tok(kp) if len(t) >= 4] or [kp.lower()[:6]]
        if any(s in al for s in stems):
            hit += 1
    return hit / len(keypoints)

# ── Длина ────────────────────────────────────────────────────────

def length_score(answer: str, target: int = 220) -> float:
    n = len(_tok(answer))
    if n == 0:
        return 0.0
    return n/target if n <= target else max(0.4, target/n)

# ── Агрегация ────────────────────────────────────────────────────

class AnswerMetrics(NamedTuple):
    rouge1: float
    rouge2: float
    rougeL: float
    semantic: float
    keypoint_coverage: float
    faithfulness: float
    length_score: float

    def composite(self) -> float:
        w = {"rouge1":0.10,"rouge2":0.10,"rougeL":0.15,"semantic":0.20,
             "keypoint_coverage":0.25,"faithfulness":0.15,"length_score":0.05}
        return float(sum(getattr(self, k)*v for k, v in w.items()))

def compute_metrics(answer: str, reference: str, keypoints: list[str],
                    faithfulness: float = 0.0) -> AnswerMetrics:
    return AnswerMetrics(
        rouge1=rouge_n(answer, reference, 1),
        rouge2=rouge_n(answer, reference, 2),
        rougeL=rouge_l(answer, reference),
        semantic=semantic_sim(answer, reference),
        keypoint_coverage=keypoint_coverage(answer, keypoints),
        faithfulness=faithfulness,
        length_score=length_score(answer),
    )

def compare_methods(answers_by_method: dict[str, list[str]], references: list[str],
                    keypoints: list[list[str]],
                    faithfulness_by_method: dict[str, list[float]] | None = None) -> dict:
    results: dict[str, dict] = {}
    fields = AnswerMetrics._fields
    for method, answers in answers_by_method.items():
        per_q = []
        for i, a in enumerate(answers):
            f = 0.0
            if faithfulness_by_method and method in faithfulness_by_method:
                f = faithfulness_by_method[method][i]
            per_q.append(compute_metrics(a, references[i], keypoints[i], faithfulness=f))
        means = {f_: float(np.mean([getattr(m, f_) for m in per_q])) for f_ in fields}
        means["composite"] = float(np.mean([m.composite() for m in per_q]))
        results[method] = {**means, "per_question": per_q}
    return results

# ── LLM-судья (для локального запуска с DeepSeek) ─────────────────

JUDGE_SYSTEM = (
    "Ты — строгий эксперт-оценщик ответов на вопросы о текстовом корпусе. "
    "Оцени ОДИН ответ по 4 критериям, каждый по шкале 1–5:\n"
    "  faithfulness — соответствие предоставленному контексту, без выдумок;\n"
    "  relevance    — релевантность вопросу;\n"
    "  coverage     — полнота охвата сути;\n"
    "  grounding    — опора на конкретику (концепты, авторы, связи).\n"
    "Верни СТРОГО JSON: {\"faithfulness\":n,\"relevance\":n,\"coverage\":n,\"grounding\":n}."
)

def _extract_json(txt: str) -> dict | None:
    """Достаёт JSON из ответа судьи, устойчиво к reasoning-моделям."""
    if not txt:
        return None
    # убрать <think>…</think> и markdown-ограждение
    txt = re.sub(r"<think>.*?</think>", " ", txt, flags=re.S | re.I)
    txt = txt.replace("```json", " ").replace("```", " ")
    # перебрать все { … } и взять последний валидный
    for m in reversed(list(re.finditer(r"\{[^{}]*\}", txt, re.S))):
        try:
            return json.loads(m.group(0))
        except Exception:
            continue
    return None


def llm_judge(answer: str, question: str, context: str, client, model: str) -> dict:
    """Оценка ответа LLM-судьёй → нормализованные [0,1] баллы. Устойчиво к reasoning-
    моделям (читает reasoning-поле, чистит <think>, ищет JSON надёжно)."""
    import os
    user = (f"Вопрос: {question}\n\nКонтекст, доступный методу:\n{context[:6000]}\n\n"
            f"Ответ для оценки:\n{answer}\n\nВерни ТОЛЬКО JSON, без рассуждений и текста вокруг.")
    try:
        r = client.chat.completions.create(
            model=model, temperature=0, max_tokens=400,
            messages=[{"role":"system","content":JUDGE_SYSTEM},
                      {"role":"user","content":user}])
        msg = r.choices[0].message
        txt = msg.content or ""
        if not txt.strip():  # reasoning-модели иногда кладут текст в reasoning-поле
            txt = getattr(msg, "reasoning", "") or getattr(msg, "reasoning_content", "") or ""
        if os.environ.get("JUDGE_DEBUG"):
            print(f"    [raw judge]: {txt[:160]!r}")
        d = _extract_json(txt)
        if not d:
            return {"error": "no-json", "raw": txt[:160]}
        norm = {}
        for k in ("faithfulness", "relevance", "coverage", "grounding"):
            if k in d:
                try:
                    norm[k] = (max(1.0, min(5.0, float(d[k]))) - 1) / 4
                except Exception:
                    pass
        norm["faithfulness_raw"] = d.get("faithfulness")
        return norm if norm else {"error": "empty", "raw": txt[:160]}
    except Exception as e:
        return {"error": str(e)[:200]}

# ── Визуализация (radar + bar) ───────────────────────────────────

METRIC_LABELS = {
    "rouge1":"ROUGE-1","rouge2":"ROUGE-2","rougeL":"ROUGE-L","semantic":"Семант. сходство",
    "keypoint_coverage":"Покрытие тезисов","faithfulness":"Достоверность (судья)",
    "length_score":"Длина","composite":"Итоговый балл",
}
METHOD_COLORS = {"discourse_graph":"#2196F3","graphrag_baseline":"#FF9800","long_context":"#9C27B0"}

def _import_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt

def plot_radar(comparison: dict, out_path: str, title="Радар: профиль методов (vs эталон)"):
    plt = _import_plt()
    metrics = list(AnswerMetrics._fields)
    labels = [METRIC_LABELS[m] for m in metrics]
    ang = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist(); ang += ang[:1]
    fig, ax = plt.subplots(figsize=(8,8), subplot_kw={"polar":True})
    for i, (method, data) in enumerate(comparison.items()):
        vals = [data.get(m,0.0) for m in metrics]; vals += vals[:1]
        c = METHOD_COLORS.get(method, None)
        ax.plot(ang, vals, "o-", lw=2, color=c, label=method)
        ax.fill(ang, vals, alpha=0.12, color=c)
    ax.set_xticks(ang[:-1]); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0,1); ax.set_title(title, fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3,1.1))
    from pathlib import Path; Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)

def plot_bars(comparison: dict, out_path: str, title="Сравнение метрик (vs эталон)"):
    plt = _import_plt()
    metrics = list(AnswerMetrics._fields)+["composite"]
    methods = list(comparison.keys())
    x = np.arange(len(metrics)); w = 0.8/max(len(methods),1)
    fig, ax = plt.subplots(figsize=(14,7))
    for i, method in enumerate(methods):
        vals=[comparison[method].get(m,0.0) for m in metrics]
        ax.bar(x+(i-len(methods)/2+0.5)*w, vals, w*0.9, label=method,
               color=METHOD_COLORS.get(method), alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels([METRIC_LABELS.get(m,m) for m in metrics], rotation=30, ha="right")
    ax.set_ylim(0,1.0); ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    from pathlib import Path; Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
