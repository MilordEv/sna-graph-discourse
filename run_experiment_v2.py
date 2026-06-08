#!/usr/bin/env python3
"""
Воспроизводимый корректный пайплайн (v2): одна и та же ПРОСТАЯ модель генерирует
ответы тремя методами → оценка ОТНОСИТЕЛЬНО ЭТАЛОННЫХ ответов (gold).
Одна модель на все методы → сравниваем РЕТРИВ, а не «силу» модели.

Endpoint-agnostic (OpenAI-совместимый). Три способа задать модель:

A) OpenRouter, БЕСПЛАТНАЯ модель (ключ есть, карта не нужна; лимит 20 req/min, 200/день):
     OPENROUTER_KEY=sk-or-...            # уже в .env
     export LLM_MODEL="google/gemma-4-31b-it:free"
     python run_experiment_v2.py

B) Локальная модель через Ollama (скачать самому, без интернета на инференсе):
     ollama pull qwen2.5:7b              # или llama3.1:8b, gemma2:9b
     export LLM_BASE_URL="http://localhost:11434/v1" LLM_API_KEY="ollama" LLM_MODEL="qwen2.5:7b"
     python run_experiment_v2.py

C) LM Studio / vLLM / llama.cpp server — то же, что B, со своим LLM_BASE_URL.

Бесплатные модели OpenRouter (июнь 2026), пригодные как «простой вербализатор»:
  google/gemma-4-31b-it:free            (262K ctx, лучшая общая бесплатная, ru ок)
  meta-llama/llama-3.3-70b-instruct:free(131K, general purpose)
  qwen/qwen3-next-80b-a3b-instruct:free (262K, ru ок)
  openai/gpt-oss-20b:free               (131K, компактная)
  meta-llama/llama-3.2-3b-instruct:free (самая «простенькая», 3B)
"""
from __future__ import annotations
import json, os, time, sys
from pathlib import Path

def load_dotenv(p=".env"):
    if Path(p).exists():
        for ln in Path(p).read_text(encoding="utf-8").splitlines():
            ln=ln.strip()
            if ln and not ln.startswith("#") and "=" in ln:
                k,_,v=ln.partition("="); os.environ.setdefault(k.strip(), v.strip())
load_dotenv()

import networkx as nx
from discourse_graph.eval_metrics_v2 import (compare_methods, plot_radar, plot_bars,
                                             llm_judge, AnswerMetrics, METRIC_LABELS)

BASE_URL = os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1")
API_KEY  = os.environ.get("LLM_API_KEY") or os.environ.get("OPENROUTER_KEY") or "not-needed"
MODEL    = os.environ.get("LLM_MODEL") or os.environ.get("OPENROUTER_MODEL") or "google/gemma-4-31b-it:free"
# СУДЬЯ — отдельная, более СИЛЬНАЯ модель (надёжнее оценивает). По умолчанию
# сильная бесплатная reasoning-модель; для макс. надёжности — дешёвая платная
# (напр. JUDGE_MODEL=deepseek/deepseek-chat).
JUDGE_MODEL = os.environ.get("JUDGE_MODEL") or "deepseek/deepseek-v4-flash:free"
SLEEP    = float(os.environ.get("LLM_SLEEP", "3.5"))   # троттлинг для free-тарифа (20/мин)
LONGCTX  = int(os.environ.get("LONGCTX_CHARS", "24000"))
DO_JUDGE = os.environ.get("LLM_JUDGE", "1") == "1"

DOCS=json.load(open("data/raw/russkaya_istina/documents.json", encoding="utf-8"))
G   =nx.read_graphml("data/graphs/russkaya_istina/constructor/discourse/discourse_graph.graphml")
B   =nx.read_graphml("data/graphs/russkaya_istina/constructor/graphrag_baseline/discourse_graph.graphml")
GOLD=json.load(open("data/eval/gold_answers.json", encoding="utf-8"))
questions=[g["q"] for g in GOLD]; references=[g["gold"] for g in GOLD]; keypoints=[g["keypoints"] for g in GOLD]

answers, faith, ctx_by_method = {}, {}, {}
generated=False
try:
    from openai import OpenAI
    from discourse_graph.qa import DiscourseQA, flat_context_answer
    client=OpenAI(api_key=API_KEY, base_url=BASE_URL)
    print(f"LLM endpoint: {BASE_URL}\nМодель (одна на все методы): {MODEL}\n")
    # smoke-test соединения
    client.chat.completions.create(model=MODEL, messages=[{"role":"user","content":"ок"}], max_tokens=5, timeout=60)

    qa_our =DiscourseQA(G, client, model=MODEL, strategy="lightrag", docs=DOCS)
    qa_base=DiscourseQA(B, client, model=MODEL, strategy="community", docs=DOCS)
    answers={"discourse_graph":[], "graphrag_baseline":[], "long_context":[]}
    ctx_by_method={k:[] for k in answers}
    for i,q in enumerate(questions,1):
        print(f"  [{i}/{len(questions)}] {q[:60]}")
        co=qa_our._retrieve(q);   answers["discourse_graph"].append(qa_our.answer(q));   ctx_by_method["discourse_graph"].append(co); time.sleep(SLEEP)
        cb=qa_base._retrieve(q);  answers["graphrag_baseline"].append(qa_base.answer(q)); ctx_by_method["graphrag_baseline"].append(cb); time.sleep(SLEEP)
        answers["long_context"].append(flat_context_answer(DOCS, q, client, model=MODEL, max_chars=LONGCTX)); ctx_by_method["long_context"].append("(корпус в окне)"); time.sleep(SLEEP)
    json.dump(answers, open("output/answers_by_method.json","w"), ensure_ascii=False, indent=1)
    json.dump(ctx_by_method, open("output/eval_contexts.json","w"), ensure_ascii=False, indent=1)
    generated=True

    if DO_JUDGE:
        print(f"\nLLM-судья (СИЛЬНАЯ модель: {JUDGE_MODEL})…")
        for m in answers:
            fl=[]
            for i,a in enumerate(answers[m]):
                j=llm_judge(a, questions[i], ctx_by_method[m][i], client, JUDGE_MODEL); time.sleep(SLEEP)
                vals=[v for k,v in j.items() if k in ("faithfulness","relevance","coverage","grounding")]
                fl.append(sum(vals)/len(vals) if vals else 0.0)
            faith[m]=fl
        if all(all(v==0 for v in fl) for fl in faith.values()):
            faith=None  # судья не дал валидного JSON — не учитываем
except Exception as e:
    print(f"\n[!] LLM недоступен ({type(e).__name__}: {str(e)[:160]}).")
    print("    Беру output/answers_by_method.json как есть. Задайте LLM_MODEL/LLM_BASE_URL и перезапустите.")
    pre=json.load(open("output/answers_by_method.json", encoding="utf-8"))
    answers={m:pre[m] for m in ("discourse_graph","graphrag_baseline","long_context")}
    faith=None

comp=compare_methods(answers, references, keypoints, faith)
fields=list(AnswerMetrics._fields)+["composite"]
print(f"\n{'method':20}"+"".join(f"{METRIC_LABELS.get(f,f)[:11]:>13}" for f in fields))
for m,d in comp.items(): print(f"{m:20}"+"".join(f"{d[f]:>13.3f}" for f in fields))
best=max(comp,key=lambda m:comp[m]["composite"]); print(f"\nЛучший по composite: {best} ({comp[best]['composite']:.3f})")

# markdown-отчёт
Path("output/rag_eval_v2").mkdir(parents=True, exist_ok=True)
lines=[f"# Оценка v2 — модель: `{MODEL}` ({'сгенерировано' if generated else 'из файла'})","",
       "Одна модель-вербализатор на все методы; метрики — относительно эталонных ответов.","",
       "| Метод | "+" | ".join(METRIC_LABELS.get(f,f) for f in fields)+" |",
       "|"+"|".join(["---"]*(len(fields)+1))+"|"]
for m,d in comp.items(): lines.append(f"| **{m}** | "+" | ".join(f"{d[f]:.3f}" for f in fields)+" |")
lines+=["",f"**Лучший:** {best} ({comp[best]['composite']:.3f})","","## По вопросам (composite)","",
        "| # | discourse_graph | graphrag_baseline | long_context |","|---|---|---|---|"]
for i in range(len(questions)):
    lines.append(f"| Q{i+1} | "+" | ".join(f"{comp[m]['per_question'][i].composite():.3f}" for m in answers)+" |")
Path("output/rag_metrics_report_v2.md").write_text("\n".join(lines), encoding="utf-8")
plot_radar(comp,"output/rag_eval_v2/01_radar_v2.png"); plot_bars(comp,"output/rag_eval_v2/02_bars_v2.png")
print("\nГотово → output/rag_metrics_report_v2.md, output/rag_eval_v2/, output/answers_by_method.json")
