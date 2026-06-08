#!/usr/bin/env python3
"""
Отдельный шаг СУДЕЙСТВА сильной моделью — БЕЗ повторной генерации ответов.

Зачем: генератор должен быть простым (меряем ретрив, а не модель), а судья —
сильным (надёжнее оценка). Этот скрипт берёт уже сгенерированные ответы
(output/answers_by_method.json), восстанавливает контексты детерминированно
(ретрив без LLM) и оценивает их СИЛЬНОЙ моделью-судьёй, затем пересчитывает метрики.

Запуск:
    JUDGE_MODEL="deepseek/deepseek-v4-flash:free" python run_judge.py
    # или платная для макс. надёжности:  JUDGE_MODEL="deepseek/deepseek-chat" python run_judge.py
"""
from __future__ import annotations
import json, os, time
from pathlib import Path

def load_dotenv(p=".env"):
    if Path(p).exists():
        for ln in Path(p).read_text(encoding="utf-8").splitlines():
            ln=ln.strip()
            if ln and not ln.startswith("#") and "=" in ln:
                k,_,v=ln.partition("="); os.environ.setdefault(k.strip(), v.strip())
load_dotenv()

import networkx as nx
from openai import OpenAI
from discourse_graph.eval_metrics_v2 import (compare_methods, plot_radar, plot_bars,
                                             llm_judge, AnswerMetrics, METRIC_LABELS)

BASE_URL=os.environ.get("LLM_BASE_URL","https://openrouter.ai/api/v1")
API_KEY =os.environ.get("LLM_API_KEY") or os.environ.get("OPENROUTER_KEY") or "not-needed"
JUDGE_MODEL=os.environ.get("JUDGE_MODEL") or "deepseek/deepseek-v4-flash:free"
SLEEP=float(os.environ.get("LLM_SLEEP","3.5"))

GOLD=json.load(open("data/eval/gold_answers.json", encoding="utf-8"))
Q=[g["q"] for g in GOLD]; REF=[g["gold"] for g in GOLD]; KP=[g["keypoints"] for g in GOLD]
answers=json.load(open("output/answers_by_method.json", encoding="utf-8"))
answers={m:answers[m] for m in ("discourse_graph","graphrag_baseline","long_context")}

# контексты: из файла (если есть от свежего прогона) либо восстановить детерминированно
ctx_path=Path("output/eval_contexts.json")
if ctx_path.exists():
    ctx=json.load(open(ctx_path, encoding="utf-8"))
else:
    from discourse_graph.retrieval import retrieve_lightrag, retrieve_community
    G=nx.read_graphml("data/graphs/russkaya_istina/constructor/discourse/discourse_graph.graphml")
    B=nx.read_graphml("data/graphs/russkaya_istina/constructor/graphrag_baseline/discourse_graph.graphml")
    DOCS=json.load(open("data/raw/russkaya_istina/documents.json", encoding="utf-8"))
    corpus="\n\n".join(f"[{d['title']}] {d['text']}" for d in DOCS)[:24000]
    ctx={"discourse_graph":[retrieve_lightrag(G,q,docs=DOCS) for q in Q],
         "graphrag_baseline":[retrieve_community(B,q) for q in Q],
         "long_context":[corpus for _ in Q]}

client=OpenAI(api_key=API_KEY, base_url=BASE_URL)
print(f"Судья: {JUDGE_MODEL}\n")
faith={}
for m in answers:
    fl=[]
    for i,a in enumerate(answers[m]):
        j=llm_judge(a, Q[i], ctx[m][i], client, JUDGE_MODEL); time.sleep(SLEEP)
        vals=[v for k,v in j.items() if k in ("faithfulness","relevance","coverage","grounding")]
        fl.append(sum(vals)/len(vals) if vals else 0.0)
        print(f"  {m:18} Q{i+1}: faith={fl[-1]:.2f}")
    faith[m]=fl
if all(all(v==0 for v in fl) for fl in faith.values()):
    print("[!] судья не дал валидного JSON — faithfulness не учитываем"); faith=None

comp=compare_methods(answers, REF, KP, faith)
fields=list(AnswerMetrics._fields)+["composite"]
print(f"\n{'method':20}"+"".join(f"{METRIC_LABELS.get(f,f)[:11]:>13}" for f in fields))
for m,d in comp.items(): print(f"{m:20}"+"".join(f"{d[f]:>13.3f}" for f in fields))
best=max(comp,key=lambda m:comp[m]["composite"]); print(f"\nЛучший по composite: {best} ({comp[best]['composite']:.3f})")
plot_radar(comp,"output/rag_eval_v2/01_radar_v2.png"); plot_bars(comp,"output/rag_eval_v2/02_bars_v2.png")
json.dump({"judge_model":JUDGE_MODEL,"faithfulness":faith,
           "composite":{m:comp[m]["composite"] for m in comp}},
          open("output/judge_result.json","w"), ensure_ascii=False, indent=1)
print("\nГотово → output/rag_eval_v2/01_radar_v2.png, output/judge_result.json")
