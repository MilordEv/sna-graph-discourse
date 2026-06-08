#!/usr/bin/env python3
"""
Оценка КАЧЕСТВА РЕТРИВА без LLM (модель-независимая, без bias вербализатора).

Для каждого вопроса берём КОНТЕКСТ, который реально возвращает метод, и при
ОДИНАКОВОМ бюджете символов меряем, насколько он покрывает эталонный ответ:
  - keypoint recall — доля ключевых тезисов эталона, найденных в контексте;
  - semantic        — TF-IDF косинус контекста с эталоном.
Это убирает зависимость от «силы» модели: сравниваем сам поиск, а не генерацию.
"""
import json, warnings; warnings.filterwarnings("ignore")
import networkx as nx
import numpy as np
from discourse_graph.retrieval import retrieve_lightrag, retrieve_community
from discourse_graph.eval_metrics_v2 import keypoint_coverage, semantic_sim

G=nx.read_graphml("data/graphs/russkaya_istina/constructor/discourse/discourse_graph.graphml")
B=nx.read_graphml("data/graphs/russkaya_istina/constructor/graphrag_baseline/discourse_graph.graphml")
DOCS=json.load(open("data/raw/russkaya_istina/documents.json", encoding="utf-8"))
GOLD=json.load(open("data/eval/gold_answers.json", encoding="utf-8"))
Q=[g["q"] for g in GOLD]; REF=[g["gold"] for g in GOLD]; KP=[g["keypoints"] for g in GOLD]
CORPUS="\n\n".join(f"[{d['title']}] {d['text']}" for d in DOCS)

def ctx(method, q):
    if method=="discourse_graph": return retrieve_lightrag(G, q, docs=DOCS)
    if method=="graphrag_baseline": return retrieve_community(B, q)
    return CORPUS  # long_context: весь корпус, режется бюджетом
methods=["discourse_graph","graphrag_baseline","long_context"]
budgets=[800,1500,3000,6000,12000]

# средний фактический размер контекста метода (до бюджета)
raw_sizes={m:int(np.mean([len(ctx(m,q)) for q in Q])) for m in methods}

print("Средний размер контекста метода (символов, до ограничения бюджетом):")
for m in methods: print(f"  {m:18}: {raw_sizes[m]:,}")
print("\n=== KEYPOINT RECALL при равном бюджете (символов) ===")
hdr=f"{'method':20}"+"".join(f"{b:>9}" for b in budgets); print(hdr)
recall={m:{} for m in methods}
for m in methods:
    row=f"{m:20}"
    for b in budgets:
        r=np.mean([keypoint_coverage(ctx(m,q)[:b], KP[i]) for i,q in enumerate(Q)])
        recall[m][b]=r; row+=f"{r:>9.3f}"
    print(row)
print("\n=== SEMANTIC (TF-IDF cos) контекст↔эталон при равном бюджете ===")
print(hdr)
sem={m:{} for m in methods}
for m in methods:
    row=f"{m:20}"
    for b in budgets:
        s=np.mean([semantic_sim(ctx(m,q)[:b], REF[i]) for i,q in enumerate(Q)])
        sem[m][b]=s; row+=f"{s:>9.3f}"
    print(row)

# plot
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
COL={"discourse_graph":"#2196F3","graphrag_baseline":"#FF9800","long_context":"#9C27B0"}
fig,ax=plt.subplots(1,2,figsize=(13,5))
for m in methods:
    ax[0].plot(budgets,[recall[m][b] for b in budgets],"o-",color=COL[m],label=m,lw=2)
    ax[1].plot(budgets,[sem[m][b] for b in budgets],"o-",color=COL[m],label=m,lw=2)
ax[0].set_title("Покрытие тезисов эталона ретривом",fontweight="bold"); ax[0].set_xlabel("бюджет контекста, символов"); ax[0].set_ylabel("keypoint recall")
ax[1].set_title("Семантическое сходство контекст↔эталон",fontweight="bold"); ax[1].set_xlabel("бюджет контекста, символов"); ax[1].set_ylabel("TF-IDF косинус")
for a in ax: a.legend(); a.grid(alpha=0.3)
fig.suptitle("Модель-независимое сравнение ретрива (один и тот же бюджет, без LLM)",fontsize=13,fontweight="bold")
fig.tight_layout(); fig.savefig("output/rag_eval_v2/04_retrieval_quality.png",dpi=150,bbox_inches="tight")
json.dump({"recall":recall,"semantic":sem,"raw_sizes":raw_sizes},open("output/retrieval_eval.json","w"),ensure_ascii=False,indent=1)
print("\nSaved: output/rag_eval_v2/04_retrieval_quality.png, output/retrieval_eval.json")
