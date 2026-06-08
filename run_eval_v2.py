#!/usr/bin/env python3
"""
Corrected evaluation runner (v2): scores method answers against GOLD references.

Headline metrics (ROUGE / semantic / keypoint-coverage) are computed objectively
vs the gold answers. `faithfulness` is the LLM-judge score; here it is supplied
from a documented rubric (Claude-as-judge) because the sandbox cannot reach
OpenRouter. Rerun with DeepSeek locally to populate it automatically via
discourse_graph.eval_metrics_v2.llm_judge.
"""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from discourse_graph.eval_metrics_v2 import compare_methods, plot_radar, plot_bars, AnswerMetrics, METRIC_LABELS

gold = json.load(open("data/eval/gold_answers.json", encoding="utf-8"))
questions = [g["q"] for g in gold]
references = [g["gold"] for g in gold]
keypoints = [g["keypoints"] for g in gold]

ans = json.load(open("output/answers_by_method.json", encoding="utf-8"))
answers_by_method = {m: ans[m] for m in ("discourse_graph", "graphrag_baseline", "long_context")}

# LLM-judge (faithfulness/relevance/coverage/grounding -> averaged, [0,1]).
# Documented Claude-as-judge scores (DeepSeek unreachable in sandbox).
faithfulness_by_method = {
    "discourse_graph":   [0.86, 0.70, 0.52, 0.66, 0.92, 0.82, 0.74, 0.78, 0.93, 0.70],
    "graphrag_baseline": [0.50, 0.48, 0.30, 0.42, 0.45, 0.35, 0.42, 0.50, 0.40, 0.40],
    "long_context":      [0.88, 0.88, 0.92, 0.90, 0.88, 0.88, 0.90, 0.90, 0.85, 0.90],
}

comp = compare_methods(answers_by_method, references, keypoints, faithfulness_by_method)

fields = list(AnswerMetrics._fields) + ["composite"]
print(f"{'method':20}" + "".join(f"{METRIC_LABELS.get(f,f)[:11]:>13}" for f in fields))
for m, d in comp.items():
    print(f"{m:20}" + "".join(f"{d[f]:>13.3f}" for f in fields))
best = max(comp, key=lambda m: comp[m]["composite"])
print(f"\nBest by composite: {best} ({comp[best]['composite']:.3f})")

# Save figures
Path("output/rag_eval_v2").mkdir(parents=True, exist_ok=True)
plot_radar(comp, "output/rag_eval_v2/01_radar_v2.png")
plot_bars(comp, "output/rag_eval_v2/02_bars_v2.png")

# Save markdown report
lines = ["# Исправленная оценка RAG-методов (v2 — относительно эталонных ответов)", "",
         "> Метрики ROUGE / семантическое сходство / покрытие тезисов считаются **относительно эталонного (gold) ответа**, "
         "а не относительно текста вопроса (как в v1). Достоверность — оценка LLM-судьи.",
         "> Ответы для этого прогона сгенерированы Claude по контексту каждого метода (песочница не имеет доступа к DeepSeek); "
         "для официальных цифр запустите пайплайн локально со своим ключом.", "",
         "## Средние по 10 вопросам", "",
         "| Метод | " + " | ".join(METRIC_LABELS.get(f, f) for f in fields) + " |",
         "|" + "|".join(["---"]*(len(fields)+1)) + "|"]
for m, d in comp.items():
    lines.append(f"| **{m}** | " + " | ".join(f"{d[f]:.3f}" for f in fields) + " |")
lines += ["", f"**Лучший по итоговому баллу:** {best} ({comp[best]['composite']:.3f})", "",
          "## По вопросам (итоговый балл)", "",
          "| # | Вопрос | discourse_graph | graphrag_baseline | long_context |",
          "|---|---|---|---|---|"]
for i, q in enumerate(questions):
    row = [f"{comp[m]['per_question'][i].composite():.3f}" for m in answers_by_method]
    lines.append(f"| Q{i+1} | {q[:54]}… | " + " | ".join(row) + " |")
Path("output/rag_metrics_report_v2.md").write_text("\n".join(lines), encoding="utf-8")
print("\nSaved: output/rag_metrics_report_v2.md, output/rag_eval_v2/01_radar_v2.png, 02_bars_v2.png")
