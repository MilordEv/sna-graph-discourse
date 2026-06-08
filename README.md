# Discourse Graphs for Retrieval-Augmented Question Answering

A typed **discourse graph** for analysing and querying a thematic text corpus, benchmarked against a vanilla GraphRAG baseline and a long‑context baseline. The case study is a Russian‑language corpus on the discourse of **truth, lie and justice** (*истина / правда / ложь*), collected from the conservative‑thought portal *Русская Истина* (politconservatism.ru / rusistina.ru).

The central claim, supported by both an LLM‑based and a **model‑independent** evaluation, is that a discourse graph that encodes *typed* concept relations (rhetorical contrast, emotional colouring, surprisal) is a **more efficient and more faithful** retrieval substrate than plain co‑occurrence (GraphRAG) or raw long‑context, especially when the answering model is small.

---

## Key results

**Question answering** — 10 analytical questions, 3 retrieval methods, one *fixed simple* generator and one *strong independent* judge. Metrics are computed against **gold reference answers** (not the question — see *Evaluation* below). Higher is better; composite is a weighted mean.

| Method | ROUGE‑1 | ROUGE‑2 | ROUGE‑L | Semantic | Keypoint cov. | Faithfulness | Length | **Composite** |
|---|---|---|---|---|---|---|---|---|
| **discourse_graph** | **0.257** | 0.030 | 0.059 | **0.286** | **0.650** | **0.912** | 0.740 | **0.431** |
| graphrag_baseline | 0.218 | 0.020 | **0.065** | 0.216 | 0.617 | 0.844 | **0.837** | 0.399 |
| long_context | 0.220 | **0.033** | 0.061 | 0.283 | 0.583 | 0.756 | 0.714 | 0.386 |

*Generator (identical for all methods): `google/gemma-4-31b-it:free`. Judge: `meta-llama/llama-3.3-70b-instruct:free`. Figure: `output/rag_eval_v2/01_radar_v2.png`.*

**Model‑independent retrieval quality** — no LLM in the loop: at an *equal context budget* we measure how much of the gold answer each method's retrieved context covers (keypoint recall). This removes any dependence on the answering model.

| Budget (chars) | discourse_graph | graphrag_baseline | long_context |
|---|---|---|---|
| 800  | 0.413 | 0.420 | 0.267 |
| 1500 | **0.557** | 0.470 | 0.320 |
| 3000 | **0.673** | 0.540 | 0.320 |
| 6000 | **0.690** | 0.540 | 0.413 |
| 12000| **0.690** | 0.540 | 0.483 |

The discourse graph reaches ~0.69 keypoint recall in a context **~5× smaller** than the naive corpus dump (its full retrieved context averages ≈4.7 K chars vs ≈929 K for the whole corpus). Figure: `output/rag_eval_v2/04_retrieval_quality.png`.

### Why the answering model matters

With a *strong* generator both graph and long‑context answers look similarly good, so the comparison mostly measures the model. With a *simple* generator the compressed, typed graph context helps the model more than raw text does — which is exactly when retrieval quality is supposed to matter. We therefore fix a simple generator for all methods and use a separate, stronger model only as the judge.

---

## Corpus

| | |
|---|---|
| Source | *Русская Истина* (politconservatism.ru), tag *истина/правда/ложь* |
| Documents | 44 articles (full text) |
| Size | **107,928 word tokens** (median ≈ 3,700 words/article) |
| Domain | philosophy / political theory of truth, lie, propaganda, relativism, post‑truth |

> Note: the raw export (`*.xlsx`) only contained titles + one‑line abstracts (≈ 826 words total). Full article bodies were re‑fetched from the source URLs and cleaned (navigation/comment stripping); this 130× increase in text is a prerequisite for any meaningful graph or retrieval comparison. The title‑only version is preserved at `data/raw/russkaya_istina/documents.titles_only.bak.json`.

---

## Discourse‑graph construction

`discourse_graph/` builds a concept graph in which **nodes are concepts** and **edges are typed relations**. The pipeline (`pipeline.py`, `config.py` preset `russkaya_istina`) is:

**1. Concept extraction (vertices).**
- **Document‑frequency centrality** (`vertices/extractors.py::vertices_docfreq`) selects concepts by how many documents they appear in, surfacing terms central to the discourse (*истина, ложь, пропаганда, манипуляция, …*). This deliberately replaces plain TF‑IDF, which rewards document‑idiosyncratic rare tokens (proper names, one‑article phrases) and *buries* corpus‑central concepts.
- **Part‑of‑speech filtering** via `pymorphy3` keeps only nouns/adjectives — no hand‑maintained stop‑list — and checks *all* morphological parses (so homonyms like *правда*, which `pymorphy` defaults to a particle, are retained).
- **Lemmatisation** merges inflected forms into a single node (*истина / истины / истину / истине → истина*), which previously fragmented every concept into 3–4 separate nodes.
- **Domain seeds** guarantee inclusion of core topic concepts when present.
- **YAKE** keyphrases ([Campos et al., 2020](https://doi.org/10.1016/j.ins.2019.09.013)) add multi‑word concepts.

**2. Typed edges (`edges/builders.py`).** Each concept pair is connected by one or more typed relations:
- **co‑occurrence** (sentence level),
- **rhetorical contrast** — RST‑style contrast markers (*не … а*, *однако*, *в отличие от*, *вместо истины* …), the discourse‑specific signal,
- **emotional** colouring (domain affect lexicon),
- **perplexity / surprisal** as an edge attribute for "hot‑edge" interpretation.

Edges are weighted **per document** (a pair counts once per document, not once per sentence), so a single verbose article can no longer dominate the graph; this also aligns edge weight with document frequency. Node matching inside text is done on **lemmas** (so node *истина* matches any inflected occurrence).

**3. Backbone & filtering (`utils.py::filter_graph`).** PMI is kept as an edge attribute but not used as a hard filter (it would remove links between two frequent concepts, e.g. *истина↔пропаганда*). Instead we apply a **k‑nearest‑neighbourhood backbone** ([Drieger, 2013](https://doi.org/10.1016/j.sbspro.2013.05.461)): each concept keeps its ≈10 strongest edges. This controls density while preserving structure.

**Resulting graph** (russkaya_istina preset): **109 concept nodes, ≈814 edges, 326 typed contrast edges**, 6 communities (modularity ≈ 0.41), 0 un‑merged word‑forms, 20/22 target domain concepts present. The truth–lie–justice opposition core is recovered correctly: *истина↔правда↔ложь↔обман↔заблуждение↔справедливость↔манипуляция*. Visualisation: `output/rag_eval_v2/03_discourse_graph_istina.png`.

### What was optimised (summary)

The discourse graph went from "names + inflected fragments" to a clean concept network through: lemmatisation; document‑frequency concept selection (vs TF‑IDF); POS content filtering; domain seeds; per‑document edge weighting; k‑NN backbone; real YAKE keyphrases; corpus cleaning; lemma‑based node matching; and a rewritten, lemma‑aware, edge‑first `walk` retriever (removing a noisy raw node dump that previously diluted the retrieved context).

---

## Methods compared

| Method | Retrieval |
|---|---|
| `discourse_graph` | LightRAG‑style local **walk** + global **community** over the typed discourse graph ([Guo et al., 2024](https://arxiv.org/abs/2410.05779)) |
| `graphrag_baseline` | Vanilla GraphRAG: per‑document salient terms + plain co‑occurrence, community summaries ([Edge et al., 2024](https://arxiv.org/abs/2404.16130)) |
| `long_context` | The corpus text packed into the model context window |

---

## Evaluation methodology

The evaluation was rebuilt (`discourse_graph/eval_metrics_v2.py`) to fix a methodological flaw in the original: ROUGE had been computed **against the question text**, which rewards answers that merely echo the question and penalises substantive ones (empirically, an echo answer scored ROUGE‑1 = 0.86 vs 0.14 for a correct one). v2 instead scores against **gold reference answers** (`data/eval/gold_answers.json`):

- **ROUGE‑1/2/L**, **semantic** (TF‑IDF cosine) and **keypoint coverage** vs the gold answer;
- **faithfulness** from an **LLM‑as‑judge** (faithfulness/relevance/coverage/grounding), with the judge a *stronger, independent* model than the generator;
- a **model‑independent retrieval** evaluation (`run_retrieval_eval.py`) that measures gold coverage of the retrieved context at matched budgets — no answering model at all.

Roles are separated on purpose: the **generator** is a fixed simple model (so the comparison reflects retrieval, not model strength), the **judge** is a strong model (more reliable assessment).

---

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1) (optional) rebuild the discourse graph + baseline from the corpus
python -c "from discourse_graph.config import ConstructorConfig as C; \
from discourse_graph.pipeline import DiscourseGraphConstructor as D; \
D(C.from_preset('russkaya_istina')).run('data/raw/russkaya_istina/documents.json')"

# 2) run the QA experiment — one simple generator for all methods
#    (free OpenRouter model; needs OPENROUTER_KEY in .env, no credit card)
LLM_MODEL="google/gemma-4-31b-it:free" python run_experiment_v2.py

# 3) judge with a strong, independent model (no regeneration)
JUDGE_MODEL="meta-llama/llama-3.3-70b-instruct:free" python run_judge.py

# 4) model-independent retrieval comparison (no LLM)
python run_retrieval_eval.py
```

Any OpenAI‑compatible endpoint works. For a **local** model (no API): `ollama pull qwen2.5:7b` then
`LLM_BASE_URL=http://localhost:11434/v1 LLM_API_KEY=ollama LLM_MODEL=qwen2.5:7b python run_experiment_v2.py`.
Free OpenRouter models are rate‑limited (20 req/min, 200/day); the scripts throttle accordingly (`LLM_SLEEP`), and `LLM_JUDGE=0` halves the call count.

---

## Repository structure

```
discourse_graph/            # library
  config.py                 #   presets (russkaya_istina)
  pipeline.py               #   end-to-end constructor
  vertices/extractors.py    #   concept extraction (docfreq, yake, POS, lemmas, seeds)
  edges/builders.py         #   typed edges (co-occ, rhetorical, emotional, perplexity)
  utils.py                  #   lemmatisation, k-NN backbone, graph IO
  retrieval/                #   walk / community / lightrag retrievers
  baseline_graphrag.py      #   vanilla GraphRAG baseline
  eval_metrics_v2.py        #   reference-based metrics + LLM judge
run_experiment_v2.py        # generate answers (simple model) + score
run_judge.py                # strong-model judging, no regeneration
run_retrieval_eval.py       # model-independent retrieval evaluation
data/raw/russkaya_istina/   # corpus (full texts; titles-only backup)
data/graphs/…               # built graphs (graphml / csv)
data/eval/                  # questions + gold answers
output/rag_eval_v2/         # figures (radar, bars, retrieval, concept graph)
docs/                       # project notes, EDA reports, methods write-up
```

---

## Methodological background

The discourse‑graph design draws on:

- **Connected Concept Analysis** — Lindgren, 2016, *Text & Talk* 36(3) — sentence‑level concept co‑occurrence networks.
- **Semantic network analysis / k‑next‑neighbourhood** — [Drieger, 2013](https://doi.org/10.1016/j.sbspro.2013.05.461).
- **Discourse Network Analysis** (actors × concepts) — [Leifeld](https://github.com/leifeld/dna) (a natural next step: add author/actor nodes).
- **TextRank** — [Mihalcea & Tarau, 2004](https://aclanthology.org/W04-3252/); **YAKE** — [Campos et al., 2020](https://doi.org/10.1016/j.ins.2019.09.013); **KeyBERT**.
- **Rhetorical Structure Theory** for contrast relations; Russian RST parsing via [IsaNLP RST](https://github.com/tchewik/isanlp_rst).
- **GraphRAG** — [Edge et al., 2024](https://arxiv.org/abs/2404.16130); **LightRAG** — [Guo et al., 2024](https://arxiv.org/abs/2410.05779).

Russian NLP: [`pymorphy3`](https://github.com/no-plagiarism/pymorphy3) (morphology/lemmatisation), [`razdel`](https://github.com/natasha/razdel) (segmentation), [`YAKE`](https://github.com/LIAAD/yake).

## Limitations

- Single small corpus (44 docs); the graph's advantage is expected to grow with corpus size and on global/multi‑hop questions.
- The concept graph has no author nodes, so authorship questions ("who writes most about lies") are answered only indirectly — adding DNA‑style actor↔concept edges is the natural extension.
- Two rare concepts (*релятивизм*, *постправда*; ≤3 docs) fall below the inclusion threshold.
- Gold answers and the question set are author‑curated.
