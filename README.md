# LLM Evals Demo — EvalsInc Assistant

A self-contained **evaluation harness** for an LLM agent system, covering the four
eval types that matter most in production: **agent routing**, **RAG retrieval**,
**response quality**, and **summarization (faithfulness / coverage / conciseness)**.

The whole suite runs **deterministically with zero dependencies and no API key**
(a built-in mock backend), so it executes in CI on every push. Point it at a real
model with `--backend anthropic|openai` to benchmark an actual LLM against the
baseline.

> Everything here — corpus, datasets, agents, graders — is **original and synthetic**
> (a fictional company, *EvalsInc*). No proprietary data.

---

## What it evaluates

| Suite | Question it answers | Metrics |
|-------|--------------------|---------|
| **Agent routing** | Does each query reach the agent that should handle it? | accuracy + confusion matrix |
| **RAG retrieval** | For a query about a known entity, are the *right, available* documents retrieved? | precision@k, recall@k, MRR, abstention on out-of-KB queries |
| **Response quality** | Is the answer correct and grounded? | rubric/keyword score, pass rate |
| **Summarization** | Is the summary faithful, complete, and concise? | faithfulness, coverage, conciseness + a faithfulness-**detector** accuracy on planted-hallucination probes |

## The system under test (`sut/`)

A small, original assistant:
- **Router** (`router.py`) — rule-based classifier → `knowledge` / `billing` /
  `technical_support` / `general`.
- **Retriever** (`retriever.py`) — pure-Python TF-IDF over the synthetic corpus.
- **Backend** (`backend.py`) — pluggable: `mock` (deterministic, extractive) or a
  real LLM (`anthropic` / `openai`, lazy-imported).

## Quick start

```bash
# No install needed for the mock backend (stdlib only)
python -m evals.run                      # all suites, mock backend
python -m evals.run --suite routing      # one suite
python -m evals.run --suite retrieval --k 5

# Against a real model (optional):
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...             # or OPENAI_API_KEY
python -m evals.run --backend anthropic
```

A full per-case report is written to `reports/last_run.json`.

## Baseline results (mock backend)

These are the deterministic results CI produces — a **deliberately simple lexical
baseline**, not a tuned model:

| Suite | Result |
|-------|--------|
| Agent routing | accuracy **1.00** (incl. 3 trap/ambiguous queries) |
| RAG retrieval | recall@3 **0.91**, MRR **0.82**, abstention **1.00**, precision@3 0.39\* |
| Response quality | avg score **0.35**, pass rate 0.40 |
| Summarization | faithfulness **1.00**, coverage 0.33, conciseness 1.00, detector acc **0.67** |

\* precision@3 is low *by design*: most queries have a single relevant document, so
precision caps at 0.33 at k=3 — recall@k and MRR are the meaningful retrieval metrics here.

### Reading the baseline (this is the point of the harness)
- **Routing & retrieval-recall are strong** — lexical methods do well at matching
  entities and surfacing the right documents.
- **Response synthesis is weak (0.35)** — a lexical extractor can't disambiguate
  *"the Pro plan price"* from other numeric lines. The harness **correctly surfaces**
  this; a real model is expected to beat it (run `--backend anthropic`).
- **Faithfulness detector = 0.67** — the lexical check catches *entity/number*
  hallucinations (wrong city, wrong year) but misses *semantic* contradictions
  (e.g. "shipped" vs "deferred"). That gap is exactly why an LLM-judge backend exists.

A good eval suite isn't one where everything scores 100% — it's one that **measures
the right things and exposes where a system fails.**

## Datasets (`data/`)

- `corpus/` — 14 documents (3 companies × profile + 2 employees each, plus 5 product
  docs) + 1 meeting transcript.
- `corpus_manifest.json` — ground truth: each doc → its entity and owning agent.
- `datasets/*.jsonl` — labeled cases per suite (routing 18, retrieval 12, quality 10,
  summarization 6 incl. faithfulness probes).

## Project layout

```
sut/      router · retriever · backend · assistant
evals/    routing · retrieval · response_quality · summarization · metrics · run
data/     corpus + manifest + datasets
.github/  CI: runs the suite in mock mode on every push
```

## License

MIT — see [LICENSE](LICENSE).
