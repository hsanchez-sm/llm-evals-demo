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

### How failing cases arise (nothing is rigged)

The failing cases are **not hand-authored to fail** — they *emerge* from running the
simple mock baseline against the ground truth, under fixed pass/fail criteria. Two
things are kept separate:

- **Expected result** — hand-labeled ground truth, derived from the corpus facts
  (revenue, CTO, price, …). Fixed and known.
- **Obtained result** — whatever the mock SUT produces (rule-based router, TF-IDF
  retriever, extractive answerer, lexical faithfulness check).

A case **fails** when *obtained ≠ expected* under the suite's pass rule. The SUT
decides that — not the dataset.

**Two kinds of hard case:**
- *Deliberately hard probes* authored to stress a scenario — routing **traps**
  (billing-sounding wording about a knowledge-base company), the retrieval
  **negative** (out-of-KB query that must abstain), and summarization
  **faithfulness probes** (pre-written summaries labeled `is_faithful=false`).
  Whether they pass is still up to the SUT.
- *Emergent failures* — the lexical baseline simply isn't strong enough. These are
  the most informative.

**Pass criteria (defined in `evals/report.py`):**

| Suite | A case passes when… |
|-------|---------------------|
| Agent routing | predicted agent == expected agent |
| RAG retrieval | recall@k == 1.0 (all expected docs in top-k); negative case → abstains (top score < 0.15) |
| Response quality | score ≥ 0.5 (keyword-recall proxy for the rubric) |
| Summarization | reference → faithful **and** coverage ≥ 0.5; probe → predicted label == labeled |

**Why each baseline failure happens:**

| Case | Why it fails |
|------|--------------|
| `rag_008` — *"What does the Pro plan include?"* | TF-IDF ranks `prod_refund` / `prod_api` above `prod_pricing`: "Pro plan" also appears in the refund policy and "include" is generic — lexical overlap is misleading. |
| `qual_003/004/005/008/009/010` | The extractive answerer returns the doc line with the most query-word overlap, which is often a header rather than the fact. E.g. *"founding year"* matches `Company name: Nimbus Analytics` (shares "Nimbus/Analytics") over `Founded: 2016` (zero overlap — "founding" ≠ "founded"), so "2016" never appears in the answer. |
| `sum_003` (TL;DR) | The extractive summary covers < 50% of the gold key points. |
| `sum_004` (faithfulness probe) | The lexical detector misses a **semantic** contradiction (*"Slack shipped this quarter"* reuses words that are present in the source) — exactly the gap an LLM-judge backend closes. |

Run `--backend anthropic|openai` and most quality/summarization failures resolve,
because a real model answers concisely and judges semantically — the harness then
quantifies that lift.

## Datasets (`data/`)

- `corpus/` — 14 documents (3 companies × profile + 2 employees each, plus 5 product
  docs) + 1 meeting transcript.
- `corpus_manifest.json` — ground truth: each doc → its entity and owning agent.
- `datasets/*.jsonl` — labeled cases per suite (routing 18, retrieval 12, quality 10,
  summarization 6 incl. faithfulness probes).

See [`docs/DATASETS.md`](docs/DATASETS.md) for every case and its expected result.

## Project layout

```
sut/      router · retriever · backend · assistant
evals/    routing · retrieval · response_quality · summarization · metrics · run
data/     corpus + manifest + datasets
.github/  CI: runs the suite in mock mode on every push
```

## License

MIT — see [LICENSE](LICENSE).
