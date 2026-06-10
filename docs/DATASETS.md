# Evaluation Datasets

Every evaluation case and its expected (ground-truth) result, grouped by suite.
All data is synthetic. Source datasets live in [`../data/datasets/`](../data/datasets/);
the document corpus and its ground-truth mapping live in
[`../data/corpus/`](../data/corpus/) and [`../data/corpus_manifest.json`](../data/corpus_manifest.json).

---

## 1. Agent Routing — `data/datasets/agent_routing.jsonl` (18 cases)

Expected result = the agent that should handle the query.

| ID | Query | Expected agent | Note |
|----|-------|----------------|------|
| route_001 | What industry is Nimbus Analytics in? | knowledge_agent | |
| route_002 | Who is the CTO of Kestrel Robotics? | knowledge_agent | |
| route_003 | What is Verdant Foods' annual revenue? | knowledge_agent | |
| route_004 | What is Maya Okonkwo's job title? | knowledge_agent | |
| route_005 | How much does the Pro plan cost? | billing_agent | |
| route_006 | I was charged twice this month, what do I do? | billing_agent | |
| route_007 | Can I get a refund on my annual subscription? | billing_agent | |
| route_008 | How do I upgrade from Free to Pro? | billing_agent | |
| route_009 | The API is returning a 401 error, why? | technical_support_agent | |
| route_010 | How do I reset my password? | technical_support_agent | |
| route_011 | My dashboard won't load, it's blank. | technical_support_agent | |
| route_012 | How do I create an API key? | technical_support_agent | |
| route_013 | Hi there! | general_agent | |
| route_014 | What can you do? | general_agent | |
| route_015 | Thanks, that's all — bye. | general_agent | |
| route_016 | What's the price of Kestrel Robotics? | knowledge_agent | TRAP: "price" sounds like billing, but it asks about a knowledge-base company |
| route_017 | I need programmatic access to pull data — how? | technical_support_agent | AMBIGUOUS: API access also touches billing; primary intent is a technical "how-to" |
| route_018 | Does Verdant Foods have an enterprise plan with us? | knowledge_agent | TRAP: "enterprise plan" is billing wording, but the subject is a knowledge-base company |

**Metric:** routing accuracy + confusion matrix.

---

## 2. RAG Retrieval — `data/datasets/rag_retrieval.jsonl` (12 cases)

Expected result = the document IDs that should be retrieved.

| ID | Query | Expected documents |
|----|-------|--------------------|
| rag_001 | Tell me everything about Nimbus Analytics. | comp_nimbus, emp_nimbus_maya, emp_nimbus_raj |
| rag_002 | Who works at Kestrel Robotics? | emp_kestrel_aiko, emp_kestrel_daniel |
| rag_003 | What is Verdant Foods' annual revenue? | comp_verdant |
| rag_004 | Who is the COO of Verdant Foods? | emp_verdant_lena |
| rag_005 | What is the website of Kestrel Robotics? | comp_kestrel |
| rag_006 | What is Raj Patel's email address? | emp_nimbus_raj |
| rag_007 | Tell me about the supply chain manager at Verdant Foods. | emp_verdant_tomas |
| rag_008 | What does the Pro plan include? | prod_pricing |
| rag_009 | How do I request a refund? | prod_refund |
| rag_010 | What does a 429 error mean? | prod_api_errors |
| rag_011 | How do I enable two-factor authentication? | prod_troubleshoot |
| rag_012 | What is the stock price of Apple Inc.? | *(none)* — NEGATIVE: out-of-knowledge-base entity, should abstain |

**Metric:** precision@k, recall@k, MRR (plus an abstention check for the negative case).

---

## 3. Response Quality — `data/datasets/response_quality.jsonl` (10 cases)

Expected result = reference answer + grading rubric.

| ID | Query | Reference answer | Rubric |
|----|-------|------------------|--------|
| qual_001 | What is Nimbus Analytics' annual revenue? | USD 48 million in FY2024. | States 48M USD · mentions FY2024 · concise |
| qual_002 | Who co-founded Kestrel Robotics and what is their role? | Aiko Tanaka co-founded Kestrel Robotics and is the CTO. | Names Aiko Tanaka · CTO/co-founder · no hallucinated co-founders |
| qual_003 | How much is the Pro plan per user per month? | $49 per user per month (or $470/user/year). | States $49/user/month · correct currency · (optional) annual price |
| qual_004 | What does a 401 error from the API mean and how do I fix it? | The API key is missing, invalid, or expired. Generate a new key in Settings > API Keys and send it as a Bearer token. | Explains invalid/expired key · gives the fix · no invented causes |
| qual_005 | Where is Verdant Foods headquartered? | Portland, Oregon, USA. | States Portland, Oregon · no wrong location · concise |
| qual_006 | What is Raj Patel's role at Nimbus Analytics? | Head of Data Science. | States Head of Data Science · correct company · concise |
| qual_007 | Can I get my money back if I cancel Pro after 20 days? | Not a full refund — the 14-day window has passed; renewal stops at the end of the paid term, no refund of unused months unless local law requires it. | Recognizes the 14-day window passed · no prorated refund · faithful to the policy |
| qual_008 | How many employees does Kestrel Robotics have? | 85 employees. | States 85 · correct company · no fabricated figure |
| qual_009 | What is the founding year of Nimbus Analytics? | 2016. | States 2016 · concise · no hallucination |
| qual_010 | I forgot my password. What should I do? | Use 'Forgot password' on the login page; you'll get a reset link by email valid for 30 minutes (check spam). | Mentions Forgot-password flow · email reset link · no invented steps |

**Metric:** rubric satisfaction score (LLM-as-judge in real mode; lexical heuristic in mock mode).

---

## 4. Summarization — `data/datasets/summarization.jsonl` (6 cases)

Two kinds:
- **reference** — the system generates a summary; it is scored for coverage, faithfulness and conciseness.
- **faithfulness_probe** — a pre-written summary; the faithfulness detector must predict the correct `is_faithful` label.

| ID | Kind | Source | Expected result |
|----|------|--------|-----------------|
| sum_001 | reference | transcript_q2_planning | 3-bullet summary of decisions/action items. Key points: ship REST API in Q2 · perf pass instead of redesign · rate limiting in API v1 · Slack moved to Q3 · redesign parked · action items (Marcus/Ken/Sofia/Priya) |
| sum_002 | reference | comp_verdant | 2-sentence summary. Key points: organic food manufacturer · Portland, founded 2009 · 1,150 employees, USD 310M · USDA Organic + B-Corp |
| sum_003 | reference | comp_nimbus | 1-sentence TL;DR. Key points: data analytics SaaS · Austin, 2016, 240 employees, USD 48M · product NimbusBoard, no-SQL dashboards |
| sum_004 | faithfulness_probe | transcript_q2_planning | **is_faithful = false** — claims Slack shipped this quarter; it was moved to the Q3 backlog (contradiction) |
| sum_005 | faithfulness_probe | comp_nimbus | **is_faithful = false** — wrong location (Boston), year (2012), size (500+) and industry |
| sum_006 | faithfulness_probe | comp_verdant | **is_faithful = true** — Portland, 2009, organic; all supported (control case) |

**Metric:** coverage + conciseness + faithfulness on reference cases; detector accuracy on the probes.
