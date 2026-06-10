"""Response-quality eval.

Mock grader (deterministic): scores an answer by how many of the reference answer's
'salient' facts (numbers, proper nouns) appear in it — a keyword-recall proxy.
With a real backend you'd swap `grade_answer` for an LLM-as-judge that scores the
rubric criteria directly. The dataset already carries per-case rubrics for that.
"""
from sut.text_utils import salient_tokens, all_tokens
from .metrics import mean

PASS_THRESHOLD = 0.5


def grade_answer(answer, reference):
    keys = set(salient_tokens(reference))
    if not keys:
        return 1.0
    present = all_tokens(answer)
    return sum(1 for kkey in keys if kkey in present) / len(keys)


def run(assistant, dataset):
    rows, scores = [], []
    for case in dataset:
        out = assistant.answer(case["query"])
        score = grade_answer(out["answer"], case["reference_answer"])
        scores.append(score)
        rows.append({
            "id": case["id"], "query": case["query"],
            "answer": out["answer"], "agent": out["agent"],
            "score": round(score, 3), "passed": score >= PASS_THRESHOLD,
        })

    passed = sum(1 for r in rows if r["passed"])
    n = len(dataset)
    return {
        "suite": "response_quality",
        "n": n,
        "metrics": {
            "avg_score": round(mean(scores), 3),
            "pass_rate": round(passed / n, 3) if n else 0.0,
        },
        "rows": rows,
    }
