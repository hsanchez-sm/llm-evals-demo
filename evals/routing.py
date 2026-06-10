"""Agent-routing eval: accuracy + confusion matrix over expected vs predicted agent."""
from collections import defaultdict

from sut.router import AGENTS


def run(assistant, dataset):
    rows = []
    correct = 0
    confusion = {a: defaultdict(int) for a in AGENTS}

    for case in dataset:
        predicted = assistant.router.route(case["query"])
        expected = case["expected_agent"]
        ok = predicted == expected
        correct += ok
        confusion[expected][predicted] += 1
        rows.append({
            "id": case["id"], "query": case["query"],
            "expected": expected, "predicted": predicted, "correct": ok,
        })

    n = len(dataset)
    return {
        "suite": "agent_routing",
        "n": n,
        "metrics": {"accuracy": correct / n if n else 0.0},
        "confusion": {e: dict(preds) for e, preds in confusion.items()},
        "rows": rows,
    }
