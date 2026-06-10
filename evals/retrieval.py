"""RAG-retrieval eval: precision@k, recall@k, MRR, plus abstention on out-of-KB queries."""
from .metrics import precision_at_k, recall_at_k, reciprocal_rank, mean

ABSTAIN_THRESHOLD = 0.15  # top similarity below this == "nothing relevant found"


def run(assistant, dataset, k=3):
    rows, p_list, r_list, rr_list = [], [], [], []
    abstain_total, abstain_ok = 0, 0

    for case in dataset:
        hits = assistant.retrieve(case["query"], k)
        predicted = [doc_id for doc_id, _ in hits]
        top_score = hits[0][1] if hits else 0.0
        expected = case["expected_doc_ids"]

        if not expected:
            # negative case: success = the retriever abstains (low top score)
            abstain_total += 1
            ok = top_score < ABSTAIN_THRESHOLD
            abstain_ok += ok
            rows.append({"id": case["id"], "query": case["query"], "type": "negative",
                         "top_score": round(top_score, 4), "abstained": ok})
            continue

        p = precision_at_k(predicted, expected, k)
        r = recall_at_k(predicted, expected, k)
        rr = reciprocal_rank(predicted, expected)
        p_list.append(p); r_list.append(r); rr_list.append(rr)
        rows.append({
            "id": case["id"], "query": case["query"], "type": "positive",
            "expected": expected, "predicted": predicted,
            "precision@k": round(p, 3), "recall@k": round(r, 3), "rr": round(rr, 3),
        })

    metrics = {
        "k": k,
        "precision@k": round(mean(p_list), 3),
        "recall@k": round(mean(r_list), 3),
        "mrr": round(mean(rr_list), 3),
    }
    if abstain_total:
        metrics["abstention_accuracy"] = round(abstain_ok / abstain_total, 3)

    return {"suite": "rag_retrieval", "n": len(dataset), "metrics": metrics, "rows": rows}
