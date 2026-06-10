"""Retrieval metrics: precision@k, recall@k, MRR."""


def precision_at_k(predicted, relevant, k):
    pred = predicted[:k]
    if not pred:
        return 0.0
    hits = sum(1 for p in pred if p in relevant)
    return hits / len(pred)


def recall_at_k(predicted, relevant, k):
    if not relevant:
        return None  # undefined when there are no relevant docs (negative case)
    hits = sum(1 for p in predicted[:k] if p in relevant)
    return hits / len(relevant)


def reciprocal_rank(predicted, relevant):
    for i, p in enumerate(predicted, start=1):
        if p in relevant:
            return 1.0 / i
    return 0.0


def mean(values):
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else 0.0
