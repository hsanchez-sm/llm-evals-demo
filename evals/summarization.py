"""Summarization eval — two parts:

1. reference cases: the SUT produces a summary, scored on
     - faithfulness: every checkable fact (number/proper noun) appears in the source
     - coverage:     fraction of the gold key_points present in the summary
     - conciseness:  summary length vs source (shorter is better, to a point)
2. faithfulness_probe cases: pre-written summaries (some with planted hallucinations)
   used to measure how well the faithfulness *detector* itself works.

NOTE (documented limitation): the mock faithfulness check is lexical — it catches
hallucinated entities/numbers (e.g. wrong city or year) but can miss *semantic*
contradictions (e.g. "shipped" vs "deferred"). Those need the LLM-judge backend.
"""
from sut.text_utils import salient_tokens, all_tokens, tokenize
from .metrics import mean


def faithfulness(summary, source):
    """Return (is_faithful, unsupported_tokens) using lexical fact-checking."""
    src = all_tokens(source)
    unsupported = sorted({t for t in salient_tokens(summary) if t not in src})
    return (len(unsupported) == 0, unsupported)


def coverage(summary, key_points):
    if not key_points:
        return 1.0
    present = all_tokens(summary)
    covered = 0
    for kp in key_points:
        keys = set(salient_tokens(kp)) or set(tokenize(kp))
        if keys and sum(1 for kkey in keys if kkey in present) / len(keys) >= 0.5:
            covered += 1
    return covered / len(key_points)


def conciseness(summary, source):
    s = len(tokenize(summary))
    src = len(tokenize(source)) or 1
    ratio = s / src
    # full marks up to 50% of source length, linearly decaying after.
    return 1.0 if ratio <= 0.5 else max(0.0, 1.0 - (ratio - 0.5))


def run(assistant, dataset, source_texts):
    ref_rows, probe_rows = [], []
    faith_scores, cov_scores, conc_scores = [], [], []
    probe_total, probe_correct = 0, 0

    for case in dataset:
        source = source_texts[case["source_doc_id"]]

        if case.get("kind") == "faithfulness_probe":
            pred_faithful, unsupported = faithfulness(case["candidate_summary"], source)
            ok = pred_faithful == case["is_faithful"]
            probe_total += 1
            probe_correct += ok
            probe_rows.append({
                "id": case["id"], "labeled_faithful": case["is_faithful"],
                "predicted_faithful": pred_faithful, "detector_correct": ok,
                "unsupported_tokens": unsupported,
            })
            continue

        summary = assistant.backend.summarize(source, case["instruction"])
        is_faithful, _ = faithfulness(summary, source)
        cov = coverage(summary, case.get("key_points", []))
        conc = conciseness(summary, source)
        faith_scores.append(1.0 if is_faithful else 0.0)
        cov_scores.append(cov)
        conc_scores.append(conc)
        ref_rows.append({
            "id": case["id"], "summary": summary,
            "faithful": is_faithful, "coverage": round(cov, 3),
            "conciseness": round(conc, 3),
        })

    metrics = {
        "faithfulness": round(mean(faith_scores), 3),
        "coverage": round(mean(cov_scores), 3),
        "conciseness": round(mean(conc_scores), 3),
    }
    if probe_total:
        metrics["faithfulness_detector_accuracy"] = round(probe_correct / probe_total, 3)

    return {
        "suite": "summarization",
        "n": len(dataset),
        "metrics": metrics,
        "rows": ref_rows + probe_rows,
    }
