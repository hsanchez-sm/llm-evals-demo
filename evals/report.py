"""Write the run report to JSON and print a readable console summary.

Each suite's `rows` already carry the expected vs. obtained result per case; this
module normalizes them into a uniform (expected, obtained, passed) shape so it can
print, per dataset: totals / passed / failed / pass-rate, plus a per-case list.
The same normalized data (`case_summary` + `cases`) is added to the JSON report.
"""
import json

from sut.corpus import ROOT

PASS_TAG, FAIL_TAG = "PASS", "FAIL"


def _clip(value, width=42):
    text = str(value).replace("\n", " ").strip()
    return text if len(text) <= width else text[: width - 3] + "..."


def _normalize_rows(res):
    """Map each suite's per-case row to {id, expected, obtained, passed}."""
    suite = res["suite"]
    cases = []
    for r in res["rows"]:
        if suite == "agent_routing":
            expected, obtained, ok = r["expected"], r["predicted"], r["correct"]
        elif suite == "rag_retrieval":
            if r.get("type") == "negative":
                expected = "(abstain / no relevant doc)"
                obtained = f"top_score={r['top_score']}"
                ok = r["abstained"]
            else:
                expected = ", ".join(r["expected"])
                obtained = ", ".join(r["predicted"]) or "(none)"
                ok = r.get("recall@k", 0) >= 1.0  # pass = all expected docs in top-k
        elif suite == "response_quality":
            expected = r.get("reference", "")
            obtained = r["answer"]
            ok = r["passed"]
        elif suite == "summarization":
            if "detector_correct" in r:  # faithfulness probe
                expected = f"is_faithful={r['labeled_faithful']}"
                obtained = f"is_faithful={r['predicted_faithful']}"
                ok = r["detector_correct"]
            else:                         # reference summary
                expected = "faithful + cover key points"
                obtained = r["summary"]
                ok = bool(r["faithful"]) and r["coverage"] >= 0.5
        else:
            expected, obtained, ok = "", "", True
        cases.append({"id": r["id"], "expected": expected,
                      "obtained": obtained, "passed": bool(ok)})
    return cases


def _summarize(cases):
    total = len(cases)
    passed = sum(1 for c in cases if c["passed"])
    failed = total - passed
    pass_pct = round(100 * passed / total, 1) if total else 0.0
    return {"total": total, "passed": passed, "failed": failed, "pass_pct": pass_pct}


def save_and_print(suite_results, backend_name):
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)
    out_path = reports_dir / "last_run.json"

    # Enrich each suite with normalized per-case results + counts (also in JSON).
    for res in suite_results:
        res["cases"] = _normalize_rows(res)
        res["case_summary"] = _summarize(res["cases"])

    payload = {"backend": backend_name, "suites": suite_results}
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 72)
    print(f" EvalsInc - LLM Eval Report   (backend: {backend_name})")
    print("=" * 72)

    grand = {"total": 0, "passed": 0, "failed": 0}
    for res in suite_results:
        cs = res["case_summary"]
        grand["total"] += cs["total"]
        grand["passed"] += cs["passed"]
        grand["failed"] += cs["failed"]

        print(f"\n  {res['suite']}")
        metrics = "  ".join(f"{k}={v}" for k, v in res["metrics"].items())
        print(f"    metrics: {metrics}")
        print(f"    cases:   {cs['total']} total | {cs['passed']} passed | "
              f"{cs['failed']} failed | {cs['pass_pct']}% pass")

        if res["suite"] == "agent_routing":
            print("    confusion (expected -> predicted):")
            for expected, preds in res["confusion"].items():
                if preds:
                    detail = ", ".join(f"{p}:{c}" for p, c in preds.items())
                    print(f"      {expected}: {detail}")

        print(f"    {'result':6}  {'id':18}  {'expected':44}  obtained")
        print(f"    {'-'*6}  {'-'*18}  {'-'*44}  {'-'*30}")
        for c in res["cases"]:
            tag = PASS_TAG if c["passed"] else FAIL_TAG
            print(f"    {tag:6}  {c['id']:18}  {_clip(c['expected'], 44):44}  "
                  f"{_clip(c['obtained'], 30)}")

    g_pct = round(100 * grand["passed"] / grand["total"], 1) if grand["total"] else 0.0
    print("\n" + "-" * 72)
    print(f"  OVERALL: {grand['total']} total | {grand['passed']} passed | "
          f"{grand['failed']} failed | {g_pct}% pass")
    print("=" * 72)
    print(f" Full per-case detail (JSON): {out_path}")
    print("=" * 72 + "\n")
    return out_path
