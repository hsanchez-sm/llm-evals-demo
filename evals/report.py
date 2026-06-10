"""Write the run report to JSON and print a readable console summary."""
import json

from sut.corpus import ROOT


def _fmt_metrics(metrics):
    return "  ".join(f"{k}={v}" for k, v in metrics.items())


def save_and_print(suite_results, backend_name):
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)
    out_path = reports_dir / "last_run.json"

    payload = {"backend": backend_name, "suites": suite_results}
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 64)
    print(f" EvalsInc — LLM Eval Report   (backend: {backend_name})")
    print("=" * 64)
    for res in suite_results:
        print(f"\n  {res['suite']}  (n={res['n']})")
        print(f"    {_fmt_metrics(res['metrics'])}")
        if res["suite"] == "agent_routing":
            print("    confusion (expected -> predicted):")
            for expected, preds in res["confusion"].items():
                if preds:
                    detail = ", ".join(f"{p}:{c}" for p, c in preds.items())
                    print(f"      {expected}: {detail}")
    print("\n" + "=" * 64)
    print(f" Full per-case detail written to: {out_path}")
    print("=" * 64 + "\n")
    return out_path
