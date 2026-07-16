"""Quality gate evaluation for benchmark reports."""
import json


def load_gates(path):
    with open(path) as f:
        return json.load(f)


def evaluate_gates(report, gates, scenario=None):
    if scenario is None:
        scenario = report.get("scenario", {}).get("mode", "batch")
    gate = gates.get(scenario, gates.get("batch", {}))
    results = []

    wer = report.get("wer", {}).get("corpus_wer_pct")
    max_wer = gate.get("max_wer_pct")
    if wer is not None and max_wer is not None:
        results.append({"gate": "max_wer_pct", "threshold": max_wer,
                        "actual": wer, "passed": wer <= max_wer})

    sweep = report.get("concurrency_sweep", [])
    total_failures = sum(e.get("failures", 0) for e in sweep)
    max_fail = gate.get("max_failure_rate", 0.0)
    total_requests = sum(
        e.get("total", e.get("ok", 0) + e.get("failures", 0)) for e in sweep
    )
    fail_rate = total_failures / max(total_requests, 1)
    results.append({"gate": "max_failure_rate", "threshold": max_fail,
                    "actual": round(fail_rate, 4), "passed": fail_rate <= max_fail})

    sustained = report.get("sustained_load", {})
    if sustained:
        sus_failures = sustained.get("failures", 0)
        results.append({"gate": "sustained_failures", "threshold": 0,
                        "actual": sus_failures, "passed": sus_failures == 0})

    min_rtfx = gate.get("min_rtfx")
    if min_rtfx is not None and sweep:
        peak_rtfx = max(e.get("rtfx", 0) for e in sweep)
        results.append({"gate": "min_rtfx", "threshold": min_rtfx,
                        "actual": round(peak_rtfx, 2), "passed": peak_rtfx >= min_rtfx})

    max_p99 = gate.get("max_p99_ms")
    if max_p99 is not None and sweep:
        max_observed = max(e.get("p99_s", 0) * 1000 for e in sweep)
        results.append({"gate": "max_p99_ms", "threshold": max_p99,
                        "actual": round(max_observed, 1),
                        "passed": max_observed <= max_p99})

    return {"scenario": scenario, "gates": results,
            "all_passed": all(g["passed"] for g in results)}


def exit_code_for_gates(result):
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Evaluate quality gates for a benchmark report")
    parser.add_argument("--report", required=True, help="Path to benchmark report JSON")
    parser.add_argument("--scenario", default=None, help="Override scenario (batch, streaming-realtime, combined)")
    parser.add_argument("--gates", default=str(Path(__file__).parent.parent / "config" / "quality-gates.json"),
                        help="Path to quality gates config")
    args = parser.parse_args()

    gates = load_gates(args.gates)
    with open(args.report) as f:
        report = json.load(f)

    result = evaluate_gates(report, gates, scenario=args.scenario)
    for g in result["gates"]:
        status = "PASS" if g["passed"] else "FAIL"
        print(f"  {status}: {g['gate']} — threshold={g['threshold']}, actual={g['actual']}")
    print(f"\nOverall: {'PASS' if result['all_passed'] else 'FAIL'} (scenario={result['scenario']})")
    sys.exit(exit_code_for_gates(result))
