"""Quality gate evaluation for benchmark reports."""
import json


def load_gates(path):
    with open(path) as f:
        return json.load(f)


def _extract_wer(report):
    """Extract WER % from either live-runner or v1alpha2 format."""
    wer = report.get("wer", {}).get("corpus_wer_pct")
    if wer is not None:
        return wer
    return report.get("quality", {}).get("wer")


def _extract_failure_rate(report):
    """Extract failure rate from either live-runner or v1alpha2 format."""
    rel = report.get("reliability", {})
    if "failure_rate" in rel:
        return rel["failure_rate"]
    sweep = report.get("concurrency_sweep", [])
    total_failures = sum(e.get("failures", 0) for e in sweep)
    total_requests = sum(
        e.get("total", e.get("ok", 0) + e.get("failures", 0)) for e in sweep
    )
    return total_failures / max(total_requests, 1)


def _extract_rtfx(report):
    """Extract peak RTFx from either live-runner or v1alpha2 format."""
    perf = report.get("performance", {})
    if "rtfx" in perf:
        return perf["rtfx"]
    sweep = report.get("concurrency_sweep", [])
    if sweep:
        return max(e.get("rtfx", 0) for e in sweep)
    return None


def _extract_p99_ms(report):
    """Extract max p99 latency in ms from either format."""
    perf = report.get("performance", {})
    if "p99_ms" in perf:
        return perf["p99_ms"]
    sweep = report.get("concurrency_sweep", [])
    if sweep:
        return max(e.get("p99_s", 0) * 1000 for e in sweep)
    return None


def evaluate_gates(report, gates, scenario=None):
    if scenario is None:
        scenario = report.get("scenario", {}).get("mode", "batch")
    gate = gates.get(scenario, gates.get("batch", {}))
    results = []

    wer = _extract_wer(report)
    max_wer = gate.get("max_wer_pct")
    if wer is not None and max_wer is not None:
        results.append({"gate": "max_wer_pct", "threshold": max_wer,
                        "actual": wer, "passed": wer <= max_wer})

    fail_rate = _extract_failure_rate(report)
    max_fail = gate.get("max_failure_rate", 0.0)
    results.append({"gate": "max_failure_rate", "threshold": max_fail,
                    "actual": round(fail_rate, 4), "passed": fail_rate <= max_fail})

    sustained = report.get("sustained_load", {})
    if sustained:
        sus_failures = sustained.get("failures", 0)
        results.append({"gate": "sustained_failures", "threshold": 0,
                        "actual": sus_failures, "passed": sus_failures == 0})

    min_rtfx = gate.get("min_rtfx")
    rtfx = _extract_rtfx(report)
    if min_rtfx is not None and rtfx is not None:
        results.append({"gate": "min_rtfx", "threshold": min_rtfx,
                        "actual": round(rtfx, 2), "passed": rtfx >= min_rtfx})

    max_p99 = gate.get("max_p99_ms")
    p99_ms = _extract_p99_ms(report)
    if max_p99 is not None and p99_ms is not None:
        results.append({"gate": "max_p99_ms", "threshold": max_p99,
                        "actual": round(p99_ms, 1),
                        "passed": p99_ms <= max_p99})

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
