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
    if not sweep:
        return None
    if not all("failures" in e for e in sweep):
        return None
    total_failures = sum(e["failures"] for e in sweep)
    total_requests = sum(
        e.get("total", e.get("ok", 0) + e["failures"]) for e in sweep
    )
    if total_requests == 0:
        return None
    return total_failures / total_requests


def _extract_rtfx(report):
    """Extract peak RTFx from either live-runner or v1alpha2 format."""
    perf = report.get("performance", {})
    if "rtfx" in perf:
        return perf["rtfx"]
    sweep = report.get("concurrency_sweep", [])
    if sweep:
        rtfx_values = [e["rtfx"] for e in sweep if "rtfx" in e]
        return max(rtfx_values) if rtfx_values else None
    return None


def _extract_p99_ms(report):
    """Extract max p99 latency in ms from either format."""
    perf = report.get("performance", {})
    if "p99_ms" in perf:
        return perf["p99_ms"]
    sweep = report.get("concurrency_sweep", [])
    if sweep:
        p99_values = [e["p99_s"] * 1000 for e in sweep if "p99_s" in e]
        return max(p99_values) if p99_values else None
    return None


def evaluate_gates(report, gates, scenario=None):
    if scenario is None:
        scenario = report.get("scenario", {}).get("mode", "batch")
    gate = gates.get(scenario, gates.get("batch", {}))
    results = []

    wer = _extract_wer(report)
    max_wer = gate.get("max_wer_pct")
    if max_wer is not None:
        results.append({"gate": "max_wer_pct", "threshold": max_wer,
                        "actual": wer,
                        "passed": wer is not None and wer <= max_wer})

    fail_rate = _extract_failure_rate(report)
    max_fail = gate.get("max_failure_rate", 0.0)
    results.append({"gate": "max_failure_rate", "threshold": max_fail,
                    "actual": round(fail_rate, 4) if fail_rate is not None else None,
                    "passed": fail_rate is not None and fail_rate <= max_fail})

    sustained = report.get("sustained_load", {})
    if sustained:
        sus_failures = sustained.get("failures", 0)
        results.append({"gate": "sustained_failures", "threshold": 0,
                        "actual": sus_failures, "passed": sus_failures == 0})

    min_rtfx = gate.get("min_rtfx")
    rtfx = _extract_rtfx(report)
    if min_rtfx is not None:
        results.append({"gate": "min_rtfx", "threshold": min_rtfx,
                        "actual": round(rtfx, 2) if rtfx is not None else None,
                        "passed": rtfx is not None and rtfx >= min_rtfx})

    max_p99 = gate.get("max_p99_ms")
    p99_ms = _extract_p99_ms(report)
    if max_p99 is not None:
        results.append({"gate": "max_p99_ms", "threshold": max_p99,
                        "actual": round(p99_ms, 1) if p99_ms is not None else None,
                        "passed": p99_ms is not None and p99_ms <= max_p99})

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
