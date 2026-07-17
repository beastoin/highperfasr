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


def _extract_reference_wer(report):
    """Extract reference WER % from either live-runner or v1alpha2 format."""
    ref_wer = report.get("wer", {}).get("reference_wer_pct")
    if ref_wer is not None:
        return ref_wer
    return report.get("quality", {}).get("reference_wer")


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


def _extract_sustained_duration_s(report):
    """Extract sustained load wall-clock duration in seconds."""
    sustained = report.get("sustained_load", {})
    if "actual_duration_s" in sustained:
        return sustained["actual_duration_s"]
    if "duration_s" in sustained:
        return sustained["duration_s"]
    if "wall_s" in sustained:
        return sustained["wall_s"]
    return report.get("scenario", {}).get("duration_seconds")


def _extract_rt_compliance_pct(report):
    """Extract realtime compliance % from live-runner or v1alpha2 format."""
    streaming = report.get("streaming", {})
    if "rt_compliance_pct" in streaming:
        return streaming["rt_compliance_pct"]
    perf = report.get("performance", {})
    if "rt_compliance_pct" in perf:
        return perf["rt_compliance_pct"]
    sustained = report.get("sustained_load", {})
    if "rt_compliance_pct" in sustained:
        return sustained["rt_compliance_pct"]
    sweep = report.get("concurrency_sweep", [])
    values = [e["rt_compliance_pct"] for e in sweep if "rt_compliance_pct" in e]
    return min(values) if values else None


def _extract_stream_lag_p95_ms(report):
    """Extract worst observed stream lag p95 in milliseconds."""
    streaming = report.get("streaming", {})
    if "lag_p95_ms" in streaming:
        return streaming["lag_p95_ms"]
    perf = report.get("performance", {})
    if "lag_p95_ms" in perf:
        return perf["lag_p95_ms"]
    sustained = report.get("sustained_load", {})
    if "lag_p95_ms" in sustained:
        return sustained["lag_p95_ms"]
    if "lag_p95_s" in sustained:
        return sustained["lag_p95_s"] * 1000
    sweep = report.get("concurrency_sweep", [])
    values = []
    for entry in sweep:
        if "lag_p95_ms" in entry:
            values.append(entry["lag_p95_ms"])
        elif "lag_p95_s" in entry:
            values.append(entry["lag_p95_s"] * 1000)
    return max(values) if values else None


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

    wer_delta_cfg = gate.get("wer_delta")
    if wer_delta_cfg is not None:
        ref_wer = _extract_reference_wer(report)
        if wer is not None and ref_wer is not None:
            delta = wer - ref_wer
            max_delta = max(wer_delta_cfg["max_absolute_pp"], wer_delta_cfg["max_relative_pct"] / 100.0 * ref_wer)
            results.append({"gate": "wer_delta", "threshold": round(max_delta, 3),
                            "actual": round(delta, 3),
                            "passed": delta <= max_delta})
        else:
            results.append({"gate": "wer_delta", "threshold": None,
                            "actual": None, "passed": False})

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

    max_vram_growth = gate.get("max_vram_growth_mb")
    if max_vram_growth is not None:
        vram_growth = report.get("resources", {}).get("vram_growth_mb")
        if vram_growth is None:
            vram_growth = report.get("sustained_load", {}).get("vram_growth_mb")
        results.append({"gate": "max_vram_growth_mb", "threshold": max_vram_growth,
                        "actual": round(vram_growth, 1) if vram_growth is not None else None,
                        "passed": vram_growth is not None and vram_growth <= max_vram_growth})

    min_rt_compliance = gate.get("min_rt_compliance_pct")
    if min_rt_compliance is not None:
        rt_compliance = _extract_rt_compliance_pct(report)
        results.append({"gate": "min_rt_compliance_pct", "threshold": min_rt_compliance,
                        "actual": round(rt_compliance, 1) if rt_compliance is not None else None,
                        "passed": rt_compliance is not None and rt_compliance >= min_rt_compliance})

    min_sustained_duration = gate.get("min_sustained_duration_s")
    if min_sustained_duration is not None:
        sustained_duration = _extract_sustained_duration_s(report)
        results.append({"gate": "min_sustained_duration_s", "threshold": min_sustained_duration,
                        "actual": round(sustained_duration, 1) if sustained_duration is not None else None,
                        "passed": sustained_duration is not None and sustained_duration >= min_sustained_duration})

    max_stream_lag_p95 = gate.get("max_stream_lag_p95_ms")
    if max_stream_lag_p95 is not None:
        stream_lag_p95 = _extract_stream_lag_p95_ms(report)
        results.append({"gate": "max_stream_lag_p95_ms", "threshold": max_stream_lag_p95,
                        "actual": round(stream_lag_p95, 1) if stream_lag_p95 is not None else None,
                        "passed": stream_lag_p95 is not None and stream_lag_p95 <= max_stream_lag_p95})

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
