#!/usr/bin/env python3
"""Check benchmark reports against baseline registry for regressions."""
import argparse
import json
import sys
from pathlib import Path


def load_registry(path):
    with open(path) as f:
        return json.load(f)


def load_report(path):
    with open(path) as f:
        return json.load(f)


def get_metric(report, metric_name):
    if metric_name == "rtfx":
        perf = report.get("performance", {})
        if "rtfx" in perf:
            return perf["rtfx"]
        sweep = report.get("concurrency_sweep", [])
        if sweep:
            values = [e["rtfx"] for e in sweep if "rtfx" in e]
            return max(values) if values else None
    elif metric_name == "wer_pct":
        q = report.get("quality", {})
        if "wer" in q:
            return q["wer"]
        wer = report.get("wer", {})
        return wer.get("corpus_wer_pct")
    elif metric_name == "p99_s":
        perf = report.get("performance", {})
        if "p99_ms" in perf:
            return perf["p99_ms"] / 1000.0
        sweep = report.get("concurrency_sweep", [])
        if sweep:
            values = [e["p99_s"] for e in sweep if "p99_s" in e]
            return max(values) if values else None
    return None


def check_regression(current_report, baseline_report, rules):
    results = []
    for metric_name, rule in rules.items():
        cur = get_metric(current_report, metric_name)
        base = get_metric(baseline_report, metric_name)
        if cur is None or base is None:
            results.append({
                "metric": metric_name, "status": "skipped",
                "reason": "metric not found",
            })
            continue

        if "max_regression_pct" in rule:
            threshold = rule["max_regression_pct"]
            if metric_name in ("wer_pct", "p99_s"):
                delta_pct = ((cur - base) / max(base, 0.001)) * 100
                passed = delta_pct <= threshold
            else:
                delta_pct = ((base - cur) / max(base, 0.001)) * 100
                passed = delta_pct <= threshold
            results.append({
                "metric": metric_name, "baseline": base, "current": cur,
                "delta_pct": round(delta_pct, 2), "threshold_pct": threshold,
                "passed": passed,
            })

        if "max_absolute_delta_pct" in rule:
            threshold = rule["max_absolute_delta_pct"]
            delta = abs(cur - base)
            passed = delta <= threshold
            results.append({
                "metric": metric_name, "baseline": base, "current": cur,
                "delta": round(delta, 4), "threshold": threshold,
                "passed": passed,
            })
    return results


def has_missing_metrics(results):
    return any(r.get("status") == "skipped" for r in results)


def main():
    parser = argparse.ArgumentParser(
        description="Check benchmark regressions against baselines",
    )
    parser.add_argument("--report", help="Path to current benchmark report JSON")
    parser.add_argument(
        "--registry", default="benchmarks/baselines/registry.json",
    )
    parser.add_argument(
        "--baseline-id", help="Specific baseline ID to compare against",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Validate all baselines exist and are parseable",
    )
    args = parser.parse_args()

    registry = load_registry(args.registry)

    if args.all:
        errors = 0
        for baseline in registry["baselines"]:
            path = Path(baseline["report"])
            if not path.exists():
                print(f"FAIL: baseline report missing: {path}")
                errors += 1
                continue
            try:
                report = load_report(path)
                results = check_regression(report, report, baseline["metrics"])
                skipped = [r for r in results if r.get("status") == "skipped"]
                if skipped:
                    names = ", ".join(r["metric"] for r in skipped)
                    print(f"FAIL: {baseline['id']} -- metrics not extractable: {names}")
                    errors += 1
                print(f"OK: {baseline['id']} ({path})")
            except Exception as e:
                print(f"FAIL: {baseline['id']} -- {e}")
                errors += 1
        return 1 if errors > 0 else 0

    if not args.report:
        print("Error: --report required (or use --all)")
        return 2

    current = load_report(args.report)

    matched_baselines = registry["baselines"]
    if args.baseline_id:
        matched_baselines = [b for b in matched_baselines if b["id"] == args.baseline_id]
        if not matched_baselines:
            print(f"FAIL: unknown baseline-id '{args.baseline_id}'")
            return 1
    elif len(matched_baselines) > 1:
        print(f"WARN: --report compares against all {len(matched_baselines)} baselines; "
              "use --baseline-id to target a specific one")

    comparisons_run = 0
    for baseline in matched_baselines:
        base_path = Path(baseline["report"])
        if not base_path.exists():
            print(f"FAIL: baseline report missing: {base_path}")
            return 1
        base_report = load_report(base_path)
        results = check_regression(current, base_report, baseline["metrics"])

        all_passed = not has_missing_metrics(results) and all(r.get("passed", True) for r in results)
        status = "PASS" if all_passed else "REGRESSION"
        print(f"\n{status}: vs {baseline['id']}")
        for r in results:
            if r.get("status") == "skipped":
                print(f"  {r['metric']}: skipped ({r['reason']})")
            else:
                mark = "PASS" if r["passed"] else "FAIL"
                print(
                    f"  {mark} {r['metric']}: {r.get('baseline')} -> "
                    f"{r.get('current')} "
                    f"(delta: {r.get('delta_pct', r.get('delta'))})"
                )

        comparisons_run += 1
        if not all_passed:
            return 1

    if comparisons_run == 0:
        print("FAIL: no baseline comparisons were run")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
