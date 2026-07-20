#!/usr/bin/env python3
"""
Benchmark orchestrator — runs batch and/or streaming benchmarks in the
correct order with automatic quality gate evaluation.

Encodes operational knowledge:
- Auto-detects server mode (batch/streaming/both) and correct ports
- Enforces batch-before-streaming ordering (GPU contention causes failures)
- Runs quality gates after each benchmark
- Prints combined summary with pass/fail status
- Forces unbuffered output for nohup compatibility

Usage:
    # Auto-detect mode, run everything
    python3 run_benchmark.py --server http://localhost:8000

    # Quick validation (200 samples, fast)
    python3 run_benchmark.py --server http://localhost:8000 --quick

    # Full publishable run (all samples, 3 trials)
    python3 run_benchmark.py --server http://localhost:8000 --full --trials 3

    # Batch only
    python3 run_benchmark.py --server http://localhost:8000 --mode batch

    # Streaming only (auto-detects correct WS port)
    python3 run_benchmark.py --server http://localhost:8000 --mode streaming
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from preflight import (
    _compose_stream_url_from_batch_url,
    detect_server,
    ensure_unbuffered,
    normalize_server_mode,
    resolve_batch_url,
    resolve_stream_url,
)

ensure_unbuffered()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("orchestrator")


def run_script(script_name, args, label):
    """Run a benchmark script and return (exit_code, output_path)."""
    cmd = [sys.executable, "-u", str(SCRIPTS_DIR / script_name)] + args
    log.info(f"{'='*60}")
    log.info(f"Starting {label}")
    log.info(f"Command: {' '.join(cmd)}")
    log.info(f"{'='*60}")

    t0 = time.monotonic()
    result = subprocess.run(cmd, env={**os.environ, "PYTHONUNBUFFERED": "1"})
    elapsed = time.monotonic() - t0

    status = "PASS" if result.returncode == 0 else "FAIL"
    log.info(f"{label}: {status} ({elapsed:.0f}s)")

    return result.returncode


def run_gates(report_path, scenario):
    """Run quality gates on a report. Returns (passed, gate_results)."""
    gates_path = SCRIPTS_DIR.parent / "config" / "quality-gates.json"
    if not gates_path.exists():
        log.warning(f"No quality gates config at {gates_path}")
        return True, []

    if not os.path.exists(report_path):
        log.error(f"Report not found: {report_path}")
        return False, []

    cmd = [
        sys.executable, str(SCRIPTS_DIR / "gates.py"),
        "--report", str(report_path),
        "--scenario", scenario,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    for line in result.stdout.splitlines():
        log.info(f"  {line}")
    for line in result.stderr.splitlines():
        if "Gate" in line or "PASS" in line or "FAIL" in line:
            log.info(f"  {line}")

    return result.returncode == 0, []


def resolve_benchmark_selection(requested_mode, server_mode):
    """Return (mode, run_batch, run_streaming) for a requested/server mode pair."""
    if requested_mode == "auto":
        mode = normalize_server_mode(server_mode)
        if mode == "unknown":
            log.warning("Could not detect server mode, defaulting to 'both'")
            mode = "both"
    else:
        mode = normalize_server_mode(requested_mode)

    return mode, mode in ("batch", "both"), mode in ("streaming", "both")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark orchestrator — auto-detects server, enforces correct ordering, runs gates"
    )
    parser.add_argument("--server", required=True, help="Server base URL (http://host:port)")
    parser.add_argument(
        "--mode", choices=["auto", "batch", "streaming", "both"],
        default="auto", help="Benchmark mode (default: auto-detect from server)"
    )
    parser.add_argument("--quick", action="store_true",
                        help="Quick validation: 200 samples, fast sweep")
    parser.add_argument("--full", action="store_true",
                        help="Full publishable run: all samples, extended sustained load")
    parser.add_argument("--trials", type=int, default=1,
                        help="Number of trial runs for statistical rigor")
    parser.add_argument("--output-dir", default="/tmp", help="Directory for output reports")
    parser.add_argument("--concurrency", default=None,
                        help="Override concurrency levels (comma-separated)")
    parser.add_argument("--sustained-rounds", type=int, default=None,
                        help="Override sustained load rounds")
    parser.add_argument("--skip-wer", action="store_true", help="Skip WER evaluation")
    parser.add_argument("--dataset", default=None, help="Dataset name override")
    parser.add_argument("--reference-wer-pct", type=float, default=None,
                        help="Reference WER for delta gate")
    args = parser.parse_args()

    http_url = args.server.replace("ws://", "http://").replace("wss://", "https://").rstrip("/")

    # Preflight: detect server
    log.info("=" * 60)
    log.info("PREFLIGHT CHECK")
    log.info("=" * 60)

    server_info = detect_server(http_url)
    if not server_info["healthy"]:
        log.error(f"Server at {http_url} is not healthy. Aborting.")
        return 1

    log.info(f"Server healthy: mode={server_info['mode']}, uptime={server_info.get('uptime_s', '?')}s")
    if server_info["models"]:
        log.info(f"Models loaded: {server_info['models']}")

    # In compose, batch (:8000) and streaming (:8001) are separate services.
    # If auto-detect sees batch-only, probe the compose streaming port.
    if args.mode == "auto" and server_info["mode"] == "batch":
        compose_stream_url = _compose_stream_url_from_batch_url(http_url)
        compose_http = compose_stream_url.replace("ws://", "http://").replace("wss://", "https://")
        stream_probe = detect_server(compose_http)
        if stream_probe["healthy"]:
            log.info(f"Compose streaming service detected at {compose_http}")
            server_info["mode"] = "both"

    # Resolve mode
    mode, run_batch, run_streaming = resolve_benchmark_selection(args.mode, server_info["mode"])
    if args.mode == "auto":
        log.info(f"Auto-detected mode: {mode}")
    if not run_batch and not run_streaming:
        log.error(f"No benchmarks selected for mode '{mode}'")
        return 1

    # Resolve URLs
    batch_url = resolve_batch_url(http_url, server_info)
    stream_url = resolve_stream_url(http_url, server_info)

    # Resolve defaults based on --quick / --full
    if args.quick:
        max_samples = "200"
        batch_concurrency = args.concurrency or "1,8,16,32,64"
        stream_concurrency = args.concurrency or "4,8,16,32,64,128,256"
        batch_sustained = str(args.sustained_rounds or 4)
        stream_sustained = str(args.sustained_rounds or 2)
    elif args.full:
        max_samples = "0"
        batch_concurrency = args.concurrency or "1,8,16,32,64"
        stream_concurrency = args.concurrency or "1,4,8,16,32,64,128,256,512"
        batch_sustained = str(args.sustained_rounds or 8)
        stream_sustained = str(args.sustained_rounds or 4)
    else:
        max_samples = "0"
        batch_concurrency = args.concurrency or "1,8,16,32,64"
        stream_concurrency = args.concurrency or "1,4,8,16,32,64,128,256,512"
        batch_sustained = str(args.sustained_rounds or 4)
        stream_sustained = str(args.sustained_rounds or 4)

    ts = time.strftime("%Y%m%d-%H%M%S")
    batch_output = os.path.join(args.output_dir, f"bench_batch_{ts}.json")
    stream_output = os.path.join(args.output_dir, f"bench_stream_{ts}.json")

    results = {}
    overall_pass = True

    # Run batch FIRST (rule: batch before streaming on shared GPU)
    if run_batch:
        batch_args = [
            "--server", batch_url,
            "--concurrency", batch_concurrency,
            "--sustained-rounds", batch_sustained,
            "--output", batch_output,
            "--max-samples", max_samples,
        ]
        if args.trials > 1:
            batch_args += ["--trials", str(args.trials)]
        if args.skip_wer:
            batch_args.append("--skip-wer")
        if args.dataset:
            batch_args += ["--dataset", args.dataset]
        if args.reference_wer_pct is not None:
            batch_args += ["--reference-wer-pct", str(args.reference_wer_pct)]
        if args.quick:
            batch_args.append("--quick")

        exit_code = run_script("bench_batch.py", batch_args, "Batch Benchmark")
        results["batch"] = {"exit_code": exit_code, "output": batch_output}

        if exit_code != 0:
            log.warning(f"Batch benchmark exited with code {exit_code}")
            overall_pass = False

        if args.quick:
            results["batch"]["gates_pass"] = None
            log.info("Quick mode: skipping quality gates (use --full for gate enforcement)")
        elif os.path.exists(batch_output):
            gates_pass, _ = run_gates(batch_output, "batch")
            results["batch"]["gates_pass"] = gates_pass
            if not gates_pass:
                overall_pass = False
        else:
            results["batch"]["gates_pass"] = False
            overall_pass = False

    # Run streaming SECOND
    if run_streaming:
        stream_args = [
            "--server", stream_url,
            "--concurrency", stream_concurrency,
            "--sustained-rounds", stream_sustained,
            "--output", stream_output,
            "--max-samples", max_samples,
        ]
        if args.trials > 1:
            stream_args += ["--trials", str(args.trials)]
        if args.skip_wer:
            stream_args.append("--skip-wer")
        if args.dataset:
            stream_args += ["--dataset", args.dataset]
        if args.reference_wer_pct is not None:
            stream_args += ["--reference-wer-pct", str(args.reference_wer_pct)]
        if args.quick:
            stream_args.append("--quick")

        exit_code = run_script("bench_stream.py", stream_args, "Streaming Benchmark")
        results["streaming"] = {"exit_code": exit_code, "output": stream_output}

        if exit_code != 0:
            log.warning(f"Streaming benchmark exited with code {exit_code}")
            overall_pass = False

        if args.quick:
            results["streaming"]["gates_pass"] = None
            log.info("Quick mode: skipping quality gates (use --full for gate enforcement)")
        elif os.path.exists(stream_output):
            gates_pass, _ = run_gates(stream_output, "streaming-realtime")
            results["streaming"]["gates_pass"] = gates_pass
            if not gates_pass:
                overall_pass = False
        else:
            results["streaming"]["gates_pass"] = False
            overall_pass = False

    # Combined summary
    log.info("")
    log.info("=" * 60)
    log.info("BENCHMARK SUMMARY")
    log.info("=" * 60)
    log.info(f"Server: {http_url} (mode: {server_info['mode']})")
    log.info(f"Mode: {'quick' if args.quick else 'full' if args.full else 'standard'}")
    log.info("")

    for name, r in results.items():
        gp = r.get("gates_pass")
        status = "SKIP" if gp is None else ("PASS" if gp else "FAIL")
        log.info(f"  {name:12s}: {status}  (report: {r.get('output', 'N/A')})")

        if os.path.exists(r.get("output", "")):
            try:
                with open(r["output"]) as f:
                    report = json.load(f)
                wer = report.get("wer", {}).get("corpus_wer_pct")
                summary = report.get("summary", {})
                if wer is not None:
                    log.info(f"               WER: {wer}%")
                if "peak_rps" in summary:
                    log.info(f"               Peak: {summary['peak_rps']} RPS, {summary['peak_rtfx']}x RTFx at c={summary['peak_concurrency']}")
                if "total_failures" in summary:
                    log.info(f"               Failures: {summary['total_failures']}")
            except Exception:
                pass

    log.info("")
    if overall_pass:
        log.info("OVERALL: PASS — all quality gates passed")
    else:
        log.info("OVERALL: FAIL — one or more quality gates failed")

    log.info(f"Reports: {args.output_dir}/bench_*_{ts}.json")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
