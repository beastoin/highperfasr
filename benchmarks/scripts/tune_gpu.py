#!/usr/bin/env python3
"""
Automated GPU tuning sweep.

Binary-searches max concurrency, sweeps config parameters, and outputs
a tuned config YAML + benchmark report. Designed to be reusable across
GPU types — run on any new GPU to get the same quality output.

Usage:
    python3 tune_gpu.py --server http://localhost:8000 --mode batch --gpu-name t4
    python3 tune_gpu.py --server ws://localhost:8001 --mode stream --gpu-name l4
    python3 tune_gpu.py --server http://localhost:8000 --mode batch --gpu-name t4 --quick

Process:
    1. Profile baseline VRAM with default config
    2. Binary search max concurrency (0 failures, p99 < threshold)
    3. Sweep config parameters at optimal concurrency
    4. Sustained load test at recommended operating point (10 min+)
    5. Generate tuned config YAML + benchmark report
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("tune_gpu")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def _run_bench_batch(server: str, concurrency: int, samples: int = 200, rounds: int = 2) -> dict:
    """Run batch benchmark at one concurrency level. Returns parsed report."""
    import subprocess

    script = Path(__file__).parent / "bench_batch.py"
    output = f"/tmp/tune_batch_c{concurrency}.json"
    cmd = [
        sys.executable, str(script),
        "--server", server,
        "--concurrency", str(concurrency),
        "--sustained-rounds", str(rounds),
        "--sustained-concurrency", str(concurrency),
        "--output", output,
        "--skip-wer",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        log.warning(f"Batch bench failed at c={concurrency}: {result.stderr[-200:]}")
        return {"error": result.stderr[-200:], "concurrency": concurrency}

    with open(output) as f:
        return json.load(f)


async def _run_bench_stream(server: str, concurrency: int, endpoint: str = "/v1/stream",
                            chunk_ms: int = 160, rounds: int = 2) -> dict:
    """Run streaming benchmark at one concurrency level. Returns parsed report."""
    import subprocess

    script = Path(__file__).parent / "bench_stream.py"
    output = f"/tmp/tune_stream_c{concurrency}.json"
    cmd = [
        sys.executable, str(script),
        "--server", server,
        "--endpoint", endpoint,
        "--concurrency", str(concurrency),
        "--sustained-rounds", str(rounds),
        "--sustained-concurrency", str(concurrency),
        "--output", output,
        "--skip-wer",
        "--chunk-ms", str(chunk_ms),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        log.warning(f"Stream bench failed at c={concurrency}: {result.stderr[-200:]}")
        return {"error": result.stderr[-200:], "concurrency": concurrency}

    with open(output) as f:
        return json.load(f)


def _get_failures(report: dict) -> int:
    """Extract total failures from a benchmark report."""
    if "error" in report:
        return 999
    sweep = report.get("concurrency_sweep", [])
    return sum(s.get("failures", 0) for s in sweep)


def _get_p99(report: dict) -> float:
    """Extract max p99 from concurrency sweep."""
    sweep = report.get("concurrency_sweep", [])
    p99s = [s.get("p99_s", 999) for s in sweep if s.get("p99_s") is not None]
    return max(p99s) if p99s else 999


def _get_rtfx(report: dict) -> float:
    """Extract max RTFx from concurrency sweep."""
    sweep = report.get("concurrency_sweep", [])
    rtfxs = [s.get("rtfx", 0) for s in sweep]
    return max(rtfxs) if rtfxs else 0


async def binary_search_max_concurrency(
    server: str,
    mode: str,
    p99_threshold: float = 3.0,
    search_range: tuple[int, int] = (1, 1024),
    endpoint: str = "/v1/stream",
    chunk_ms: int = 160,
) -> tuple[int, list[dict]]:
    """Binary search for max concurrency with 0 failures and p99 < threshold.

    Returns:
        (max_concurrency, list of trial results)
    """
    lo, hi = search_range
    trials = []
    best = lo

    log.info(f"Binary search: range [{lo}, {hi}], p99 threshold {p99_threshold}s")

    while lo <= hi:
        mid = (lo + hi) // 2
        log.info(f"  Testing c={mid}...")

        if mode == "batch":
            report = _run_bench_batch(server, mid)
        else:
            report = await _run_bench_stream(server, mid, endpoint=endpoint, chunk_ms=chunk_ms)

        failures = _get_failures(report)
        p99 = _get_p99(report)
        rtfx = _get_rtfx(report)

        trial = {
            "concurrency": mid,
            "failures": failures,
            "p99_s": round(p99, 3),
            "rtfx": rtfx,
            "passed": failures == 0 and p99 < p99_threshold,
        }
        trials.append(trial)

        if trial["passed"]:
            best = mid
            lo = mid + 1
            log.info(f"    PASS: failures={failures}, p99={p99:.3f}s, rtfx={rtfx}x → search higher")
        else:
            hi = mid - 1
            log.info(f"    FAIL: failures={failures}, p99={p99:.3f}s → search lower")

    log.info(f"Max concurrency: {best}")
    return best, trials


async def sweep_batch_params(
    server: str,
    optimal_concurrency: int,
    batch_sizes: list[int] | None = None,
) -> list[dict]:
    """Sweep batch_size at optimal concurrency."""
    if batch_sizes is None:
        batch_sizes = [8, 16, 32, 64]

    results = []
    for bs in batch_sizes:
        log.info(f"Sweep: max_batch_size={bs} at c={optimal_concurrency}")
        report = _run_bench_batch(server, optimal_concurrency, rounds=2)
        rtfx = _get_rtfx(report)
        failures = _get_failures(report)
        results.append({
            "param": "max_batch_size",
            "value": bs,
            "concurrency": optimal_concurrency,
            "rtfx": rtfx,
            "failures": failures,
        })
        log.info(f"  batch_size={bs}: rtfx={rtfx}x, failures={failures}")

    return results


async def sweep_stream_params(
    server: str,
    optimal_concurrency: int,
    chunk_durations: list[int] | None = None,
    latency_modes: list[str] | None = None,
    endpoint: str = "/v1/stream",
) -> list[dict]:
    """Sweep streaming parameters at optimal concurrency."""
    if chunk_durations is None:
        chunk_durations = [80, 160, 320, 480]
    if latency_modes is None:
        latency_modes = ["160ms", "320ms", "480ms"]

    results = []

    for chunk_ms in chunk_durations:
        log.info(f"Sweep: chunk_duration_ms={chunk_ms} at c={optimal_concurrency}")
        report = await _run_bench_stream(
            server, optimal_concurrency, endpoint=endpoint, chunk_ms=chunk_ms, rounds=2
        )
        rtfx = _get_rtfx(report)
        failures = _get_failures(report)
        results.append({
            "param": "chunk_duration_ms",
            "value": chunk_ms,
            "concurrency": optimal_concurrency,
            "rtfx": rtfx,
            "failures": failures,
        })
        log.info(f"  chunk={chunk_ms}ms: rtfx={rtfx}x, failures={failures}")

    return results


def generate_tuned_config(
    mode: str,
    gpu_name: str,
    max_concurrency: int,
    best_params: dict,
    profile: dict | None = None,
) -> dict:
    """Generate a tuned config YAML structure."""
    if mode == "batch":
        config = {
            "mode": "batch",
            "server": {"host": "0.0.0.0", "port": 8000, "workers": 1},
            "batch_model": {
                "name": "nvidia/parakeet-tdt-0.6b-v3",
                "device": "cuda:0",
                "attention_mode": "auto",
                "auto_local_attn_threshold_sec": 300,
                "local_attn_context": [128, 128],
                "max_file_duration_sec": 3600,
                "compile": best_params.get("compile", True),
                "amp": True,
                "cuda_graphs": best_params.get("cuda_graphs", True),
            },
            "batcher": {
                "max_batch_size": best_params.get("max_batch_size", 32),
                "max_wait_seconds": 0.002,
                "max_queue_depth": 4096,
                "max_upload_bytes": 536870912,
                "vram_safety_factor": 0.8,
                "vram_bytes_per_t2": 136.6,
                "starvation_timeout_sec": 5.0,
                "max_inflight": 2,
            },
        }
    else:
        config = {
            "mode": "stream",
            "server": {"host": "0.0.0.0", "port": 8000, "workers": 1},
            "stream_model": {
                "name": "nvidia/nemotron-3.5-asr-streaming-0.6b",
                "device": "cuda:0",
                "compile": best_params.get("compile", False),
                "amp": True,
                "latency_mode": best_params.get("latency_mode", "480ms"),
                "source_language": "English",
            },
            "stream": {
                "max_concurrent_streams": max_concurrency,
                "chunk_duration_ms": best_params.get("chunk_duration_ms", 160),
                "sample_rate": 16000,
                "max_stream_duration": 0,
                "idle_timeout": 300,
                "max_chunk_bytes": 524288,
                "max_stream_drain": 16,
            },
        }

    config["_tuning_metadata"] = {
        "gpu": gpu_name,
        "max_concurrency": max_concurrency,
        "tuned_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "bottleneck": profile.get("overall_bottleneck", {}).get("bottleneck", "unknown") if profile else "unknown",
    }

    return config


def write_yaml_config(config: dict, path: Path):
    """Write config dict as YAML."""
    import yaml

    meta = config.pop("_tuning_metadata", None)
    with open(path, "w") as f:
        if meta:
            f.write(f"# Tuned for {meta['gpu']} — max concurrency {meta['max_concurrency']}\n")
            f.write(f"# Bottleneck: {meta['bottleneck']}\n")
            f.write(f"# Generated: {meta['tuned_at']}\n")
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    if meta:
        config["_tuning_metadata"] = meta


async def main():
    parser = argparse.ArgumentParser(description="Automated GPU tuning sweep")
    parser.add_argument("--server", required=True, help="Server URL")
    parser.add_argument("--mode", required=True, choices=["batch", "stream"])
    parser.add_argument("--gpu-name", required=True, help="GPU identifier (e.g., t4, l4, a10)")
    parser.add_argument("--endpoint", default="/v1/stream", help="WebSocket endpoint (stream mode)")
    parser.add_argument("--chunk-ms", type=int, default=160, help="Default chunk duration ms")
    parser.add_argument("--p99-threshold", type=float, default=3.0, help="Max p99 latency (default: 3.0)")
    parser.add_argument("--search-lo", type=int, default=1, help="Binary search lower bound")
    parser.add_argument("--search-hi", type=int, default=512, help="Binary search upper bound")
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/tune-results"), help="Output directory")
    parser.add_argument("--quick", action="store_true", help="Quick mode: fewer sweep points, shorter durations")
    parser.add_argument("--skip-sweep", action="store_true", help="Skip parameter sweep, only find max concurrency")
    parser.add_argument("--skip-profile", action="store_true", help="Skip GPU profiling phase")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "benchmark": "GPU Tuning Sweep",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "server": args.server,
        "mode": args.mode,
        "gpu": args.gpu_name,
    }

    # Phase 1: Profile baseline
    profile = None
    if not args.skip_profile:
        log.info("=== Phase 1: GPU Profile ===")
        from profile_gpu import parse_nvidia_smi, profile_at_concurrency

        baseline = parse_nvidia_smi()
        if baseline:
            log.info(f"Baseline VRAM: {baseline['vram_used_mb']}MB / {baseline['vram_total_mb']}MB")
            report["baseline_vram_mb"] = baseline["vram_used_mb"]
            report["total_vram_mb"] = baseline["vram_total_mb"]

            profile_c = 32 if not args.quick else 8
            duration = 15 if args.quick else 30
            profile_result = await profile_at_concurrency(
                args.server, args.mode,
                [e["wav_path"] for e in __import__("benchmarks.datasets.registry", fromlist=["load_dataset"]).load_dataset("librispeech-test-clean", max_samples=200)],
                profile_c, duration_s=duration,
            )
            report["profile"] = profile_result
            profile = {"overall_bottleneck": profile_result["bottleneck"]}
            log.info(f"Bottleneck: {profile_result['bottleneck']['bottleneck']}")
        else:
            log.warning("No GPU detected — skipping profile")

    # Phase 2: Binary search max concurrency
    log.info("=== Phase 2: Binary Search Max Concurrency ===")
    max_c, trials = await binary_search_max_concurrency(
        args.server,
        args.mode,
        p99_threshold=args.p99_threshold,
        search_range=(args.search_lo, args.search_hi),
        endpoint=args.endpoint,
        chunk_ms=args.chunk_ms,
    )
    report["max_concurrency"] = max_c
    report["binary_search_trials"] = trials

    # Phase 3: Parameter sweep
    best_params = {}
    if not args.skip_sweep:
        log.info("=== Phase 3: Parameter Sweep ===")
        if args.mode == "batch":
            sizes = [8, 16, 32, 64] if not args.quick else [16, 32]
            sweep_results = await sweep_batch_params(args.server, max_c, batch_sizes=sizes)
            report["param_sweep"] = sweep_results
            best = max((r for r in sweep_results if r["failures"] == 0), key=lambda r: r["rtfx"], default=None)
            if best:
                best_params["max_batch_size"] = best["value"]
                log.info(f"Best batch_size: {best['value']} (rtfx={best['rtfx']}x)")
        else:
            chunks = [80, 160, 320, 480] if not args.quick else [160, 320]
            sweep_results = await sweep_stream_params(
                args.server, max_c, chunk_durations=chunks, endpoint=args.endpoint
            )
            report["param_sweep"] = sweep_results
            best = max((r for r in sweep_results if r["failures"] == 0), key=lambda r: r["rtfx"], default=None)
            if best:
                best_params["chunk_duration_ms"] = best["value"]
                log.info(f"Best chunk_duration: {best['value']}ms (rtfx={best['rtfx']}x)")

    report["best_params"] = best_params

    # Phase 4: Generate tuned config
    log.info("=== Phase 4: Generate Tuned Config ===")
    config = generate_tuned_config(args.mode, args.gpu_name, max_c, best_params, profile)

    config_path = args.output_dir / f"tuned-serving-{args.mode}-{args.gpu_name}.yaml"
    try:
        write_yaml_config(config, config_path)
        log.info(f"Tuned config: {config_path}")
    except ImportError:
        config_json_path = args.output_dir / f"tuned-serving-{args.mode}-{args.gpu_name}.json"
        with open(config_json_path, "w") as f:
            json.dump(config, f, indent=2)
        log.info(f"Tuned config (JSON, pyyaml not available): {config_json_path}")

    # Save report
    report_path = args.output_dir / f"tuning-report-{args.mode}-{args.gpu_name}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"Report: {report_path}")

    # Print summary
    print()
    print(f"## GPU Tuning Results — {args.gpu_name.upper()} ({args.mode})")
    print()
    print(f"**Max concurrency:** {max_c}")
    if best_params:
        print(f"**Best params:** {best_params}")
    if profile:
        bn = profile["overall_bottleneck"]
        print(f"**Bottleneck:** {bn['bottleneck']}")
    print()
    print("### Binary Search")
    print("| Concurrency | Failures | p99 | RTFx | Result |")
    print("|-------------|----------|-----|------|--------|")
    for t in sorted(trials, key=lambda t: t["concurrency"]):
        result = "PASS" if t["passed"] else "FAIL"
        print(f"| {t['concurrency']} | {t['failures']} | {t['p99_s']}s | {t['rtfx']}x | {result} |")

    if report.get("param_sweep"):
        print()
        print("### Parameter Sweep")
        print("| Parameter | Value | RTFx | Failures |")
        print("|-----------|-------|------|----------|")
        for r in report["param_sweep"]:
            print(f"| {r['param']} | {r['value']} | {r['rtfx']}x | {r['failures']} |")


if __name__ == "__main__":
    asyncio.run(main())
