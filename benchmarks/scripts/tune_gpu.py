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


def _run_bench_batch(
    server: str,
    concurrency: int,
    samples: int = 0,
    rounds: int = 2,
    dataset: str = "librispeech-test-clean",
    dataset_dir: Path | None = None,
    skip_wer: bool = False,
) -> dict:
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
        "--dataset", dataset,
        "--max-samples", str(samples),
    ]
    if skip_wer:
        cmd.append("--skip-wer")
    if dataset_dir is not None:
        cmd.extend(["--dataset-dir", str(dataset_dir)])
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        log.warning(f"Batch bench timed out at c={concurrency}")
        return {"error": "timeout after 300s", "concurrency": concurrency}
    if result.returncode != 0 and not os.path.exists(output):
        log.warning(f"Batch bench failed at c={concurrency}: {result.stderr[-200:]}")
        return {"error": result.stderr[-200:], "concurrency": concurrency}

    with open(output) as f:
        return json.load(f)


async def _run_bench_stream(server: str, concurrency: int, endpoint: str = "/v1/stream",
                            chunk_ms: int = 160, rounds: int = 2,
                            dataset: str = "librispeech-test-clean",
                            dataset_dir: Path | None = None,
                            samples: int = 0,
                            skip_wer: bool = False) -> dict:
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
        "--chunk-ms", str(chunk_ms),
        "--dataset", dataset,
        "--max-samples", str(samples),
    ]
    if skip_wer:
        cmd.append("--skip-wer")
    if dataset_dir is not None:
        cmd.extend(["--dataset-dir", str(dataset_dir)])
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        log.warning(f"Stream bench timed out at c={concurrency}")
        return {"error": "timeout after 600s", "concurrency": concurrency}
    if result.returncode != 0 and not os.path.exists(output):
        log.warning(f"Stream bench failed at c={concurrency}: {result.stderr[-200:]}")
        return {"error": result.stderr[-200:], "concurrency": concurrency}

    with open(output) as f:
        return json.load(f)


def _get_failures(report: dict) -> int:
    """Extract total failures from a benchmark report."""
    if "error" in report:
        return 999
    sweep = report.get("concurrency_sweep", [])
    sweep_failures = sum(s.get("failures", 0) for s in sweep)
    sustained_failures = report.get("sustained_load", {}).get("failures", 0)
    if sweep or "sustained_load" in report:
        return sweep_failures + sustained_failures
    return report.get("summary", {}).get("total_failures", 0)


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


def _get_wer_pct(report: dict) -> float | None:
    """Extract corpus WER percentage from a benchmark report."""
    wer = report.get("wer")
    if not isinstance(wer, dict):
        return None
    value = wer.get("corpus_wer_pct")
    return float(value) if value is not None else None


def _quality_passed(report: dict, wer_threshold_pct: float, skip_wer: bool = False) -> bool:
    """Return whether report quality is acceptable for tuned config selection."""
    if skip_wer:
        return "error" not in report
    wer_pct = _get_wer_pct(report)
    return wer_pct is not None and wer_pct <= wer_threshold_pct


async def binary_search_max_concurrency(
    server: str,
    mode: str,
    p99_threshold: float = 3.0,
    search_range: tuple[int, int] = (1, 1024),
    endpoint: str = "/v1/stream",
    chunk_ms: int = 160,
    dataset: str = "librispeech-test-clean",
    dataset_dir: Path | None = None,
    samples: int = 0,
    wer_threshold_pct: float = 20.0,
    skip_wer: bool = False,
) -> tuple[int, list[dict]]:
    """Binary search for max concurrency with 0 failures and p99 < threshold.

    Returns:
        (max_concurrency, list of trial results)
    """
    lo, hi = search_range
    trials = []
    best = 0

    log.info(f"Binary search: range [{lo}, {hi}], p99 threshold {p99_threshold}s")

    while lo <= hi:
        mid = (lo + hi) // 2
        log.info(f"  Testing c={mid}...")

        if mode == "batch":
            report = _run_bench_batch(
                server,
                mid,
                samples=samples,
                dataset=dataset,
                dataset_dir=dataset_dir,
                skip_wer=skip_wer,
            )
        else:
            report = await _run_bench_stream(
                server,
                mid,
                endpoint=endpoint,
                chunk_ms=chunk_ms,
                dataset=dataset,
                dataset_dir=dataset_dir,
                samples=samples,
                skip_wer=skip_wer,
            )

        failures = _get_failures(report)
        p99 = _get_p99(report)
        rtfx = _get_rtfx(report)
        wer_pct = _get_wer_pct(report)
        quality_passed = _quality_passed(report, wer_threshold_pct, skip_wer=skip_wer)

        trial = {
            "concurrency": mid,
            "failures": failures,
            "p99_s": round(p99, 3),
            "rtfx": rtfx,
            "wer_pct": wer_pct,
            "quality_passed": quality_passed,
            "passed": failures == 0 and p99 < p99_threshold and quality_passed,
        }
        trials.append(trial)

        if trial["passed"]:
            best = mid
            lo = mid + 1
            quality = "skipped" if skip_wer else f"WER={wer_pct}%"
            log.info(
                f"    PASS: failures={failures}, p99={p99:.3f}s, rtfx={rtfx}x, {quality} → search higher"
            )
        else:
            hi = mid - 1
            quality = "skipped" if skip_wer else f"WER={wer_pct}%"
            log.info(f"    FAIL: failures={failures}, p99={p99:.3f}s, {quality} → search lower")

    log.info(f"Max concurrency: {best}")
    return best, trials


async def sweep_batch_params(
    server: str,
    optimal_concurrency: int,
    batch_sizes: list[int] | None = None,
    dataset: str = "librispeech-test-clean",
    dataset_dir: Path | None = None,
    samples: int = 0,
    skip_wer: bool = False,
) -> list[dict]:
    """Benchmark the currently running batch server config once.

    max_batch_size is a server-side startup setting. This script talks to an
    already running server, so it cannot honestly sweep candidate values unless
    an external harness restarts the server per candidate.
    """
    if batch_sizes:
        log.warning(
            "Skipping max_batch_size candidates %s: live server config is unchanged by tune_gpu.py",
            batch_sizes,
        )

    report = _run_bench_batch(
        server,
        optimal_concurrency,
        rounds=2,
        dataset=dataset,
        dataset_dir=dataset_dir,
        samples=samples,
        skip_wer=skip_wer,
    )
    rtfx = _get_rtfx(report)
    failures = _get_failures(report)
    result = {
        "param": "current_server_config",
        "value": "unchanged",
        "concurrency": optimal_concurrency,
        "rtfx": rtfx,
        "failures": failures,
        "note": "max_batch_size requires restarting the server with a candidate config",
    }
    log.info(f"  current server config: rtfx={rtfx}x, failures={failures}")
    return [result]


async def sweep_stream_params(
    server: str,
    optimal_concurrency: int,
    chunk_durations: list[int] | None = None,
    latency_modes: list[str] | None = None,
    endpoint: str = "/v1/stream",
    dataset: str = "librispeech-test-clean",
    dataset_dir: Path | None = None,
    samples: int = 0,
    skip_wer: bool = False,
) -> list[dict]:
    """Sweep client streaming chunk sizes at optimal concurrency.

    stream_model.latency_mode and stream.chunk_duration_ms are server-side
    startup settings. The live benchmark can vary only the client chunk size.
    """
    if chunk_durations is None:
        chunk_durations = [80, 160, 320, 480]
    if latency_modes:
        log.warning(
            "Skipping latency_mode candidates %s: live server config is unchanged by tune_gpu.py",
            latency_modes,
        )

    results = []

    for chunk_ms in chunk_durations:
        log.info(f"Sweep: client_chunk_ms={chunk_ms} at c={optimal_concurrency}")
        report = await _run_bench_stream(
            server,
            optimal_concurrency,
            endpoint=endpoint,
            chunk_ms=chunk_ms,
            rounds=2,
            dataset=dataset,
            dataset_dir=dataset_dir,
            samples=samples,
            skip_wer=skip_wer,
        )
        rtfx = _get_rtfx(report)
        failures = _get_failures(report)
        results.append({
            "param": "client_chunk_ms",
            "value": chunk_ms,
            "concurrency": optimal_concurrency,
            "rtfx": rtfx,
            "failures": failures,
            "note": "server stream.chunk_duration_ms and stream_model.latency_mode were unchanged",
        })
        log.info(f"  client_chunk={chunk_ms}ms: rtfx={rtfx}x, failures={failures}")

    return results


def generate_tuned_config(
    mode: str,
    gpu_name: str,
    max_concurrency: int,
    best_params: dict,
    profile: dict | None = None,
) -> dict:
    """Generate a tuned config YAML structure."""
    if max_concurrency < 1:
        raise ValueError("max_concurrency must be >= 1")

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
            "server": {"host": "0.0.0.0", "port": 8001, "workers": 1},
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
    parser.add_argument("--skip-wer", action="store_true", help="Skip WER quality gate for exploratory tuning only")
    parser.add_argument("--wer-threshold", type=float, default=20.0, help="Max WER percent for tuned candidates")
    parser.add_argument("--dataset", default="librispeech-test-clean", help="Benchmark dataset for tuning runs")
    parser.add_argument("--max-samples", type=int, default=0, help="Max dataset samples per run (0=full dataset)")
    parser.add_argument("--dataset-dir", type=Path, default=None, help="Dataset cache directory")
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=None,
        help="Externally validated batcher.max_batch_size to write into generated batch config",
    )
    parser.add_argument(
        "--stream-chunk-ms",
        type=int,
        default=None,
        help="Externally validated stream.chunk_duration_ms to write into generated stream config",
    )
    parser.add_argument(
        "--latency-mode",
        choices=["80ms", "160ms", "480ms", "1040ms"],
        default=None,
        help="Externally validated stream_model.latency_mode to write into generated stream config",
    )
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
                [
                    e["wav_path"]
                    for e in __import__("benchmarks.datasets.registry", fromlist=["load_dataset"]).load_dataset(
                        args.dataset, cache_dir=args.dataset_dir, max_samples=args.max_samples
                    )
                ],
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
        dataset=args.dataset,
        dataset_dir=args.dataset_dir,
        samples=args.max_samples,
        wer_threshold_pct=args.wer_threshold,
        skip_wer=args.skip_wer,
    )
    if max_c < 1:
        report["error"] = "no concurrency level passed"
        report_path = args.output_dir / f"tuning-report-{args.mode}-{args.gpu_name}.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        raise SystemExit(
            f"No concurrency level passed in range [{args.search_lo}, {args.search_hi}]; "
            f"not writing tuned config. Report: {report_path}"
        )
    report["max_concurrency"] = max_c
    report["binary_search_trials"] = trials

    # Phase 3: Parameter sweep
    best_params = {}
    if not args.skip_sweep:
        log.info("=== Phase 3: Parameter Sweep ===")
        if args.mode == "batch":
            sizes = [8, 16, 32, 64] if not args.quick else [16, 32]
            sweep_results = await sweep_batch_params(
                args.server,
                max_c,
                batch_sizes=sizes,
                dataset=args.dataset,
                dataset_dir=args.dataset_dir,
                samples=args.max_samples,
                skip_wer=args.skip_wer,
            )
            report["param_sweep"] = sweep_results
        else:
            chunks = [80, 160, 320, 480] if not args.quick else [160, 320]
            sweep_results = await sweep_stream_params(
                args.server,
                max_c,
                chunk_durations=chunks,
                endpoint=args.endpoint,
                dataset=args.dataset,
                dataset_dir=args.dataset_dir,
                samples=args.max_samples,
                skip_wer=args.skip_wer,
            )
            report["param_sweep"] = sweep_results
            best = max((r for r in sweep_results if r["failures"] == 0), key=lambda r: r["rtfx"], default=None)
            if best:
                report["recommended_client_params"] = {"chunk_ms": best["value"]}
                log.info(f"Best client chunk: {best['value']}ms (rtfx={best['rtfx']}x)")

    explicit_config_params = {}
    if args.mode == "batch" and args.max_batch_size is not None:
        best_params["max_batch_size"] = args.max_batch_size
        explicit_config_params["max_batch_size"] = args.max_batch_size
    if args.mode == "stream":
        if args.stream_chunk_ms is not None:
            best_params["chunk_duration_ms"] = args.stream_chunk_ms
            explicit_config_params["chunk_duration_ms"] = args.stream_chunk_ms
        if args.latency_mode is not None:
            best_params["latency_mode"] = args.latency_mode
            explicit_config_params["latency_mode"] = args.latency_mode
    if explicit_config_params:
        report["explicit_config_params"] = explicit_config_params

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
    print("| Concurrency | Failures | p99 | WER | RTFx | Result |")
    print("|-------------|----------|-----|-----|------|--------|")
    for t in sorted(trials, key=lambda t: t["concurrency"]):
        result = "PASS" if t["passed"] else "FAIL"
        wer = "skipped" if args.skip_wer else (f"{t['wer_pct']}%" if t["wer_pct"] is not None else "missing")
        print(f"| {t['concurrency']} | {t['failures']} | {t['p99_s']}s | {wer} | {t['rtfx']}x | {result} |")

    if report.get("param_sweep"):
        print()
        print("### Parameter Sweep")
        print("| Parameter | Value | RTFx | Failures |")
        print("|-----------|-------|------|----------|")
        for r in report["param_sweep"]:
            print(f"| {r['param']} | {r['value']} | {r['rtfx']}x | {r['failures']} |")


if __name__ == "__main__":
    asyncio.run(main())
