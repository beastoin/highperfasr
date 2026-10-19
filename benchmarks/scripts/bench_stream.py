#!/usr/bin/env python3
"""
Deterministic streaming ASR benchmark with WER.

Loads the frozen LibriSpeech test-clean benchmark corpus, streams audio chunks via WebSocket,
computes WER using wer_utils (Whisper normalization), runs concurrency sweep and
sustained load, and outputs a structured JSON report.

Usage:
    python3 bench_stream.py --server ws://localhost:8001
    python3 bench_stream.py --server ws://localhost:8001 --chunk-ms 480
    python3 bench_stream.py --server ws://localhost:8001 --concurrency 1,4,8,16,32
"""

import argparse
import asyncio
import json
import logging
import os
import struct
import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR.parent.parent))

from preflight import detect_server, resolve_stream_url, log_duration_estimate, log_preflight_summary, ensure_unbuffered

from benchmark_telemetry import JsonlWriter, jsonl_path_from_output, emit_sample, emit_resource, emit_report_checkpoint
from profile_gpu import GPUProfiler, CPUProfiler

ensure_unbuffered()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("bench_stream")

SR = 16000


def ensure_librispeech():
    """Reuse bench_batch's download logic."""
    from bench_batch import ensure_librispeech as _ensure

    _ensure()


def load_references():
    """Load reference transcripts."""
    from bench_batch import load_references as _load

    return _load()


def compute_wer(references, hypotheses):
    """Compute WER using wer_utils."""
    from bench_batch import compute_wer as _compute

    return _compute(references, hypotheses)


def summarize_wer_results(results, refs):
    """Compute report-ready WER fields through bench_batch's shared helper."""
    from bench_batch import summarize_wer_results as _summarize

    return _summarize(results, refs)


def collect_system_info(gpu_override=None, image_tag=None):
    from bench_batch import collect_system_info as _collect

    return _collect(gpu_override=gpu_override, image_tag=image_tag)


def collect_gpu_memory_used_mb():
    """Collect max used GPU memory through bench_batch helper."""
    from bench_batch import collect_gpu_memory_used_mb as _collect

    return _collect()


def reference_wer_pct(mode, dataset_name, override=None, baseline_report=None):
    """Resolve reference-model WER through bench_batch helper."""
    from bench_batch import reference_wer_pct as _reference

    return _reference(mode, dataset_name, override=override, baseline_report=baseline_report)


def manifest_from_wavs(wav_files, refs):
    """Build manifest entries for the legacy LibriSpeech subset path."""
    return [
        {
            "utt_id": Path(wav).stem,
            "wav_path": str(wav),
            "reference": refs.get(Path(wav).stem),
        }
        for wav in wav_files
    ]


def load_dataset_manifest(dataset_name: str, max_samples: int = 0, cache_dir=None):
    """Load streaming benchmark data from the shared corpus registry."""
    from benchmarks.datasets.registry import load_dataset

    manifest = load_dataset(dataset_name, cache_dir=cache_dir, max_samples=max_samples)
    refs = {e["utt_id"]: e["reference"] for e in manifest if e.get("reference")}
    log.info(f"Dataset '{dataset_name}': {len(manifest)} files, {len(refs)} references")
    return manifest, refs


def select_round_robin_entries(manifest, concurrency: int, target_count: int):
    """Select benchmark work using RoundRobinLoader batches.

    Falls back to simple cycling when concurrency exceeds manifest size.
    """
    if concurrency > len(manifest):
        log.warning(f"concurrency {concurrency} > dataset size {len(manifest)}: audio reused within wave")
        return [manifest[i % len(manifest)] for i in range(target_count)]

    from benchmarks.datasets.loader import RoundRobinLoader

    loader = RoundRobinLoader(manifest)
    selected = []
    while len(selected) < target_count:
        selected.extend(loader.next_round(concurrency))
    return selected[:target_count]


async def stream_file(ws_url, wav_path, chunk_ms, semaphore):
    """Stream one file via WebSocket, return transcript and timing."""
    import websockets

    chunk_samples = int(SR * chunk_ms / 1000)
    chunk_bytes = chunk_samples * 2
    utt_id = Path(wav_path).stem

    async with semaphore:
        t0 = time.monotonic()
        ttfb = None
        try:
            with open(wav_path, "rb") as f:
                f.read(44)  # skip WAV header
                raw = f.read()

            async with websockets.connect(ws_url, max_size=10 * 1024 * 1024) as ws:
                config = json.dumps({"format": "pcm_s16le", "sample_rate": SR, "language": "en"})
                await ws.send(config)
                await asyncio.wait_for(ws.recv(), timeout=5)

                offset = 0
                final_parts = []
                while offset < len(raw):
                    chunk = raw[offset : offset + chunk_bytes]
                    await ws.send(chunk)
                    offset += chunk_bytes
                    await asyncio.sleep(chunk_ms / 1000.0)

                    try:
                        while True:
                            msg = await asyncio.wait_for(ws.recv(), timeout=0.01)
                            resp = json.loads(msg)
                            if ttfb is None and (resp.get("partial_transcript") or resp.get("final_transcript")):
                                ttfb = time.monotonic() - t0
                            if resp.get("final_transcript"):
                                final_parts.append(resp["final_transcript"])
                    except asyncio.TimeoutError:
                        pass

                await ws.send(json.dumps({"action": "close"}))

                try:
                    while True:
                        msg = await asyncio.wait_for(ws.recv(), timeout=5)
                        resp = json.loads(msg)
                        if ttfb is None and (resp.get("partial_transcript") or resp.get("final_transcript") or resp.get("final_text")):
                            ttfb = time.monotonic() - t0
                        if resp.get("final_transcript"):
                            final_parts.append(resp["final_transcript"])
                        if resp.get("final_text"):
                            final_parts = [resp["final_text"]]
                        if resp.get("done") or resp.get("status") == "closed":
                            break
                except (asyncio.TimeoutError, Exception):
                    pass

                elapsed = time.monotonic() - t0
                text = " ".join(final_parts).strip()
                audio_dur = len(raw) / (SR * 2)
                return {
                    "utt_id": utt_id,
                    "text": text,
                    "elapsed": elapsed,
                    "audio_dur": audio_dur,
                    "rtfx": audio_dur / elapsed if elapsed > 0 else 0,
                    "ttfb_s": round(ttfb, 4) if ttfb else None,
                    "status": "ok",
                }
        except Exception as e:
            return {
                "utt_id": utt_id,
                "error": str(e)[:200],
                "elapsed": time.monotonic() - t0,
                "status": "error",
            }


async def run_sweep(ws_url, manifest, concurrency, chunk_ms, repeat=1, target_count=None, on_result=None):
    """Run one concurrency level for streaming.

    Args:
        on_result: optional callback(result_dict, index) called as each stream completes.
    """
    if target_count is None:
        target_count = len(manifest) * repeat
    files = [Path(e["wav_path"]) for e in select_round_robin_entries(manifest, concurrency, target_count)]
    sem = asyncio.Semaphore(concurrency)
    t0 = time.monotonic()
    collected = []
    idx_counter = [0]

    async def _run_one(f):
        r = await stream_file(ws_url, f, chunk_ms, sem)
        i = idx_counter[0]
        idx_counter[0] += 1
        collected.append(r)
        if on_result:
            try:
                on_result(r, i)
            except Exception as exc:
                log.warning("on_result callback failed for sample %d: %s", i, exc)
        return r

    tasks = [_run_one(f) for f in files]
    await asyncio.gather(*tasks)
    wall = time.monotonic() - t0
    return collected, wall


def summarize_sweep(results, wall_time, concurrency):
    """Compute summary for one streaming concurrency level."""
    ok = [r for r in results if r["status"] == "ok"]
    failed = [r for r in results if r["status"] == "error"]
    latencies = sorted(r["elapsed"] for r in ok)
    total_audio = sum(r.get("audio_dur", 0) for r in ok)
    lags = sorted(max(0, r.get("elapsed", 0) - r.get("audio_dur", 0)) for r in ok)

    summary = {
        "concurrency": concurrency,
        "total": len(results),
        "ok": len(ok),
        "failures": len(failed),
        "wall_s": round(wall_time, 2),
        "rps": round(len(ok) / wall_time, 2) if wall_time > 0 else 0,
        "rtfx": round(total_audio / wall_time, 2) if wall_time > 0 else 0,
        "rtf": round(wall_time / total_audio, 3) if total_audio > 0 else 0,
        "total_audio_s": round(total_audio, 1),
        "sess_per_min": round(len(ok) / (wall_time / 60), 1) if wall_time > 0 else 0,
    }
    if latencies:
        import statistics
        summary["p50_s"] = round(latencies[len(latencies) // 2], 3)
        summary["p99_s"] = round(latencies[int(len(latencies) * 0.99)], 3)
        summary["min_s"] = round(latencies[0], 3)
        summary["max_s"] = round(latencies[-1], 3)
        summary["mean_s"] = round(statistics.mean(latencies), 3)
        if len(latencies) > 1:
            summary["stddev_s"] = round(statistics.stdev(latencies), 3)

    if lags:
        summary["lag_p50_s"] = round(lags[int((len(lags) - 1) * 0.50)], 3)
        summary["lag_p95_s"] = round(lags[int((len(lags) - 1) * 0.95)], 3)
        summary["lag_p99_s"] = round(lags[int((len(lags) - 1) * 0.99)], 3)
        realtime_ok = sum(1 for lag in lags if lag <= 1.0)
        summary["rt_compliance_pct"] = round(realtime_ok / len(lags) * 100, 1)

    ttfbs = sorted(r["ttfb_s"] for r in ok if r.get("ttfb_s") is not None)
    if ttfbs:
        summary["ttfb_p50_s"] = round(ttfbs[len(ttfbs) // 2], 4)
        summary["ttfb_p95_s"] = round(ttfbs[int(len(ttfbs) * 0.95)], 4)
        summary["ttfb_p99_s"] = round(ttfbs[int(len(ttfbs) * 0.99)], 4)

    return summary


def load_baseline(path):
    """Load a previous benchmark report for smart mode comparison."""
    try:
        with open(path) as f:
            baseline = json.load(f)
        sweep = {s["concurrency"]: s for s in baseline.get("concurrency_sweep", [])}
        log.info(f"Loaded baseline from {path}: {len(sweep)} concurrency levels")
        return baseline, sweep
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        log.warning(f"Could not load baseline from {path}: {e}")
        return None, {}


def check_regression(current, baseline_level, metric="rtfx", threshold=0.20):
    """Check if current result regressed beyond threshold vs baseline."""
    if not baseline_level:
        return False, "no baseline"
    cur_val = current.get(metric, 0)
    base_val = baseline_level.get(metric, 0)
    if base_val == 0:
        return False, "baseline zero"
    change = (cur_val - base_val) / base_val
    regressed = change < -threshold
    return regressed, f"{metric}: {base_val} -> {cur_val} ({change:+.1%})"


async def main():
    parser = argparse.ArgumentParser(description="Deterministic streaming ASR benchmark with WER")
    parser.add_argument("--server", default="ws://localhost:8001", help="Server WebSocket base URL")
    parser.add_argument("--chunk-ms", type=int, default=160, help="Chunk duration in ms (default: 160)")
    parser.add_argument(
        "--concurrency",
        default="1,4,8,16,32",
        help="Comma-separated concurrency levels (default: 1,4,8,16,32)",
    )
    parser.add_argument("--sustained-rounds", type=int, default=4, help="Sustained load rounds (default: 4)")
    parser.add_argument(
        "--sustained-concurrency", type=int, default=32, help="Sustained load concurrency (default: 32)"
    )
    parser.add_argument("--warmup", type=int, default=10, help="Warmup streams (default: 10)")
    parser.add_argument("--output", default="/tmp/bench_stream_report.json", help="Output JSON path")
    parser.add_argument("--skip-wer", action="store_true", help="Skip WER computation")
    parser.add_argument("--endpoint", default="/v1/stream", help="WebSocket endpoint path (default: /v1/stream)")
    parser.add_argument("--smart", action="store_true", help="Smart mode: sweep high-to-low, early-stop on match")
    parser.add_argument("--baseline", default=None, help="Path to previous report JSON for smart comparison")
    parser.add_argument("--reference-wer-pct", type=float, default=None,
                        help="Reference model WER %% for WER delta gate")
    parser.add_argument("--dataset", default=None,
                        help="Use multi-corpus dataset (e.g., 'librispeech-test-clean', 'all')")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Max samples from dataset (0=all)")
    parser.add_argument("--dataset-dir", type=Path, default=None, help="Dataset cache directory")
    parser.add_argument("--trials", type=int, default=1,
                        help="Number of trial runs for statistical rigor (default: 1)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick validation: 200 samples, c=4-256 sweep (use full corpus for publishable results)")
    parser.add_argument("--publish", default=None, metavar="DIR",
                        help="Publish results to DIR/<auto-named>/ using GPU-mode-timestamp naming")
    parser.add_argument("--gpu", default=None, metavar="NAME",
                        help="Override GPU name when nvidia-smi is unavailable (e.g. remote benchmarking)")
    parser.add_argument("--image-tag", default=None, metavar="TAG",
                        help="Image version tag for reports (e.g. v0.3.0). Falls back to sha-<HEAD>")
    args = parser.parse_args()

    if args.quick:
        if args.max_samples == 0:
            args.max_samples = 200
        log.info("Quick mode: 200 samples for fast validation (use --max-samples 0 for publishable results)")

    levels = [int(x) for x in args.concurrency.split(",")]

    server_info = detect_server(args.server)
    log_preflight_summary(server_info, "streaming")
    server_base = resolve_stream_url(args.server, server_info)
    ws_url = f"{server_base}{args.endpoint}"

    baseline_report, baseline_sweep = None, {}
    if args.baseline:
        baseline_report, baseline_sweep = load_baseline(args.baseline)
    elif args.smart:
        default_baseline = args.output
        if os.path.exists(default_baseline):
            baseline_report, baseline_sweep = load_baseline(default_baseline)

    if args.smart:
        levels = sorted(levels, reverse=True)
        log.info("Smart mode: sweeping high-to-low with early-stop")

    log.info("=== Deterministic Streaming ASR Benchmark ===")
    log.info(f"Server: {args.server}")
    log.info(f"Chunk: {args.chunk_ms}ms, Concurrency levels: {levels}")

    dataset_name = args.dataset or "librispeech-test-clean"
    manifest, refs = load_dataset_manifest(dataset_name, max_samples=args.max_samples, cache_dir=args.dataset_dir)
    log.info(f"Using {len(manifest)} WAV files")

    from bench_batch import get_wav_duration
    total_audio_s = 0
    for e in manifest:
        if isinstance(e, dict):
            dur = e.get("duration_s", 0)
            if dur == 0 and e.get("wav_path"):
                try:
                    dur = get_wav_duration(e["wav_path"])
                    e["duration_s"] = dur
                except Exception:
                    pass
            total_audio_s += dur
        elif hasattr(e, "stat"):
            try:
                total_audio_s += get_wav_duration(str(e))
            except Exception:
                pass
    avg_file_duration_s = round(total_audio_s / len(manifest), 2) if manifest else 0

    report = {
        "schema_version": "v1alpha2-live",
        "benchmark": "Streaming ASR Benchmark",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "server": args.server,
        "chunk_ms": args.chunk_ms,
        "samples": len(manifest),
        "dataset": dataset_name,
        "avg_file_duration_s": avg_file_duration_s,
        "total_audio_s": round(total_audio_s, 1),
        "smart_mode": args.smart,
        "system": collect_system_info(gpu_override=args.gpu, image_tag=args.image_tag),
        "command": " ".join(sys.argv),
    }
    vram_start_mb = collect_gpu_memory_used_mb()

    jsonl_path = jsonl_path_from_output(args.output)
    jsonl_writer = JsonlWriter(jsonl_path)
    jsonl_writer.emit("run_start", mode="streaming", dataset=dataset_name,
                      samples=len(manifest), avg_file_duration_s=avg_file_duration_s,
                      chunk_ms=args.chunk_ms)

    # Warmup
    warmup_c = min(4, len(manifest), max(args.warmup, 1))
    log.info(f"Warmup: {args.warmup} streams at c={warmup_c}...")
    await run_sweep(ws_url, manifest, concurrency=warmup_c, chunk_ms=args.chunk_ms, target_count=args.warmup)

    # WER evaluation (c=1)
    if not args.skip_wer:
        total_audio = sum(e.get("duration_s", 5.0) for e in manifest)
        log_duration_estimate(len(manifest), total_audio, mode="streaming")
        wer_results, _ = await run_sweep(
            ws_url, manifest, concurrency=1, chunk_ms=args.chunk_ms, target_count=len(manifest),
            on_result=lambda r, i: emit_sample(jsonl_writer, r, "wer", 1, i),
        )
        wer_summary = summarize_wer_results(wer_results, refs)

        if wer_summary:
            report["wer"] = {
                "corpus_wer_pct": wer_summary["corpus_wer_pct"],
                "c1_corpus_wer_pct": wer_summary["corpus_wer_pct"],
                "samples_evaluated": wer_summary["samples_evaluated"],
                "samples_failed": wer_summary.get("samples_failed", 0),
                "normalization": wer_summary["normalization"],
            }
            ref_wer = reference_wer_pct(
                "streaming-realtime", report["dataset"],
                override=args.reference_wer_pct,
                baseline_report=baseline_report,
            )
            if ref_wer is not None:
                report["wer"]["reference_wer_pct"] = ref_wer
            log.info(f"WER: {report['wer']['corpus_wer_pct']:.2f}%")

            if report["wer"]["samples_failed"] > 0:
                log.error(
                    f"WER incomplete: {report['wer']['samples_failed']} of "
                    f"{len(manifest)} eval items failed at c=1"
                )

            if args.smart and baseline_report and "wer" in baseline_report:
                base_wer = baseline_report["wer"]["corpus_wer_pct"]
                cur_wer = report["wer"]["corpus_wer_pct"]
                log.info(f"  vs baseline: {base_wer}% -> {cur_wer}% (delta {cur_wer - base_wer:+.2f}%)")

    # Concurrency sweep
    emit_report_checkpoint(jsonl_writer, report, "before_sweep")
    log.info("Concurrency sweep...")
    sweep_results = []
    consecutive_matches = 0
    for c in levels:
        log.info(f"  c={c}...")
        jsonl_writer.emit("level_start", phase="sweep", concurrency=c)
        gpu_profiler = GPUProfiler(interval_s=1.0,
                                   on_sample=lambda s, _c=c: emit_resource(jsonl_writer, s, None, "sweep", _c))
        cpu_profiler = CPUProfiler(interval_s=1.0,
                                   on_sample=lambda v, _c=c: emit_resource(jsonl_writer, None, v, "sweep", _c))
        gpu_profiler.start()
        cpu_profiler.start()
        try:
            results, wall = await run_sweep(
                ws_url, manifest, concurrency=c, chunk_ms=args.chunk_ms, target_count=max(len(manifest), c),
                on_result=lambda r, i, _c=c: emit_sample(jsonl_writer, r, "sweep", _c, i),
            )
        finally:
            gpu_samples = gpu_profiler.stop()
            cpu_samples = cpu_profiler.stop()
        summary = summarize_sweep(results, wall, c)
        gpu_summary = GPUProfiler.summarize(gpu_samples)
        cpu_summary = CPUProfiler.summarize(cpu_samples)
        summary["client_gpu_profile"] = gpu_summary
        summary["client_cpu_profile"] = cpu_summary
        if "error" not in gpu_summary:
            summary["client_bottleneck"] = GPUProfiler.classify_bottleneck(gpu_summary)
        ttfb_str = f"TTFB p50={summary.get('ttfb_p50_s', 'N/A')}s p99={summary.get('ttfb_p99_s', 'N/A')}s"
        rt_str = f"RT={summary.get('rt_compliance_pct', 'N/A')}%"
        gpu_info = ""
        if "error" not in gpu_summary:
            gpu_info = f", GPU={gpu_summary['gpu_util_pct']['mean']}%, VRAM={gpu_summary['vram_mb']['max']}MB"
        jsonl_writer.emit("level_end", phase="sweep", concurrency=c,
                          rtfx=summary["rtfx"], sess_per_min=summary["sess_per_min"],
                          failures=summary["failures"],
                          ttfb_p50_s=summary.get("ttfb_p50_s"),
                          ttfb_p99_s=summary.get("ttfb_p99_s"),
                          rt_compliance_pct=summary.get("rt_compliance_pct"),
                          client_gpu_util_avg=gpu_summary.get("gpu_util_pct", {}).get("mean"),
                          client_vram_peak_mb=gpu_summary.get("vram_mb", {}).get("max"))
        sweep_results.append(summary)
        report["concurrency_sweep"] = sweep_results
        emit_report_checkpoint(jsonl_writer, report, f"after_level_c{c}")
        log.info(
            f"    RTFx={summary['rtfx']}, sess/min={summary['sess_per_min']}, "
            f"failures={summary['failures']}, {ttfb_str}, {rt_str}{gpu_info}"
        )

        if args.smart and baseline_sweep:
            bl = baseline_sweep.get(c)
            reg_rtfx, msg_rtfx = check_regression(summary, bl, "rtfx")
            reg_sess, msg_sess = check_regression(summary, bl, "sess_per_min")
            log.info(f"    vs baseline: {msg_rtfx} | {msg_sess}")

            if not reg_rtfx and not reg_sess and summary["failures"] == 0:
                consecutive_matches += 1
                if consecutive_matches >= 2 and len(sweep_results) >= 2:
                    remaining = [l for l in levels if l not in {s["concurrency"] for s in sweep_results}]
                    if remaining:
                        log.info(f"  Smart early-stop: 2 consecutive levels match baseline, skipping {remaining}")
                        report["smart_skipped"] = remaining
                        break
            else:
                consecutive_matches = 0

    sweep_results.sort(key=lambda s: s["concurrency"])
    report["concurrency_sweep"] = sweep_results
    zero_fail_levels = [s for s in sweep_results if s["failures"] == 0]
    report["max_streams"] = max((s["concurrency"] for s in zero_fail_levels), default=0)
    emit_report_checkpoint(jsonl_writer, report, "after_sweep")

    # Sustained load
    sc = args.sustained_concurrency
    rounds = args.sustained_rounds
    log.info(f"Sustained load: c={sc}, {rounds} rounds...")
    jsonl_writer.emit("level_start", phase="sustained", concurrency=sc, rounds=rounds)
    sus_gpu = GPUProfiler(interval_s=1.0,
                          on_sample=lambda s: emit_resource(jsonl_writer, s, None, "sustained", sc))
    sus_cpu = CPUProfiler(interval_s=1.0,
                          on_sample=lambda v: emit_resource(jsonl_writer, None, v, "sustained", sc))
    sus_gpu.start()
    sus_cpu.start()
    try:
        sustained_results, sustained_wall = await run_sweep(
            ws_url, manifest, concurrency=sc, chunk_ms=args.chunk_ms, repeat=rounds,
            on_result=lambda r, i: emit_sample(jsonl_writer, r, "sustained", sc, i),
        )
    finally:
        sus_gpu_samples = sus_gpu.stop()
        sus_cpu_samples = sus_cpu.stop()
    sustained_summary = summarize_sweep(sustained_results, sustained_wall, sc)
    sustained_summary["rounds"] = rounds
    sustained_summary["client_gpu_profile"] = GPUProfiler.summarize(sus_gpu_samples)
    sustained_summary["client_cpu_profile"] = CPUProfiler.summarize(sus_cpu_samples)
    if "error" not in sustained_summary["client_gpu_profile"]:
        sustained_summary["client_bottleneck"] = GPUProfiler.classify_bottleneck(sustained_summary["client_gpu_profile"])
    report["sustained_load"] = sustained_summary
    jsonl_writer.emit("level_end", phase="sustained", concurrency=sc,
                      rtfx=sustained_summary["rtfx"], sess_per_min=sustained_summary["sess_per_min"],
                      failures=sustained_summary["failures"],
                      ttfb_p50_s=sustained_summary.get("ttfb_p50_s"),
                      ttfb_p99_s=sustained_summary.get("ttfb_p99_s"),
                      rt_compliance_pct=sustained_summary.get("rt_compliance_pct"))
    if not args.skip_wer:
        load_wer_summary = summarize_wer_results(sustained_results, refs)
        if load_wer_summary:
            report.setdefault("wer", {})
            report["wer"]["max_load_corpus_wer_pct"] = load_wer_summary["corpus_wer_pct"]
            report["wer"]["max_load_samples_evaluated"] = load_wer_summary["samples_evaluated"]
            sustained_summary["wer_pct"] = load_wer_summary["corpus_wer_pct"]
            log.info(
                f"Max-load WER: {load_wer_summary['corpus_wer_pct']:.2f}% "
                f"at c={sc} ({load_wer_summary['samples_evaluated']} samples)"
            )
    if "rt_compliance_pct" in sustained_summary or "lag_p95_s" in sustained_summary:
        report["streaming"] = {}
        if "rt_compliance_pct" in sustained_summary:
            report["streaming"]["rt_compliance_pct"] = sustained_summary["rt_compliance_pct"]
        if "lag_p95_s" in sustained_summary:
            report["streaming"]["lag_p95_ms"] = sustained_summary["lag_p95_s"] * 1000
    vram_end_mb = collect_gpu_memory_used_mb()
    if vram_start_mb is not None or vram_end_mb is not None:
        report["resources"] = {
            "vram_start_mb": vram_start_mb,
            "vram_end_mb": vram_end_mb,
            "vram_growth_mb": (
                vram_end_mb - vram_start_mb if vram_start_mb is not None and vram_end_mb is not None else None
            ),
            "vram_peak_mb": max(v for v in (vram_start_mb, vram_end_mb) if v is not None),
        }

    emit_report_checkpoint(jsonl_writer, report, "after_sustained")

    if args.smart and baseline_report and "sustained_load" in baseline_report:
        bl_s = baseline_report["sustained_load"]
        log.info(
            f"  Sustained vs baseline: "
            f"RTFx {bl_s.get('rtfx', '?')} -> {sustained_summary['rtfx']}, "
            f"sess/min {bl_s.get('sess_per_min', '?')} -> {sustained_summary['sess_per_min']}"
        )

    # Print markdown
    print()
    print("## Streaming Benchmark Results")
    if args.smart:
        print("**(smart mode: high-to-low sweep with early-stop)**")
    print()
    if "wer" in report:
        print(
            f"**WER:** {report['wer']['corpus_wer_pct']}% "
            f"({report['wer']['samples_evaluated']} samples, "
            f"{report['wer']['normalization']} normalization)"
        )
        print()
    print("### Concurrency Sweep")
    print("| c | TTFB p50 | TTFB p99 | RT Compliance | Failures |")
    print("|---|----------|----------|---------------|----------|")
    for s in sweep_results:
        ttfb_p50 = f"{s['ttfb_p50_s']}s" if s.get('ttfb_p50_s') is not None else "N/A"
        ttfb_p99 = f"{s['ttfb_p99_s']}s" if s.get('ttfb_p99_s') is not None else "N/A"
        rt = f"{s['rt_compliance_pct']}%" if s.get('rt_compliance_pct') is not None else "N/A"
        print(
            f"| {s['concurrency']} | {ttfb_p50} | {ttfb_p99} | {rt} | {s['failures']} |"
        )
    if report.get("smart_skipped"):
        print(f"\n*Smart early-stop: skipped c={report['smart_skipped']} (matched baseline)*")
    print()
    print("### Sustained Load")
    print(f"| Metric | Value |")
    print(f"|--------|-------|")
    print(f"| Concurrency | {sustained_summary['concurrency']} |")
    print(f"| RTFx | {sustained_summary['rtfx']}x |")
    print(f"| sess/min | {sustained_summary['sess_per_min']} |")
    print(f"| Failures | {sustained_summary['failures']} |")

    # Multi-trial aggregation
    if args.trials > 1:
        from benchmarks.scripts.stats import summarize_trials

        peak = max(sweep_results, key=lambda x: x["rtfx"])
        all_peak_rtfx = [peak["rtfx"]]
        all_sustained_rtfx = [sustained_summary["rtfx"]]
        all_sustained_sess = [sustained_summary["sess_per_min"]]
        trial_total_failures = sum(s["failures"] for s in sweep_results) + sustained_summary["failures"]
        all_trial_fail_by_c = {}
        for s in sweep_results:
            all_trial_fail_by_c[s["concurrency"]] = s["failures"]

        for trial in range(2, args.trials + 1):
            log.info(f"=== Trial {trial}/{args.trials} ===")
            jsonl_writer.emit("trial_start", trial=trial)
            t_sweep = []
            for c in levels:
                jsonl_writer.emit("level_start", phase="sweep", concurrency=c, trial=trial)
                t_gpu = GPUProfiler(interval_s=1.0,
                                    on_sample=lambda s, _c=c: emit_resource(jsonl_writer, s, None, "sweep", _c))
                t_cpu = CPUProfiler(interval_s=1.0,
                                    on_sample=lambda v, _c=c: emit_resource(jsonl_writer, None, v, "sweep", _c))
                t_gpu.start()
                t_cpu.start()
                try:
                    results, wall = await run_sweep(
                        ws_url, manifest, concurrency=c, chunk_ms=args.chunk_ms, target_count=max(len(manifest), c),
                        on_result=lambda r, i, _c=c: emit_sample(jsonl_writer, r, "sweep", _c, i),
                    )
                finally:
                    t_gpu.stop()
                    t_cpu.stop()
                t_sweep.append(summarize_sweep(results, wall, c))
                jsonl_writer.emit("level_end", phase="sweep", concurrency=c, trial=trial,
                                  rtfx=t_sweep[-1]["rtfx"], failures=t_sweep[-1]["failures"])
            t_peak = max(t_sweep, key=lambda x: x["rtfx"])
            all_peak_rtfx.append(t_peak["rtfx"])

            jsonl_writer.emit("level_start", phase="sustained", concurrency=sc, trial=trial)
            ts_gpu = GPUProfiler(interval_s=1.0,
                                 on_sample=lambda s: emit_resource(jsonl_writer, s, None, "sustained", sc))
            ts_cpu = CPUProfiler(interval_s=1.0,
                                 on_sample=lambda v: emit_resource(jsonl_writer, None, v, "sustained", sc))
            ts_gpu.start()
            ts_cpu.start()
            try:
                t_sus_results, t_sus_wall = await run_sweep(
                    ws_url, manifest, concurrency=sc, chunk_ms=args.chunk_ms, repeat=rounds,
                    on_result=lambda r, i: emit_sample(jsonl_writer, r, "sustained", sc, i),
                )
            finally:
                ts_gpu.stop()
                ts_cpu.stop()
            t_sus = summarize_sweep(t_sus_results, t_sus_wall, sc)
            all_sustained_rtfx.append(t_sus["rtfx"])
            all_sustained_sess.append(t_sus["sess_per_min"])
            for s in t_sweep:
                all_trial_fail_by_c[s["concurrency"]] = all_trial_fail_by_c.get(s["concurrency"], 0) + s["failures"]
            t_failures = sum(s["failures"] for s in t_sweep) + t_sus["failures"]
            trial_total_failures += t_failures
            jsonl_writer.emit("level_end", phase="sustained", concurrency=sc, trial=trial,
                              rtfx=t_sus["rtfx"], failures=t_sus["failures"])
            jsonl_writer.emit("trial_end", trial=trial, failures=t_failures)
            log.info(f"  Trial {trial}: peak RTFx={t_peak['rtfx']}, sustained={t_sus['rtfx']}, failures={t_failures}")

        report["trials"] = {
            "count": args.trials,
            "peak_rtfx": summarize_trials(all_peak_rtfx),
            "sustained_rtfx": summarize_trials(all_sustained_rtfx),
            "sustained_sess_per_min": summarize_trials(all_sustained_sess),
        }
        log.info(f"Trials summary: peak RTFx={report['trials']['peak_rtfx']}")
        zero_fail_levels = [c for c, f in all_trial_fail_by_c.items() if f == 0]
        report["max_streams"] = max(zero_fail_levels, default=0)

    # Quality gate evaluation (skipped in quick mode — gates require full sustained load)
    gates_path = Path(__file__).parent.parent / "config" / "quality-gates.json"
    if gates_path.exists() and not args.quick:
        from benchmarks.scripts.gates import load_gates, evaluate_gates

        gates = load_gates(str(gates_path))
        gate_result = evaluate_gates(report, gates, scenario="streaming-realtime")
        report["quality_gates"] = gate_result
        for g in gate_result["gates"]:
            status = "PASS" if g["passed"] else "FAIL"
            log.info(f"  Gate {status}: {g['gate']} — threshold={g['threshold']}, actual={g['actual']}")
    elif args.quick:
        log.info("Quick mode: skipping quality gates (use full corpus for gate enforcement)")

    total_failures = sum(s["failures"] for s in sweep_results) + sustained_summary["failures"]
    total_failures = trial_total_failures if args.trials > 1 else total_failures
    report["summary"] = {"total_failures": total_failures, "max_streams": report.get("max_streams", 0)}
    emit_report_checkpoint(jsonl_writer, report, "final")
    jsonl_writer.emit("run_end", total_failures=total_failures)
    jsonl_writer.close()
    log.info(f"JSONL log saved to {jsonl_path}")

    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"Report saved to {args.output}")

    if args.publish:
        from bench_batch import publish_results
        publish_results(report, args.output, args.publish, mode="stream")
    wer_failures = report.get("wer", {}).get("samples_failed", 0)
    if total_failures > 0:
        log.error(f"FAIL: {total_failures} total failures across sweep + sustained")
        return 1
    if wer_failures > 0:
        log.error(f"FAIL: {wer_failures} WER eval items failed at c=1 — incomplete coverage")
        return 1

    if "quality_gates" in report and not report["quality_gates"]["all_passed"]:
        failed = [g for g in report["quality_gates"]["gates"] if not g["passed"]]
        log.error(f"FAIL: {len(failed)} quality gate(s) failed")
        return 1

    if args.smart and baseline_sweep:
        regressions = []
        for s in sweep_results:
            bl = baseline_sweep.get(s["concurrency"])
            if bl:
                reg_rtfx, _ = check_regression(s, bl, "rtfx")
                reg_sess, _ = check_regression(s, bl, "sess_per_min")
                if reg_rtfx or reg_sess:
                    regressions.append(s["concurrency"])
        if regressions:
            log.error(f"FAIL: regression detected at concurrency levels {regressions}")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
