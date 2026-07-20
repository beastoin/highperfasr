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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from preflight import detect_server, resolve_stream_url, log_duration_estimate, log_preflight_summary, ensure_unbuffered

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


def collect_system_info():
    """Collect system metadata for report reproducibility."""
    from bench_batch import collect_system_info as _collect

    return _collect()


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


async def run_sweep(ws_url, manifest, concurrency, chunk_ms, repeat=1, target_count=None):
    """Run one concurrency level for streaming."""
    if target_count is None:
        target_count = len(manifest) * repeat
    files = [Path(e["wav_path"]) for e in select_round_robin_entries(manifest, concurrency, target_count)]
    sem = asyncio.Semaphore(concurrency)
    t0 = time.monotonic()
    tasks = [stream_file(ws_url, f, chunk_ms, sem) for f in files]
    results = await asyncio.gather(*tasks)
    wall = time.monotonic() - t0
    return list(results), wall


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

    report = {
        "schema_version": "v1alpha2-live",
        "benchmark": "Streaming ASR Benchmark",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "server": args.server,
        "chunk_ms": args.chunk_ms,
        "samples": len(manifest),
        "dataset": dataset_name,
        "smart_mode": args.smart,
        "system": collect_system_info(),
        "command": " ".join(sys.argv),
    }
    vram_start_mb = collect_gpu_memory_used_mb()

    # Warmup
    warmup_c = min(4, len(manifest), max(args.warmup, 1))
    log.info(f"Warmup: {args.warmup} streams at c={warmup_c}...")
    await run_sweep(ws_url, manifest, concurrency=warmup_c, chunk_ms=args.chunk_ms, target_count=args.warmup)

    # WER evaluation (c=1)
    if not args.skip_wer:
        total_audio = sum(e.get("duration_s", 5.0) for e in manifest)
        log_duration_estimate(len(manifest), total_audio, mode="streaming")
        wer_results, _ = await run_sweep(ws_url, manifest, concurrency=1, chunk_ms=args.chunk_ms, target_count=len(manifest))
        wer_summary = summarize_wer_results(wer_results, refs)

        if wer_summary:
            report["wer"] = {
                "corpus_wer_pct": wer_summary["corpus_wer_pct"],
                "c1_corpus_wer_pct": wer_summary["corpus_wer_pct"],
                "samples_evaluated": wer_summary["samples_evaluated"],
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

            if args.smart and baseline_report and "wer" in baseline_report:
                base_wer = baseline_report["wer"]["corpus_wer_pct"]
                cur_wer = report["wer"]["corpus_wer_pct"]
                log.info(f"  vs baseline: {base_wer}% -> {cur_wer}% (delta {cur_wer - base_wer:+.2f}%)")

    # Concurrency sweep
    log.info("Concurrency sweep...")
    sweep_results = []
    consecutive_matches = 0
    for c in levels:
        log.info(f"  c={c}...")
        results, wall = await run_sweep(
            ws_url, manifest, concurrency=c, chunk_ms=args.chunk_ms, target_count=max(len(manifest), c)
        )
        summary = summarize_sweep(results, wall, c)
        sweep_results.append(summary)
        log.info(f"    RTFx={summary['rtfx']}, sess/min={summary['sess_per_min']}, failures={summary['failures']}")

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

    # Sustained load
    sc = args.sustained_concurrency
    rounds = args.sustained_rounds
    log.info(f"Sustained load: c={sc}, {rounds} rounds...")
    sustained_results, sustained_wall = await run_sweep(
        ws_url, manifest, concurrency=sc, chunk_ms=args.chunk_ms, repeat=rounds
    )
    sustained_summary = summarize_sweep(sustained_results, sustained_wall, sc)
    sustained_summary["rounds"] = rounds
    report["sustained_load"] = sustained_summary
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
    print("| c | RTFx | sess/min | p50 | p99 | Failures |")
    print("|---|------|----------|-----|-----|----------|")
    for s in sweep_results:
        print(
            f"| {s['concurrency']} | {s['rtfx']}x | {s['sess_per_min']} | "
            f"{s.get('p50_s', '?')}s | {s.get('p99_s', '?')}s | {s['failures']} |"
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

        for trial in range(2, args.trials + 1):
            log.info(f"=== Trial {trial}/{args.trials} ===")
            t_sweep = []
            for c in levels:
                results, wall = await run_sweep(
                    ws_url, manifest, concurrency=c, chunk_ms=args.chunk_ms, target_count=max(len(manifest), c)
                )
                t_sweep.append(summarize_sweep(results, wall, c))
            t_peak = max(t_sweep, key=lambda x: x["rtfx"])
            all_peak_rtfx.append(t_peak["rtfx"])

            t_sus_results, t_sus_wall = await run_sweep(
                ws_url, manifest, concurrency=sc, chunk_ms=args.chunk_ms, repeat=rounds
            )
            t_sus = summarize_sweep(t_sus_results, t_sus_wall, sc)
            all_sustained_rtfx.append(t_sus["rtfx"])
            all_sustained_sess.append(t_sus["sess_per_min"])
            t_failures = sum(s["failures"] for s in t_sweep) + t_sus["failures"]
            trial_total_failures += t_failures
            log.info(f"  Trial {trial}: peak RTFx={t_peak['rtfx']}, sustained={t_sus['rtfx']}, failures={t_failures}")

        report["trials"] = {
            "count": args.trials,
            "peak_rtfx": summarize_trials(all_peak_rtfx),
            "sustained_rtfx": summarize_trials(all_sustained_rtfx),
            "sustained_sess_per_min": summarize_trials(all_sustained_sess),
        }
        log.info(f"Trials summary: peak RTFx={report['trials']['peak_rtfx']}")

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

    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"Report saved to {args.output}")

    total_failures = trial_total_failures if args.trials > 1 else (
        sum(s["failures"] for s in sweep_results) + sustained_summary["failures"])
    if total_failures > 0:
        log.error(f"FAIL: {total_failures} total failures across sweep + sustained")
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
