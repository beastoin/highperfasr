#!/usr/bin/env python3
"""
Deterministic streaming ASR benchmark with WER.

Downloads LibriSpeech test-clean (200 samples), streams audio chunks via WebSocket,
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


def collect_system_info():
    """Collect system metadata for report reproducibility."""
    from bench_batch import collect_system_info as _collect

    return _collect()


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


def select_round_robin_entries(manifest, concurrency: int, target_count: int):
    """Select benchmark work using RoundRobinLoader batches.

    Falls back to simple cycling when concurrency exceeds manifest size.
    """
    if concurrency > len(manifest):
        return [manifest[i % len(manifest)] for i in range(target_count)]

    parent = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(parent))
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
                            if resp.get("final_transcript"):
                                final_parts.append(resp["final_transcript"])
                    except asyncio.TimeoutError:
                        pass

                await ws.send(json.dumps({"action": "close"}))

                try:
                    while True:
                        msg = await asyncio.wait_for(ws.recv(), timeout=5)
                        resp = json.loads(msg)
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
    parser.add_argument("--dataset", default=None,
                        help="Use multi-corpus dataset (e.g., 'librispeech-test-clean', 'all')")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Max samples from dataset (0=all)")
    parser.add_argument("--dataset-dir", type=Path, default=None, help="Dataset cache directory")
    args = parser.parse_args()

    levels = [int(x) for x in args.concurrency.split(",")]
    ws_url = f"{args.server}{args.endpoint}"

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

    if args.dataset:
        from pathlib import Path as _P
        parent = _P(__file__).resolve().parent.parent.parent
        sys.path.insert(0, str(parent))
        from benchmarks.datasets.registry import load_dataset
        manifest = load_dataset(args.dataset, cache_dir=args.dataset_dir, max_samples=args.max_samples)
        refs = {e["utt_id"]: e["reference"] for e in manifest if e.get("reference")}
    else:
        ensure_librispeech()
        refs = load_references()
        wav_dir = Path("/tmp/librispeech-test-clean/wav")
        wav_files = sorted(wav_dir.glob("*.wav"))[:200]
        manifest = manifest_from_wavs(wav_files, refs)
    log.info(f"Using {len(manifest)} WAV files")

    report = {
        "benchmark": "Streaming ASR Benchmark",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "server": args.server,
        "chunk_ms": args.chunk_ms,
        "samples": len(manifest),
        "dataset": args.dataset or "LibriSpeech test-clean (200 subset)",
        "smart_mode": args.smart,
        "system": collect_system_info(),
        "command": " ".join(sys.argv),
    }

    # Warmup
    warmup_c = min(4, len(manifest), max(args.warmup, 1))
    log.info(f"Warmup: {args.warmup} streams at c={warmup_c}...")
    await run_sweep(ws_url, manifest, concurrency=warmup_c, chunk_ms=args.chunk_ms, target_count=args.warmup)

    # WER evaluation (c=1)
    if not args.skip_wer:
        log.info("WER evaluation: c=1...")
        wer_results, _ = await run_sweep(ws_url, manifest, concurrency=1, chunk_ms=args.chunk_ms, target_count=len(manifest))
        ok_results = [r for r in wer_results if r["status"] == "ok"]

        ref_texts, hyp_texts = [], []
        for r in ok_results:
            if r["utt_id"] in refs:
                ref_texts.append(refs[r["utt_id"]])
                hyp_texts.append(r["text"])

        if ref_texts:
            wer_val, per_utt = compute_wer(ref_texts, hyp_texts)
            report["wer"] = {
                "corpus_wer_pct": round(wer_val * 100, 2),
                "samples_evaluated": len(ref_texts),
                "normalization": "whisper_english",
            }
            log.info(f"WER: {wer_val*100:.2f}%")

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

    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"Report saved to {args.output}")

    total_failures = sum(s["failures"] for s in sweep_results) + sustained_summary["failures"]
    if total_failures > 0:
        log.error(f"FAIL: {total_failures} total failures across sweep + sustained")
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
