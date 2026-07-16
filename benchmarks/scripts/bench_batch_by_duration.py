#!/usr/bin/env python3
"""
Duration-stratified batch ASR benchmark.

Tests batch throughput and max safe concurrency per audio duration bracket.
Complements bench_batch.py (which uses mixed-duration LibriSpeech) by isolating
duration as the independent variable.

Usage:
    python3 bench_batch_by_duration.py --server http://localhost:8000
    python3 bench_batch_by_duration.py --server http://localhost:8000 --durations 10,30,60,120
    python3 bench_batch_by_duration.py --server http://localhost:8000 --durations 60 --concurrency 1,4,8,16,24
"""

import argparse
import asyncio
import json
import logging
import os
import struct
import subprocess
import sys
import time
import wave
from pathlib import Path

import aiohttp

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("bench_batch_dur")

SAFE_CONCURRENCY = {
    10: [1, 8, 32, 64, 128, 256],
    30: [1, 8, 32, 64, 128, 256],
    60: [1, 4, 8, 16, 24],
    120: [1, 4, 8, 16, 32, 64, 128],
}

SUSTAINED_CONCURRENCY = {
    10: 128,
    30: 128,
    60: 16,
    120: 64,
}

FILES_PER_DURATION = 100


def collect_system_info():
    """Collect system metadata for report reproducibility."""
    import platform as _platform

    def _run(cmd):
        try:
            return subprocess.check_output(cmd, shell=True, text=True, timeout=5).strip()
        except Exception:
            return None

    info = {
        "python_version": _platform.python_version(),
        "platform": _platform.platform(),
    }
    try:
        import torch
        info["pytorch_version"] = torch.__version__
        info["cuda_version"] = torch.version.cuda or "N/A"
    except ImportError:
        pass

    git_sha = _run("git rev-parse --short HEAD")
    if git_sha:
        info["git_sha"] = git_sha

    gpu = _run("nvidia-smi --query-gpu=name --format=csv,noheader")
    if gpu:
        info["gpu"] = gpu.split("\n")[0]

    gpu_mem = _run("nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits")
    if gpu_mem:
        info["gpu_memory_mb"] = int(gpu_mem.split("\n")[0])

    driver = _run("nvidia-smi --query-gpu=driver_version --format=csv,noheader")
    if driver:
        info["driver_version"] = driver.split("\n")[0]

    gpu_util = _run("nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits")
    if gpu_util:
        info["gpu_utilization_pct"] = int(gpu_util.split("\n")[0])

    if os.path.exists("/.dockerenv") or os.path.exists("/run/.containerenv"):
        image = os.environ.get("NVIDIA_PYTORCH_VERSION") or os.environ.get("BASE_IMAGE")
        if image:
            info["container_image"] = image

    return info


def get_wav_duration(wav_path):
    with wave.open(str(wav_path), "rb") as w:
        return w.getnframes() / w.getframerate()


def generate_wavs(duration_sec, count, out_dir):
    """Generate speech WAVs using espeak-ng at the target duration."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(out_dir.glob("*.wav"))
    if len(existing) >= count:
        avg_dur = sum(get_wav_duration(f) for f in existing[:5]) / 5
        log.info(f"  Using {len(existing)} cached WAVs (avg {avg_dur:.1f}s)")
        return existing[:count]

    sentences = [
        "The quick brown fox jumps over the lazy dog near the riverbank.",
        "A journey of a thousand miles begins with a single step forward.",
        "Knowledge is power and wisdom is the application of that knowledge.",
        "The early bird catches the worm but the second mouse gets the cheese.",
        "In the middle of difficulty lies opportunity for those who seek it.",
        "Science is organized knowledge and wisdom is organized life itself.",
        "The only way to do great work is to love what you do every day.",
        "Time flies over us but leaves its shadow behind for all to see.",
    ]

    words_per_sec = 2.5
    target_words = int(duration_sec * words_per_sec)

    log.info(f"  Generating {count} WAVs at ~{duration_sec}s...")
    for i in range(count):
        wav_path = out_dir / f"speech_{duration_sec}s_{i:04d}.wav"
        if wav_path.exists():
            continue
        text = ""
        while len(text.split()) < target_words:
            text += " " + sentences[len(text.split()) % len(sentences)]
        text = " ".join(text.split()[:target_words])

        tmp_raw = out_dir / f"_tmp_{i}.wav"
        subprocess.run(
            ["espeak-ng", "-w", str(tmp_raw), text],
            capture_output=True, timeout=30,
        )
        if tmp_raw.exists():
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(tmp_raw), "-ar", "16000", "-ac", "1",
                 "-sample_fmt", "s16", str(wav_path)],
                capture_output=True, timeout=30,
            )
            tmp_raw.unlink(missing_ok=True)

    wavs = sorted(out_dir.glob("*.wav"))
    if wavs:
        avg_dur = sum(get_wav_duration(f) for f in wavs[:5]) / 5
        log.info(f"  Generated {len(wavs)} WAVs (avg {avg_dur:.1f}s)")
    return wavs[:count]


async def transcribe_file(session, url, wav_path, semaphore, timeout_s=300):
    async with semaphore:
        t0 = time.monotonic()
        audio_dur = get_wav_duration(wav_path)
        try:
            data = aiohttp.FormData()
            data.add_field(
                "file", open(wav_path, "rb"),
                filename=os.path.basename(wav_path),
                content_type="audio/wav",
            )
            async with session.post(
                url, data=data,
                timeout=aiohttp.ClientTimeout(total=timeout_s),
            ) as resp:
                elapsed = time.monotonic() - t0
                if resp.status == 200:
                    result = await resp.json()
                    return {
                        "status": "ok",
                        "elapsed": elapsed,
                        "audio_dur": audio_dur,
                        "text_len": len(result.get("text", "")),
                    }
                else:
                    return {"status": "error", "elapsed": elapsed, "error": f"HTTP {resp.status}"}
        except Exception as e:
            return {"status": "error", "elapsed": time.monotonic() - t0, "error": str(e)[:200]}


async def run_level(url, wavs, concurrency, count=None):
    if count is None:
        count = max(len(wavs), concurrency)
    sem = asyncio.Semaphore(concurrency)
    files = [wavs[i % len(wavs)] for i in range(count)]
    timeout_s = max(300, get_wav_duration(wavs[0]) * 4)

    t0 = time.monotonic()
    async with aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(limit=concurrency + 10)
    ) as session:
        tasks = [transcribe_file(session, url, f, sem, timeout_s) for f in files]
        results = await asyncio.gather(*tasks)
    wall = time.monotonic() - t0
    return list(results), wall


def summarize(results, wall_time, concurrency):
    ok = [r for r in results if r["status"] == "ok"]
    failed = [r for r in results if r["status"] == "error"]
    latencies = sorted(r["elapsed"] for r in ok)
    total_audio = sum(r.get("audio_dur", 0) for r in ok)

    s = {
        "concurrency": concurrency,
        "total": len(results),
        "ok": len(ok),
        "failures": len(failed),
        "wall_s": round(wall_time, 2),
        "rps": round(len(ok) / wall_time, 2) if wall_time > 0 else 0,
        "rtfx": round(total_audio / wall_time, 2) if wall_time > 0 else 0,
        "rtf": round(wall_time / total_audio, 3) if total_audio > 0 else 0,
        "total_audio_s": round(total_audio, 1),
    }
    if latencies:
        import statistics
        s["p50_s"] = round(latencies[len(latencies) // 2], 3)
        s["p99_s"] = round(latencies[int(len(latencies) * 0.99)], 3)
        s["min_s"] = round(latencies[0], 3)
        s["max_s"] = round(latencies[-1], 3)
        s["mean_s"] = round(statistics.mean(latencies), 3)
        if len(latencies) > 1:
            s["stddev_s"] = round(statistics.stdev(latencies), 3)
    return s


async def main():
    parser = argparse.ArgumentParser(description="Duration-stratified batch ASR benchmark")
    parser.add_argument("--server", required=True, help="Server base URL")
    parser.add_argument("--durations", default="10,30,60,120",
                        help="Comma-separated durations in seconds (default: 10,30,60,120)")
    parser.add_argument("--concurrency", default=None,
                        help="Override concurrency levels (default: per-duration safe limits)")
    parser.add_argument("--files", type=int, default=FILES_PER_DURATION,
                        help=f"Files per duration (default: {FILES_PER_DURATION})")
    parser.add_argument("--wav-dir", default="/tmp",
                        help="Parent directory for WAV caches (default: /tmp)")
    parser.add_argument("--sustained-rounds", type=int, default=4,
                        help="Sustained load rounds (default: 4)")
    parser.add_argument("--endpoint", default="/v1/transcriptions",
                        help="Transcription endpoint (default: /v1/transcriptions)")
    parser.add_argument("--output", default="/tmp/bench_batch_by_duration.json",
                        help="Output JSON path")
    parser.add_argument("--warmup", type=int, default=10,
                        help="Warmup requests per duration (default: 10)")
    args = parser.parse_args()

    durations = [int(d) for d in args.durations.split(",")]
    url = f"{args.server}{args.endpoint}"

    log.info("=== Duration-Stratified Batch ASR Benchmark ===")
    log.info(f"Server: {args.server}")
    log.info(f"Durations: {durations}s")
    log.info(f"Files per duration: {args.files}")

    report = {
        "benchmark": "Batch ASR Benchmark — Duration Stratified",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "server": args.server,
        "files_per_duration": args.files,
        "system": collect_system_info(),
        "command": " ".join(sys.argv),
        "durations": [],
    }

    for dur in durations:
        log.info(f"\n{'='*60}")
        log.info(f"Duration: {dur}s")
        log.info(f"{'='*60}")

        wav_dir = Path(args.wav_dir) / f"bench-{dur}s-wavs"
        wavs = sorted(wav_dir.glob("*.wav"))
        if not wavs:
            wavs = generate_wavs(dur, args.files, wav_dir)
        if not wavs:
            log.error(f"  No WAVs for {dur}s, skipping")
            continue

        wavs = wavs[:args.files]
        avg_dur = sum(get_wav_duration(f) for f in wavs[:5]) / 5
        log.info(f"  Using {len(wavs)} files (avg {avg_dur:.1f}s)")

        if args.concurrency:
            levels = [int(c) for c in args.concurrency.split(",")]
        else:
            levels = SAFE_CONCURRENCY.get(dur, [1, 8, 16, 32])

        # Warmup
        log.info(f"  Warmup: {args.warmup} requests...")
        await run_level(url, wavs, concurrency=min(4, len(wavs)), count=args.warmup)

        # Concurrency sweep
        sweep = []
        for c in levels:
            log.info(f"  c={c}...")
            count = max(len(wavs), c)
            results, wall = await run_level(url, wavs, c, count)
            s = summarize(results, wall, c)
            sweep.append(s)
            log.info(
                f"    {s['rps']} RPS | {s['rtfx']}x RTFx | "
                f"p50={s.get('p50_s', '?')}s p99={s.get('p99_s', '?')}s | "
                f"failures={s['failures']}"
            )
            if s["failures"] > 0:
                log.warning(f"    Failures at c={c} — stopping sweep for {dur}s")
                break

        # Sustained load
        sustained_c = SUSTAINED_CONCURRENCY.get(dur, min(levels[-1], 32))
        if args.concurrency:
            sustained_c = min(levels)
        rounds = args.sustained_rounds
        sustained_count = len(wavs) * rounds
        log.info(f"  Sustained: c={sustained_c}, {rounds} rounds ({sustained_count} files)...")
        sus_results, sus_wall = await run_level(url, wavs, sustained_c, sustained_count)
        sus_summary = summarize(sus_results, sus_wall, sustained_c)
        sus_summary["rounds"] = rounds

        # Max safe concurrency
        max_safe = 0
        for s in sweep:
            if s["failures"] == 0:
                max_safe = s["concurrency"]

        peak = max(sweep, key=lambda x: x["rps"])

        dur_report = {
            "duration_sec": dur,
            "avg_file_duration_sec": round(avg_dur, 1),
            "file_count": len(wavs),
            "concurrency_sweep": sweep,
            "sustained_load": sus_summary,
            "max_safe_concurrency": max_safe,
            "peak_rps": peak["rps"],
            "peak_rtfx": peak["rtfx"],
            "peak_concurrency": peak["concurrency"],
        }
        report["durations"].append(dur_report)

    # Summary table
    print()
    print("## Batch Benchmark — Duration Stratified")
    print()
    print("### Summary")
    print("| Duration | Avg File | Max Safe c | Peak RPS | Peak RTFx | Sustained RPS | Sustained RTFx |")
    print("|----------|----------|-----------|----------|-----------|---------------|----------------|")
    for d in report["durations"]:
        sus = d["sustained_load"]
        print(
            f"| {d['duration_sec']}s | {d['avg_file_duration_sec']}s | "
            f"c={d['max_safe_concurrency']} | {d['peak_rps']} | "
            f"{d['peak_rtfx']}x | {sus['rps']} | {sus['rtfx']}x |"
        )

    for d in report["durations"]:
        print(f"\n### {d['duration_sec']}s Audio ({d['avg_file_duration_sec']}s avg)")
        print("| c | RPS | RTFx | RTF | p50 | p99 | Failures |")
        print("|---|-----|------|-----|-----|-----|----------|")
        for s in d["concurrency_sweep"]:
            print(
                f"| {s['concurrency']} | {s['rps']} | {s['rtfx']}x | {s['rtf']} | "
                f"{s.get('p50_s', '?')}s | {s.get('p99_s', '?')}s | {s['failures']} |"
            )
        sus = d["sustained_load"]
        print(f"\nSustained (c={sus['concurrency']}, {sus.get('rounds', '?')} rounds): "
              f"{sus['rps']} RPS / {sus['rtfx']}x RTFx / {sus['failures']} failures")

    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"\nReport saved to {args.output}")

    total_failures = sum(
        s["failures"]
        for d in report["durations"]
        for s in d["concurrency_sweep"]
    ) + sum(d["sustained_load"]["failures"] for d in report["durations"])
    if total_failures > 0:
        log.error(f"FAIL: {total_failures} total failures across all durations")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
