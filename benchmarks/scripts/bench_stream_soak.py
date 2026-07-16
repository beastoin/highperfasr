#!/usr/bin/env python3
"""
Long-lived streaming ASR benchmark — sustained concurrent WebSocket connections.

Simulates always-on WebSocket traffic at target concurrency for configurable
durations. Streams real LibriSpeech audio at realtime pace, measuring WER,
time-to-first-byte (TTFB), connection stability, and VRAM growth over time.

Usage:
    python3 bench_stream_longlive.py --server ws://localhost:8000 --concurrency 96 --durations 300,900,1800
    python3 bench_stream_longlive.py --server ws://localhost:8000 --concurrency 96 --durations 1800 --vram-interval 10
"""

import argparse
import asyncio
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("bench_longlive")

SR = 16000


def ensure_librispeech(max_samples=None):
    from bench_batch import ensure_librispeech as _ensure

    _ensure(max_samples=max_samples)


def load_references():
    from bench_batch import load_references as _load

    return _load()


def compute_wer(references, hypotheses):
    from bench_batch import compute_wer as _compute

    return _compute(references, hypotheses)


def get_vram_mb():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return int(result.stdout.strip())
    except Exception:
        return -1


async def vram_sampler(interval_s, samples, stop_event, t0):
    while not stop_event.is_set():
        mb = get_vram_mb()
        samples.append({"elapsed_s": round(time.monotonic() - t0, 1), "vram_mb": mb})
        await asyncio.sleep(interval_s)


SKIP_HANDSHAKE = False


async def stream_file_with_ttfb(ws_url, wav_path, chunk_ms, semaphore, refs):
    import websockets

    chunk_samples = int(SR * chunk_ms / 1000)
    chunk_bytes = chunk_samples * 2
    utt_id = Path(wav_path).stem

    async with semaphore:
        t0 = time.monotonic()
        ttfb = None
        try:
            with open(wav_path, "rb") as f:
                f.read(44)
                raw = f.read()

            async with websockets.connect(ws_url, max_size=10 * 1024 * 1024) as ws:
                if not SKIP_HANDSHAKE:
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
                    "ttfb_s": round(ttfb, 3) if ttfb else None,
                    "status": "ok",
                }
        except Exception as e:
            return {
                "utt_id": utt_id,
                "error": str(e)[:200],
                "elapsed": time.monotonic() - t0,
                "ttfb_s": round(ttfb, 3) if ttfb else None,
                "status": "error",
            }


async def persistent_stream(ws_url, wav_files, chunk_ms, duration_s, stream_id):
    """One persistent WebSocket that stays open for the full duration, continuously streaming audio."""
    import websockets

    chunk_samples = int(SR * chunk_ms / 1000)
    chunk_bytes = chunk_samples * 2

    t0 = time.monotonic()
    ttfb = None
    files_streamed = 0
    total_audio_s = 0
    partial_count = 0
    errors = []
    transcripts = []

    try:
        async with websockets.connect(ws_url, max_size=10 * 1024 * 1024, ping_interval=None) as ws:
            if not SKIP_HANDSHAKE:
                config = json.dumps({"format": "pcm_s16le", "sample_rate": SR, "language": "en"})
                await ws.send(config)
                await asyncio.wait_for(ws.recv(), timeout=10)

            file_idx = stream_id
            prev_cumulative = ""
            while time.monotonic() - t0 < duration_s:
                wav_path = wav_files[file_idx % len(wav_files)]
                utt_id = Path(wav_path).stem
                file_idx += 1

                with open(wav_path, "rb") as f:
                    f.read(44)
                    raw = f.read()

                audio_dur = len(raw) / (SR * 2)
                total_audio_s += audio_dur

                cumulative_text = prev_cumulative
                offset = 0
                while offset < len(raw) and time.monotonic() - t0 < duration_s:
                    chunk = raw[offset : offset + chunk_bytes]
                    await ws.send(chunk)
                    offset += chunk_bytes
                    await asyncio.sleep(chunk_ms / 1000.0)

                    try:
                        while True:
                            msg = await asyncio.wait_for(ws.recv(), timeout=0.01)
                            resp = json.loads(msg)
                            if ttfb is None and (
                                resp.get("partial_transcript") or resp.get("final_transcript") or resp.get("text")
                            ):
                                ttfb = time.monotonic() - t0
                            txt = resp.get("final_transcript") or resp.get("partial_transcript") or resp.get("text")
                            if txt:
                                cumulative_text = txt
                            partial_count += 1
                    except asyncio.TimeoutError:
                        pass

                # Brief pause between files so the model can flush its decoder
                await asyncio.sleep(0.5)
                try:
                    while True:
                        msg = await asyncio.wait_for(ws.recv(), timeout=0.3)
                        resp = json.loads(msg)
                        txt = resp.get("final_transcript") or resp.get("partial_transcript") or resp.get("text")
                        if txt:
                            cumulative_text = txt
                        partial_count += 1
                except asyncio.TimeoutError:
                    pass

                # Extract this file's transcript by removing the previous cumulative text
                file_text = cumulative_text
                if prev_cumulative and cumulative_text.startswith(prev_cumulative):
                    file_text = cumulative_text[len(prev_cumulative) :].strip()
                prev_cumulative = cumulative_text

                transcripts.append({"utt_id": utt_id, "text": file_text})
                files_streamed += 1

    except Exception as e:
        errors.append(str(e)[:200])

    elapsed = time.monotonic() - t0
    return {
        "stream_id": stream_id,
        "duration_s": round(elapsed, 1),
        "files_streamed": files_streamed,
        "total_audio_s": round(total_audio_s, 1),
        "ttfb_s": round(ttfb, 3) if ttfb else None,
        "partial_count": partial_count,
        "transcripts": transcripts,
        "errors": errors,
        "status": "ok" if not errors else "error",
    }


async def run_persistent_test(ws_url, wav_files, refs, concurrency, duration_s, chunk_ms, vram_interval):
    """Run N persistent WebSocket connections for duration_s, each looping audio continuously."""
    vram_samples = []
    stop_vram = asyncio.Event()
    t0 = time.monotonic()

    vram_task = asyncio.create_task(vram_sampler(vram_interval, vram_samples, stop_vram, t0))
    vram_start = get_vram_mb()

    log.info(f"  Starting {concurrency} persistent connections for {duration_s}s...")

    tasks = [
        asyncio.create_task(persistent_stream(ws_url, wav_files, chunk_ms, duration_s, i)) for i in range(concurrency)
    ]

    # Log progress every 60s
    while not all(t.done() for t in tasks):
        await asyncio.sleep(10)
        elapsed = time.monotonic() - t0
        done_count = sum(1 for t in tasks if t.done())
        vram_now = get_vram_mb()
        if int(elapsed) % 60 < 10 and int(elapsed) > 0:
            log.info(f"    {int(elapsed)}s: {concurrency - done_count}/{concurrency} active, VRAM={vram_now}MB")

    results = [t.result() for t in tasks]

    stop_vram.set()
    await vram_task
    vram_end = get_vram_mb()
    wall = time.monotonic() - t0

    ok = [r for r in results if r["status"] == "ok"]
    failed = [r for r in results if r["status"] == "error"]

    ttfbs = sorted(r["ttfb_s"] for r in ok if r.get("ttfb_s") is not None)
    total_files = sum(r["files_streamed"] for r in results)
    total_audio = sum(r["total_audio_s"] for r in results)
    total_partials = sum(r["partial_count"] for r in results)

    ref_texts, hyp_texts = [], []
    for r in ok:
        for t in r.get("transcripts", []):
            if t["utt_id"] in refs and t.get("text"):
                ref_texts.append(refs[t["utt_id"]])
                hyp_texts.append(t["text"])

    wer_val = None
    if ref_texts:
        wer_val, _ = compute_wer(ref_texts, hyp_texts)

    vram_mbs = [s["vram_mb"] for s in vram_samples if s["vram_mb"] > 0]
    vram_growth = vram_end - vram_start if vram_start > 0 and vram_end > 0 else None

    summary = {
        "mode": "persistent",
        "concurrency": concurrency,
        "target_duration_s": duration_s,
        "actual_duration_s": round(wall, 1),
        "connections_ok": len(ok),
        "connections_failed": len(failed),
        "failed_errors": [e for r in failed for e in r.get("errors", [])[:2]],
        "total_files_streamed": total_files,
        "total_partials": total_partials,
        "rtfx": round(total_audio / wall, 2) if wall > 0 else 0,
        "wer_pct": round(wer_val * 100, 2) if wer_val is not None else None,
        "wer_samples": len(ref_texts),
    }

    if ttfbs:
        summary["ttfb_p50_s"] = round(ttfbs[len(ttfbs) // 2], 3)
        summary["ttfb_p99_s"] = round(ttfbs[int(len(ttfbs) * 0.99)], 3)

    summary["vram_start_mb"] = vram_start
    summary["vram_end_mb"] = vram_end
    summary["vram_growth_mb"] = vram_growth
    summary["vram_peak_mb"] = max(vram_mbs) if vram_mbs else None
    summary["vram_min_mb"] = min(vram_mbs) if vram_mbs else None
    summary["vram_samples"] = vram_samples

    stream_details = []
    for r in results:
        detail = {
            "stream_id": r["stream_id"],
            "status": r["status"],
            "duration_s": r["duration_s"],
            "files_streamed": r["files_streamed"],
            "total_audio_s": r["total_audio_s"],
            "ttfb_s": r.get("ttfb_s"),
            "partial_count": r["partial_count"],
            "errors": r.get("errors", []),
            "transcripts": r.get("transcripts", []),
        }
        stream_details.append(detail)
    summary["streams"] = stream_details

    return summary


async def run_duration_test(ws_url, wav_files, refs, concurrency, duration_s, chunk_ms, vram_interval):
    sem = asyncio.Semaphore(concurrency)
    vram_samples = []
    stop_vram = asyncio.Event()
    t0 = time.monotonic()

    vram_task = asyncio.create_task(vram_sampler(vram_interval, vram_samples, stop_vram, t0))

    results = []
    file_idx = 0
    active_tasks = set()

    log.info(f"  Starting c={concurrency} for {duration_s}s...")
    vram_start = get_vram_mb()

    while time.monotonic() - t0 < duration_s or active_tasks:
        while len(active_tasks) < concurrency * 2 and time.monotonic() - t0 < duration_s:
            wav = wav_files[file_idx % len(wav_files)]
            file_idx += 1
            task = asyncio.create_task(stream_file_with_ttfb(ws_url, wav, chunk_ms, sem, refs))
            active_tasks.add(task)
            task.add_done_callback(active_tasks.discard)

        if active_tasks:
            done, _ = await asyncio.wait(active_tasks, timeout=1.0, return_when=asyncio.FIRST_COMPLETED)
            for t in done:
                active_tasks.discard(t)
                r = t.result()
                results.append(r)

        elapsed = time.monotonic() - t0
        if int(elapsed) % 60 == 0 and int(elapsed) > 0:
            ok_count = sum(1 for r in results if r["status"] == "ok")
            fail_count = sum(1 for r in results if r["status"] == "error")
            vram_now = get_vram_mb()
            if elapsed == int(elapsed):
                log.info(f"    {int(elapsed)}s: {ok_count} ok, {fail_count} fail, VRAM={vram_now}MB")

    if active_tasks:
        done, _ = await asyncio.wait(active_tasks, timeout=30)
        for t in done:
            results.append(t.result())

    stop_vram.set()
    await vram_task
    vram_end = get_vram_mb()

    wall = time.monotonic() - t0

    ok = [r for r in results if r["status"] == "ok"]
    failed = [r for r in results if r["status"] == "error"]
    ttfbs = sorted(r["ttfb_s"] for r in ok if r.get("ttfb_s") is not None)
    latencies = sorted(r["elapsed"] for r in ok)
    total_audio = sum(r.get("audio_dur", 0) for r in ok)

    ref_texts, hyp_texts = [], []
    for r in ok:
        if r["utt_id"] in refs and r.get("text"):
            ref_texts.append(refs[r["utt_id"]])
            hyp_texts.append(r["text"])

    wer_val = None
    if ref_texts:
        wer_val, _ = compute_wer(ref_texts, hyp_texts)

    vram_mbs = [s["vram_mb"] for s in vram_samples if s["vram_mb"] > 0]
    vram_growth = vram_end - vram_start if vram_start > 0 and vram_end > 0 else None

    summary = {
        "concurrency": concurrency,
        "target_duration_s": duration_s,
        "actual_duration_s": round(wall, 1),
        "total_streams": len(results),
        "ok": len(ok),
        "failures": len(failed),
        "failed_errors": [r.get("error", "") for r in failed[:5]],
        "sess_per_min": round(len(ok) / (wall / 60), 1) if wall > 0 else 0,
        "rtfx": round(total_audio / wall, 2) if wall > 0 else 0,
        "wer_pct": round(wer_val * 100, 2) if wer_val is not None else None,
        "wer_samples": len(ref_texts),
    }

    if ttfbs:
        summary["ttfb_p50_s"] = round(ttfbs[len(ttfbs) // 2], 3)
        summary["ttfb_p99_s"] = round(ttfbs[int(len(ttfbs) * 0.99)], 3)
        summary["ttfb_min_s"] = round(ttfbs[0], 3)
        summary["ttfb_max_s"] = round(ttfbs[-1], 3)

    if latencies:
        summary["latency_p50_s"] = round(latencies[len(latencies) // 2], 3)
        summary["latency_p99_s"] = round(latencies[int(len(latencies) * 0.99)], 3)

    summary["vram_start_mb"] = vram_start
    summary["vram_end_mb"] = vram_end
    summary["vram_growth_mb"] = vram_growth
    summary["vram_peak_mb"] = max(vram_mbs) if vram_mbs else None
    summary["vram_min_mb"] = min(vram_mbs) if vram_mbs else None
    summary["vram_samples"] = vram_samples

    return summary


async def main():
    parser = argparse.ArgumentParser(description="Long-lived streaming ASR benchmark")
    parser.add_argument("--server", default="ws://localhost:8001", help="Server WebSocket base URL")
    parser.add_argument("--chunk-ms", type=int, default=160, help="Chunk duration in ms (default: 160)")
    parser.add_argument("--concurrency", type=int, default=96, help="Concurrent WebSocket connections (default: 96)")
    parser.add_argument(
        "--durations",
        default="300,900,1800",
        help="Comma-separated test durations in seconds (default: 300,900,1800 = 5,15,30 min)",
    )
    parser.add_argument(
        "--vram-interval", type=int, default=10, help="VRAM sampling interval in seconds (default: 10)"
    )
    parser.add_argument("--warmup", type=int, default=10, help="Warmup streams (default: 10)")
    parser.add_argument(
        "--persistent",
        action="store_true",
        help="Keep each WebSocket open for the full duration, looping audio (true always-on test)",
    )
    parser.add_argument("--output", default="/tmp/bench_longlive_report.json", help="Output JSON path")
    parser.add_argument("--endpoint", default="/v1/stream", help="WebSocket endpoint path (default: /v1/stream)")
    parser.add_argument("--no-handshake", action="store_true", help="Skip JSON config handshake (for v3/v4 endpoints that use query params)")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=200,
        help="Max WAV files to extract (default: 200, 0=all ~2620 files)",
    )
    args = parser.parse_args()

    global SKIP_HANDSHAKE
    SKIP_HANDSHAKE = args.no_handshake

    durations = [int(x) for x in args.durations.split(",")]
    ws_url = f"{args.server}{args.endpoint}"

    mode_label = "persistent" if args.persistent else "rotating"
    log.info(f"=== Long-Lived Streaming ASR Benchmark ({mode_label}) ===")
    log.info(f"Server: {args.server}")
    log.info(f"Concurrency: {args.concurrency}")
    log.info(f"Durations: {durations}s")
    log.info(f"Mode: {mode_label} (each WS {'stays open' if args.persistent else 'opens/closes per file'})")
    log.info(f"Chunk: {args.chunk_ms}ms, VRAM interval: {args.vram_interval}s")

    ensure_librispeech(max_samples=args.max_samples)
    refs = load_references()

    wav_dir = Path("/tmp/librispeech-test-clean/wav")
    all_wavs = sorted(wav_dir.glob("*.wav"))
    if args.max_samples > 0:
        all_wavs = all_wavs[: args.max_samples]
    wav_files = list(all_wavs)
    log.info(f"Using {len(wav_files)} WAV files (looped for duration)")

    log.info(f"Initial VRAM: {get_vram_mb()} MB")

    # Warmup
    log.info(f"Warmup: {args.warmup} streams...")
    sem = asyncio.Semaphore(4)
    warmup_tasks = [stream_file_with_ttfb(ws_url, f, args.chunk_ms, sem, refs) for f in wav_files[: args.warmup]]
    await asyncio.gather(*warmup_tasks)
    log.info(f"VRAM after warmup: {get_vram_mb()} MB")

    report = {
        "benchmark": "Streaming Soak ASR Benchmark",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "server": args.server,
        "concurrency": args.concurrency,
        "chunk_ms": args.chunk_ms,
        "dataset": f"LibriSpeech test-clean ({len(wav_files)} files, looped)",
        "durations": [],
    }

    for dur in durations:
        log.info(f"=== Duration test: {dur}s ({dur//60}min) at c={args.concurrency} ({mode_label}) ===")
        if args.persistent:
            summary = await run_persistent_test(
                ws_url, wav_files, refs, args.concurrency, dur, args.chunk_ms, args.vram_interval
            )
        else:
            summary = await run_duration_test(
                ws_url, wav_files, refs, args.concurrency, dur, args.chunk_ms, args.vram_interval
            )
        report["durations"].append(summary)

        ok_key = "connections_ok" if args.persistent else "ok"
        fail_key = "connections_failed" if args.persistent else "failures"
        log.info(f"  Results: {summary[ok_key]} ok, {summary[fail_key]} fail")
        if args.persistent:
            log.info(
                f"  RTFx={summary['rtfx']}, files={summary.get('total_files_streamed', '?')}, "
                f"partials={summary.get('total_partials', '?')}"
            )
        else:
            log.info(f"  sess/min={summary['sess_per_min']}, RTFx={summary['rtfx']}")
        if summary.get("wer_pct") is not None:
            log.info(f"  WER={summary['wer_pct']}%")
        if summary.get("ttfb_p50_s") is not None:
            log.info(f"  TTFB p50={summary['ttfb_p50_s']}s, p99={summary['ttfb_p99_s']}s")
        log.info(
            f"  VRAM: {summary['vram_start_mb']}MB -> {summary['vram_end_mb']}MB "
            f"(growth={summary['vram_growth_mb']}MB, peak={summary['vram_peak_mb']}MB)"
        )

    # Print markdown
    print()
    print(f"## Long-Lived Streaming Benchmark Results ({mode_label})")
    print(f"**Concurrency:** {args.concurrency} | **Chunk:** {args.chunk_ms}ms")
    print()
    if args.persistent:
        print("### Summary by Duration (persistent — each WS open for full duration)")
        print("| Duration | Conns OK | Failed | Files | RTFx | WER | TTFB p50 | TTFB p99 |")
        print("|----------|----------|--------|-------|------|-----|----------|----------|")
        for s in report["durations"]:
            dur_label = f"{s['target_duration_s']//60}min"
            print(
                f"| {dur_label} | {s['connections_ok']} | {s['connections_failed']} "
                f"| {s.get('total_files_streamed', '?')} | {s['rtfx']}x "
                f"| {s.get('wer_pct', '?')}% "
                f"| {s.get('ttfb_p50_s', '?')}s "
                f"| {s.get('ttfb_p99_s', '?')}s |"
            )
    else:
        print("### Summary by Duration (rotating — open/close per file)")
        print("| Duration | Streams | Failures | sess/min | RTFx | WER | TTFB p50 | TTFB p99 |")
        print("|----------|---------|----------|----------|------|-----|----------|----------|")
        for s in report["durations"]:
            dur_label = f"{s['target_duration_s']//60}min"
            print(
                f"| {dur_label} | {s['ok']} | {s['failures']} "
                f"| {s['sess_per_min']} | {s['rtfx']}x "
                f"| {s.get('wer_pct', '?')}% "
                f"| {s.get('ttfb_p50_s', '?')}s "
                f"| {s.get('ttfb_p99_s', '?')}s |"
            )
    print()
    print("### VRAM Stability")
    print("| Duration | Start | End | Growth | Peak |")
    print("|----------|-------|-----|--------|------|")
    for s in report["durations"]:
        dur_label = f"{s['target_duration_s']//60}min"
        print(
            f"| {dur_label} | {s['vram_start_mb']}MB | {s['vram_end_mb']}MB "
            f"| {s['vram_growth_mb']}MB | {s['vram_peak_mb']}MB |"
        )

    # Per-stream detail (persistent mode)
    if args.persistent:
        for s in report["durations"]:
            dur_label = f"{s['target_duration_s']//60}min"
            streams = s.get("streams", [])
            if not streams:
                continue
            print()
            print(f"### Per-Stream Detail ({dur_label}, {len(streams)} streams)")
            print("| Stream | Status | Duration | Files | Audio (s) | TTFB (s) | Partials | Transcripts |")
            print("|:------:|:------:|:--------:|:-----:|:---------:|:--------:|:--------:|:-----------:|")
            for st in sorted(streams, key=lambda x: x["stream_id"]):
                n_transcripts = len(st.get("transcripts", []))
                print(
                    f"| {st['stream_id']} | {st['status']} | {st['duration_s']}s "
                    f"| {st['files_streamed']} | {st['total_audio_s']} "
                    f"| {st.get('ttfb_s', '—')} | {st['partial_count']} | {n_transcripts} |"
                )
            # Sample transcripts from first 3 streams
            print()
            print(f"#### Sample Transcripts ({dur_label}, streams 0-2)")
            for st in sorted(streams, key=lambda x: x["stream_id"])[:3]:
                transcripts = st.get("transcripts", [])
                if not transcripts:
                    continue
                print(f"\n**Stream {st['stream_id']}** ({len(transcripts)} files):")
                for t in transcripts[:5]:
                    text_preview = t.get("text", "")[:120]
                    if len(t.get("text", "")) > 120:
                        text_preview += "..."
                    print(f"- `{t['utt_id']}`: {text_preview}")
                if len(transcripts) > 5:
                    print(f"- ... and {len(transcripts) - 5} more files")

    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"Report saved to {args.output}")
    log.info(f"Full per-stream transcripts available in JSON: {args.output}")

    total_failures = sum(d.get("failures", 0) + d.get("connections_failed", 0)
                         for d in report.get("durations", []))
    vram_values = [d.get("vram_growth_mb") for d in report.get("durations", [])
                   if d.get("vram_growth_mb") is not None]
    vram_growth = max(vram_values, default=0)
    return 1 if total_failures > 0 or vram_growth > 100 else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
