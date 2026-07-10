#!/usr/bin/env python3
"""Per-utterance WER evaluation with full error breakdown for independent verification."""
import asyncio
import csv
import json
import sys
import time
from pathlib import Path

SR = 16000


async def stream_file(ws_url, wav_path, chunk_ms):
    import websockets

    chunk_samples = int(SR * chunk_ms / 1000)
    chunk_bytes = chunk_samples * 2
    utt_id = Path(wav_path).stem
    t0 = time.monotonic()
    ttfb = None
    try:
        with open(wav_path, "rb") as f:
            f.read(44)
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
        return {"utt_id": utt_id, "error": str(e)[:200], "elapsed": time.monotonic() - t0, "status": "error"}


def edit_distance_detail(ref_words, hyp_words):
    """Compute edit distance with S/I/D counts."""
    n, m = len(ref_words), len(hyp_words)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    # Backtrace for S/I/D
    i, j = n, m
    subs, ins, dels = 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_words[i - 1] == hyp_words[j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            subs += 1
            i -= 1
            j -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            ins += 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            dels += 1
            i -= 1
        else:
            break
    return dp[n][m], subs, ins, dels


async def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="ws://localhost:8000")
    parser.add_argument("--chunk-ms", type=int, default=160)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--output-dir", default="/tmp/bench_artifact")
    parser.add_argument("--endpoint", default="/v1/stream", help="WebSocket endpoint path (default: /v1/stream)")
    args = parser.parse_args()

    ws_url = f"{args.server}{args.endpoint}"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load references
    ref_file = Path("/tmp/librispeech-test-clean/references.tsv")
    refs = {}
    with open(ref_file) as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                refs[parts[0]] = parts[1]

    wav_dir = Path("/tmp/librispeech-test-clean/wav")
    wav_files = sorted(wav_dir.glob("*.wav"))[:200]
    print(f"Evaluating {len(wav_files)} files at c={args.concurrency}")

    # Normalize function
    try:
        from whisper_normalizer.english import EnglishTextNormalizer

        normalizer = EnglishTextNormalizer()
        norm_name = "whisper_english"
    except ImportError:
        import re

        normalizer = lambda t: re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", t.lower())).strip()
        norm_name = "basic_lowercase"

    # Stream all files
    sem = asyncio.Semaphore(args.concurrency)

    async def bounded_stream(wav):
        async with sem:
            return await stream_file(ws_url, wav, args.chunk_ms)

    t0 = time.monotonic()
    tasks = [bounded_stream(f) for f in wav_files]
    results = await asyncio.gather(*tasks)
    total_wall = time.monotonic() - t0

    # Per-utterance WER with S/I/D
    rows = []
    total_ref_words = 0
    total_errors = 0
    total_subs = 0
    total_ins = 0
    total_dels = 0
    ok_count = 0
    fail_count = 0

    for r in results:
        if r["status"] != "ok":
            fail_count += 1
            continue
        ok_count += 1
        utt_id = r["utt_id"]
        hyp_raw = r.get("text", "")
        ref_raw = refs.get(utt_id, "")
        if not ref_raw:
            continue

        ref_norm = normalizer(ref_raw)
        hyp_norm = normalizer(hyp_raw)
        ref_words = ref_norm.split()
        hyp_words = hyp_norm.split()

        errs, subs, ins, dels = edit_distance_detail(ref_words, hyp_words)
        utt_wer = errs / max(len(ref_words), 1)

        total_ref_words += len(ref_words)
        total_errors += errs
        total_subs += subs
        total_ins += ins
        total_dels += dels

        rows.append(
            {
                "utt_id": utt_id,
                "ref_raw": ref_raw,
                "hyp_raw": hyp_raw,
                "ref_normalized": ref_norm,
                "hyp_normalized": hyp_norm,
                "ref_words": len(ref_words),
                "hyp_words": len(hyp_words),
                "errors": errs,
                "substitutions": subs,
                "insertions": ins,
                "deletions": dels,
                "wer": round(utt_wer, 6),
                "audio_duration_s": round(r.get("audio_dur", 0), 3),
                "latency_s": round(r["elapsed"], 4),
                "rtfx": round(r.get("rtfx", 0), 2),
                "ttfb_s": r.get("ttfb_s"),
            }
        )

    corpus_wer = total_errors / max(total_ref_words, 1)

    # Write per-utterance CSV
    csv_path = out_dir / f"per_utterance_wer_c{args.concurrency}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "utt_id",
                "ref_raw",
                "hyp_raw",
                "ref_normalized",
                "hyp_normalized",
                "ref_words",
                "hyp_words",
                "errors",
                "substitutions",
                "insertions",
                "deletions",
                "wer",
                "audio_duration_s",
                "latency_s",
                "rtfx",
                "ttfb_s",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    # Write summary JSON
    latencies = sorted(r["latency_s"] for r in rows)
    rtfxs = sorted(r["rtfx"] for r in rows)
    ttfbs = sorted(r["ttfb_s"] for r in rows if r.get("ttfb_s"))
    audio_durs = [r["audio_duration_s"] for r in rows]

    summary = {
        "benchmark": "Streaming WER Evaluation",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": "LibriSpeech test-clean",
        "samples": len(wav_files),
        "evaluated": len(rows),
        "failures": fail_count,
        "concurrency": args.concurrency,
        "chunk_ms": args.chunk_ms,
        "normalization": norm_name,
        "quality": {
            "corpus_wer": round(corpus_wer, 6),
            "corpus_wer_pct": round(corpus_wer * 100, 2),
            "total_ref_words": total_ref_words,
            "total_errors": total_errors,
            "total_substitutions": total_subs,
            "total_insertions": total_ins,
            "total_deletions": total_dels,
            "ser": round(sum(1 for r in rows if r["wer"] > 0) / max(len(rows), 1), 4),
        },
        "latency": {
            "wall_time_s": round(total_wall, 2),
            "p50_s": round(latencies[len(latencies) // 2], 4) if latencies else None,
            "p90_s": round(latencies[int(len(latencies) * 0.90)], 4) if latencies else None,
            "p95_s": round(latencies[int(len(latencies) * 0.95)], 4) if latencies else None,
            "p99_s": round(latencies[int(len(latencies) * 0.99)], 4) if latencies else None,
            "max_s": round(latencies[-1], 4) if latencies else None,
            "min_s": round(latencies[0], 4) if latencies else None,
        },
        "throughput": {
            "total_audio_s": round(sum(audio_durs), 1),
            "rtfx_aggregate": round(sum(audio_durs) / total_wall, 2) if total_wall > 0 else 0,
            "rtfx_p50": round(rtfxs[len(rtfxs) // 2], 2) if rtfxs else None,
            "rtfx_min": round(rtfxs[0], 2) if rtfxs else None,
            "rtfx_max": round(rtfxs[-1], 2) if rtfxs else None,
            "sessions_per_min": round(len(rows) / (total_wall / 60), 1) if total_wall > 0 else 0,
        },
        "ttfb": {
            "p50_s": round(ttfbs[len(ttfbs) // 2], 4) if ttfbs else None,
            "p95_s": round(ttfbs[int(len(ttfbs) * 0.95)], 4) if ttfbs else None,
            "p99_s": round(ttfbs[int(len(ttfbs) * 0.99)], 4) if ttfbs else None,
            "max_s": round(ttfbs[-1], 4) if ttfbs else None,
        },
        "audio_stats": {
            "mean_duration_s": round(sum(audio_durs) / max(len(audio_durs), 1), 2),
            "min_duration_s": round(min(audio_durs), 2) if audio_durs else None,
            "max_duration_s": round(max(audio_durs), 2) if audio_durs else None,
            "total_duration_s": round(sum(audio_durs), 1),
        },
    }

    summary_path = out_dir / f"summary_c{args.concurrency}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nCorpus WER: {corpus_wer*100:.2f}% ({total_errors}/{total_ref_words} errors)")
    print(f"  Substitutions: {total_subs}, Insertions: {total_ins}, Deletions: {total_dels}")
    print(f"  SER: {summary['quality']['ser']*100:.1f}%")
    print(f"Latency p50={summary['latency']['p50_s']}s p99={summary['latency']['p99_s']}s")
    print(f"RTFx aggregate: {summary['throughput']['rtfx_aggregate']}x")
    print(f"Per-utterance CSV: {csv_path}")
    print(f"Summary JSON: {summary_path}")


asyncio.run(main())
