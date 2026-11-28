#!/usr/bin/env python3
"""
Deterministic embedding benchmark for highperfasr.

Tests the /v1/embeddings endpoint with audio and text inputs at various
concurrency levels. Measures throughput (embeds/sec), latency (p50/p99),
VRAM stability, and optionally cross-modal retrieval quality (Recall@K).

Usage:
    python3 bench_embed.py --server http://localhost:8000
    python3 bench_embed.py --server http://localhost:8000 --modality audio
    python3 bench_embed.py --server http://localhost:8000 --modality text --concurrency 1,16,64,256
    python3 bench_embed.py --server http://localhost:8000 --smart
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

import aiohttp

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR.parent.parent))

from preflight import detect_server, ensure_unbuffered, log_preflight_summary

ensure_unbuffered()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("bench_embed")

DATA_DIR = Path("/tmp/librispeech-test-clean")
WAV_DIR = DATA_DIR / "wav"
REF_FILE = DATA_DIR / "references.tsv"
MAX_SAMPLES = 200

DEFAULT_TEXT_QUERIES = [
    "quarterly revenue discussion",
    "technical architecture review",
    "customer support escalation",
    "product roadmap planning",
    "engineering hiring pipeline",
    "infrastructure cost optimization",
    "security incident response",
    "user onboarding experience",
    "performance monitoring alert",
    "data privacy compliance",
    "marketing campaign analysis",
    "team standup meeting notes",
    "bug triage and prioritization",
    "deployment rollback procedure",
    "API rate limiting strategy",
    "database migration planning",
    "load testing methodology",
    "feature flag management",
    "error budget tracking",
    "service level objectives",
]


def ensure_librispeech(max_samples=None):
    """Download and extract LibriSpeech test-clean if not cached."""
    limit = MAX_SAMPLES if max_samples is None else max_samples
    target = limit if limit > 0 else 2620

    if WAV_DIR.exists() and len(list(WAV_DIR.glob("*.wav"))) >= target and REF_FILE.exists():
        n = len(list(WAV_DIR.glob("*.wav")))
        log.info(f"LibriSpeech test-clean cached: {n} WAV files")
        return

    import tarfile
    import urllib.request

    LIBRISPEECH_URL = "https://www.openslr.org/resources/12/test-clean.tar.gz"
    log.info("Downloading LibriSpeech test-clean...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    WAV_DIR.mkdir(exist_ok=True)
    tar_path = DATA_DIR / "test-clean.tar.gz"

    if not tar_path.exists():
        urllib.request.urlretrieve(LIBRISPEECH_URL, tar_path)
        log.info(f"Downloaded {tar_path.stat().st_size / 1e6:.0f}MB")

    refs = {}
    count = 0
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar:
            if member.name.endswith(".trans.txt"):
                f = tar.extractfile(member)
                if f:
                    for line in f.read().decode().strip().split("\n"):
                        parts = line.strip().split(" ", 1)
                        if len(parts) == 2:
                            refs[parts[0]] = parts[1]

            if member.name.endswith(".flac"):
                utt_id = Path(member.name).stem
                if limit > 0 and count >= limit:
                    continue
                wav_path = WAV_DIR / f"{utt_id}.wav"
                if wav_path.exists():
                    count += 1
                    continue
                f = tar.extractfile(member)
                if f:
                    import io
                    import struct
                    import soundfile as sf
                    audio, sr = sf.read(io.BytesIO(f.read()), dtype="int16")
                    num_samples = len(audio)
                    data_size = num_samples * 2
                    with open(wav_path, "wb") as wf:
                        wf.write(b"RIFF")
                        wf.write(struct.pack("<I", 36 + data_size))
                        wf.write(b"WAVE")
                        wf.write(b"fmt ")
                        wf.write(struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16))
                        wf.write(b"data")
                        wf.write(struct.pack("<I", data_size))
                        wf.write(audio.tobytes())
                    count += 1

    with open(REF_FILE, "w") as f:
        for utt_id in sorted(refs):
            f.write(f"{utt_id}\t{refs[utt_id]}\n")

    n = len(list(WAV_DIR.glob("*.wav")))
    log.info(f"Extracted {n} WAV files, {len(refs)} references")


def load_references():
    refs = {}
    with open(REF_FILE) as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                refs[parts[0]] = parts[1]
    return refs


def collect_system_info(gpu_override=None, image_tag=None):
    import platform as _platform
    import subprocess as _sp

    def _run(cmd):
        try:
            return _sp.check_output(cmd, shell=True, text=True, timeout=5).strip()
        except Exception:
            return None

    info = {"python_version": _platform.python_version(), "platform": _platform.platform()}
    try:
        import torch
        info["pytorch_version"] = torch.__version__
        info["cuda_version"] = torch.version.cuda or "N/A"
    except ImportError:
        pass

    git_sha = _run("git rev-parse --short HEAD")
    if git_sha:
        info["git_sha"] = git_sha
    if image_tag:
        info["container_image"] = image_tag
    if gpu_override:
        info["gpu"] = gpu_override
    else:
        gpu = _run("nvidia-smi --query-gpu=name --format=csv,noheader")
        if gpu:
            info["gpu"] = gpu.split("\n")[0]
    gpu_mem = _run("nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits")
    if gpu_mem:
        info["gpu_memory_mb"] = int(gpu_mem.split("\n")[0])
    return info


def collect_gpu_memory_used_mb():
    import subprocess as _sp
    try:
        output = _sp.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True, timeout=5,
        ).strip()
    except Exception:
        return None
    values = []
    for line in output.splitlines():
        line = line.strip()
        if line:
            try:
                values.append(int(line))
            except ValueError:
                return None
    return max(values) if values else None


def _validate_embedding(body):
    """Validate embedding response has a non-empty finite numeric vector."""
    import math
    emb = body.get("embedding")
    if not emb or not isinstance(emb, list) or len(emb) == 0:
        return None, "missing or empty embedding"
    for v in emb:
        if not isinstance(v, (int, float)):
            return None, "non-numeric values in embedding"
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None, "non-finite values in embedding"
    return emb, None


async def embed_audio(session, url, wav_path, semaphore):
    """Send one audio file to the embedding endpoint."""
    async with semaphore:
        t0 = time.monotonic()
        try:
            data = aiohttp.FormData()
            data.add_field(
                "file",
                open(wav_path, "rb"),
                filename=os.path.basename(wav_path),
                content_type="audio/wav",
            )
            async with session.post(url, data=data, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                if resp.status == 200:
                    body = await resp.json()
                    elapsed = time.monotonic() - t0
                    emb, err = _validate_embedding(body)
                    if err:
                        return {
                            "status": "error",
                            "utt_id": Path(wav_path).stem,
                            "modality": "audio",
                            "error": err,
                            "elapsed_s": round(elapsed, 4),
                        }
                    return {
                        "status": "ok",
                        "utt_id": Path(wav_path).stem,
                        "modality": "audio",
                        "dimensions": body.get("dimensions", len(emb)),
                        "elapsed_s": round(elapsed, 4),
                        "embedding": emb,
                    }
                else:
                    text = await resp.text()
                    elapsed = time.monotonic() - t0
                    return {
                        "status": "error",
                        "utt_id": Path(wav_path).stem,
                        "modality": "audio",
                        "error": f"HTTP {resp.status}: {text[:200]}",
                        "elapsed_s": round(elapsed, 4),
                    }
        except Exception as exc:
            elapsed = time.monotonic() - t0
            return {
                "status": "error",
                "utt_id": Path(wav_path).stem,
                "modality": "audio",
                "error": str(exc)[:200],
                "elapsed_s": round(elapsed, 4),
            }


async def embed_text(session, url, text_input, semaphore, idx=0):
    """Send one text input to the embedding endpoint."""
    async with semaphore:
        t0 = time.monotonic()
        try:
            payload = {"input": text_input, "normalize": True}
            async with session.post(
                url, json=payload, timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                if resp.status == 200:
                    body = await resp.json()
                    elapsed = time.monotonic() - t0
                    emb, err = _validate_embedding(body)
                    if err:
                        return {
                            "status": "error",
                            "idx": idx,
                            "modality": "text",
                            "error": err,
                            "elapsed_s": round(elapsed, 4),
                        }
                    return {
                        "status": "ok",
                        "idx": idx,
                        "modality": "text",
                        "dimensions": body.get("dimensions", len(emb)),
                        "elapsed_s": round(elapsed, 4),
                        "embedding": emb,
                    }
                else:
                    text = await resp.text()
                    elapsed = time.monotonic() - t0
                    return {
                        "status": "error",
                        "idx": idx,
                        "modality": "text",
                        "error": f"HTTP {resp.status}: {text[:200]}",
                        "elapsed_s": round(elapsed, 4),
                    }
        except Exception as exc:
            elapsed = time.monotonic() - t0
            return {
                "status": "error",
                "idx": idx,
                "modality": "text",
                "error": str(exc)[:200],
                "elapsed_s": round(elapsed, 4),
            }


def compute_latency_stats(results):
    """Compute p50/p99/mean latency from results."""
    latencies = sorted(r["elapsed_s"] for r in results if r["status"] == "ok")
    if not latencies:
        return {}
    n = len(latencies)
    return {
        "count": n,
        "mean_s": round(sum(latencies) / n, 4),
        "p50_s": round(latencies[int(n * 0.5)], 4),
        "p99_s": round(latencies[int(n * 0.99)], 4),
        "min_s": round(latencies[0], 4),
        "max_s": round(latencies[-1], 4),
    }


MIN_RETRIEVAL_PAIRS = 10


def compute_retrieval_metrics(audio_embeddings, text_embeddings, audio_utt_ids, refs, k=10):
    """Compute cross-modal Recall@K: for each audio, check if matching text is in top-K."""
    import numpy as np

    if not audio_embeddings or not text_embeddings:
        return None

    audio_dim = len(audio_embeddings[0])
    text_dim = len(text_embeddings[0])
    if audio_dim != text_dim:
        return {"error": f"dimension mismatch: audio={audio_dim}, text={text_dim}"}

    audio_filtered = []
    audio_ids_filtered = []
    for emb, uid in zip(audio_embeddings, audio_utt_ids):
        if len(emb) == audio_dim:
            audio_filtered.append(emb)
            audio_ids_filtered.append(uid)

    text_labels = list(refs.keys())
    text_filtered = []
    text_labels_filtered = []
    for emb, label in zip(text_embeddings, text_labels):
        if len(emb) == text_dim:
            text_filtered.append(emb)
            text_labels_filtered.append(label)

    if not audio_filtered or not text_filtered:
        return None

    try:
        audio_matrix = np.array(audio_filtered)
        text_matrix = np.array(text_filtered)
    except ValueError:
        return {"error": "inconsistent embedding dimensions"}

    audio_norms = np.linalg.norm(audio_matrix, axis=1, keepdims=True)
    text_norms = np.linalg.norm(text_matrix, axis=1, keepdims=True)
    audio_matrix = audio_matrix / np.maximum(audio_norms, 1e-8)
    text_matrix = text_matrix / np.maximum(text_norms, 1e-8)

    similarity = audio_matrix @ text_matrix.T

    text_label_set = set(text_labels_filtered)

    hits = 0
    evaluated = 0
    for i, utt_id in enumerate(audio_ids_filtered):
        if utt_id not in text_label_set:
            continue
        target_idx = text_labels_filtered.index(utt_id)
        top_k_indices = np.argsort(similarity[i])[::-1][:k]
        if target_idx in top_k_indices:
            hits += 1
        evaluated += 1

    if evaluated == 0:
        return None

    trivial_k = k >= len(text_filtered)

    return {
        "recall_at_k": round(hits / evaluated, 4),
        "k": k,
        "evaluated_pairs": evaluated,
        "total_audio": len(audio_filtered),
        "total_text": len(text_filtered),
        "insufficient_coverage": evaluated < MIN_RETRIEVAL_PAIRS,
        "trivial_k": trivial_k,
    }


async def run_concurrency_level(session, url, modality, concurrency, wav_files, texts, rounds=1):
    """Run embedding benchmark at a specific concurrency level."""
    semaphore = asyncio.Semaphore(concurrency)
    all_results = []

    for round_idx in range(rounds):
        tasks = []
        if modality in ("audio", "mixed"):
            for i in range(concurrency):
                wav = wav_files[i % len(wav_files)] if wav_files else None
                if wav:
                    tasks.append(embed_audio(session, url, wav, semaphore))

        if modality in ("text", "mixed"):
            for i in range(concurrency):
                text = texts[i % len(texts)]
                tasks.append(embed_text(session, url, text, semaphore, idx=i))

        results = await asyncio.gather(*tasks)
        all_results.extend(results)

    return all_results


async def main():
    parser = argparse.ArgumentParser(description="Deterministic embedding benchmark for highperfasr")
    parser.add_argument("--server", default="http://localhost:8000", help="Server base URL")
    parser.add_argument(
        "--concurrency", default="1,16,64,128,256",
        help="Comma-separated concurrency levels (default: 1,16,64,128,256)",
    )
    parser.add_argument("--modality", default="audio", choices=["audio", "text", "mixed"],
                        help="Embedding modality to benchmark (default: audio)")
    parser.add_argument("--sustained-rounds", type=int, default=4,
                        help="Sustained load rounds at peak concurrency (default: 4)")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup requests (default: 20)")
    parser.add_argument("--output", default="/tmp/bench_embed_report.json", help="Output JSON path")
    parser.add_argument("--endpoint", default="/v1/embeddings",
                        help="Embedding endpoint path (default: /v1/embeddings)")
    parser.add_argument("--smart", action="store_true",
                        help="Smart mode: sweep high-to-low, early-stop on first failure")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Max audio samples (0 = use MAX_SAMPLES default of 200)")
    parser.add_argument("--retrieval", action="store_true",
                        help="Run cross-modal retrieval quality test (Recall@K)")
    parser.add_argument("--retrieval-k", type=int, default=10,
                        help="K for Recall@K metric (default: 10)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: c=1 only, 50 samples, no sustained")
    parser.add_argument("--publish", default=None, metavar="DIR",
                        help="Copy report to auto-named dir under DIR")
    parser.add_argument("--gpu", default=None, metavar="NAME", help="Override GPU name in report")
    parser.add_argument("--image-tag", default=None, metavar="TAG",
                        help="Container image tag for report metadata")
    args = parser.parse_args()

    concurrency_levels = [int(c) for c in args.concurrency.split(",")]
    if args.quick:
        concurrency_levels = [1]
        args.sustained_rounds = 0
        args.max_samples = 50
        log.info("Quick mode: c=1, 50 samples, no sustained")

    if args.smart:
        concurrency_levels = sorted(concurrency_levels, reverse=True)
        log.info(f"Smart mode: sweeping high-to-low {concurrency_levels}")

    # --- Preflight ---
    server_info = detect_server(args.server)
    log_preflight_summary(server_info, mode_requested="embedding")

    endpoint = args.endpoint if args.endpoint.startswith("/") else f"/{args.endpoint}"
    embed_url = f"{args.server.rstrip('/')}{endpoint}"
    log.info(f"Embedding endpoint: {embed_url}")

    # --- Prepare data ---
    wav_files = []
    refs = {}
    if args.modality in ("audio", "mixed"):
        max_s = args.max_samples if args.max_samples > 0 else MAX_SAMPLES
        ensure_librispeech(max_s)
        wav_files = sorted(WAV_DIR.glob("*.wav"))[:max_s]
        refs = load_references()
        log.info(f"Audio files: {len(wav_files)}, references: {len(refs)}")

    texts = list(DEFAULT_TEXT_QUERIES)
    if args.modality in ("text", "mixed") and refs:
        texts = list(refs.values())[:MAX_SAMPLES]
    log.info(f"Text inputs: {len(texts)}")

    # --- System info ---
    sys_info = collect_system_info(gpu_override=args.gpu, image_tag=args.image_tag)
    vram_before = collect_gpu_memory_used_mb()

    report = {
        "benchmark": "embedding",
        "modality": args.modality,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": " ".join(sys.argv),
        "system": sys_info,
        "endpoint": args.endpoint,
        "concurrency_sweep": [],
        "vram": {"before_mb": vram_before},
    }

    max_c = max(concurrency_levels) if concurrency_levels else 256
    connector = aiohttp.TCPConnector(limit=max(max_c + 16, 128))
    async with aiohttp.ClientSession(connector=connector) as session:
        # --- Warmup ---
        log.info(f"Warming up with {args.warmup} requests...")
        warmup_sem = asyncio.Semaphore(4)
        warmup_tasks = []
        for i in range(args.warmup):
            if args.modality in ("audio", "mixed") and wav_files:
                wav = wav_files[i % len(wav_files)]
                warmup_tasks.append(embed_audio(session, embed_url, wav, warmup_sem))
            else:
                text = texts[i % len(texts)]
                warmup_tasks.append(embed_text(session, embed_url, text, warmup_sem, idx=i))
        warmup_results = await asyncio.gather(*warmup_tasks)
        warmup_ok = sum(1 for r in warmup_results if r["status"] == "ok")
        warmup_err = sum(1 for r in warmup_results if r["status"] != "ok")
        log.info(f"Warmup: {warmup_ok} ok, {warmup_err} errors")

        if warmup_ok == 0:
            log.error("All warmup requests failed — aborting")
            report["error"] = "All warmup requests failed"
            with open(args.output, "w") as f:
                json.dump(report, f, indent=2)
            print(f"\nReport: {args.output}")
            return

        vram_after_warmup = collect_gpu_memory_used_mb()
        report["vram"]["after_warmup_mb"] = vram_after_warmup

        # --- Check dimensions from warmup ---
        for r in warmup_results:
            if r["status"] == "ok" and r.get("dimensions"):
                report["embedding_dimensions"] = r["dimensions"]
                log.info(f"Embedding dimensions: {r['dimensions']}")
                break

        # --- Concurrency sweep ---
        log.info(f"\n## Concurrency Sweep ({args.modality})")
        print(f"\n{'Concurrency':>12} {'Requests':>10} {'OK':>8} {'Fail':>8} "
              f"{'Embeds/s':>10} {'p50(ms)':>10} {'p99(ms)':>10}")
        print("-" * 80)

        for c in concurrency_levels:
            n_requests = max(c, 50)
            rounds = max(1, n_requests // max(c, 1))

            t0 = time.monotonic()
            results = await run_concurrency_level(
                session, embed_url, args.modality, c, wav_files, texts, rounds=rounds,
            )
            wall_time = time.monotonic() - t0

            ok_count = sum(1 for r in results if r["status"] == "ok")
            fail_count = sum(1 for r in results if r["status"] != "ok")
            total = len(results)
            embeds_per_sec = ok_count / wall_time if wall_time > 0 else 0

            stats = compute_latency_stats(results)
            p50_ms = round(stats.get("p50_s", 0) * 1000, 1)
            p99_ms = round(stats.get("p99_s", 0) * 1000, 1)

            print(f"{c:>12} {total:>10} {ok_count:>8} {fail_count:>8} "
                  f"{embeds_per_sec:>10.1f} {p50_ms:>10.1f} {p99_ms:>10.1f}")

            level_data = {
                "concurrency": c,
                "requests": total,
                "successes": ok_count,
                "failures": fail_count,
                "wall_time_s": round(wall_time, 2),
                "embeds_per_sec": round(embeds_per_sec, 2),
                "failure_rate": round(fail_count / total, 4) if total > 0 else 0,
                **stats,
            }
            report["concurrency_sweep"].append(level_data)

            if args.smart and fail_count > 0:
                log.warning(f"Smart mode: {fail_count} failures at c={c}, continuing to find working level")

        # --- Sustained load ---
        if args.sustained_rounds > 0 and report["concurrency_sweep"]:
            passing = [s for s in report["concurrency_sweep"] if s["failures"] == 0]
            if passing:
                peak_c = max(s["concurrency"] for s in passing)
                log.info(f"\n## Sustained Load: c={peak_c}, {args.sustained_rounds} rounds")

                sustained_results = []
                t0 = time.monotonic()
                for rnd in range(args.sustained_rounds):
                    round_results = await run_concurrency_level(
                        session, embed_url, args.modality, peak_c, wav_files, texts, rounds=1,
                    )
                    sustained_results.extend(round_results)
                    ok = sum(1 for r in round_results if r["status"] == "ok")
                    fail = sum(1 for r in round_results if r["status"] != "ok")
                    log.info(f"  Round {rnd + 1}/{args.sustained_rounds}: {ok} ok, {fail} fail")

                sustained_wall = time.monotonic() - t0
                sustained_ok = sum(1 for r in sustained_results if r["status"] == "ok")
                sustained_fail = sum(1 for r in sustained_results if r["status"] != "ok")
                sustained_stats = compute_latency_stats(sustained_results)

                report["sustained_load"] = {
                    "concurrency": peak_c,
                    "rounds": args.sustained_rounds,
                    "total_requests": len(sustained_results),
                    "successes": sustained_ok,
                    "failures": sustained_fail,
                    "wall_time_s": round(sustained_wall, 2),
                    "embeds_per_sec": round(sustained_ok / sustained_wall, 2) if sustained_wall > 0 else 0,
                    **sustained_stats,
                }

        # --- VRAM check ---
        vram_after = collect_gpu_memory_used_mb()
        report["vram"]["after_benchmark_mb"] = vram_after
        if vram_after_warmup and vram_after:
            report["vram"]["growth_mb"] = vram_after - vram_after_warmup

        # --- Retrieval quality ---
        if args.retrieval and args.modality == "text":
            report["retrieval_error"] = {"reason": "retrieval requires audio modality (use --modality audio or mixed)"}
            log.warning("Retrieval skipped — requires audio modality")
        elif args.retrieval and args.modality in ("audio", "mixed") and not refs:
            report["retrieval_error"] = {"reason": "no reference transcripts available for retrieval"}
            log.warning("Retrieval skipped — no reference transcripts")

        if args.retrieval and args.modality in ("audio", "mixed") and refs:
            log.info("\n## Cross-Modal Retrieval Quality (Recall@K)")
            retrieval_sem = asyncio.Semaphore(8)
            audio_embeddings = []
            audio_utt_ids = []
            text_embeddings = []

            sample_wavs = wav_files[:min(100, len(wav_files))]
            sample_refs = {}
            for wav in sample_wavs:
                utt_id = Path(wav).stem
                if utt_id in refs:
                    sample_refs[utt_id] = refs[utt_id]

            log.info(f"Embedding {len(sample_wavs)} audio files...")
            audio_tasks = [embed_audio(session, embed_url, wav, retrieval_sem) for wav in sample_wavs]
            audio_results = await asyncio.gather(*audio_tasks)
            for r in audio_results:
                if r["status"] == "ok" and r.get("embedding"):
                    audio_embeddings.append(r["embedding"])
                    audio_utt_ids.append(r["utt_id"])

            ref_utt_ids = list(sample_refs.keys())
            ref_texts = [sample_refs[uid] for uid in ref_utt_ids]
            log.info(f"Embedding {len(ref_texts)} reference texts...")
            text_tasks = [
                embed_text(session, embed_url, t, retrieval_sem, idx=i)
                for i, t in enumerate(ref_texts)
            ]
            text_results = await asyncio.gather(*text_tasks)
            successful_text_refs = {}
            for i, r in enumerate(text_results):
                if r["status"] == "ok" and r.get("embedding"):
                    text_embeddings.append(r["embedding"])
                    successful_text_refs[ref_utt_ids[i]] = sample_refs[ref_utt_ids[i]]

            audio_failures = sum(1 for r in audio_results if r["status"] != "ok")
            text_failures = sum(1 for r in text_results if r["status"] != "ok")

            retrieval = compute_retrieval_metrics(
                audio_embeddings, text_embeddings, audio_utt_ids, successful_text_refs, k=args.retrieval_k,
            )
            if retrieval and retrieval.get("error"):
                report["retrieval"] = retrieval
                log.warning(f"Retrieval error: {retrieval['error']}")
            elif retrieval and retrieval.get("recall_at_k") is not None:
                retrieval["audio_failures"] = audio_failures
                retrieval["text_failures"] = text_failures
                report["retrieval"] = retrieval
                log.info(
                    f"Recall@{retrieval['k']}: {retrieval['recall_at_k']:.4f} "
                    f"({retrieval['evaluated_pairs']} pairs evaluated, "
                    f"{audio_failures} audio failures, {text_failures} text failures)"
                )
            else:
                report["retrieval_error"] = {
                    "reason": "no evaluable pairs",
                    "audio_attempted": len(sample_wavs),
                    "audio_succeeded": len(audio_embeddings),
                    "text_attempted": len(ref_texts),
                    "text_succeeded": len(text_embeddings),
                }
                log.warning("Retrieval test failed — no evaluable pairs")

    # --- Quality gates ---
    report["quality_gates"] = check_quality_gates(report)

    # --- Write report ---
    report_clean = json.loads(json.dumps(report, default=str))
    with open(args.output, "w") as f:
        json.dump(report_clean, f, indent=2)
    log.info(f"\nReport written to {args.output}")

    # --- Publish ---
    if args.publish:
        publish_results(report_clean, args.output, args.publish, mode=f"embed-{args.modality}")

    # --- Print summary ---
    print_summary(report)


def check_quality_gates(report):
    """Check embedding quality gates from config."""
    gates_path = SCRIPTS_DIR.parent / "config" / "quality-gates.json"
    if not gates_path.exists():
        return {"status": "skipped", "reason": "quality-gates.json not found"}

    with open(gates_path) as f:
        all_gates = json.load(f)

    gates = all_gates.get("embedding", {})
    if not gates:
        return {"status": "skipped", "reason": "no embedding gates defined"}

    results = {"status": "pass", "checks": []}
    modality = report.get("modality", "audio")

    min_throughput = gates.get(f"min_throughput_{modality}", gates.get("min_throughput_audio", 100))
    max_p99_ms = gates.get(f"max_p99_ms_{modality}", gates.get("max_p99_ms_audio", 100))
    max_failure_rate = gates.get("max_failure_rate", 0.0)
    max_vram_growth_mb = gates.get("max_vram_growth_mb", 100)
    min_recall = gates.get("min_recall_at_k", 0.8)

    sweep = report.get("concurrency_sweep", [])
    if not sweep:
        results["status"] = "fail"
        results["checks"].append({
            "gate": "sweep_evidence",
            "value": 0,
            "threshold": 1,
            "passed": False,
        })
        return results

    for level in sweep:
        if level.get("failure_rate", 0) > max_failure_rate:
            results["status"] = "fail"
            results["checks"].append({
                "gate": "failure_rate",
                "concurrency": level["concurrency"],
                "value": level["failure_rate"],
                "threshold": max_failure_rate,
                "passed": False,
            })

    sustained = report.get("sustained_load", {})
    if sustained and sustained.get("failures", 0) > 0:
        results["status"] = "fail"
        results["checks"].append({
            "gate": "sustained_failures",
            "value": sustained["failures"],
            "threshold": 0,
            "passed": False,
        })

    if sweep:
        peak = max(sweep, key=lambda s: s.get("embeds_per_sec", 0))
        if peak.get("embeds_per_sec", 0) < min_throughput:
            results["status"] = "fail"
            results["checks"].append({
                "gate": "min_throughput",
                "value": peak["embeds_per_sec"],
                "threshold": min_throughput,
                "passed": False,
            })
        else:
            results["checks"].append({
                "gate": "min_throughput",
                "value": peak["embeds_per_sec"],
                "threshold": min_throughput,
                "passed": True,
            })

    c1_results = [s for s in sweep if s["concurrency"] == 1]
    if not c1_results:
        results["checks"].append({
            "gate": "max_p99_ms",
            "value": "skipped (no c=1 in sweep)",
            "threshold": max_p99_ms,
            "passed": True,
        })
    if c1_results:
        p99 = c1_results[0].get("p99_s", 0) * 1000
        passed = p99 <= max_p99_ms
        if not passed:
            results["status"] = "fail"
        results["checks"].append({
            "gate": "max_p99_ms",
            "value": round(p99, 1),
            "threshold": max_p99_ms,
            "passed": passed,
        })

    vram_growth = report.get("vram", {}).get("growth_mb")
    if vram_growth is not None:
        passed = vram_growth <= max_vram_growth_mb
        if not passed:
            results["status"] = "fail"
        results["checks"].append({
            "gate": "max_vram_growth_mb",
            "value": vram_growth,
            "threshold": max_vram_growth_mb,
            "passed": passed,
        })

    retrieval = report.get("retrieval", {})
    if retrieval and retrieval.get("error"):
        results["status"] = "fail"
        results["checks"].append({
            "gate": "retrieval_evidence",
            "value": retrieval["error"],
            "threshold": "valid retrieval results",
            "passed": False,
        })
    elif retrieval and retrieval.get("recall_at_k") is not None:
        if retrieval.get("insufficient_coverage"):
            results["status"] = "fail"
            results["checks"].append({
                "gate": "retrieval_coverage",
                "value": retrieval["evaluated_pairs"],
                "threshold": MIN_RETRIEVAL_PAIRS,
                "passed": False,
            })
        if retrieval.get("trivial_k"):
            results["checks"].append({
                "gate": "retrieval_trivial_k",
                "value": f"k={retrieval['k']} >= candidates={retrieval['total_text']}",
                "threshold": "k < total_text",
                "passed": False,
            })
            results["status"] = "fail"
        passed = retrieval["recall_at_k"] >= min_recall
        if not passed:
            results["status"] = "fail"
        results["checks"].append({
            "gate": "min_recall_at_k",
            "value": retrieval["recall_at_k"],
            "threshold": min_recall,
            "passed": passed,
        })
    elif report.get("retrieval_error"):
        results["status"] = "fail"
        results["checks"].append({
            "gate": "retrieval_evidence",
            "value": report["retrieval_error"].get("reason", "no evaluable pairs"),
            "threshold": "retrieval must produce results",
            "passed": False,
        })

    return results


def publish_results(report, output_path, publish_dir, mode="embed-audio"):
    import shutil

    ts = report.get("timestamp", "")
    ts_slug = ts.replace(":", "").replace("-", "")[:15]
    sys_info = report.get("system", {})
    gpu_raw = sys_info.get("gpu", "gpu")
    gpu_slug = gpu_raw.lower().split()[-1] if gpu_raw else "gpu"

    dir_name = f"{gpu_slug}-{mode}-{ts_slug}"
    dest = os.path.join(publish_dir, dir_name)
    os.makedirs(dest, exist_ok=True)

    shutil.copy2(output_path, os.path.join(dest, "report.json"))

    sweep = report.get("concurrency_sweep", [])
    if sweep:
        sweep_out = {
            "scenario": f"{mode}-concurrency",
            "modality": report.get("modality", ""),
            "gpu": sys_info.get("gpu", ""),
            "git_sha": sys_info.get("git_sha", ""),
            "timestamp": ts,
            "levels": [{
                "concurrency": s["concurrency"],
                "embeds_per_sec": s.get("embeds_per_sec", 0),
                "failures": s.get("failures", 0),
                "p50_ms": round(s.get("p50_s", 0) * 1000),
                "p99_ms": round(s.get("p99_s", 0) * 1000),
            } for s in sweep],
        }
        with open(os.path.join(dest, "concurrency-sweep.json"), "w") as f:
            json.dump(sweep_out, f, indent=2)

    log.info(f"Published to {dest}/")


def print_summary(report):
    """Print human-readable markdown summary."""
    print(f"\n## Embedding Benchmark Summary ({report.get('modality', 'unknown')})")
    print(f"Timestamp: {report.get('timestamp', 'N/A')}")
    print(f"Endpoint: {report.get('endpoint', 'N/A')}")

    dims = report.get("embedding_dimensions")
    if dims:
        print(f"Dimensions: {dims}")

    sweep = report.get("concurrency_sweep", [])
    if sweep:
        peak = max(sweep, key=lambda s: s.get("embeds_per_sec", 0))
        print(f"\nPeak throughput: {peak['embeds_per_sec']:.1f} embeds/sec at c={peak['concurrency']}")
        total_failures = sum(s.get("failures", 0) for s in sweep)
        max_c_ok = max((s["concurrency"] for s in sweep if s.get("failures", 0) == 0), default=0)
        print(f"Max clean concurrency: c={max_c_ok}")
        print(f"Total failures across sweep: {total_failures}")

    sustained = report.get("sustained_load", {})
    if sustained:
        print(f"\nSustained: {sustained.get('embeds_per_sec', 0):.1f} embeds/sec "
              f"at c={sustained.get('concurrency', 0)} for {sustained.get('rounds', 0)} rounds, "
              f"{sustained.get('failures', 0)} failures")

    vram = report.get("vram", {})
    growth = vram.get("growth_mb")
    if growth is not None:
        print(f"\nVRAM growth after warmup: {growth} MB")

    retrieval = report.get("retrieval", {})
    if retrieval:
        print(f"\nRetrieval Recall@{retrieval.get('k', 10)}: {retrieval.get('recall_at_k', 0):.4f} "
              f"({retrieval.get('evaluated_pairs', 0)} pairs)")

    gates = report.get("quality_gates", {})
    if gates.get("checks"):
        print(f"\nQuality Gates: {gates['status'].upper()}")
        for check in gates["checks"]:
            icon = "PASS" if check["passed"] else "FAIL"
            print(f"  [{icon}] {check['gate']}: {check['value']} (threshold: {check['threshold']})")


if __name__ == "__main__":
    asyncio.run(main())
