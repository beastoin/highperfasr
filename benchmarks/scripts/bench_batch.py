#!/usr/bin/env python3
"""
Deterministic batch ASR benchmark with WER.

Downloads LibriSpeech test-clean (200 samples), runs warmup, concurrency sweep,
sustained load, computes WER using wer_utils (Whisper normalization), and outputs
a structured JSON report.

Usage:
    python3 bench_batch.py --server http://localhost:8000
    python3 bench_batch.py --server http://localhost:8000 --sustained-rounds 8
    python3 bench_batch.py --server http://localhost:8000 --concurrency 1,8,16,32,64
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

import aiohttp

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR.parent.parent))

from preflight import detect_server, resolve_batch_url, log_duration_estimate, log_preflight_summary, ensure_unbuffered
from benchmark_telemetry import JsonlWriter, jsonl_path_from_output, emit_sample, emit_resource, emit_report_checkpoint
from profile_gpu import GPUProfiler, CPUProfiler

ensure_unbuffered()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("bench_batch")

LIBRISPEECH_URL = "https://www.openslr.org/resources/12/test-clean.tar.gz"
DATA_DIR = Path("/tmp/librispeech-test-clean")
WAV_DIR = DATA_DIR / "wav"
REF_FILE = DATA_DIR / "references.tsv"
MAX_SAMPLES = 200
REFERENCE_WER_PCT = {
    ("batch", "librispeech-test-clean"): 1.92,
    ("streaming-realtime", "librispeech-test-clean"): 3.21,
}


def load_dataset_manifest(dataset_name: str, max_samples: int = 0, cache_dir=None):
    """Load dataset from the multi-corpus registry. Returns (manifest, refs_dict)."""
    from benchmarks.datasets.registry import load_dataset

    manifest = load_dataset(dataset_name, cache_dir=cache_dir, max_samples=max_samples)
    refs = {e["utt_id"]: e["reference"] for e in manifest if e.get("reference")}
    log.info(f"Dataset '{dataset_name}': {len(manifest)} files, {len(refs)} references")
    return manifest, refs


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

    Falls back to simple cycling when concurrency exceeds manifest size
    (high-c sweeps don't need per-round uniqueness).
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


def ensure_librispeech(max_samples=None):
    """Download and extract LibriSpeech test-clean if not cached.

    Args:
        max_samples: Max WAV files to extract. None uses MAX_SAMPLES (200).
                     0 means extract all (~2620 files).
    """
    limit = MAX_SAMPLES if max_samples is None else max_samples
    target = limit if limit > 0 else 2620

    if WAV_DIR.exists() and len(list(WAV_DIR.glob("*.wav"))) >= target and REF_FILE.exists():
        n = len(list(WAV_DIR.glob("*.wav")))
        log.info(f"LibriSpeech test-clean cached: {n} WAV files")
        return

    log.info("Downloading LibriSpeech test-clean...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    WAV_DIR.mkdir(exist_ok=True)
    tar_path = DATA_DIR / "test-clean.tar.gz"

    if not tar_path.exists():
        urllib.request.urlretrieve(LIBRISPEECH_URL, tar_path)
        log.info(f"Downloaded {tar_path.stat().st_size / 1e6:.0f}MB")
        import hashlib
        sha256 = hashlib.sha256(tar_path.read_bytes()).hexdigest()
        log.info(f"SHA256: {sha256}")

    log.info("Extracting...")
    refs = {}
    count = 0
    existing = len(list(WAV_DIR.glob("*.wav")))
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
    """Load reference transcripts as {utt_id: text}."""
    refs = {}
    with open(REF_FILE) as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                refs[parts[0]] = parts[1]
    return refs


def compute_wer(references, hypotheses):
    """Compute WER using wer_utils (Whisper normalization). Hard-fails if unavailable."""
    try:
        from wer_utils import corpus_wer, pair_wer
    except ImportError:
        log.error("wer_utils requires jiwer and whisper-normalizer. Install: pip install jiwer whisper-normalizer")
        raise SystemExit(1)

    wer_val = corpus_wer(references, hypotheses)
    per_utt = [pair_wer(r, h) for r, h in zip(references, hypotheses)]
    return wer_val, per_utt


def summarize_wer_results(results, refs):
    """Compute report-ready WER fields from successful benchmark results."""
    ref_texts = []
    hyp_texts = []
    wer_failures = sum(1 for r in results if r["status"] != "ok")
    for r in (r for r in results if r["status"] == "ok"):
        utt_id = r["utt_id"]
        if utt_id in refs:
            ref_texts.append(refs[utt_id])
            hyp_texts.append(r["text"])

    if not ref_texts:
        return None

    wer_val, per_utt = compute_wer(ref_texts, hyp_texts)
    high_wer_count = sum(1 for per_utt_wer in per_utt if per_utt_wer > 0.1)
    return {
        "corpus_wer_pct": round(wer_val * 100, 2),
        "samples_evaluated": len(ref_texts),
        "samples_failed": wer_failures,
        "normalization": "whisper_english",
        "high_wer_count": high_wer_count,
    }


def collect_system_info(gpu_override=None, image_tag=None):
    """Collect system metadata for report reproducibility."""
    import platform as _platform
    import subprocess as _sp

    def _run(cmd):
        try:
            return _sp.check_output(cmd, shell=True, text=True, timeout=5).strip()
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

    if image_tag:
        info["container_image"] = image_tag
    elif not image_tag:
        sha = git_sha or _run("git rev-parse --short HEAD")
        if sha:
            info["container_image"] = f"sha-{sha}"

    if gpu_override:
        info["gpu"] = gpu_override
    else:
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

    return info


def publish_results(report, output_path, publish_dir, mode="batch"):
    """Copy report to an auto-named directory under publish_dir.

    Naming convention: {gpu}-{mode}-{timestamp}
    e.g. l4-batch-20260716T040510
    """
    import shutil

    ts = report.get("timestamp", "")
    if not ts:
        raise ValueError("Report missing 'timestamp' — cannot publish without benchmark start time")
    ts_slug = ts.replace(":", "").replace("-", "")[:15]

    sys_info = report.get("system", {})
    gpu_raw = sys_info.get("gpu", "gpu")
    gpu_slug = gpu_raw.lower().split()[-1] if gpu_raw else "gpu"

    dir_name = f"{gpu_slug}-{mode}-{ts_slug}"
    dest = os.path.join(publish_dir, dir_name)
    os.makedirs(dest, exist_ok=True)

    report_dest = os.path.join(dest, "report.json")
    shutil.copy2(output_path, report_dest)

    sweep = report.get("concurrency_sweep", [])
    if sweep:
        sweep_out = {
            "scenario": f"{mode}-concurrency",
            "dataset": report.get("dataset", ""),
            "gpu": sys_info.get("gpu", ""),
            "model": report.get("command", "").split("--model")[-1].split()[0] if "--model" in report.get("command", "") else "",
            "git_sha": sys_info.get("git_sha", ""),
            "timestamp": ts,
            "levels": [],
        }
        for s in sweep:
            level = {"concurrency": s["concurrency"], "rtfx": s.get("rtfx", 0), "failures": s.get("failures", 0)}
            if "rps" in s:
                level["rps"] = s["rps"]
            if "sess_per_min" in s:
                level["sessions_per_min"] = s["sess_per_min"]
            if "p50_s" in s:
                level["p50_ms"] = round(s["p50_s"] * 1000)
            if "p99_s" in s:
                level["p99_ms"] = round(s["p99_s"] * 1000)
            sweep_out["levels"].append(level)

        if "wer" in report:
            sweep_out["wer"] = report["wer"].get("corpus_wer_pct")
            sweep_out["normalization"] = report["wer"].get("normalization", "")

        with open(os.path.join(dest, "concurrency-sweep.json"), "w") as f:
            json.dump(sweep_out, f, indent=2)

    log.info(f"Published to {dest}/")


def collect_gpu_memory_used_mb():
    """Return max used GPU memory across visible GPUs, or None when unavailable."""
    import subprocess as _sp

    try:
        output = _sp.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
            timeout=5,
        ).strip()
    except Exception:
        return None

    values = []
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            values.append(int(line))
        except ValueError:
            return None
    return max(values) if values else None


def reference_wer_pct(mode, dataset_name, override=None, baseline_report=None):
    """Resolve reference-model WER for WER delta gates."""
    if override is not None:
        return override
    if baseline_report:
        live_ref = baseline_report.get("wer", {}).get("reference_wer_pct")
        if live_ref is not None:
            return live_ref
        quality_ref = baseline_report.get("quality", {}).get("reference_wer")
        if quality_ref is not None:
            return quality_ref
        live_wer = baseline_report.get("wer", {}).get("corpus_wer_pct")
        if live_wer is not None:
            return live_wer
        quality_wer = baseline_report.get("quality", {}).get("wer")
        if quality_wer is not None:
            return quality_wer
    return REFERENCE_WER_PCT.get((mode, dataset_name))


def get_wav_duration(wav_path):
    """Get audio duration in seconds from WAV file."""
    import wave

    with wave.open(str(wav_path), "rb") as w:
        return w.getnframes() / w.getframerate()


async def transcribe_file(session, url, wav_path, semaphore):
    """Send one file to the batch transcription endpoint, return result dict."""
    async with semaphore:
        t0 = time.monotonic()
        audio_dur = get_wav_duration(wav_path)
        try:
            data = aiohttp.FormData()
            data.add_field(
                "file",
                open(wav_path, "rb"),
                filename=os.path.basename(wav_path),
                content_type="audio/wav",
            )
            async with session.post(url, data=data, timeout=aiohttp.ClientTimeout(total=120)) as resp:
                elapsed = time.monotonic() - t0
                if resp.status == 200:
                    result = await resp.json()
                    return {
                        "utt_id": Path(wav_path).stem,
                        "text": result.get("text", ""),
                        "elapsed": elapsed,
                        "audio_dur": audio_dur,
                        "status": "ok",
                    }
                else:
                    body = await resp.text()
                    return {
                        "utt_id": Path(wav_path).stem,
                        "error": f"HTTP {resp.status}: {body[:200]}",
                        "elapsed": elapsed,
                        "status": "error",
                    }
        except Exception as e:
            return {
                "utt_id": Path(wav_path).stem,
                "error": str(e)[:200],
                "elapsed": time.monotonic() - t0,
                "status": "error",
            }


async def run_sweep(url, manifest, concurrency, repeat=1, target_count=None, on_result=None):
    """Run one concurrency level, return results list.

    Args:
        on_result: optional callback(result_dict, index) called as each request completes.
    """
    if target_count is None:
        target_count = len(manifest) * repeat
    files = [Path(e["wav_path"]) for e in select_round_robin_entries(manifest, concurrency, target_count)]
    sem = asyncio.Semaphore(concurrency)
    t0 = time.monotonic()
    collected = []
    idx_counter = [0]

    async def _run_one(f):
        r = await transcribe_file(session, url, f, sem)
        i = idx_counter[0]
        idx_counter[0] += 1
        collected.append(r)
        if on_result:
            try:
                on_result(r, i)
            except Exception as exc:
                log.warning("on_result callback failed for sample %d: %s", i, exc)
        return r

    async with aiohttp.ClientSession() as session:
        tasks = [_run_one(f) for f in files]
        await asyncio.gather(*tasks)
    wall = time.monotonic() - t0
    return collected, wall


def summarize_sweep(results, wall_time, concurrency):
    """Compute throughput/latency/RTFx summary for one concurrency level."""
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


def check_regression(current, baseline_level, metric="rps", threshold=0.20):
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
    parser = argparse.ArgumentParser(description="Deterministic batch ASR benchmark with WER")
    parser.add_argument("--server", default="http://localhost:8000", help="Server base URL")
    parser.add_argument(
        "--concurrency",
        default="1,8,16,32,64",
        help="Comma-separated concurrency levels (default: 1,8,16,32,64)",
    )
    parser.add_argument("--sustained-rounds", type=int, default=8, help="Sustained load rounds (default: 8)")
    parser.add_argument(
        "--sustained-concurrency", type=int, default=64, help="Sustained load concurrency (default: 64)"
    )
    parser.add_argument("--warmup", type=int, default=50, help="Warmup requests (default: 50)")
    parser.add_argument("--output", default="/tmp/bench_batch_report.json", help="Output JSON path")
    parser.add_argument("--skip-wer", action="store_true", help="Skip WER computation")
    parser.add_argument("--endpoint", default="/v1/transcriptions", help="Transcription endpoint path (default: /v1/transcriptions)")
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
                        help="Quick validation: 200 samples (use full corpus for publishable results)")
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
    log_preflight_summary(server_info, "batch")
    server_base = resolve_batch_url(args.server, server_info)
    url = f"{server_base}{args.endpoint}"

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

    log.info("=== Deterministic Batch ASR Benchmark ===")
    log.info(f"Server: {args.server}")
    log.info(f"Concurrency levels: {levels}")

    # Step 1: Load dataset (always via registry)
    dataset_name = args.dataset or "librispeech-test-clean"
    max_samples = args.max_samples
    manifest, refs = load_dataset_manifest(
        dataset_name, max_samples=max_samples, cache_dir=args.dataset_dir
    )
    log.info(f"Using {len(manifest)} WAV files, {len(refs)} references")

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
        "benchmark": "Batch ASR Benchmark",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "server": args.server,
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
    jsonl_writer.emit("run_start", mode="batch", dataset=dataset_name,
                      samples=len(manifest), avg_file_duration_s=avg_file_duration_s,
                      levels=[int(x) for x in args.concurrency.split(",")])

    # Step 2: Warmup
    warmup_c = min(16, len(manifest), max(args.warmup, 1))
    log.info(f"Warmup: {args.warmup} requests at c={warmup_c}...")
    await run_sweep(url, manifest, concurrency=warmup_c, target_count=args.warmup)
    log.info("Warmup complete")

    # Step 3: WER evaluation (c=1 for deterministic ordering)
    if not args.skip_wer:
        total_audio = sum(e.get("duration_s", 5.0) for e in manifest)
        log_duration_estimate(len(manifest), total_audio, mode="batch")
        wer_results, _ = await run_sweep(
            url, manifest, concurrency=1, target_count=len(manifest),
            on_result=lambda r, i: emit_sample(jsonl_writer, r, "wer", 1, i),
        )

        wer_summary = summarize_wer_results(wer_results, refs)

        if wer_summary:
            report["wer"] = wer_summary
            report["wer"]["c1_corpus_wer_pct"] = report["wer"]["corpus_wer_pct"]
            ref_wer = reference_wer_pct(
                "batch", dataset_name, override=args.reference_wer_pct, baseline_report=baseline_report
            )
            if ref_wer is not None:
                report["wer"]["reference_wer_pct"] = ref_wer
            log.info(
                f"WER: {report['wer']['corpus_wer_pct']:.2f}% "
                f"({report['wer']['samples_evaluated']} samples, "
                f"{report['wer']['high_wer_count']} with >10% WER)"
            )

            if report["wer"]["samples_failed"] > 0:
                log.error(
                    f"WER incomplete: {report['wer']['samples_failed']} of "
                    f"{len(manifest)} eval items failed at c=1"
                )

            if args.smart and baseline_report and "wer" in baseline_report:
                base_wer = baseline_report["wer"]["corpus_wer_pct"]
                cur_wer = report["wer"]["corpus_wer_pct"]
                log.info(f"  vs baseline: {base_wer}% -> {cur_wer}% (delta {cur_wer - base_wer:+.2f}%)")
        else:
            log.warning("No matching references found for WER")
    else:
        log.info("WER evaluation skipped (--skip-wer)")

    # Step 4: Concurrency sweep (warm model)
    emit_report_checkpoint(jsonl_writer, report, "before_sweep")
    log.info("Concurrency sweep (warm model)...")
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
                url, manifest, concurrency=c, target_count=max(len(manifest), c),
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
        jsonl_writer.emit("level_end", phase="sweep", concurrency=c,
                          rtfx=summary["rtfx"], rps=summary["rps"],
                          failures=summary["failures"],
                          client_gpu_util_avg=gpu_summary.get("gpu_util_pct", {}).get("mean"),
                          client_vram_peak_mb=gpu_summary.get("vram_mb", {}).get("max"),
                          client_cpu_avg=cpu_summary.get("mean"))
        sweep_results.append(summary)
        report["concurrency_sweep"] = sweep_results
        emit_report_checkpoint(jsonl_writer, report, f"after_level_c{c}")
        gpu_info = ""
        if "error" not in gpu_summary:
            gpu_info = f", GPU={gpu_summary['gpu_util_pct']['mean']}%, VRAM={gpu_summary['vram_mb']['max']}MB"
        log.info(
            f"    RTFx={summary['rtfx']}x, p99={summary.get('p99_s', '?')}s, "
            f"p50={summary.get('p50_s', '?')}s, failures={summary['failures']}, "
            f"RPS={summary['rps']}{gpu_info}"
        )

        if args.smart and baseline_sweep:
            bl = baseline_sweep.get(c)
            reg_rps, msg_rps = check_regression(summary, bl, "rps")
            reg_rtfx, msg_rtfx = check_regression(summary, bl, "rtfx")
            log.info(f"    vs baseline: {msg_rps} | {msg_rtfx}")

            if not reg_rps and not reg_rtfx and summary["failures"] == 0:
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
    emit_report_checkpoint(jsonl_writer, report, "after_sweep")

    # Step 5: Sustained load
    sustained_c = args.sustained_concurrency
    rounds = args.sustained_rounds
    log.info(f"Sustained load: c={sustained_c}, {rounds} rounds x {len(manifest)} files...")
    jsonl_writer.emit("level_start", phase="sustained", concurrency=sustained_c, rounds=rounds)
    sus_gpu = GPUProfiler(interval_s=1.0,
                          on_sample=lambda s: emit_resource(jsonl_writer, s, None, "sustained", sustained_c))
    sus_cpu = CPUProfiler(interval_s=1.0,
                          on_sample=lambda v: emit_resource(jsonl_writer, None, v, "sustained", sustained_c))
    sus_gpu.start()
    sus_cpu.start()
    try:
        sustained_results, sustained_wall = await run_sweep(
            url, manifest, concurrency=sustained_c, repeat=rounds,
            on_result=lambda r, i: emit_sample(jsonl_writer, r, "sustained", sustained_c, i),
        )
    finally:
        sus_gpu_samples = sus_gpu.stop()
        sus_cpu_samples = sus_cpu.stop()
    sustained_summary = summarize_sweep(sustained_results, sustained_wall, sustained_c)
    sustained_summary["rounds"] = rounds
    sustained_summary["total_files"] = len(manifest) * rounds
    sustained_summary["client_gpu_profile"] = GPUProfiler.summarize(sus_gpu_samples)
    sustained_summary["client_cpu_profile"] = CPUProfiler.summarize(sus_cpu_samples)
    if "error" not in sustained_summary["client_gpu_profile"]:
        sustained_summary["client_bottleneck"] = GPUProfiler.classify_bottleneck(sustained_summary["client_gpu_profile"])
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
                f"at c={sustained_c} ({load_wer_summary['samples_evaluated']} samples)"
            )
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
    sus_gpu_info = ""
    if "error" not in sustained_summary.get("client_gpu_profile", {"error": True}):
        sg = sustained_summary["client_gpu_profile"]
        sus_gpu_info = f", GPU={sg['gpu_util_pct']['mean']}%, VRAM={sg['vram_mb']['max']}MB"
    log.info(
        f"  Sustained: RTFx={sustained_summary['rtfx']}x, "
        f"{sustained_summary['total_files']} files, "
        f"failures={sustained_summary['failures']}, "
        f"RPS={sustained_summary['rps']}{sus_gpu_info}"
    )
    jsonl_writer.emit("level_end", phase="sustained", concurrency=sustained_c,
                      rtfx=sustained_summary["rtfx"], rps=sustained_summary["rps"],
                      failures=sustained_summary["failures"])

    if args.smart and baseline_report and "sustained_load" in baseline_report:
        bl_s = baseline_report["sustained_load"]
        log.info(
            f"  Sustained vs baseline: "
            f"RPS {bl_s.get('rps', '?')} -> {sustained_summary['rps']}, "
            f"RTFx {bl_s.get('rtfx', '?')} -> {sustained_summary['rtfx']}"
        )

    # Step 6: Summary
    peak = max(sweep_results, key=lambda x: x["rtfx"])
    peak_rps_entry = max(sweep_results, key=lambda x: x["rps"])
    max_conc_zero_fail = max(
        (s for s in sweep_results if s["failures"] == 0),
        key=lambda x: x["concurrency"],
        default=peak,
    )
    report["summary"] = {
        "peak_rtfx": peak["rtfx"],
        "peak_concurrency": peak["concurrency"],
        "peak_rps": peak_rps_entry["rps"],
        "max_concurrency_zero_fail": max_conc_zero_fail["concurrency"],
        "sustained_rtfx": sustained_summary["rtfx"],
        "sustained_rps": sustained_summary["rps"],
        "total_failures": sum(s["failures"] for s in sweep_results) + sustained_summary["failures"],
        "wer_pct": report.get("wer", {}).get("corpus_wer_pct"),
    }

    # Print markdown report
    print()
    print("## Batch Benchmark Results")
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
    print("| c | RTFx | p50 | p99 | Failures | RPS | RTF |")
    print("|---|------|-----|-----|----------|-----|-----|")
    for s in sweep_results:
        print(
            f"| {s['concurrency']} | {s['rtfx']}x | {s.get('p50_s', '?')}s | {s.get('p99_s', '?')}s "
            f"| {s['failures']} | {s['rps']} | {s['rtf']} |"
        )
    if report.get("smart_skipped"):
        print(f"\n*Smart early-stop: skipped c={report['smart_skipped']} (matched baseline)*")
    print()
    print(
        f"**Peak:** {peak['rtfx']}x RTFx at c={peak['concurrency']} "
        f"({peak['rps']} RPS) | **Max concurrency (0 failures):** c={max_conc_zero_fail['concurrency']}"
    )
    print()
    print("### Sustained Load")
    print(f"| Metric | Value |")
    print(f"|--------|-------|")
    print(f"| Concurrency | {sustained_summary['concurrency']} |")
    print(f"| Total files | {sustained_summary['total_files']} |")
    print(f"| RTFx | {sustained_summary['rtfx']}x |")
    print(f"| p50 / p99 | {sustained_summary.get('p50_s', '?')}s / {sustained_summary.get('p99_s', '?')}s |")
    print(f"| Failures | {sustained_summary['failures']} |")
    print(f"| RPS | {sustained_summary['rps']} |")

    # Multi-trial aggregation
    if args.trials > 1:
        from benchmarks.scripts.stats import summarize_trials

        all_peak_rps = [peak_rps_entry["rps"]]
        all_peak_rtfx = [peak["rtfx"]]
        all_sustained_rps = [sustained_summary["rps"]]
        all_sustained_rtfx = [sustained_summary["rtfx"]]
        trial_total_failures = report["summary"]["total_failures"]

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
                        url, manifest, concurrency=c, target_count=max(len(manifest), c),
                        on_result=lambda r, i, _c=c: emit_sample(jsonl_writer, r, "sweep", _c, i),
                    )
                finally:
                    t_gpu.stop()
                    t_cpu.stop()
                t_sweep.append(summarize_sweep(results, wall, c))
                jsonl_writer.emit("level_end", phase="sweep", concurrency=c, trial=trial,
                                  rtfx=t_sweep[-1]["rtfx"], rps=t_sweep[-1]["rps"],
                                  failures=t_sweep[-1]["failures"])
            t_peak_rtfx = max(t_sweep, key=lambda x: x["rtfx"])
            t_peak_rps = max(t_sweep, key=lambda x: x["rps"])
            all_peak_rps.append(t_peak_rps["rps"])
            all_peak_rtfx.append(t_peak_rtfx["rtfx"])

            jsonl_writer.emit("level_start", phase="sustained", concurrency=sustained_c, trial=trial)
            ts_gpu = GPUProfiler(interval_s=1.0,
                                 on_sample=lambda s: emit_resource(jsonl_writer, s, None, "sustained", sustained_c))
            ts_cpu = CPUProfiler(interval_s=1.0,
                                 on_sample=lambda v: emit_resource(jsonl_writer, None, v, "sustained", sustained_c))
            ts_gpu.start()
            ts_cpu.start()
            try:
                t_sus_results, t_sus_wall = await run_sweep(
                    url, manifest, concurrency=sustained_c, repeat=rounds,
                    on_result=lambda r, i: emit_sample(jsonl_writer, r, "sustained", sustained_c, i),
                )
            finally:
                ts_gpu.stop()
                ts_cpu.stop()
            t_sus = summarize_sweep(t_sus_results, t_sus_wall, sustained_c)
            all_sustained_rps.append(t_sus["rps"])
            all_sustained_rtfx.append(t_sus["rtfx"])
            t_failures = sum(s["failures"] for s in t_sweep) + t_sus["failures"]
            trial_total_failures += t_failures
            jsonl_writer.emit("level_end", phase="sustained", concurrency=sustained_c, trial=trial,
                              rtfx=t_sus["rtfx"], rps=t_sus["rps"], failures=t_sus["failures"])
            jsonl_writer.emit("trial_end", trial=trial, failures=t_failures)
            log.info(f"  Trial {trial}: peak={t_peak_rps['rps']} RPS, sustained={t_sus['rps']} RPS, failures={t_failures}")

        report["summary"]["total_failures"] = trial_total_failures

        report["trials"] = {
            "count": args.trials,
            "peak_rps": summarize_trials(all_peak_rps),
            "peak_rtfx": summarize_trials(all_peak_rtfx),
            "sustained_rps": summarize_trials(all_sustained_rps),
            "sustained_rtfx": summarize_trials(all_sustained_rtfx),
        }
        log.info(f"Trials summary: peak RPS={report['trials']['peak_rps']}")

    # Quality gate evaluation (skipped in quick mode — gates require full sustained load)
    gates_path = Path(__file__).parent.parent / "config" / "quality-gates.json"
    if gates_path.exists() and not args.quick:
        from benchmarks.scripts.gates import load_gates, evaluate_gates, exit_code_for_gates

        gates = load_gates(str(gates_path))
        gate_result = evaluate_gates(report, gates)
        report["quality_gates"] = gate_result
        for g in gate_result["gates"]:
            status = "PASS" if g["passed"] else "FAIL"
            log.info(f"  Gate {status}: {g['gate']} — threshold={g['threshold']}, actual={g['actual']}")
    elif args.quick:
        log.info("Quick mode: skipping quality gates (use full corpus for gate enforcement)")

    emit_report_checkpoint(jsonl_writer, report, "final")
    jsonl_writer.emit("run_end", total_failures=report["summary"]["total_failures"])
    jsonl_writer.close()
    log.info(f"JSONL log saved to {jsonl_path}")

    # Save report
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"Report saved to {args.output}")

    if args.publish:
        publish_results(report, args.output, args.publish, mode="batch")


    total_failures = report["summary"]["total_failures"]
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
                reg_rps, _ = check_regression(s, bl, "rps")
                reg_rtfx, _ = check_regression(s, bl, "rtfx")
                if reg_rps or reg_rtfx:
                    regressions.append(s["concurrency"])
        if regressions:
            log.error(f"FAIL: regression detected at concurrency levels {regressions}")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
