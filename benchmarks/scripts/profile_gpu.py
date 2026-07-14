#!/usr/bin/env python3
"""
GPU VRAM profiling and bottleneck identification.

Collects VRAM usage, GPU compute utilization, and CPU utilization at each
concurrency level during a benchmark run. Classifies the bottleneck as
VRAM-limited, compute-limited, or CPU-limited.

Usage:
    python3 profile_gpu.py --server http://localhost:8000 --mode batch
    python3 profile_gpu.py --server ws://localhost:8001 --mode stream
    python3 profile_gpu.py --server http://localhost:8000 --mode batch --concurrency 1,8,16,32,64
"""

import argparse
import asyncio
import json
import logging
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("profile_gpu")

NVIDIA_SMI = shutil.which("nvidia-smi")


def parse_nvidia_smi() -> dict | None:
    """Query nvidia-smi for current GPU state."""
    if not NVIDIA_SMI:
        return None
    try:
        result = subprocess.run(
            [
                NVIDIA_SMI,
                "--query-gpu=memory.used,memory.total,memory.free,utilization.gpu,utilization.memory,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return None
        parts = [p.strip() for p in result.stdout.strip().split(",")]
        if len(parts) < 7:
            return None
        return {
            "vram_used_mb": int(parts[0]),
            "vram_total_mb": int(parts[1]),
            "vram_free_mb": int(parts[2]),
            "gpu_util_pct": int(parts[3]),
            "mem_bw_util_pct": int(parts[4]),
            "temp_c": int(parts[5]),
            "power_w": float(parts[6]),
        }
    except (subprocess.TimeoutExpired, ValueError, IndexError):
        return None


class GPUProfiler:
    """Background thread that samples nvidia-smi at regular intervals."""

    def __init__(self, interval_s: float = 1.0):
        self._interval = interval_s
        self._samples: list[dict] = []
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self):
        self._samples = []
        self._running = True
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> list[dict]:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        return self._samples

    def _sample_loop(self):
        while self._running:
            sample = parse_nvidia_smi()
            if sample:
                sample["timestamp"] = time.monotonic()
                self._samples.append(sample)
            time.sleep(self._interval)

    @staticmethod
    def summarize(samples: list[dict]) -> dict:
        """Compute summary statistics from GPU samples."""
        if not samples:
            return {"error": "no GPU samples collected"}

        vram = [s["vram_used_mb"] for s in samples]
        gpu = [s["gpu_util_pct"] for s in samples]
        mem_bw = [s["mem_bw_util_pct"] for s in samples]
        power = [s["power_w"] for s in samples]
        temp = [s["temp_c"] for s in samples]

        return {
            "samples": len(samples),
            "vram_mb": {
                "min": min(vram),
                "max": max(vram),
                "mean": round(sum(vram) / len(vram), 1),
                "total": samples[0]["vram_total_mb"],
                "highwater_pct": round(max(vram) / samples[0]["vram_total_mb"] * 100, 1),
            },
            "gpu_util_pct": {
                "min": min(gpu),
                "max": max(gpu),
                "mean": round(sum(gpu) / len(gpu), 1),
            },
            "mem_bw_util_pct": {
                "min": min(mem_bw),
                "max": max(mem_bw),
                "mean": round(sum(mem_bw) / len(mem_bw), 1),
            },
            "power_w": {
                "min": round(min(power), 1),
                "max": round(max(power), 1),
                "mean": round(sum(power) / len(power), 1),
            },
            "temp_c": {
                "min": min(temp),
                "max": max(temp),
            },
        }

    @staticmethod
    def classify_bottleneck(summary: dict) -> dict:
        """Classify the bottleneck based on GPU metrics.

        Returns:
            {"bottleneck": "vram"|"compute"|"cpu"|"unknown", "confidence": float, "reasoning": str}
        """
        if "error" in summary:
            return {"bottleneck": "unknown", "confidence": 0.0, "reasoning": summary["error"]}

        vram_pct = summary["vram_mb"]["highwater_pct"]
        gpu_mean = summary["gpu_util_pct"]["mean"]
        mem_bw_mean = summary["mem_bw_util_pct"]["mean"]

        if vram_pct > 90:
            return {
                "bottleneck": "vram",
                "confidence": min(1.0, vram_pct / 100),
                "reasoning": f"VRAM at {vram_pct}% — approaching OOM. "
                f"GPU compute at {gpu_mean}%, memory bandwidth at {mem_bw_mean}%.",
            }

        if gpu_mean > 85:
            return {
                "bottleneck": "compute",
                "confidence": min(1.0, gpu_mean / 100),
                "reasoning": f"GPU compute at {gpu_mean}% — saturated. "
                f"VRAM headroom: {100 - vram_pct:.0f}%. Memory bandwidth at {mem_bw_mean}%.",
            }

        if mem_bw_mean > 80:
            return {
                "bottleneck": "memory_bandwidth",
                "confidence": min(1.0, mem_bw_mean / 100),
                "reasoning": f"Memory bandwidth at {mem_bw_mean}% — saturated. "
                f"GPU compute at {gpu_mean}%, VRAM at {vram_pct}%.",
            }

        if gpu_mean < 50 and vram_pct < 70:
            return {
                "bottleneck": "cpu",
                "confidence": 0.7,
                "reasoning": f"GPU underutilized (compute {gpu_mean}%, VRAM {vram_pct}%) — "
                f"likely CPU or I/O bottleneck. Check host CPU and network.",
            }

        return {
            "bottleneck": "balanced",
            "confidence": 0.5,
            "reasoning": f"No clear bottleneck. GPU compute {gpu_mean}%, VRAM {vram_pct}%, "
            f"memory bandwidth {mem_bw_mean}%. System is reasonably balanced.",
        }


def _read_cpu_times() -> tuple[int, int] | None:
    """Read aggregate CPU idle and total jiffies from /proc/stat."""
    try:
        with open("/proc/stat") as f:
            line = f.readline()
        parts = line.split()
        if not parts or parts[0] != "cpu":
            return None
        idle = int(parts[4])
        total = sum(int(p) for p in parts[1:])
        return idle, total
    except (FileNotFoundError, ValueError, IndexError):
        return None


def _cpu_util_between(before: tuple[int, int], after: tuple[int, int]) -> float:
    """Compute CPU utilization between two /proc/stat samples."""
    idle, total = before
    idle2, total2 = after
    return round((1 - (idle2 - idle) / max(total2 - total, 1)) * 100, 1)


def get_cpu_utilization() -> float | None:
    """Get current system CPU utilization percentage."""
    before = _read_cpu_times()
    if before is None:
        return None
    time.sleep(0.5)
    after = _read_cpu_times()
    if after is None:
        return None
    return _cpu_util_between(before, after)


class CPUProfiler:
    """Background thread that samples host CPU utilization during load."""

    def __init__(self, interval_s: float = 1.0):
        self._interval = interval_s
        self._samples: list[float] = []
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self):
        self._samples = []
        self._running = True
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> list[float]:
        self._running = False
        if self._thread:
            self._thread.join(timeout=max(5, self._interval * 2))
        return self._samples

    def _sample_loop(self):
        previous = _read_cpu_times()
        while self._running:
            time.sleep(self._interval)
            current = _read_cpu_times()
            if previous is not None and current is not None:
                self._samples.append(_cpu_util_between(previous, current))
            previous = current

    @staticmethod
    def summarize(samples: list[float]) -> dict:
        """Compute summary statistics from CPU samples."""
        if not samples:
            return {"error": "no CPU samples collected"}
        return {
            "samples": len(samples),
            "min": min(samples),
            "max": max(samples),
            "mean": round(sum(samples) / len(samples), 1),
        }


async def run_batch_load(server: str, wav_files: list[str], concurrency: int, duration_s: float = 30,
                         endpoint: str = "/v1/transcriptions"):
    """Run batch load at given concurrency for duration."""
    import aiohttp

    url = f"{server}{endpoint}"
    sem = asyncio.Semaphore(concurrency)
    results = []
    stop_time = time.monotonic() + duration_s
    pool: list[str] = []
    rng = random.Random(42)

    def next_wav() -> str:
        nonlocal pool
        if not pool:
            pool = list(wav_files)
            rng.shuffle(pool)
        return pool.pop()

    async def send_one():
        async with sem:
            if time.monotonic() > stop_time:
                return
            wav = next_wav()
            try:
                data = aiohttp.FormData()
                data.add_field("file", open(wav, "rb"), filename=os.path.basename(wav), content_type="audio/wav")
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, data=data, timeout=aiohttp.ClientTimeout(total=120)) as resp:
                        results.append(resp.status == 200)
            except Exception:
                results.append(False)

    tasks = []
    while time.monotonic() < stop_time:
        tasks.append(asyncio.create_task(send_one()))
        await asyncio.sleep(0.05)

    await asyncio.gather(*tasks, return_exceptions=True)
    return results


async def run_stream_load(server: str, wav_files: list[str], concurrency: int, duration_s: float = 30,
                          endpoint: str = "/v1/stream"):
    """Run streaming load at given concurrency for duration."""
    import websockets

    endpoint = f"{server}{endpoint}"
    sem = asyncio.Semaphore(concurrency)
    results = []
    stop_time = time.monotonic() + duration_s
    pool: list[str] = []
    rng = random.Random(42)

    def next_wav() -> str:
        nonlocal pool
        if not pool:
            pool = list(wav_files)
            rng.shuffle(pool)
        return pool.pop()

    async def stream_one():
        async with sem:
            if time.monotonic() > stop_time:
                return
            wav = next_wav()
            try:
                with open(wav, "rb") as f:
                    f.read(44)
                    raw = f.read()

                async with websockets.connect(endpoint, max_size=10 * 1024 * 1024) as ws:
                    config = json.dumps({"format": "pcm_s16le", "sample_rate": 16000, "language": "en"})
                    await ws.send(config)
                    await asyncio.wait_for(ws.recv(), timeout=5)

                    chunk_size = 16000 * 2 * 160 // 1000
                    offset = 0
                    while offset < len(raw) and time.monotonic() < stop_time:
                        chunk = raw[offset : offset + chunk_size]
                        await ws.send(chunk)
                        offset += chunk_size
                        await asyncio.sleep(0.16)

                    await ws.send(json.dumps({"action": "close"}))
                    try:
                        while True:
                            msg = await asyncio.wait_for(ws.recv(), timeout=3)
                            resp = json.loads(msg)
                            if resp.get("done") or resp.get("status") == "closed":
                                break
                    except (asyncio.TimeoutError, Exception):
                        pass
                    results.append(True)
            except Exception:
                results.append(False)

    tasks = []
    while time.monotonic() < stop_time:
        tasks.append(asyncio.create_task(stream_one()))
        await asyncio.sleep(0.1)

    await asyncio.gather(*tasks, return_exceptions=True)
    return results


async def profile_at_concurrency(
    server: str,
    mode: str,
    wav_files: list[str],
    concurrency: int,
    duration_s: float = 30,
    sample_interval: float = 1.0,
    endpoint: str | None = None,
) -> dict:
    """Profile GPU at one concurrency level."""
    log.info(f"Profiling c={concurrency} for {duration_s}s...")

    profiler = GPUProfiler(interval_s=sample_interval)
    cpu_profiler = CPUProfiler(interval_s=sample_interval)

    baseline = parse_nvidia_smi()

    profiler.start()
    cpu_profiler.start()

    if mode == "batch":
        kwargs = {"endpoint": endpoint} if endpoint else {}
        results = await run_batch_load(server, wav_files, concurrency, duration_s, **kwargs)
    else:
        kwargs = {"endpoint": endpoint} if endpoint else {}
        results = await run_stream_load(server, wav_files, concurrency, duration_s, **kwargs)

    samples = profiler.stop()
    cpu_samples = cpu_profiler.stop()

    gpu_summary = GPUProfiler.summarize(samples)
    cpu_summary = CPUProfiler.summarize(cpu_samples)
    bottleneck = GPUProfiler.classify_bottleneck(gpu_summary)

    ok = sum(1 for r in results if r)
    return {
        "concurrency": concurrency,
        "duration_s": duration_s,
        "requests": len(results),
        "successes": ok,
        "failures": len(results) - ok,
        "gpu": gpu_summary,
        "cpu": cpu_summary,
        "cpu_util_pct": cpu_summary.get("mean") if "error" not in cpu_summary else None,
        "baseline_vram_mb": baseline["vram_used_mb"] if baseline else None,
        "bottleneck": bottleneck,
    }


async def main():
    parser = argparse.ArgumentParser(description="GPU VRAM profiling and bottleneck identification")
    parser.add_argument("--server", required=True, help="Server URL (http:// for batch, ws:// for stream)")
    parser.add_argument("--mode", required=True, choices=["batch", "stream"], help="Benchmark mode")
    parser.add_argument("--concurrency", default="1,8,16,32,64", help="Concurrency levels (default: 1,8,16,32,64)")
    parser.add_argument("--duration", type=float, default=30, help="Duration per level in seconds (default: 30)")
    parser.add_argument("--sample-interval", type=float, default=1.0, help="GPU sampling interval (default: 1.0)")
    parser.add_argument("--output", default="/tmp/gpu_profile.json", help="Output JSON path")
    parser.add_argument("--dataset", default="librispeech-test-clean", help="Benchmark dataset for load generation")
    parser.add_argument("--max-samples", type=int, default=0, help="Max dataset samples (0=full dataset)")
    parser.add_argument("--dataset-dir", type=Path, default=None, help="Dataset cache directory")
    parser.add_argument("--endpoint", default=None, help="Override endpoint path (e.g., /v1/transcribe for batch)")
    args = parser.parse_args()

    levels = [int(x) for x in args.concurrency.split(",")]

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from benchmarks.datasets.registry import load_dataset

    log.info("Loading dataset...")
    manifest = load_dataset(args.dataset, cache_dir=args.dataset_dir, max_samples=args.max_samples)
    wav_files = [e["wav_path"] for e in manifest]
    log.info(f"Using {len(wav_files)} WAV files for load generation")

    baseline = parse_nvidia_smi()
    log.info(f"Baseline VRAM: {baseline['vram_used_mb']}MB / {baseline['vram_total_mb']}MB" if baseline else "No GPU detected")

    report = {
        "benchmark": "GPU Profile",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "server": args.server,
        "mode": args.mode,
        "baseline": baseline,
        "levels": [],
    }

    for c in levels:
        result = await profile_at_concurrency(
            args.server, args.mode, wav_files, c, args.duration, args.sample_interval,
            endpoint=args.endpoint,
        )
        report["levels"].append(result)
        bn = result["bottleneck"]
        gpu = result["gpu"]
        if "error" in gpu:
            log.info(
                f"  c={c}: GPU samples unavailable ({gpu['error']}), "
                f"bottleneck={bn['bottleneck']} ({bn['confidence']:.0%})"
            )
        else:
            log.info(
                f"  c={c}: VRAM {gpu['vram_mb']['max']}MB "
                f"({gpu['vram_mb']['highwater_pct']}%), "
                f"GPU {gpu['gpu_util_pct']['mean']}%, "
                f"bottleneck={bn['bottleneck']} ({bn['confidence']:.0%})"
            )

    final_bn = (
        report["levels"][-1]["bottleneck"]
        if report["levels"]
        else {"bottleneck": "unknown", "reasoning": "no concurrency levels were profiled"}
    )
    report["overall_bottleneck"] = final_bn

    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"Profile saved to {args.output}")

    print()
    print("## GPU Profile Results")
    print()
    print(f"**Mode:** {args.mode}")
    if baseline:
        print(f"**GPU:** {baseline['vram_total_mb']}MB VRAM")
        print(f"**Baseline VRAM:** {baseline['vram_used_mb']}MB")
    print()
    print("| Concurrency | VRAM (max) | VRAM % | GPU Util | Mem BW | Bottleneck |")
    print("|-------------|-----------|--------|----------|--------|------------|")
    for level in report["levels"]:
        g = level["gpu"]
        bn = level["bottleneck"]["bottleneck"]
        if "error" in g:
            print(f"| {level['concurrency']} | unavailable | unavailable | unavailable | unavailable | {bn} |")
        else:
            print(
                f"| {level['concurrency']} | {g['vram_mb']['max']}MB | "
                f"{g['vram_mb']['highwater_pct']}% | {g['gpu_util_pct']['mean']}% | "
                f"{g['mem_bw_util_pct']['mean']}% | {bn} |"
            )
    print()
    print(f"**Overall bottleneck:** {final_bn['bottleneck']} — {final_bn.get('reasoning', 'unknown')}")


if __name__ == "__main__":
    asyncio.run(main())
