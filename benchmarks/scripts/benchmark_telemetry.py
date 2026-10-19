#!/usr/bin/env python3
"""
Structured JSONL telemetry for benchmark scripts.

Writes one JSON object per line, flushed immediately for crash safety.
Events: run_start, level_start, sample, resource, level_end, run_end.
"""

import atexit
import json
import os
import threading
import time


class JsonlWriter:
    """Thread-safe, crash-safe JSONL log writer."""

    def __init__(self, path: str):
        self._path = path
        self._lock = threading.Lock()
        self._fd = open(path, "a", encoding="utf-8")
        self._closed = False
        self._run_id = f"{os.getpid()}-{int(time.time())}"
        self._t0 = time.monotonic()
        atexit.register(self.close)

    @property
    def path(self) -> str:
        return self._path

    def emit(self, event: str, **fields):
        record = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "elapsed_s": round(time.monotonic() - self._t0, 3),
            "run_id": self._run_id,
            "event": event,
            **fields,
        }
        line = json.dumps(record, default=str) + "\n"
        with self._lock:
            if not self._closed:
                self._fd.write(line)
                self._fd.flush()

    def close(self):
        with self._lock:
            if not self._closed:
                self._closed = True
                self._fd.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def jsonl_path_from_output(output_path: str) -> str:
    """Derive JSONL sidecar path from report output path."""
    base, _ = os.path.splitext(output_path)
    return f"{base}.jsonl"


def emit_sample(writer: "JsonlWriter | None", result: dict, phase: str, concurrency: int, index: int = 0):
    """Log one benchmark sample result."""
    if writer is None:
        return
    writer.emit(
        "sample",
        phase=phase,
        concurrency=concurrency,
        index=index,
        utt_id=result.get("utt_id", ""),
        status=result.get("status", "unknown"),
        elapsed_s=round(result.get("elapsed", 0), 4),
        audio_dur_s=round(result.get("audio_dur", 0), 3),
        ttfb_s=result.get("ttfb_s"),
        text_len=len(result.get("text", "")),
        error=result.get("error"),
    )


def emit_resource(writer: "JsonlWriter | None", gpu_sample: dict | None, cpu_pct: float | None,
                  phase: str, concurrency: int):
    """Log one resource snapshot (client-side — measures the machine running the benchmark)."""
    if writer is None:
        return
    fields = {"phase": phase, "concurrency": concurrency, "source": "client"}
    if gpu_sample:
        fields.update({
            "vram_used_mb": gpu_sample.get("vram_used_mb"),
            "gpu_util_pct": gpu_sample.get("gpu_util_pct"),
            "mem_bw_util_pct": gpu_sample.get("mem_bw_util_pct"),
            "temp_c": gpu_sample.get("temp_c"),
            "power_w": gpu_sample.get("power_w"),
        })
    if cpu_pct is not None:
        fields["cpu_pct"] = cpu_pct
    writer.emit("resource", **fields)


def emit_report_checkpoint(writer: "JsonlWriter | None", report: dict, phase: str):
    """Snapshot partial report state at key milestones for crash recovery."""
    if writer is None:
        return
    checkpoint = {}
    for key in ("wer", "concurrency_sweep", "sustained_load", "quality_gates", "summary"):
        if key in report:
            checkpoint[key] = report[key]
    writer.emit("report_checkpoint", phase=phase, report=checkpoint)
