#!/usr/bin/env python3
"""
Structured JSONL telemetry for benchmark scripts.

Writes one JSON object per line, flushed immediately for crash safety.
Report checkpoints preserve every completed phase, including WER and metadata.
Recover a report with:
    python3 benchmark_telemetry.py run.jsonl --output recovered.json
"""

import atexit
import json
import os
import threading
import time
import uuid


class JsonlWriter:
    """Thread-safe, crash-safe JSONL log writer."""

    def __init__(self, path: str):
        self._path = path
        self._lock = threading.Lock()
        self._fd = open(path, "a", encoding="utf-8")
        self._closed = False
        self._run_id = uuid.uuid4().hex
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

    def checkpoint(self, report: dict):
        """Flush the complete report-so-far before starting the next phase."""
        self.emit("report_checkpoint", report=report)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def jsonl_path_from_output(output_path: str) -> str:
    """Derive JSONL sidecar path from report output path."""
    base, _ = os.path.splitext(output_path)
    return f"{base}.jsonl"


def emit_sample(writer: "JsonlWriter | None", result: dict, phase: str, concurrency: int,
                index: int = 0, trial: int = 1):
    """Log one benchmark sample result."""
    if writer is None:
        return
    writer.emit(
        "sample",
        phase=phase,
        concurrency=concurrency,
        index=index,
        trial=trial,
        utt_id=result.get("utt_id", ""),
        status=result.get("status", "unknown"),
        elapsed_s=round(result.get("elapsed", 0), 4),
        audio_dur_s=round(result.get("audio_dur", 0), 3),
        ttfb_s=result.get("ttfb_s"),
        text_len=len(result.get("text", "")),
        text=result.get("text", ""),
        result=result,
        error=result.get("error"),
    )


def emit_resource(writer: "JsonlWriter | None", gpu_sample: dict | None, cpu_pct: float | None,
                  phase: str, concurrency: int, trial: int = 1):
    """Log one resource snapshot."""
    if writer is None:
        return
    fields = {"phase": phase, "concurrency": concurrency, "trial": trial,
              "resource_scope": "benchmark_client"}
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


def reconstruct_report(path: str, run_id: str | None = None) -> dict:
    """Recover the latest completed checkpoint, or select an older appended run.

    An interrupted phase's raw samples remain in JSONL; no complete summary or
    quality-gate result is invented for that phase. Ignore only a torn last line.
    """
    report = None
    selected_run = run_id
    with open(path, encoding="utf-8") as source:
        for line in source:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                if not line.endswith("\n"):
                    break
                raise
            if run_id is None and record["event"] == "run_start":
                selected_run = record["run_id"]
                report = None
            if record["run_id"] == selected_run and record["event"] == "report_checkpoint":
                report = record["report"]
    if report is None:
        raise ValueError("No report checkpoint found for the selected run")
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Recover a benchmark report from JSONL checkpoints")
    parser.add_argument("jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-id", help="Recover a specific run (default: latest)")
    args = parser.parse_args()
    report = reconstruct_report(args.jsonl, args.run_id)
    with open(args.output, "w", encoding="utf-8") as output:
        json.dump(report, output, indent=2)
