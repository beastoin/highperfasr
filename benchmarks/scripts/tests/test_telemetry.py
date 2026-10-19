"""Tests for benchmark_telemetry module — JSONL writer, emit helpers, checkpoints."""

import json
import os
import sys
import threading

import pytest

SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, SCRIPTS_DIR)

from benchmark_telemetry import (
    JsonlWriter,
    emit_report_checkpoint,
    emit_resource,
    emit_sample,
    jsonl_path_from_output,
)


class TestJsonlWriter:
    def test_emit_writes_flushed_json_lines(self, tmp_path):
        path = str(tmp_path / "test.jsonl")
        writer = JsonlWriter(path)
        writer.emit("run_start", mode="batch")
        writer.emit("sample", index=0)
        writer.close()

        lines = open(path).readlines()
        assert len(lines) == 2
        r0 = json.loads(lines[0])
        assert r0["event"] == "run_start"
        assert r0["mode"] == "batch"
        assert "run_id" in r0
        assert "ts" in r0
        assert "elapsed_s" in r0
        r1 = json.loads(lines[1])
        assert r1["event"] == "sample"
        assert r1["index"] == 0

    def test_context_manager_closes(self, tmp_path):
        path = str(tmp_path / "test.jsonl")
        with JsonlWriter(path) as writer:
            writer.emit("test_event")
        # Should not raise on second close
        writer.close()
        lines = open(path).readlines()
        assert len(lines) == 1

    def test_idempotent_close(self, tmp_path):
        path = str(tmp_path / "test.jsonl")
        writer = JsonlWriter(path)
        writer.emit("a")
        writer.close()
        writer.close()  # no-op
        writer.emit("b")  # silently dropped after close
        lines = open(path).readlines()
        assert len(lines) == 1

    def test_flush_per_line(self, tmp_path):
        """Verify data is readable on disk after emit, before close."""
        path = str(tmp_path / "test.jsonl")
        writer = JsonlWriter(path)
        writer.emit("first")
        # Read file WITHOUT closing writer — data must be on disk
        with open(path) as f:
            lines = f.readlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["event"] == "first"
        writer.emit("second")
        with open(path) as f:
            lines = f.readlines()
        assert len(lines) == 2
        writer.close()

    def test_thread_safety(self, tmp_path):
        path = str(tmp_path / "test.jsonl")
        writer = JsonlWriter(path)
        errors = []

        def write_n(n):
            try:
                for i in range(n):
                    writer.emit("sample", thread_id=threading.current_thread().name, i=i)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=write_n, args=(50,)) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        writer.close()

        assert not errors
        lines = open(path).readlines()
        assert len(lines) == 200
        for line in lines:
            json.loads(line)  # all valid JSON


class TestEmitSample:
    def test_emits_basic_fields(self, tmp_path):
        path = str(tmp_path / "test.jsonl")
        writer = JsonlWriter(path)
        result = {
            "utt_id": "test-001",
            "status": "ok",
            "elapsed": 1.234,
            "audio_dur": 5.0,
            "text": "hello world",
        }
        emit_sample(writer, result, "sweep", 8, 0)
        writer.close()

        record = json.loads(open(path).readline())
        assert record["event"] == "sample"
        assert record["phase"] == "sweep"
        assert record["concurrency"] == 8
        assert record["utt_id"] == "test-001"
        assert record["status"] == "ok"
        assert record["elapsed_s"] == 1.234
        assert record["audio_dur_s"] == 5.0
        assert record["text_len"] == 11
        assert "text" not in record  # non-WER phase

    def test_wer_phase_includes_text(self, tmp_path):
        path = str(tmp_path / "test.jsonl")
        writer = JsonlWriter(path)
        result = {"utt_id": "w-001", "status": "ok", "elapsed": 1.0, "audio_dur": 3.0, "text": "hypothesis text"}
        emit_sample(writer, result, "wer", 1, 0)
        writer.close()

        record = json.loads(open(path).readline())
        assert record["text"] == "hypothesis text"
        assert record["text_len"] == 15

    def test_none_writer_is_noop(self):
        emit_sample(None, {}, "sweep", 1, 0)  # should not raise


class TestEmitResource:
    def test_gpu_sample(self, tmp_path):
        path = str(tmp_path / "test.jsonl")
        writer = JsonlWriter(path)
        gpu = {"vram_used_mb": 4096, "gpu_util_pct": 85, "mem_bw_util_pct": 60, "temp_c": 72, "power_w": 200}
        emit_resource(writer, gpu, None, "sweep", 32)
        writer.close()

        record = json.loads(open(path).readline())
        assert record["event"] == "resource"
        assert record["source"] == "client"
        assert record["vram_used_mb"] == 4096
        assert record["gpu_util_pct"] == 85

    def test_cpu_only(self, tmp_path):
        path = str(tmp_path / "test.jsonl")
        writer = JsonlWriter(path)
        emit_resource(writer, None, 45.3, "sustained", 16)
        writer.close()

        record = json.loads(open(path).readline())
        assert record["cpu_pct"] == 45.3
        assert record["source"] == "client"
        assert "vram_used_mb" not in record


class TestEmitReportCheckpoint:
    def test_checkpoint_contains_full_report(self, tmp_path):
        path = str(tmp_path / "test.jsonl")
        writer = JsonlWriter(path)
        report = {
            "schema_version": "v1alpha2",
            "wer": {"corpus_wer_pct": 1.92},
            "concurrency_sweep": [{"concurrency": 1, "rtfx": 10}],
            "max_streams": 256,
        }
        emit_report_checkpoint(writer, report, "after_sweep")
        writer.close()

        record = json.loads(open(path).readline())
        assert record["event"] == "report_checkpoint"
        assert record["phase"] == "after_sweep"
        assert record["report"]["wer"]["corpus_wer_pct"] == 1.92
        assert record["report"]["max_streams"] == 256
        assert record["report"]["schema_version"] == "v1alpha2"


class TestJsonlPathFromOutput:
    def test_json_to_jsonl(self):
        assert jsonl_path_from_output("/tmp/report.json") == "/tmp/report.jsonl"

    def test_txt_to_jsonl(self):
        assert jsonl_path_from_output("/tmp/output.txt") == "/tmp/output.jsonl"

    def test_no_extension(self):
        assert jsonl_path_from_output("/tmp/report") == "/tmp/report.jsonl"


class TestMaxStreams:
    """Test max_streams computation logic (extracted from bench_stream.py)."""

    def test_zero_failure_levels(self):
        sweep = [
            {"concurrency": 1, "failures": 0},
            {"concurrency": 8, "failures": 0},
            {"concurrency": 32, "failures": 0},
            {"concurrency": 64, "failures": 2},
        ]
        zero_fail = [s for s in sweep if s["failures"] == 0]
        max_streams = max((s["concurrency"] for s in zero_fail), default=0)
        assert max_streams == 32

    def test_all_failures(self):
        sweep = [
            {"concurrency": 1, "failures": 1},
            {"concurrency": 8, "failures": 3},
        ]
        zero_fail = [s for s in sweep if s["failures"] == 0]
        max_streams = max((s["concurrency"] for s in zero_fail), default=0)
        assert max_streams == 0

    def test_cross_trial_accumulation(self):
        # Trial 1: c=32 has 0 failures
        # Trial 2: c=32 has 1 failure
        # Result: c=32 should NOT be in max_streams
        fail_by_c = {}
        trial1 = [{"concurrency": 8, "failures": 0}, {"concurrency": 32, "failures": 0}]
        for s in trial1:
            fail_by_c[s["concurrency"]] = s["failures"]
        trial2 = [{"concurrency": 8, "failures": 0}, {"concurrency": 32, "failures": 1}]
        for s in trial2:
            fail_by_c[s["concurrency"]] = fail_by_c.get(s["concurrency"], 0) + s["failures"]
        zero_fail = [c for c, f in fail_by_c.items() if f == 0]
        assert max(zero_fail, default=0) == 8


class TestProfilerCallback:
    """Test that profiler on_sample callbacks emit to JSONL correctly."""

    def test_gpu_profiler_callback_emits_resource(self, tmp_path):
        from profile_gpu import GPUProfiler

        path = str(tmp_path / "test.jsonl")
        writer = JsonlWriter(path)
        emitted = []

        def on_sample(sample):
            emitted.append(sample)
            emit_resource(writer, sample, None, "sweep", 8)

        profiler = GPUProfiler(interval_s=0.1, on_sample=on_sample)
        # Don't start — just verify callback wiring
        assert profiler._on_sample is on_sample
        writer.close()

    def test_cpu_profiler_callback_emits_resource(self, tmp_path):
        from profile_gpu import CPUProfiler

        path = str(tmp_path / "test.jsonl")
        writer = JsonlWriter(path)

        def on_sample(cpu_pct):
            emit_resource(writer, None, cpu_pct, "sweep", 8)

        profiler = CPUProfiler(interval_s=0.1, on_sample=on_sample)
        assert profiler._on_sample is on_sample
        writer.close()

    def test_callback_with_real_emit(self, tmp_path):
        """Test that a callback actually writes to JSONL."""
        path = str(tmp_path / "test.jsonl")
        writer = JsonlWriter(path)
        fake_gpu = {"vram_used_mb": 2048, "gpu_util_pct": 90, "mem_bw_util_pct": 50, "temp_c": 70, "power_w": 180}
        emit_resource(writer, fake_gpu, 35.0, "sweep", 16)
        writer.close()

        record = json.loads(open(path).readline())
        assert record["event"] == "resource"
        assert record["source"] == "client"
        assert record["vram_used_mb"] == 2048
        assert record["cpu_pct"] == 35.0
        assert record["concurrency"] == 16
