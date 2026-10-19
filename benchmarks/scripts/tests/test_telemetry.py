"""Tests for benchmark_telemetry module — JSONL writer, emit helpers, checkpoints, production paths."""

import asyncio
import importlib.util
import json
import os
import sys
import threading
import time

import pytest

SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, SCRIPTS_DIR)


def _load_script(name):
    path = os.path.join(SCRIPTS_DIR, f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

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


class TestBenchStreamMainMaxStreams:
    """Production-path: exercise bench_stream.main() and verify max_streams in report."""

    def _run_main(self, monkeypatch, tmp_path, sweep_failures):
        """Run bench_stream.main() with monkeypatched run_sweep returning specified failures.

        sweep_failures: dict mapping concurrency -> failures count
        """
        bench_stream = _load_script("bench_stream")

        def fake_load_dataset_manifest(dataset_name, max_samples=0, cache_dir=None):
            return [{"utt_id": "s1", "wav_path": str(tmp_path / "s.wav"),
                     "reference": "ref", "duration_s": 5.0}], {"s1": "ref"}

        call_count = {"sweep": 0}

        async def fake_run_sweep(*_args, **kwargs):
            call_count["sweep"] += 1
            c = kwargs.get("concurrency", 1)
            return [{"utt_id": "s1", "status": "ok", "elapsed": 0.5,
                     "audio_dur": 5.0, "text": "ref"}], 1.0

        def fake_summarize_sweep(_results, _wall, concurrency):
            return {
                "concurrency": concurrency,
                "rtfx": 10.0,
                "sess_per_min": 60.0,
                "failures": sweep_failures.get(concurrency, 0),
            }

        monkeypatch.setattr(bench_stream, "load_dataset_manifest", fake_load_dataset_manifest)
        monkeypatch.setattr(bench_stream, "run_sweep", fake_run_sweep)
        monkeypatch.setattr(bench_stream, "summarize_sweep", fake_summarize_sweep)
        monkeypatch.setattr(bench_stream, "collect_system_info", lambda **kw: {})
        monkeypatch.setattr(bench_stream, "collect_gpu_memory_used_mb", lambda: 0)
        monkeypatch.setattr(bench_stream, "detect_server", lambda *a, **kw: {"streaming": True})
        monkeypatch.setattr(bench_stream, "resolve_stream_url", lambda u, i: u)
        monkeypatch.setattr(bench_stream, "log_preflight_summary", lambda *a, **kw: None)

        report_path = str(tmp_path / "report.json")
        monkeypatch.setattr(
            sys, "argv",
            ["bench_stream.py", "--server", "ws://localhost:8001",
             "--concurrency", ",".join(str(c) for c in sweep_failures),
             "--sustained-rounds", "1", "--sustained-concurrency", "1",
             "--warmup", "1", "--skip-wer", "--output", report_path],
        )

        asyncio.run(bench_stream.main())

        report = json.loads(open(report_path).read())
        jsonl_path = report_path.replace(".json", ".jsonl")
        return report, jsonl_path

    def test_max_streams_zero_failures(self, monkeypatch, tmp_path):
        """All levels zero failures → max_streams = highest concurrency."""
        report, _ = self._run_main(monkeypatch, tmp_path, {1: 0, 8: 0, 32: 0})
        assert report["max_streams"] == 32
        assert report["summary"]["max_streams"] == 32

    def test_max_streams_partial_failures(self, monkeypatch, tmp_path):
        """c=32 has failures → max_streams = 8."""
        report, _ = self._run_main(monkeypatch, tmp_path, {1: 0, 8: 0, 32: 2})
        assert report["max_streams"] == 8

    def test_max_streams_all_failures(self, monkeypatch, tmp_path):
        """All levels fail → max_streams = 0."""
        report, _ = self._run_main(monkeypatch, tmp_path, {1: 1, 8: 3})
        assert report["max_streams"] == 0
        assert report["summary"]["max_streams"] == 0

    def test_jsonl_has_run_start_and_end(self, monkeypatch, tmp_path):
        """Production main() writes JSONL with run_start and run_end events."""
        _, jsonl_path = self._run_main(monkeypatch, tmp_path, {1: 0})
        events = [json.loads(line)["event"] for line in open(jsonl_path)]
        assert events[0] == "run_start"
        assert events[-1] == "run_end"

    def test_jsonl_has_report_checkpoints(self, monkeypatch, tmp_path):
        """Production main() emits report_checkpoint events at key phases."""
        _, jsonl_path = self._run_main(monkeypatch, tmp_path, {1: 0, 4: 0})
        checkpoints = [json.loads(line) for line in open(jsonl_path)
                       if '"report_checkpoint"' in line]
        phases = [cp["phase"] for cp in checkpoints]
        assert "before_sweep" in phases
        assert "after_sweep" in phases
        assert "final" in phases


class TestProfilerSamplingLoop:
    """Production-path: exercise actual profiler sampling loops with mocked system inputs."""

    def test_gpu_profiler_samples_with_mocked_nvidia_smi(self, monkeypatch, tmp_path):
        """GPUProfiler._sample_loop fires on_sample callback with mocked nvidia-smi data."""
        import profile_gpu

        fake_gpu = {
            "vram_used_mb": 4096, "vram_total_mb": 24576, "vram_free_mb": 20480,
            "gpu_util_pct": 85, "mem_bw_util_pct": 60, "temp_c": 72, "power_w": 200.0,
        }
        monkeypatch.setattr(profile_gpu, "parse_nvidia_smi", lambda: dict(fake_gpu))

        path = str(tmp_path / "gpu.jsonl")
        writer = JsonlWriter(path)
        collected = []

        def on_sample(sample):
            collected.append(sample)
            emit_resource(writer, sample, None, "sweep", 16)

        profiler = profile_gpu.GPUProfiler(interval_s=0.05, on_sample=on_sample)
        profiler.start()
        time.sleep(0.25)
        samples = profiler.stop()
        writer.close()

        assert len(collected) >= 2
        assert collected[0]["vram_used_mb"] == 4096
        assert collected[0]["gpu_util_pct"] == 85
        assert "timestamp" in collected[0]

        lines = open(path).readlines()
        assert len(lines) >= 2
        for line in lines:
            record = json.loads(line)
            assert record["event"] == "resource"
            assert record["source"] == "client"
            assert record["vram_used_mb"] == 4096
            assert record["concurrency"] == 16

    def test_cpu_profiler_samples_with_mocked_proc_stat(self, monkeypatch, tmp_path):
        """CPUProfiler._sample_loop fires on_sample with mocked /proc/stat reads."""
        import profile_gpu

        call_count = {"n": 0}

        def fake_read_cpu_times():
            call_count["n"] += 1
            idle = 1000 + call_count["n"] * 10
            total = 5000 + call_count["n"] * 100
            return idle, total

        monkeypatch.setattr(profile_gpu, "_read_cpu_times", fake_read_cpu_times)

        path = str(tmp_path / "cpu.jsonl")
        writer = JsonlWriter(path)
        collected = []

        def on_sample(cpu_pct):
            collected.append(cpu_pct)
            emit_resource(writer, None, cpu_pct, "sustained", 32)

        profiler = profile_gpu.CPUProfiler(interval_s=0.05, on_sample=on_sample)
        profiler.start()
        time.sleep(0.25)
        samples = profiler.stop()
        writer.close()

        assert len(collected) >= 2
        assert all(isinstance(v, float) for v in collected)
        assert all(0.0 <= v <= 100.0 for v in collected)

        lines = open(path).readlines()
        assert len(lines) >= 2
        for line in lines:
            record = json.loads(line)
            assert record["event"] == "resource"
            assert record["source"] == "client"
            assert "cpu_pct" in record
            assert record["concurrency"] == 32

    def test_gpu_profiler_summarize_from_real_loop(self, monkeypatch):
        """GPUProfiler.summarize produces valid stats from sampling loop output."""
        import profile_gpu

        fake_gpu = {
            "vram_used_mb": 8000, "vram_total_mb": 24576, "vram_free_mb": 16576,
            "gpu_util_pct": 70, "mem_bw_util_pct": 45, "temp_c": 65, "power_w": 180.0,
        }
        monkeypatch.setattr(profile_gpu, "parse_nvidia_smi", lambda: dict(fake_gpu))

        profiler = profile_gpu.GPUProfiler(interval_s=0.05)
        profiler.start()
        time.sleep(0.2)
        samples = profiler.stop()

        assert len(samples) >= 2
        summary = profile_gpu.GPUProfiler.summarize(samples)
        assert summary["vram_mb"]["max"] == 8000
        assert summary["gpu_util_pct"]["mean"] == 70.0
        bottleneck = profile_gpu.GPUProfiler.classify_bottleneck(summary)
        assert bottleneck["bottleneck"] in ("vram", "compute", "memory_bandwidth", "cpu", "balanced")


class TestPerSampleTelemetry:
    """Verify on_result callbacks actually emit sample events in production main() paths."""

    def test_sweep_emits_sample_events(self, monkeypatch, tmp_path):
        """bench_stream.main() sweep phase emits per-sample JSONL events via on_result."""
        bench_stream = _load_script("bench_stream")

        def fake_load_dataset_manifest(dataset_name, max_samples=0, cache_dir=None):
            return [{"utt_id": f"s{i}", "wav_path": str(tmp_path / f"s{i}.wav"),
                     "reference": f"ref{i}", "duration_s": 5.0} for i in range(3)], \
                   {f"s{i}": f"ref{i}" for i in range(3)}

        async def fake_run_sweep(*_args, **kwargs):
            on_result = kwargs.get("on_result")
            results = []
            for i in range(3):
                r = {"utt_id": f"s{i}", "status": "ok", "elapsed": 0.5,
                     "audio_dur": 5.0, "text": f"ref{i}"}
                results.append(r)
                if on_result:
                    on_result(r, i)
            return results, 1.5

        def fake_summarize_sweep(_results, _wall, concurrency):
            return {"concurrency": concurrency, "rtfx": 10.0, "sess_per_min": 60.0, "failures": 0}

        monkeypatch.setattr(bench_stream, "load_dataset_manifest", fake_load_dataset_manifest)
        monkeypatch.setattr(bench_stream, "run_sweep", fake_run_sweep)
        monkeypatch.setattr(bench_stream, "summarize_sweep", fake_summarize_sweep)
        monkeypatch.setattr(bench_stream, "collect_system_info", lambda **kw: {})
        monkeypatch.setattr(bench_stream, "collect_gpu_memory_used_mb", lambda: 0)
        monkeypatch.setattr(bench_stream, "detect_server", lambda *a, **kw: {"streaming": True})
        monkeypatch.setattr(bench_stream, "resolve_stream_url", lambda u, i: u)
        monkeypatch.setattr(bench_stream, "log_preflight_summary", lambda *a, **kw: None)

        report_path = str(tmp_path / "report.json")
        monkeypatch.setattr(sys, "argv", [
            "bench_stream.py", "--server", "ws://localhost:8001",
            "--concurrency", "4", "--sustained-rounds", "1",
            "--sustained-concurrency", "1", "--warmup", "1",
            "--skip-wer", "--output", report_path,
        ])
        asyncio.run(bench_stream.main())

        jsonl_path = report_path.replace(".json", ".jsonl")
        samples = [json.loads(line) for line in open(jsonl_path)
                   if '"sample"' in line]
        sweep_samples = [s for s in samples if s["phase"] == "sweep"]
        assert len(sweep_samples) >= 3
        assert all(s["utt_id"].startswith("s") for s in sweep_samples)
        assert all(s["status"] == "ok" for s in sweep_samples)


class TestCrossTrialMaxStreams:
    """Verify cross-trial failure accumulation correctly reduces max_streams."""

    def _run_trials(self, monkeypatch, tmp_path, trial_failure_map):
        """Run bench_stream.main() with --trials 2 and per-trial/concurrency failures.

        trial_failure_map: dict of {(trial, concurrency): failures}
        """
        bench_stream = _load_script("bench_stream")

        def fake_load_dataset_manifest(dataset_name, max_samples=0, cache_dir=None):
            return [{"utt_id": "s1", "wav_path": str(tmp_path / "s.wav"),
                     "reference": "ref", "duration_s": 5.0}], {"s1": "ref"}

        trial_counter = {"current": 0}

        async def fake_run_sweep(*_args, **kwargs):
            return [{"utt_id": "s1", "status": "ok", "elapsed": 0.5,
                     "audio_dur": 5.0, "text": "ref"}], 1.0

        def fake_summarize_sweep(_results, _wall, concurrency):
            trial = trial_counter["current"]
            failures = trial_failure_map.get((trial, concurrency), 0)
            return {"concurrency": concurrency, "rtfx": 10.0,
                    "sess_per_min": 60.0, "failures": failures}

        original_emit = bench_stream.JsonlWriter.emit

        def tracking_emit(self, event, **fields):
            if event == "trial_start":
                trial_counter["current"] = fields.get("trial", 1)
            original_emit(self, event, **fields)

        monkeypatch.setattr(bench_stream, "load_dataset_manifest", fake_load_dataset_manifest)
        monkeypatch.setattr(bench_stream, "run_sweep", fake_run_sweep)
        monkeypatch.setattr(bench_stream, "summarize_sweep", fake_summarize_sweep)
        monkeypatch.setattr(bench_stream, "collect_system_info", lambda **kw: {})
        monkeypatch.setattr(bench_stream, "collect_gpu_memory_used_mb", lambda: 0)
        monkeypatch.setattr(bench_stream, "detect_server", lambda *a, **kw: {"streaming": True})
        monkeypatch.setattr(bench_stream, "resolve_stream_url", lambda u, i: u)
        monkeypatch.setattr(bench_stream, "log_preflight_summary", lambda *a, **kw: None)
        monkeypatch.setattr(bench_stream.JsonlWriter, "emit", tracking_emit)

        report_path = str(tmp_path / "report.json")
        monkeypatch.setattr(sys, "argv", [
            "bench_stream.py", "--server", "ws://localhost:8001",
            "--concurrency", "8,32", "--sustained-rounds", "1",
            "--sustained-concurrency", "1", "--warmup", "1",
            "--skip-wer", "--trials", "2", "--output", report_path,
        ])
        asyncio.run(bench_stream.main())
        return json.loads(open(report_path).read())

    def test_trial2_failures_reduce_max_streams(self, monkeypatch, tmp_path):
        """Trial 1: c=32 zero failures. Trial 2: c=32 has failures. max_streams = 8."""
        report = self._run_trials(monkeypatch, tmp_path, {
            (0, 8): 0, (0, 32): 0,   # trial 1 (tracked as 0): all clean
            (2, 8): 0, (2, 32): 2,    # trial 2: c=32 fails
        })
        assert report["max_streams"] == 8

    def test_trial1_failures_retained_even_if_trial2_succeeds(self, monkeypatch, tmp_path):
        """Trial 1: c=32 fails. Trial 2: c=32 succeeds. Accumulation keeps total > 0."""
        report = self._run_trials(monkeypatch, tmp_path, {
            (0, 8): 0, (0, 32): 3,    # trial 1: c=32 fails
            (2, 8): 0, (2, 32): 0,    # trial 2: c=32 clean — but total is 3, not 0
        })
        assert report["max_streams"] == 8


class TestProfilerLifecycleInMain:
    """Verify profiler start/stop lifecycle produces populated profiles in production reports."""

    def test_sweep_populates_client_gpu_profile(self, monkeypatch, tmp_path):
        """bench_stream.main() sweep phase collects GPU samples into client_gpu_profile."""
        bench_stream = _load_script("bench_stream")
        import profile_gpu

        fake_gpu = {
            "vram_used_mb": 4096, "vram_total_mb": 24576, "vram_free_mb": 20480,
            "gpu_util_pct": 85, "mem_bw_util_pct": 60, "temp_c": 72, "power_w": 200.0,
        }
        monkeypatch.setattr(profile_gpu, "parse_nvidia_smi", lambda: dict(fake_gpu))

        def fake_load_dataset_manifest(dataset_name, max_samples=0, cache_dir=None):
            return [{"utt_id": "s1", "wav_path": str(tmp_path / "s.wav"),
                     "reference": "ref", "duration_s": 5.0}], {"s1": "ref"}

        async def fake_run_sweep(*_args, **kwargs):
            await asyncio.sleep(0.15)
            return [{"utt_id": "s1", "status": "ok", "elapsed": 0.5,
                     "audio_dur": 5.0, "text": "ref"}], 0.15

        monkeypatch.setattr(bench_stream, "load_dataset_manifest", fake_load_dataset_manifest)
        monkeypatch.setattr(bench_stream, "run_sweep", fake_run_sweep)
        monkeypatch.setattr(bench_stream, "collect_system_info", lambda **kw: {})
        monkeypatch.setattr(bench_stream, "collect_gpu_memory_used_mb", lambda: 0)
        monkeypatch.setattr(bench_stream, "detect_server", lambda *a, **kw: {"streaming": True})
        monkeypatch.setattr(bench_stream, "resolve_stream_url", lambda u, i: u)
        monkeypatch.setattr(bench_stream, "log_preflight_summary", lambda *a, **kw: None)

        report_path = str(tmp_path / "report.json")
        monkeypatch.setattr(sys, "argv", [
            "bench_stream.py", "--server", "ws://localhost:8001",
            "--concurrency", "4", "--sustained-rounds", "1",
            "--sustained-concurrency", "1", "--warmup", "1",
            "--skip-wer", "--output", report_path,
        ])
        asyncio.run(bench_stream.main())

        report = json.loads(open(report_path).read())
        sweep_entry = report["concurrency_sweep"][0]
        gpu_profile = sweep_entry["client_gpu_profile"]
        assert "error" not in gpu_profile
        assert gpu_profile["samples"] >= 1
        assert gpu_profile["vram_mb"]["max"] == 4096
        assert gpu_profile["gpu_util_pct"]["mean"] == 85.0

        sus = report["sustained_load"]
        sus_gpu = sus["client_gpu_profile"]
        assert "error" not in sus_gpu
        assert sus_gpu["samples"] >= 1

        jsonl_path = report_path.replace(".json", ".jsonl")
        resource_events = [json.loads(line) for line in open(jsonl_path)
                           if '"resource"' in line]
        assert len(resource_events) >= 2
        assert all(e["source"] == "client" for e in resource_events)


class TestBatchPeakSelection:
    """Verify bench_batch.py selects peak RTFx and peak RPS independently."""

    def test_peak_rtfx_differs_from_peak_rps(self, monkeypatch, tmp_path):
        """When RTFx and RPS peaks are at different concurrency levels, both are reported."""
        bench_batch = _load_script("bench_batch")

        def fake_load_dataset_manifest(dataset_name, max_samples=0, cache_dir=None):
            return [{"utt_id": "s1", "wav_path": str(tmp_path / "s.wav"),
                     "reference": "ref", "duration_s": 5.0}], {"s1": "ref"}

        async def fake_run_sweep(*_args, **kwargs):
            return [{"utt_id": "s1", "status": "ok", "elapsed": 0.5,
                     "audio_dur": 5.0, "text": "ref"}], 1.0

        def fake_summarize_sweep(_results, _wall, concurrency):
            if concurrency == 8:
                return {"concurrency": 8, "rtfx": 20.0, "rps": 3.0,
                        "sess_per_min": 60.0, "failures": 0, "total": 1, "ok": 1,
                        "wall_s": 1.0, "rtf": 0.05, "total_audio_s": 5.0}
            else:
                return {"concurrency": 32, "rtfx": 15.0, "rps": 12.0,
                        "sess_per_min": 120.0, "failures": 0, "total": 4, "ok": 4,
                        "wall_s": 1.0, "rtf": 0.07, "total_audio_s": 5.0}

        monkeypatch.setattr(bench_batch, "load_dataset_manifest", fake_load_dataset_manifest)
        monkeypatch.setattr(bench_batch, "run_sweep", fake_run_sweep)
        monkeypatch.setattr(bench_batch, "summarize_sweep", fake_summarize_sweep)
        monkeypatch.setattr(bench_batch, "collect_system_info", lambda **kw: {})
        monkeypatch.setattr(bench_batch, "collect_gpu_memory_used_mb", lambda: 0)
        monkeypatch.setattr(bench_batch, "detect_server", lambda *a, **kw: {"batch": True})
        monkeypatch.setattr(bench_batch, "resolve_batch_url", lambda u, i: u)
        monkeypatch.setattr(bench_batch, "log_preflight_summary", lambda *a, **kw: None)

        report_path = str(tmp_path / "report.json")
        monkeypatch.setattr(sys, "argv", [
            "bench_batch.py", "--server", "http://localhost:8000",
            "--concurrency", "8,32", "--sustained-rounds", "1",
            "--sustained-concurrency", "1", "--warmup", "1",
            "--skip-wer", "--output", report_path,
        ])
        asyncio.run(bench_batch.main())

        report = json.loads(open(report_path).read())
        assert report["summary"]["peak_rtfx"] == 20.0
        assert report["summary"]["peak_concurrency"] == 8
        assert report["summary"]["peak_rps"] == 12.0
