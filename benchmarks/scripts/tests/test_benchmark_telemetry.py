"""Recovery and capacity regressions for PR #41's review feedback."""

import asyncio
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_run(mode, monkeypatch, tmp_path):
    """Exercise the real sweeps, summaries and WER, replacing only external I/O."""
    module = load_script(f"bench_{mode}")
    import benchmark_telemetry as telemetry
    import profile_gpu

    manifest = [{"utt_id": name, "wav_path": str(tmp_path / f"{name}.wav"),
                 "duration_s": 2.0, "reference": "hello world"} for name in ("one", "two")]
    refs = {item["utt_id"]: item["reference"] for item in manifest}
    monkeypatch.setattr(module, "load_dataset_manifest", lambda *a, **kw: (manifest, refs))
    monkeypatch.setattr(module, "detect_server", lambda *a: {"raw": {"model": "test-model"}})
    monkeypatch.setattr(module, "log_preflight_summary", lambda *a: None)
    resolver = "resolve_batch_url" if mode == "batch" else "resolve_stream_url"
    monkeypatch.setattr(module, resolver, lambda server, info: server)
    monkeypatch.setattr(module, "collect_system_info", lambda **kw: {
        "container_image": kw["image_tag"], "gpu": "client-gpu", "git_sha": "abc123"})
    monkeypatch.setattr(module, "collect_gpu_memory_used_mb", lambda: 200)

    gpu_sample = {"vram_used_mb": 200, "vram_total_mb": 1000, "gpu_util_pct": 98,
                  "mem_bw_util_pct": 20, "temp_c": 50, "power_w": 40}

    class GPUProfiler:
        summarize = staticmethod(profile_gpu.GPUProfiler.summarize)

        def __init__(self, interval_s, on_sample):
            self.on_sample = on_sample

        def start(self):
            self.on_sample(gpu_sample)

        def stop(self):
            return [gpu_sample]

        @staticmethod
        def classify_bottleneck(summary):
            pytest.fail("Client GPU samples must not classify a server bottleneck")

    class CPUProfiler(GPUProfiler):
        summarize = staticmethod(profile_gpu.CPUProfiler.summarize)

        def start(self):
            self.on_sample(25)

        def stop(self):
            return [25]

    monkeypatch.setattr(module, "GPUProfiler", GPUProfiler)
    monkeypatch.setattr(module, "CPUProfiler", CPUProfiler)
    state = {"call": 0, "item": 0, "fail_calls": set(), "interrupt_call": None}
    real_sweep = module.run_sweep

    async def sweep(*args, **kwargs):
        state["call"] += 1
        state["item"] = 0
        return await real_sweep(*args, **kwargs)

    async def result(wav_path):
        state["item"] += 1
        if state["call"] == state["interrupt_call"] and state["item"] == 2:
            raise RuntimeError("interrupted benchmark")
        if state["call"] in state["fail_calls"]:
            return {"utt_id": wav_path.stem, "status": "error", "elapsed": 0.1,
                    "error": "server rejected request"}
        return {"utt_id": wav_path.stem, "status": "ok", "elapsed": 2.1,
                "audio_dur": 2.0, "text": "hello world", "ttfb_s": 0.25}

    async def transcribe(session, url, wav_path, semaphore):
        return await result(wav_path)

    async def stream(url, wav_path, chunk_ms, semaphore):
        return await result(wav_path)

    monkeypatch.setattr(module, "run_sweep", sweep)
    monkeypatch.setattr(module, "transcribe_file" if mode == "batch" else "stream_file",
                        transcribe if mode == "batch" else stream)
    output = tmp_path / "report.json"
    argv = [f"bench_{mode}.py", "--server", "http://remote.example:8000" if mode == "batch"
            else "ws://remote.example:8001", "--concurrency", "1,2", "--sustained-concurrency", "1",
            "--sustained-rounds", "1", "--warmup", "1", "--image-tag", "v0.3.0", "--output", str(output)]
    monkeypatch.setattr(sys, "argv", argv)
    return module, telemetry, state, output, argv


@pytest.fixture(params=["batch", "stream"])
def bench_run(request, monkeypatch, tmp_path):
    return prepare_run(request.param, monkeypatch, tmp_path)


@pytest.fixture
def stream_run(monkeypatch, tmp_path):
    return prepare_run("stream", monkeypatch, tmp_path)


def test_report_round_trip_includes_wer_profiles_trials_and_gates(bench_run):
    module, telemetry, state, output, argv = bench_run
    argv.extend(["--trials", "3"])
    # No server resource telemetry: the VRAM gate must fail, not use client VRAM.
    assert asyncio.run(module.main()) == 1
    report = json.loads(output.read_text())
    recovered = telemetry.reconstruct_report(str(output.with_suffix(".jsonl")))
    assert recovered == report
    assert report["wer"]["normalization"] == "whisper_english"
    assert report["wer"]["corpus_wer_pct"] == 0
    assert report["wer"]["max_load_corpus_wer_pct"] == 0
    assert report["system"]["container_image"] == "v0.3.0"
    assert report["avg_file_duration_s"] == 2
    assert report["resource_scope"] == "benchmark_client"
    assert "resources" not in report
    assert report["client_resources"]["vram_growth_mb"] == 0
    vram_gate = next(g for g in report["quality_gates"]["gates"] if g["gate"] == "max_vram_growth_mb")
    assert vram_gate["actual"] is None
    assert not vram_gate["passed"]
    for trial in [report, *report["additional_trials"]]:
        for level in [*trial["concurrency_sweep"], trial["sustained_load"]]:
            assert level["resource_scope"] == "benchmark_client"
            assert level["gpu_profile"]["gpu_util_pct"]["mean"] == 98
            assert level["cpu_profile"]["mean"] == 25
            assert "bottleneck" not in level
    events = [json.loads(line) for line in output.with_suffix(".jsonl").read_text().splitlines()]
    assert events[0]["references"] == {"one": "hello world", "two": "hello world"}
    assert events[0]["configuration"]["image_tag"] == "v0.3.0"
    samples = [e for e in events if e["event"] == "sample"]
    assert {e["phase"] for e in samples} == {"warmup", "wer", "sweep", "sustained"}
    assert {e["trial"] for e in samples} == {1, 2, 3}
    assert all(e["text"] == e["result"]["text"] == "hello world" for e in samples)
    assert all(e["resource_scope"] == "benchmark_client" for e in events if e["event"] == "resource")


def test_interruption_preserves_completed_wer_level_and_partial_samples(bench_run):
    module, telemetry, state, output, argv = bench_run
    state["interrupt_call"] = 4  # warmup, WER, c=1, then interrupt c=2
    with pytest.raises(RuntimeError, match="interrupted benchmark"):
        asyncio.run(module.main())
    assert not output.exists()
    jsonl = output.with_suffix(".jsonl")
    # All records are readable before writer close/atexit.
    recovered = telemetry.reconstruct_report(str(jsonl))
    assert recovered["wer"]["corpus_wer_pct"] == 0
    assert recovered["system"]["container_image"] == "v0.3.0"
    assert [s["concurrency"] for s in recovered["concurrency_sweep"]] == [1]
    assert "quality_gates" not in recovered
    events = [json.loads(line) for line in jsonl.read_text().splitlines()]
    partial = [e for e in events if e["event"] == "sample" and e["phase"] == "sweep" and e["concurrency"] == 2]
    assert len(partial) == 1
    assert partial[0]["result"]["text"] == "hello world"


@pytest.mark.parametrize("fail_calls,capacity", [({4}, 1), ({3, 4}, 0), (set(), 2)])
def test_stream_capacity_excludes_failed_levels(stream_run, fail_calls, capacity, capsys):
    module, telemetry, state, output, argv = stream_run
    argv.append("--quick")
    state["fail_calls"] = fail_calls
    assert asyncio.run(module.main()) == int(bool(fail_calls))
    report = json.loads(output.read_text())
    assert report["summary"]["max_concurrent_streams_zero_fail"] == capacity
    stdout = capsys.readouterr().out
    assert f"**Max concurrent streams (0 failures across trials):** {capacity}" in stdout
    assert "| Max Streams |" not in stdout
    assert telemetry.reconstruct_report(str(output.with_suffix(".jsonl"))) == report


def test_capacity_must_pass_same_level_in_every_trial(stream_run):
    module, telemetry, state, output, argv = stream_run
    argv.extend(["--quick", "--trials", "2"])
    state["fail_calls"] = {3, 7}  # trial 1 fails c=1; trial 2 fails c=2
    assert asyncio.run(module.main()) == 1
    report = json.loads(output.read_text())
    assert report["summary"]["max_concurrent_streams_zero_fail"] == 0
    assert report["summary"]["total_failures"] == 4


def test_recovery_selects_appended_runs_and_ignores_only_torn_tail(tmp_path):
    telemetry = load_script("benchmark_telemetry")
    path = tmp_path / "runs.jsonl"
    for value in (1, 2):
        with telemetry.JsonlWriter(str(path)) as writer:
            writer.emit("run_start")
            writer.checkpoint({"samples": value})
    events = [json.loads(line) for line in path.read_text().splitlines()]
    first_id = events[0]["run_id"]
    assert first_id != events[2]["run_id"]
    with path.open("a") as tail:
        tail.write('{"event": "report_checkpoint", "report":')
    assert telemetry.reconstruct_report(str(path)) == {"samples": 2}
    assert telemetry.reconstruct_report(str(path), first_id) == {"samples": 1}
    output = tmp_path / "recovered.json"
    subprocess.run([sys.executable, str(SCRIPTS / "benchmark_telemetry.py"), str(path),
                    "--output", str(output), "--run-id", first_id], check=True)
    assert json.loads(output.read_text()) == {"samples": 1}
    with path.open("a") as tail:
        tail.write("\n")
    with pytest.raises(json.JSONDecodeError):
        telemetry.reconstruct_report(str(path))
