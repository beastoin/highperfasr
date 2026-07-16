"""Tests for GPU tuning helpers that do not require a live ASR server."""

import asyncio
import importlib.util
import json
import subprocess
from pathlib import Path


def _load_tune_gpu():
    path = Path(__file__).resolve().parents[1] / "tune_gpu.py"
    spec = importlib.util.spec_from_file_location("tune_gpu", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_binary_search_reports_zero_when_lowest_concurrency_fails(monkeypatch):
    tune_gpu = _load_tune_gpu()

    def failing_batch(_server, concurrency, **_kwargs):
        return {"concurrency_sweep": [{"concurrency": concurrency, "failures": 1, "p99_s": 999, "rtfx": 0}]}

    monkeypatch.setattr(tune_gpu, "_run_bench_batch", failing_batch)

    max_c, trials = asyncio.run(
        tune_gpu.binary_search_max_concurrency(
            "http://localhost:8000",
            "batch",
            search_range=(1, 1),
        )
    )

    assert max_c == 0
    assert trials == [
        {
            "concurrency": 1,
            "failures": 1,
            "p99_s": 999,
            "rtfx": 0,
            "wer_pct": None,
            "quality_passed": False,
            "passed": False,
        }
    ]


def test_binary_search_requires_wer_by_default(monkeypatch):
    tune_gpu = _load_tune_gpu()

    def no_wer_batch(_server, concurrency, **_kwargs):
        return {"concurrency_sweep": [{"concurrency": concurrency, "failures": 0, "p99_s": 1.0, "rtfx": 12.0}]}

    monkeypatch.setattr(tune_gpu, "_run_bench_batch", no_wer_batch)

    max_c, trials = asyncio.run(
        tune_gpu.binary_search_max_concurrency(
            "http://localhost:8000",
            "batch",
            search_range=(1, 1),
        )
    )

    assert max_c == 0
    assert trials[0]["quality_passed"] is False
    assert trials[0]["passed"] is False


def test_binary_search_can_explicitly_skip_wer(monkeypatch):
    tune_gpu = _load_tune_gpu()

    def no_wer_batch(_server, concurrency, **_kwargs):
        return {"concurrency_sweep": [{"concurrency": concurrency, "failures": 0, "p99_s": 1.0, "rtfx": 12.0}]}

    monkeypatch.setattr(tune_gpu, "_run_bench_batch", no_wer_batch)

    max_c, trials = asyncio.run(
        tune_gpu.binary_search_max_concurrency(
            "http://localhost:8000",
            "batch",
            search_range=(1, 1),
            skip_wer=True,
        )
    )

    assert max_c == 1
    assert trials[0]["quality_passed"] is True
    assert trials[0]["passed"] is True


def test_binary_search_rejects_high_wer(monkeypatch):
    tune_gpu = _load_tune_gpu()

    def high_wer_batch(_server, concurrency, **_kwargs):
        return {
            "wer": {"corpus_wer_pct": 25.0},
            "concurrency_sweep": [{"concurrency": concurrency, "failures": 0, "p99_s": 1.0, "rtfx": 12.0}],
        }

    monkeypatch.setattr(tune_gpu, "_run_bench_batch", high_wer_batch)

    max_c, trials = asyncio.run(
        tune_gpu.binary_search_max_concurrency(
            "http://localhost:8000",
            "batch",
            search_range=(1, 1),
            wer_threshold_pct=20.0,
        )
    )

    assert max_c == 0
    assert trials[0]["wer_pct"] == 25.0
    assert trials[0]["quality_passed"] is False


def test_batch_sweep_does_not_fake_server_side_batch_sizes(monkeypatch):
    tune_gpu = _load_tune_gpu()

    def bench(_server, concurrency, **_kwargs):
        return {"concurrency_sweep": [{"concurrency": concurrency, "failures": 0, "p99_s": 1.0, "rtfx": 12.5}]}

    monkeypatch.setattr(tune_gpu, "_run_bench_batch", bench)

    results = asyncio.run(
        tune_gpu.sweep_batch_params(
            "http://localhost:8000",
            16,
            batch_sizes=[8, 16, 32],
        )
    )

    assert len(results) == 1
    assert results[0]["param"] == "current_server_config"
    assert results[0]["value"] == "unchanged"
    assert "requires restarting the server" in results[0]["note"]


def test_stream_config_uses_external_stream_port():
    tune_gpu = _load_tune_gpu()

    config = tune_gpu.generate_tuned_config(
        "stream",
        "l4",
        max_concurrency=512,
        best_params={"chunk_duration_ms": 160, "latency_mode": "480ms"},
    )

    assert config["server"]["port"] == 8001
    assert config["stream"]["chunk_duration_ms"] == 160
    assert config["stream_model"]["latency_mode"] == "480ms"


def test_config_rejects_zero_concurrency():
    tune_gpu = _load_tune_gpu()

    try:
        tune_gpu.generate_tuned_config("stream", "l4", max_concurrency=0, best_params={})
    except ValueError as exc:
        assert "max_concurrency" in str(exc)
    else:
        raise AssertionError("generate_tuned_config should reject zero concurrency")


def test_failures_include_sustained_load_failures():
    tune_gpu = _load_tune_gpu()

    report = {
        "concurrency_sweep": [{"failures": 0}],
        "sustained_load": {"failures": 2},
    }

    assert tune_gpu._get_failures(report) == 2


def test_run_bench_batch_uses_full_dataset_by_default(monkeypatch, tmp_path):
    tune_gpu = _load_tune_gpu()
    output = Path("/tmp/tune_batch_c8.json")
    output.write_text(json.dumps({"concurrency_sweep": []}))
    captured = {}

    def fake_run(cmd, **_kwargs):
        captured["cmd"] = cmd

        class Result:
            returncode = 0
            stderr = ""

        return Result()

    monkeypatch.setattr(subprocess, "run", fake_run)

    tune_gpu._run_bench_batch("http://localhost:8000", 8, dataset_dir=tmp_path)

    cmd = captured["cmd"]
    assert "--dataset" in cmd
    assert cmd[cmd.index("--dataset") + 1] == "librispeech-test-clean"
    assert "--max-samples" in cmd
    assert cmd[cmd.index("--max-samples") + 1] == "0"
    assert "--dataset-dir" in cmd
    assert cmd[cmd.index("--dataset-dir") + 1] == str(tmp_path)
    assert "--skip-wer" not in cmd


def test_run_bench_batch_only_skips_wer_when_requested(monkeypatch, tmp_path):
    tune_gpu = _load_tune_gpu()
    output = Path("/tmp/tune_batch_c8.json")
    output.write_text(json.dumps({"concurrency_sweep": []}))
    captured = {}

    def fake_run(cmd, **_kwargs):
        captured["cmd"] = cmd

        class Result:
            returncode = 0
            stderr = ""

        return Result()

    monkeypatch.setattr(subprocess, "run", fake_run)

    tune_gpu._run_bench_batch("http://localhost:8000", 8, dataset_dir=tmp_path, skip_wer=True)

    assert "--skip-wer" in captured["cmd"]


def test_main_exits_without_config_when_no_concurrency_passes(monkeypatch, tmp_path):
    tune_gpu = _load_tune_gpu()

    async def failing_search(*_args, **_kwargs):
        return 0, [{"concurrency": 1, "failures": 1, "p99_s": 999, "rtfx": 0, "passed": False}]

    monkeypatch.setattr(tune_gpu, "binary_search_max_concurrency", failing_search)
    monkeypatch.setattr(
        tune_gpu.sys,
        "argv",
        [
            "tune_gpu.py",
            "--server",
            "http://localhost:8000",
            "--mode",
            "batch",
            "--gpu-name",
            "l4",
            "--skip-profile",
            "--skip-sweep",
            "--search-lo",
            "1",
            "--search-hi",
            "1",
            "--output-dir",
            str(tmp_path),
        ],
    )

    try:
        asyncio.run(tune_gpu.main())
    except SystemExit as exc:
        assert "No concurrency level passed" in str(exc)
    else:
        raise AssertionError("main() should exit when no concurrency level passes")

    assert (tmp_path / "tuning-report-batch-l4.json").exists()
    assert not (tmp_path / "tuned-serving-batch-l4.yaml").exists()
