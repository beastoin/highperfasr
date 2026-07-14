"""Tests for GPU tuning helpers that do not require a live ASR server."""

import asyncio
import importlib.util
from pathlib import Path


def _load_tune_gpu():
    path = Path(__file__).resolve().parents[1] / "tune_gpu.py"
    spec = importlib.util.spec_from_file_location("tune_gpu", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_binary_search_reports_zero_when_lowest_concurrency_fails(monkeypatch):
    tune_gpu = _load_tune_gpu()

    def failing_batch(_server, concurrency, samples=200, rounds=2):
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
    assert trials == [{"concurrency": 1, "failures": 1, "p99_s": 999, "rtfx": 0, "passed": False}]


def test_batch_sweep_does_not_fake_server_side_batch_sizes(monkeypatch):
    tune_gpu = _load_tune_gpu()

    def bench(_server, concurrency, samples=200, rounds=2):
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
