"""Regression coverage for PR review findings."""

import importlib.util
import json
import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1]


def _load_script(name):
    path = SCRIPTS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_quality_gate_fails_when_configured_wer_is_missing():
    gates = _load_script("gates")

    result = gates.evaluate_gates(
        {
            "scenario": {"mode": "batch"},
            "concurrency_sweep": [{"total": 1, "failures": 0, "rtfx": 2.0}],
        },
        {"batch": {"max_wer_pct": 2.5, "max_failure_rate": 0.0, "min_rtfx": 1.0}},
    )

    wer_gate = next(g for g in result["gates"] if g["gate"] == "max_wer_pct")
    assert wer_gate["actual"] is None
    assert wer_gate["passed"] is False
    assert result["all_passed"] is False


def test_quality_gate_fails_when_failure_rate_data_is_missing():
    gates = _load_script("gates")

    result = gates.evaluate_gates(
        {"scenario": {"mode": "batch"}, "quality": {"wer": 1.5}},
        {"batch": {"max_wer_pct": 2.5, "max_failure_rate": 0.0}},
    )

    fr_gate = next(g for g in result["gates"] if g["gate"] == "max_failure_rate")
    assert fr_gate["actual"] is None
    assert fr_gate["passed"] is False
    assert result["all_passed"] is False


def test_quality_gate_fails_when_sweep_omits_failures_key():
    gates = _load_script("gates")

    result = gates.evaluate_gates(
        {
            "scenario": {"mode": "batch"},
            "quality": {"wer": 1.5},
            "concurrency_sweep": [{"total": 10, "rtfx": 2.0}],
        },
        {"batch": {"max_wer_pct": 2.5, "max_failure_rate": 0.0, "min_rtfx": 1.0}},
    )

    fr_gate = next(g for g in result["gates"] if g["gate"] == "max_failure_rate")
    assert fr_gate["actual"] is None
    assert fr_gate["passed"] is False


def test_quality_gate_fails_when_mixed_sweep_omits_failures_key():
    gates = _load_script("gates")

    result = gates.evaluate_gates(
        {
            "scenario": {"mode": "batch"},
            "quality": {"wer": 1.5},
            "concurrency_sweep": [
                {"total": 10, "failures": 0, "rtfx": 2.0},
                {"total": 10, "rtfx": 3.0},
            ],
        },
        {"batch": {"max_wer_pct": 2.5, "max_failure_rate": 0.0, "min_rtfx": 1.0}},
    )

    fr_gate = next(g for g in result["gates"] if g["gate"] == "max_failure_rate")
    assert fr_gate["actual"] is None
    assert fr_gate["passed"] is False


def test_quality_gate_fails_when_configured_rtfx_is_missing():
    gates = _load_script("gates")

    result = gates.evaluate_gates(
        {
            "scenario": {"mode": "batch"},
            "quality": {"wer": 1.5},
            "concurrency_sweep": [{"total": 1, "failures": 0}],
        },
        {"batch": {"max_wer_pct": 2.5, "max_failure_rate": 0.0, "min_rtfx": 1.0}},
    )

    rtfx_gate = next(g for g in result["gates"] if g["gate"] == "min_rtfx")
    assert rtfx_gate["actual"] is None
    assert rtfx_gate["passed"] is False
    assert result["all_passed"] is False


def test_quality_gate_config_matches_project_wer_thresholds():
    config = json.loads((SCRIPTS_DIR.parent / "config" / "quality-gates.json").read_text())

    assert config["batch"]["max_wer_pct"] == 2.5
    assert config["streaming-realtime"]["max_wer_pct"] == 4.0
    assert config["streaming-realtime"]["max_stream_lag_p95_ms"] == 5000


def test_streaming_gate_fails_when_sustained_duration_is_missing():
    gates = _load_script("gates")

    result = gates.evaluate_gates(
        {
            "scenario": {"mode": "streaming-realtime"},
            "wer": {"corpus_wer_pct": 3.0, "reference_wer_pct": 3.0},
            "concurrency_sweep": [{"total": 1, "failures": 0, "rtfx": 2.0, "rt_compliance_pct": 100}],
            "sustained_load": {"failures": 0, "vram_growth_mb": 0, "lag_p95_s": 1.0},
        },
        {
            "streaming-realtime": {
                "max_wer_pct": 4.0,
                "max_failure_rate": 0.0,
                "min_rt_compliance_pct": 95.0,
                "wer_delta": {"max_absolute_pp": 0.3, "max_relative_pct": 5.0},
                "max_vram_growth_mb": 100,
                "min_sustained_duration_s": 600,
                "max_stream_lag_p95_ms": 5000,
            }
        },
        scenario="streaming-realtime",
    )

    duration_gate = next(g for g in result["gates"] if g["gate"] == "min_sustained_duration_s")
    assert duration_gate["actual"] is None
    assert duration_gate["passed"] is False
    assert result["all_passed"] is False


def test_streaming_gate_fails_when_lag_exceeds_threshold():
    gates = _load_script("gates")

    result = gates.evaluate_gates(
        {
            "scenario": {"mode": "streaming-realtime"},
            "wer": {"corpus_wer_pct": 3.0, "reference_wer_pct": 3.0},
            "concurrency_sweep": [{"total": 1, "failures": 0, "rtfx": 2.0, "rt_compliance_pct": 100}],
            "sustained_load": {"failures": 0, "wall_s": 600, "vram_growth_mb": 0, "lag_p95_s": 6.0},
        },
        {
            "streaming-realtime": {
                "max_wer_pct": 4.0,
                "max_failure_rate": 0.0,
                "min_rt_compliance_pct": 95.0,
                "wer_delta": {"max_absolute_pp": 0.3, "max_relative_pct": 5.0},
                "max_vram_growth_mb": 100,
                "min_sustained_duration_s": 600,
                "max_stream_lag_p95_ms": 5000,
            }
        },
        scenario="streaming-realtime",
    )

    lag_gate = next(g for g in result["gates"] if g["gate"] == "max_stream_lag_p95_ms")
    assert lag_gate["actual"] == 6000
    assert lag_gate["passed"] is False
    assert result["all_passed"] is False


def test_combined_exit_status_sees_nested_sweep_failures():
    bench_combined = _load_script("bench_combined")

    assert bench_combined._has_combined_entry_failures(
        {"batch": {"failures": 1}, "stream": {"failures": 0}}
    )
    assert bench_combined._has_combined_entry_failures(
        {"batch": {"failures": 0}, "stream": {"failures": 2}}
    )
    assert not bench_combined._has_combined_entry_failures(
        {"batch": {"failures": 0}, "stream": {"failures": 0}}
    )


def test_check_regression_treats_missing_metrics_as_failure():
    check_regression = _load_script("check_regression")

    results = check_regression.check_regression({}, {}, {"wer_pct": {"max_absolute_delta_pct": 0.5}})

    assert check_regression.has_missing_metrics(results)


def test_bench_batch_zero_max_samples_means_full_dataset(monkeypatch, tmp_path):
    bench_batch = _load_script("bench_batch")
    captured = {}

    def fake_load_dataset_manifest(dataset_name, max_samples=0, cache_dir=None):
        captured["dataset_name"] = dataset_name
        captured["max_samples"] = max_samples
        return [tmp_path / "sample.wav"], {"sample": "reference"}

    async def fake_run_sweep(*_args, **_kwargs):
        return [], 0.0

    monkeypatch.setattr(bench_batch, "load_dataset_manifest", fake_load_dataset_manifest)
    monkeypatch.setattr(bench_batch, "run_sweep", fake_run_sweep)
    monkeypatch.setattr(
        bench_batch,
        "summarize_sweep",
        lambda _results, _wall, concurrency: {
            "concurrency": concurrency,
            "rps": 0,
            "rtfx": 0,
            "rtf": 0,
            "sess_per_min": 0,
            "failures": 0,
        },
    )
    monkeypatch.setattr(bench_batch, "compute_wer", lambda *_args, **_kwargs: (0.0, []))
    monkeypatch.setattr(bench_batch, "collect_system_info", lambda: {})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench_batch.py",
            "--server",
            "http://localhost:8000",
            "--concurrency",
            "1",
            "--sustained-rounds",
            "1",
            "--sustained-concurrency",
            "1",
            "--warmup",
            "1",
            "--skip-wer",
            "--output",
            str(tmp_path / "report.json"),
        ],
    )

    import asyncio

    asyncio.run(bench_batch.main())

    assert captured["dataset_name"] == "librispeech-test-clean"
    assert captured["max_samples"] == 0
