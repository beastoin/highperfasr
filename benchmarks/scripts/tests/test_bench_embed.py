"""Tests for bench_embed.py — embedding benchmark script."""

import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench_embed import (
    check_quality_gates,
    compute_latency_stats,
    compute_retrieval_metrics,
    DEFAULT_TEXT_QUERIES,
)


def test_latency_stats_empty():
    assert compute_latency_stats([]) == {}
    assert compute_latency_stats([{"status": "error", "elapsed_s": 1.0}]) == {}


def test_latency_stats_basic():
    results = [
        {"status": "ok", "elapsed_s": 0.1},
        {"status": "ok", "elapsed_s": 0.2},
        {"status": "ok", "elapsed_s": 0.3},
        {"status": "ok", "elapsed_s": 0.4},
        {"status": "ok", "elapsed_s": 0.5},
    ]
    stats = compute_latency_stats(results)
    assert stats["count"] == 5
    assert stats["min_s"] == 0.1
    assert stats["max_s"] == 0.5
    assert 0.2 <= stats["mean_s"] <= 0.4
    assert stats["p50_s"] == 0.3


def test_latency_stats_ignores_errors():
    results = [
        {"status": "ok", "elapsed_s": 0.1},
        {"status": "error", "elapsed_s": 99.0},
        {"status": "ok", "elapsed_s": 0.2},
    ]
    stats = compute_latency_stats(results)
    assert stats["count"] == 2
    assert stats["max_s"] == 0.2


def test_retrieval_metrics_basic():
    audio_embs = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    text_embs = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    utt_ids = ["utt1", "utt2", "utt3"]
    refs = {"utt1": "text one", "utt2": "text two", "utt3": "text three"}

    result = compute_retrieval_metrics(audio_embs, text_embs, utt_ids, refs, k=1)
    assert result is not None
    assert result["recall_at_k"] == 1.0
    assert result["k"] == 1
    assert result["evaluated_pairs"] == 3


def test_retrieval_metrics_empty():
    assert compute_retrieval_metrics([], [], [], {}) is None
    assert compute_retrieval_metrics([[1, 0]], [], ["utt1"], {"utt1": "t"}) is None


def test_retrieval_metrics_no_matching_utt_ids():
    audio_embs = [[1.0, 0.0]]
    text_embs = [[1.0, 0.0]]
    utt_ids = ["utt1"]
    refs = {"other_utt": "text"}
    result = compute_retrieval_metrics(audio_embs, text_embs, utt_ids, refs, k=1)
    assert result is None


def test_retrieval_metrics_partial_match():
    audio_embs = [[1.0, 0.0], [0.0, 1.0]]
    text_embs = [[0.9, 0.1], [0.1, 0.9]]
    utt_ids = ["utt1", "utt2"]
    refs = {"utt1": "text one", "utt2": "text two"}

    result = compute_retrieval_metrics(audio_embs, text_embs, utt_ids, refs, k=2)
    assert result is not None
    assert result["recall_at_k"] == 1.0
    assert result["k"] == 2


def test_default_text_queries_exist():
    assert len(DEFAULT_TEXT_QUERIES) >= 20


def test_quality_gates_pass():
    report = {
        "modality": "audio",
        "concurrency_sweep": [
            {
                "concurrency": 1,
                "embeds_per_sec": 200,
                "failure_rate": 0.0,
                "failures": 0,
                "p99_s": 0.05,
            },
            {
                "concurrency": 64,
                "embeds_per_sec": 500,
                "failure_rate": 0.0,
                "failures": 0,
                "p99_s": 0.08,
            },
        ],
        "vram": {"growth_mb": 50},
    }
    gates = check_quality_gates(report)
    assert gates["status"] == "pass"
    for check in gates["checks"]:
        assert check["passed"]


def test_quality_gates_fail_throughput():
    report = {
        "modality": "audio",
        "concurrency_sweep": [
            {
                "concurrency": 1,
                "embeds_per_sec": 10,
                "failure_rate": 0.0,
                "failures": 0,
                "p99_s": 0.05,
            },
        ],
        "vram": {"growth_mb": 50},
    }
    gates = check_quality_gates(report)
    assert gates["status"] == "fail"
    throughput_checks = [c for c in gates["checks"] if c["gate"] == "min_throughput"]
    assert len(throughput_checks) == 1
    assert not throughput_checks[0]["passed"]


def test_quality_gates_fail_p99():
    report = {
        "modality": "audio",
        "concurrency_sweep": [
            {
                "concurrency": 1,
                "embeds_per_sec": 200,
                "failure_rate": 0.0,
                "failures": 0,
                "p99_s": 0.5,
            },
        ],
        "vram": {"growth_mb": 50},
    }
    gates = check_quality_gates(report)
    assert gates["status"] == "fail"
    p99_checks = [c for c in gates["checks"] if c["gate"] == "max_p99_ms"]
    assert len(p99_checks) == 1
    assert not p99_checks[0]["passed"]


def test_quality_gates_fail_vram():
    report = {
        "modality": "audio",
        "concurrency_sweep": [
            {
                "concurrency": 1,
                "embeds_per_sec": 200,
                "failure_rate": 0.0,
                "failures": 0,
                "p99_s": 0.05,
            },
        ],
        "vram": {"growth_mb": 500},
    }
    gates = check_quality_gates(report)
    assert gates["status"] == "fail"
    vram_checks = [c for c in gates["checks"] if c["gate"] == "max_vram_growth_mb"]
    assert len(vram_checks) == 1
    assert not vram_checks[0]["passed"]


def test_quality_gates_fail_failure_rate():
    report = {
        "modality": "audio",
        "concurrency_sweep": [
            {
                "concurrency": 64,
                "embeds_per_sec": 200,
                "failure_rate": 0.05,
                "failures": 5,
                "p99_s": 0.05,
            },
        ],
        "vram": {"growth_mb": 50},
    }
    gates = check_quality_gates(report)
    assert gates["status"] == "fail"


def test_quality_gates_text_modality_thresholds():
    report = {
        "modality": "text",
        "concurrency_sweep": [
            {
                "concurrency": 1,
                "embeds_per_sec": 200,
                "failure_rate": 0.0,
                "failures": 0,
                "p99_s": 0.03,
            },
        ],
        "vram": {"growth_mb": 50},
    }
    gates = check_quality_gates(report)
    assert gates["status"] == "fail"
    throughput_checks = [c for c in gates["checks"] if c["gate"] == "min_throughput"]
    assert len(throughput_checks) == 1
    assert throughput_checks[0]["threshold"] == 500


def test_quality_gates_retrieval_pass():
    report = {
        "modality": "audio",
        "concurrency_sweep": [
            {
                "concurrency": 1,
                "embeds_per_sec": 200,
                "failure_rate": 0.0,
                "failures": 0,
                "p99_s": 0.05,
            },
        ],
        "vram": {"growth_mb": 50},
        "retrieval": {"recall_at_k": 0.95, "k": 10, "evaluated_pairs": 100},
    }
    gates = check_quality_gates(report)
    assert gates["status"] == "pass"
    recall_checks = [c for c in gates["checks"] if c["gate"] == "min_recall_at_k"]
    assert len(recall_checks) == 1
    assert recall_checks[0]["passed"]


def test_quality_gates_retrieval_fail():
    report = {
        "modality": "audio",
        "concurrency_sweep": [
            {
                "concurrency": 1,
                "embeds_per_sec": 200,
                "failure_rate": 0.0,
                "failures": 0,
                "p99_s": 0.05,
            },
        ],
        "vram": {"growth_mb": 50},
        "retrieval": {"recall_at_k": 0.5, "k": 10, "evaluated_pairs": 100},
    }
    gates = check_quality_gates(report)
    assert gates["status"] == "fail"


def test_quality_gates_missing_config():
    with patch("bench_embed.SCRIPTS_DIR", Path("/nonexistent")):
        report = {"modality": "audio", "concurrency_sweep": []}
        gates = check_quality_gates(report)
        assert gates["status"] == "skipped"
