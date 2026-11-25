"""Tests for bench_embed.py — embedding benchmark script."""

import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench_embed import (
    _validate_embedding,
    check_quality_gates,
    compute_latency_stats,
    compute_retrieval_metrics,
    DEFAULT_TEXT_QUERIES,
    MIN_RETRIEVAL_PAIRS,
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


def test_quality_gates_empty_sweep_fails():
    report = {
        "modality": "audio",
        "concurrency_sweep": [],
        "vram": {"growth_mb": 50},
    }
    gates = check_quality_gates(report)
    assert gates["status"] == "fail"
    sweep_checks = [c for c in gates["checks"] if c["gate"] == "sweep_evidence"]
    assert len(sweep_checks) == 1
    assert not sweep_checks[0]["passed"]


def test_quality_gates_sustained_failures():
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
        "sustained_load": {"failures": 3, "successes": 97},
        "vram": {"growth_mb": 50},
    }
    gates = check_quality_gates(report)
    assert gates["status"] == "fail"
    sustained_checks = [c for c in gates["checks"] if c["gate"] == "sustained_failures"]
    assert len(sustained_checks) == 1
    assert not sustained_checks[0]["passed"]


def test_validate_embedding_valid():
    body = {"embedding": [0.1, 0.2, 0.3], "dimensions": 3}
    emb, err = _validate_embedding(body)
    assert emb == [0.1, 0.2, 0.3]
    assert err is None


def test_validate_embedding_missing():
    emb, err = _validate_embedding({})
    assert emb is None
    assert "missing" in err


def test_validate_embedding_empty():
    emb, err = _validate_embedding({"embedding": []})
    assert emb is None
    assert "empty" in err


def test_validate_embedding_non_numeric():
    emb, err = _validate_embedding({"embedding": ["a", "b", "c"]})
    assert emb is None
    assert "non-numeric" in err


def test_validate_embedding_nan():
    emb, err = _validate_embedding({"embedding": [0.1, float("nan"), 0.3]})
    assert emb is None
    assert "non-finite" in err


def test_validate_embedding_inf():
    emb, err = _validate_embedding({"embedding": [0.1, float("inf"), 0.3]})
    assert emb is None
    assert "non-finite" in err


def test_quality_gates_retrieval_error():
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
        "retrieval_error": {"reason": "no evaluable pairs"},
    }
    gates = check_quality_gates(report)
    assert gates["status"] == "fail"
    retrieval_checks = [c for c in gates["checks"] if c["gate"] == "retrieval_evidence"]
    assert len(retrieval_checks) == 1
    assert not retrieval_checks[0]["passed"]


def test_quality_gates_insufficient_retrieval_coverage():
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
        "retrieval": {
            "recall_at_k": 1.0,
            "k": 10,
            "evaluated_pairs": 3,
            "insufficient_coverage": True,
        },
    }
    gates = check_quality_gates(report)
    assert gates["status"] == "fail"
    coverage_checks = [c for c in gates["checks"] if c["gate"] == "retrieval_coverage"]
    assert len(coverage_checks) == 1
    assert not coverage_checks[0]["passed"]


def test_retrieval_metrics_dimension_mismatch():
    audio_embs = [[1.0, 0.0, 0.0]]
    text_embs = [[1.0, 0.0]]
    utt_ids = ["utt1"]
    refs = {"utt1": "text one"}
    result = compute_retrieval_metrics(audio_embs, text_embs, utt_ids, refs, k=1)
    assert result is not None
    assert "error" in result
    assert "dimension mismatch" in result["error"]


def test_retrieval_metrics_reports_insufficient_coverage():
    audio_embs = [[1.0, 0.0, 0.0]]
    text_embs = [[1.0, 0.0, 0.0]]
    utt_ids = ["utt1"]
    refs = {"utt1": "text one"}
    result = compute_retrieval_metrics(audio_embs, text_embs, utt_ids, refs, k=1)
    assert result is not None
    assert result["insufficient_coverage"] is True
    assert result["evaluated_pairs"] == 1


def test_quality_gates_missing_config():
    with patch("bench_embed.SCRIPTS_DIR", Path("/nonexistent")):
        report = {"modality": "audio", "concurrency_sweep": []}
        gates = check_quality_gates(report)
        assert gates["status"] == "skipped"
