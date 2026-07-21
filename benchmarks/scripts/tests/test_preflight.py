"""Tests for preflight server detection and duration estimation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from preflight import detect_server, estimate_wer_duration, normalize_server_mode, resolve_batch_url, resolve_stream_url


def test_detect_server_handles_unreachable():
    info = detect_server("http://localhost:99999", timeout=1)
    assert not info["healthy"]
    assert info["mode"] == "unknown"
    assert info["batch_url"] is None
    assert info["stream_url"] is None


def test_resolve_stream_url_passthrough_when_unhealthy():
    info = {"healthy": False, "stream_url": None}
    assert resolve_stream_url("ws://localhost:8001", info) == "ws://localhost:8001"


def test_resolve_stream_url_corrects_port_in_both_mode():
    info = {"healthy": True, "mode": "both", "stream_url": "ws://localhost:8000"}
    result = resolve_stream_url("ws://localhost:8001", info)
    assert result == "ws://localhost:8000"


def test_resolve_stream_url_uses_compose_stream_port_when_batch_server_detected():
    info = {"healthy": True, "mode": "batch", "stream_url": None}
    result = resolve_stream_url("http://localhost:8000", info)
    assert result == "ws://localhost:8001"


def test_normalize_server_mode_accepts_runtime_stream_mode():
    assert normalize_server_mode("stream") == "streaming"


def test_resolve_batch_url_passthrough_when_unhealthy():
    info = {"healthy": False, "batch_url": None}
    assert resolve_batch_url("http://localhost:8000", info) == "http://localhost:8000"


def test_resolve_batch_url_uses_detected():
    info = {"healthy": True, "mode": "batch", "batch_url": "http://localhost:8000"}
    result = resolve_batch_url("http://localhost:8000", info)
    assert result == "http://localhost:8000"


def test_duration_estimate_streaming():
    est_s, human = estimate_wer_duration(200, 9.0, concurrency=1)
    assert 2000 < est_s < 2500
    assert "min" in human


def test_duration_estimate_short():
    est_s, human = estimate_wer_duration(10, 5.0, concurrency=1)
    assert est_s < 120
    assert "s" in human


def test_duration_estimate_long():
    est_s, human = estimate_wer_duration(2620, 7.5, concurrency=1)
    assert est_s > 3600
    assert "h" in human


def test_duration_estimate_batch_faster():
    stream_s, _ = estimate_wer_duration(200, 9.0, concurrency=1, overhead_s=2.0)
    batch_s, _ = estimate_wer_duration(200, 0.9, concurrency=1, overhead_s=0.5)
    assert batch_s < stream_s
