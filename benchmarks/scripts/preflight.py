#!/usr/bin/env python3
"""
Server auto-detection and preflight checks for benchmark scripts.

Probes the server health endpoint to detect mode (batch/streaming/both),
available endpoints, and model info. Provides duration estimates and
operational warnings.
"""

import json
import logging
import os
import sys
import urllib.request
import urllib.error
from urllib.parse import urlsplit, urlunsplit

log = logging.getLogger("preflight")


def normalize_server_mode(mode):
    """Normalize serving modes to benchmark scenario names."""
    aliases = {
        "stream": "streaming",
        "streams": "streaming",
        "streaming-realtime": "streaming",
        "offline": "batch",
    }
    if mode is None:
        return "unknown"
    return aliases.get(str(mode).strip().lower(), str(mode).strip().lower() or "unknown")


def _http_to_ws_url(url):
    parts = urlsplit(url)
    scheme = "wss" if parts.scheme == "https" else "ws"
    return urlunsplit((scheme, parts.netloc, parts.path, parts.query, parts.fragment))


def _compose_stream_url_from_batch_url(url):
    parts = urlsplit(url)
    if parts.hostname and parts.port == 8000:
        host = parts.hostname
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        netloc = f"{host}:8001"
        scheme = "wss" if parts.scheme == "https" else "ws"
        return urlunsplit((scheme, netloc, parts.path, parts.query, parts.fragment))
    return _http_to_ws_url(url)


def detect_server(base_url, timeout=5):
    """Probe server health endpoint and return server info dict.

    Returns:
        {
            "healthy": bool,
            "mode": "batch" | "streaming" | "both" | "unknown",
            "models": [...],
            "uptime_s": float,
            "batch_url": "http://host:port" or None,
            "stream_url": "ws://host:port" or None,
            "raw": dict,  # full health response
        }
    """
    http_url = base_url.replace("ws://", "http://").replace("wss://", "https://")
    http_url = http_url.rstrip("/")

    result = {
        "healthy": False,
        "mode": "unknown",
        "models": [],
        "uptime_s": None,
        "batch_url": None,
        "stream_url": None,
        "raw": {},
    }

    for path in ["/health", "/healthz", "/"]:
        try:
            req = urllib.request.Request(f"{http_url}{path}", method="GET")
            resp = urllib.request.urlopen(req, timeout=timeout)
            data = json.loads(resp.read())
            result["raw"] = data

            status = data.get("status", data.get("state", "ready"))
            is_ready = data.get("ready", True)
            if str(status).lower() in ("loading", "starting") or is_ready is False:
                log.warning(f"Server responding but not ready: status={status}, ready={is_ready}")
                continue

            result["healthy"] = True

            mode = normalize_server_mode(data.get("mode", data.get("server_mode", "unknown")))
            result["mode"] = mode

            models = data.get("models", data.get("loaded_models", []))
            if isinstance(models, list):
                result["models"] = models

            result["uptime_s"] = data.get("uptime_seconds", data.get("uptime_s"))

            parts = urlsplit(http_url)
            scheme = parts.scheme
            host_port = parts.netloc
            batch_scheme = scheme
            ws_scheme = "wss" if scheme == "https" else "ws"
            if mode in ("batch", "both"):
                result["batch_url"] = f"{batch_scheme}://{host_port}"
            if mode in ("streaming", "both"):
                result["stream_url"] = f"{ws_scheme}://{host_port}"

            break
        except (urllib.error.URLError, json.JSONDecodeError, Exception):
            continue

    return result


def resolve_stream_url(user_url, server_info):
    """Return the correct WebSocket URL, auto-correcting port if needed.

    In both-mode, batch and streaming share the same port. In Docker Compose,
    the stream service listens on container port 8000 but is published on host
    port 8001, so a batch URL on :8000 must not be reused for streaming.
    """
    if not server_info["healthy"]:
        return _http_to_ws_url(user_url)

    if server_info["stream_url"]:
        detected = server_info["stream_url"]
        if user_url != detected:
            log.warning(
                f"Server is in '{server_info['mode']}' mode. "
                f"Streaming URL corrected: {user_url} -> {detected}"
            )
        return detected

    return _compose_stream_url_from_batch_url(user_url)


def resolve_batch_url(user_url, server_info):
    """Return the correct HTTP URL for batch endpoints."""
    if not server_info["healthy"]:
        return user_url

    if server_info["batch_url"]:
        detected = server_info["batch_url"]
        if user_url != detected:
            log.warning(
                f"Server is in '{server_info['mode']}' mode. "
                f"Batch URL corrected: {user_url} -> {detected}"
            )
        return detected

    return user_url


def estimate_wer_duration(n_files, avg_duration_s, concurrency=1, overhead_s=2.0):
    """Estimate wall-clock time for WER evaluation.

    Args:
        n_files: Number of files to evaluate
        avg_duration_s: Average audio duration per file
        concurrency: Number of concurrent streams (1 for WER)
        overhead_s: Per-file overhead (connect, config, drain)

    Returns:
        (estimated_seconds, human_readable_string)
    """
    per_file = avg_duration_s + overhead_s
    total_s = (n_files / concurrency) * per_file
    if total_s < 120:
        human = f"{total_s:.0f}s"
    elif total_s < 7200:
        human = f"{total_s / 60:.0f}min"
    else:
        human = f"{total_s / 3600:.1f}h"
    return total_s, human


def log_duration_estimate(n_files, total_audio_s, mode="streaming"):
    """Log estimated WER evaluation duration with --quick suggestion."""
    avg_dur = total_audio_s / n_files if n_files > 0 else 5.0

    if mode == "streaming":
        est_s, est_human = estimate_wer_duration(n_files, avg_dur, concurrency=1)
    else:
        est_s, est_human = estimate_wer_duration(n_files, avg_dur / 10, concurrency=1, overhead_s=0.5)

    log.info(f"WER evaluation: {n_files} files, ~{est_human} estimated at c=1")

    if est_s > 3600 and n_files > 500:
        log.warning(
            f"Full corpus WER at c=1 will take ~{est_human}. "
            f"Use --quick (200 samples) for validation, --max-samples 0 for publishable runs."
        )
    elif est_s > 1800 and n_files > 500:
        log.info(
            f"Tip: use --quick for faster validation (~{estimate_wer_duration(200, avg_dur)[1]})"
        )


def log_preflight_summary(server_info, mode_requested):
    """Log a preflight summary before starting the benchmark."""
    if not server_info["healthy"]:
        log.error("Server health check failed — benchmark may not work")
        return

    log.info(f"Server: mode={server_info['mode']}, uptime={server_info.get('uptime_s', '?')}s")
    if server_info["models"]:
        log.info(f"Models: {', '.join(str(m) for m in server_info['models'])}")

    if server_info["mode"] == "both" and mode_requested in ("batch", "streaming"):
        log.info(
            f"Server is in 'both' mode — batch and streaming share the same port. "
            f"Running {mode_requested} benchmark only."
        )


def ensure_unbuffered():
    """Force unbuffered stdout/stderr for nohup compatibility."""
    if not os.environ.get("PYTHONUNBUFFERED"):
        os.environ["PYTHONUNBUFFERED"] = "1"
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(line_buffering=True)
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(line_buffering=True)
