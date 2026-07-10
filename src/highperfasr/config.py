"""Configuration loading with CLI > env > YAML > defaults precedence."""

import os
from pathlib import Path

import yaml


def load_config(config_path: str | None = None) -> dict:
    """Load YAML config, falling back to built-in defaults."""
    if config_path is None:
        config_path = os.path.join(Path(__file__).parent.parent.parent, "configs", "serving.yaml")
        if not os.path.exists(config_path):
            return _defaults()
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


def _defaults() -> dict:
    return {
        "mode": "stream",
        "server": {"host": "0.0.0.0", "port": 8000, "workers": 1},
        "stream_model": {
            "name": "nvidia/nemotron-3.5-asr-streaming-0.6b",
            "device": "cuda:0",
            "compile": False,
            "amp": True,
            "latency_mode": "480ms",
            "source_language": "English",
        },
        "stream": {
            "max_concurrent_streams": 512,
            "chunk_duration_ms": 160,
            "sample_rate": 16000,
            "max_stream_duration": 0,
            "idle_timeout": 300,
            "max_chunk_bytes": 524288,
            "max_stream_drain": 16,
        },
    }
