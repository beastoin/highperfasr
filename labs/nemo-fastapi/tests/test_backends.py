"""Tests for model backend abstraction."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from highperfasr.backends import EG2Backend, ModelBackend, NeMoBackend, resolve_backend


class FakeBackend:
    """Minimal backend for testing GPUWorker delegation."""

    def __init__(self):
        self.load_calls = []
        self.transcribe_calls = []
        self.stream_calls = []

    @property
    def supports_streaming(self) -> bool:
        return False

    @property
    def supports_embeddings(self) -> bool:
        return False

    def load_batch_model(self, config: dict, device: str, pool_index: int = 0) -> Any:
        self.load_calls.append((config, device, pool_index))
        return MagicMock(name=f"fake-model-{pool_index}")

    def transcribe_batch(self, model: Any, payload: dict) -> list:
        self.transcribe_calls.append((model, payload))
        return ["fake transcription"]

    def build_stream_pipeline(self, config: dict) -> tuple[Any, int, str] | None:
        self.stream_calls.append(config)
        return None

    def load_embedding_model(self, config: dict, device: str) -> Any:
        return None

    def embed_audio(self, model: Any, audio_paths: list[str], normalize: bool = True, dimensions: int = 0) -> list[list[float]]:
        return []

    def embed_text(self, model: Any, texts: list[str], normalize: bool = True, dimensions: int = 0) -> list[list[float]]:
        return []


class TestModelBackendProtocol:
    def test_nemo_backend_satisfies_protocol(self):
        assert isinstance(NeMoBackend(), ModelBackend)

    def test_fake_backend_satisfies_protocol(self):
        assert isinstance(FakeBackend(), ModelBackend)

    def test_nemo_backend_properties(self):
        backend = NeMoBackend()
        assert backend.supports_streaming is True
        assert backend.supports_embeddings is False
        assert backend.attn_mode == "full"
        assert backend.attn_auto_threshold_sec == 600

    def test_eg2_backend_properties(self):
        backend = EG2Backend()
        assert backend.supports_streaming is False
        assert backend.supports_embeddings is True

    def test_eg2_backend_no_transcription(self):
        backend = EG2Backend()
        with pytest.raises(NotImplementedError):
            backend.load_batch_model({}, "cpu")
        with pytest.raises(NotImplementedError):
            backend.transcribe_batch(None, {})

    def test_eg2_backend_stream_returns_none(self):
        backend = EG2Backend()
        assert backend.build_stream_pipeline({}) is None

    def test_eg2_float16_rejected(self):
        backend = EG2Backend()
        import logging
        with patch("highperfasr.backends.log") as mock_log:
            try:
                backend.load_embedding_model({"name": "test", "dtype": "float16"}, "cpu")
            except Exception:
                pass
            assert any("float16" in str(c) for c in mock_log.warning.call_args_list)


class TestResolveBackend:
    def test_resolve_nemo(self):
        backend = resolve_backend("nemo")
        assert isinstance(backend, NeMoBackend)

    def test_resolve_eg2(self):
        backend = resolve_backend("eg2")
        assert isinstance(backend, EG2Backend)

    def test_resolve_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown backend type"):
            resolve_backend("nonexistent")


class TestNeMoBackendExtractResults:
    def test_extract_plain_text(self):
        results = ["hello world", "test"]
        out = NeMoBackend._extract_results(results, timestamps=False)
        assert out == ["hello world", "test"]

    def test_extract_hypothesis_object(self):
        hyp = MagicMock()
        hyp.text = "hello"
        hyp.timestamp = {"word": [{"word": "hello", "start_offset": 0.0, "end_offset": 0.5}]}
        out = NeMoBackend._extract_results([hyp], timestamps=True)
        assert len(out) == 1
        assert out[0]["text"] == "hello"
        assert "word" in out[0]["timestamp"]

    def test_extract_hypothesis_no_timestamps(self):
        hyp = MagicMock()
        hyp.text = "hello"
        out = NeMoBackend._extract_results([hyp], timestamps=False)
        assert out == ["hello"]

    def test_extract_timestep_excluded(self):
        hyp = MagicMock()
        hyp.text = "hello"
        hyp.timestamp = {
            "timestep": [{"ts": 1}],
            "word": [{"word": "hello", "start_offset": 0.1, "end_offset": 0.5}],
        }
        out = NeMoBackend._extract_results([hyp], timestamps=True)
        assert "timestep" not in out[0]["timestamp"]
        assert "word" in out[0]["timestamp"]


class TestNeMoBackendBuildStreamPipeline:
    def test_no_config_returns_none(self):
        backend = NeMoBackend()
        assert backend.build_stream_pipeline({}) is None
        assert backend.build_stream_pipeline({"name": ""}) is None

    def test_no_name_returns_none(self):
        backend = NeMoBackend()
        assert backend.build_stream_pipeline({"device": "cuda:0"}) is None


class TestGPUWorkerBackendInjection:
    def test_default_backend_is_nemo(self):
        from highperfasr.gpu_worker import GPUWorker

        worker = GPUWorker()
        assert isinstance(worker._backend, NeMoBackend)

    def test_custom_backend_injection(self):
        from highperfasr.gpu_worker import GPUWorker

        fake = FakeBackend()
        worker = GPUWorker(backend=fake)
        assert worker._backend is fake

    def test_vram_info_with_nemo_backend(self):
        from highperfasr.gpu_worker import GPUWorker

        worker = GPUWorker()
        info = worker.vram_info
        assert "attention_mode" in info
        assert info["attention_mode"] == "full"

    def test_vram_info_with_fake_backend(self):
        from highperfasr.gpu_worker import GPUWorker

        worker = GPUWorker(backend=FakeBackend())
        info = worker.vram_info
        assert info["attention_mode"] == "full"
        assert info["auto_threshold_sec"] == 600
