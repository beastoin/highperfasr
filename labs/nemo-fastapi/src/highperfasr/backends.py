"""Model backend abstraction for highperfasr.

Decouples gpu_worker.py from NeMo-specific model loading and inference,
enabling alternative model families (EG2, HuggingFace transformers, etc.)
behind the same GPUWorker interface.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Protocol, runtime_checkable

import torch

log = logging.getLogger(__name__)


@runtime_checkable
class ModelBackend(Protocol):
    """Interface that every model backend must satisfy."""

    def load_batch_model(self, config: dict, device: str, pool_index: int = 0) -> Any:
        """Load and return a batch inference model. Called on the GPU thread."""
        ...

    def transcribe_batch(self, model: Any, payload: dict) -> list:
        """Run batch transcription. Returns serialized result list."""
        ...

    def build_stream_pipeline(self, config: dict) -> tuple[Any, int, str] | None:
        """Build a streaming pipeline. Returns (pipeline, chunk_samples, source_language) or None."""
        ...

    @property
    def supports_streaming(self) -> bool:
        ...

    @property
    def supports_embeddings(self) -> bool:
        ...

    def load_embedding_model(self, config: dict, device: str) -> Any:
        """Load and return an embedding model. Called on the GPU thread."""
        ...

    def embed_audio(self, model: Any, audio_paths: list[str], normalize: bool = True, dimensions: int = 0) -> list[list[float]]:
        """Compute audio embeddings. Returns list of float vectors."""
        ...

    def embed_text(self, model: Any, texts: list[str], normalize: bool = True, dimensions: int = 0) -> list[list[float]]:
        """Compute text embeddings. Returns list of float vectors."""
        ...


class NeMoBackend:
    """Wraps existing NeMo ASR model loading and inference."""

    def __init__(self) -> None:
        self._attn_mode = "full"
        self._attn_auto_threshold_sec = 600
        self._attn_local_context = [128, 128]
        self._attn_is_local = False
        self._max_file_duration_sec = 0
        self._raw_batch_model: Any = None

    @property
    def supports_streaming(self) -> bool:
        return True

    @property
    def supports_embeddings(self) -> bool:
        return False

    def load_embedding_model(self, config: dict, device: str) -> Any:
        raise NotImplementedError("NeMoBackend does not support embeddings")

    def embed_audio(self, model: Any, audio_paths: list[str], normalize: bool = True, dimensions: int = 0) -> list[list[float]]:
        raise NotImplementedError("NeMoBackend does not support embeddings")

    def embed_text(self, model: Any, texts: list[str], normalize: bool = True, dimensions: int = 0) -> list[list[float]]:
        raise NotImplementedError("NeMoBackend does not support embeddings")

    @property
    def attn_mode(self) -> str:
        return self._attn_mode

    @property
    def attn_auto_threshold_sec(self) -> float:
        return self._attn_auto_threshold_sec

    @staticmethod
    def _load_asr_model(nemo_asr: Any, name: str, device: Any) -> Any:
        if name.endswith(".nemo") or os.path.isfile(name):
            return nemo_asr.models.ASRModel.restore_from(name, map_location=device)
        return nemo_asr.models.ASRModel.from_pretrained(name, map_location=device)

    def load_batch_model(self, config: dict, device: str, pool_index: int = 0) -> Any:
        import nemo.collections.asr as nemo_asr

        pool_size = config.get("model_pool_size", 1)
        tag = f" (pool #{pool_index})" if pool_size > 1 else ""
        log.info(f"Loading batch model{tag}: {config['name']}")

        model = self._load_asr_model(nemo_asr, config["name"], device)
        model.eval()

        self._attn_mode = config.get("attention_mode", "full")
        self._attn_local_context = config.get("local_attn_context", [128, 128])
        self._attn_auto_threshold_sec = config.get("auto_local_attn_threshold_sec", 600)
        self._max_file_duration_sec = config.get("max_file_duration_sec", 0)

        if self._attn_mode == "auto" and pool_size > 1:
            log.warning(
                f"Auto attention mode is unsafe with model_pool_size={pool_size} "
                f"(shared mutable state). Falling back to full attention mode."
            )
            self._attn_mode = "full"

        if self._attn_mode == "local":
            model.change_attention_model("rel_pos_local_attn", self._attn_local_context)
            model.change_subsampling_conv_chunking_factor(1)
            self._attn_is_local = True
            log.info(f"Attention mode: local{tag} (context={self._attn_local_context})")
        elif self._attn_mode == "auto":
            log.info(
                f"Attention mode: auto{tag} — full for <{self._attn_auto_threshold_sec}s, "
                f"local for >={self._attn_auto_threshold_sec}s (torch.compile disabled)"
            )
            self._raw_batch_model = model
        else:
            log.info(f"Attention mode: full{tag} (default)")

        if self._max_file_duration_sec > 0:
            log.info(f"Max file duration: {self._max_file_duration_sec}s")

        if not config.get("cuda_graphs", True):
            if hasattr(model, 'decoding') and hasattr(model.decoding, 'decoding'):
                disabled = model.decoding.decoding.disable_cuda_graphs()
                log.info(f"CUDA graph decoding disabled{tag} (was active: {disabled})")

        if config.get("compile", False) and self._attn_mode != "auto":
            log.info(f"Compiling batch model{tag} with torch.compile")
            model = torch.compile(model)
        elif self._attn_mode == "auto" and config.get("compile", False):
            log.info(f"Skipping torch.compile{tag} — incompatible with auto attention switching")

        return model

    def _switch_attention(self, to_local: bool) -> None:
        if to_local == self._attn_is_local:
            return
        model = self._raw_batch_model
        if model is None:
            return
        if to_local:
            model.change_attention_model("rel_pos_local_attn", self._attn_local_context)
            model.change_subsampling_conv_chunking_factor(1)
            self._attn_is_local = True
        else:
            model.change_attention_model("rel_pos")
            self._attn_is_local = False

    @staticmethod
    def _get_audio_duration_sec(path: str) -> float:
        try:
            import torchaudio

            info = torchaudio.info(path)
            return info.num_frames / info.sample_rate
        except Exception as exc:
            try:
                import wave

                with wave.open(path) as wf:
                    return wf.getnframes() / wf.getframerate()
            except Exception:
                log.warning(f"Cannot determine audio duration for {path}: {exc}")
                return 0.0

    def transcribe_batch(self, model: Any, payload: dict) -> list:
        timestamps = payload.get("timestamps", False)
        batch_size = payload.get("batch_size", 16)

        audio_input = payload.get("audio_tensors", payload["audio_paths"])

        if self._max_file_duration_sec > 0 and isinstance(audio_input, list):
            for path in audio_input:
                if isinstance(path, str):
                    dur = self._get_audio_duration_sec(path)
                    if dur > self._max_file_duration_sec:
                        raise RuntimeError(
                            f"Audio file {dur:.0f}s exceeds max_file_duration_sec "
                            f"({self._max_file_duration_sec}s). Use shorter files or "
                            f"set attention_mode: local/auto for longer audio."
                        )

        if self._attn_mode == "auto":
            durations_from_batcher = payload.get("durations")
            if durations_from_batcher:
                max_dur = max(durations_from_batcher)
            elif isinstance(audio_input, list):
                max_dur = max(
                    (self._get_audio_duration_sec(p) for p in audio_input if isinstance(p, str)),
                    default=0.0,
                )
            else:
                max_dur = 0.0
            need_local = max_dur >= self._attn_auto_threshold_sec
            if need_local != self._attn_is_local:
                mode_name = "local" if need_local else "full"
                log.info(f"Auto-switching attention to {mode_name} (longest file: {max_dur:.0f}s)")
                self._switch_attention(need_local)

        results = model.transcribe(
            audio_input,
            batch_size=batch_size,
            timestamps=timestamps,
            return_hypotheses=timestamps,
            num_workers=0,
            verbose=False,
        )
        serialized = self._extract_results(results, timestamps)
        del results
        return serialized

    @staticmethod
    def _extract_results(results: Any, timestamps: bool) -> list:
        out = []
        items = results if isinstance(results, list) else [results]
        for r in items:
            if timestamps and hasattr(r, 'text') and hasattr(r, 'timestamp'):
                ts = {}
                if isinstance(r.timestamp, dict):
                    for k, entries in r.timestamp.items():
                        if k == 'timestep':
                            continue
                        ts[k] = [
                            {
                                ek: (
                                    round(ev, 4)
                                    if isinstance(ev, float)
                                    else str(ev) if not isinstance(ev, (int, str)) else ev
                                )
                                for ek, ev in e.items()
                            }
                            for e in entries
                        ]
                out.append({"text": str(r.text), "timestamp": ts})
            elif hasattr(r, 'text'):
                out.append(str(r.text))
            else:
                out.append(str(r))
        return out

    _LATENCY_MODE_TO_RIGHT_CONTEXT = {
        "80ms": 0,
        "160ms": 1,
        "480ms": 6,
        "1040ms": 13,
    }

    def build_stream_pipeline(self, config: dict) -> Any | None:
        if not config or not config.get("name"):
            log.info("No stream model configured, streaming will be unavailable")
            return None

        from omegaconf import OmegaConf

        from nemo.collections.asr.inference.factory.pipeline_builder import PipelineBuilder

        device = config.get("device", "cuda:0")
        device_parts = device.split(":")
        device_name = device_parts[0]
        device_id = int(device_parts[1]) if len(device_parts) > 1 else 0

        ref_config_path = config.get("pipeline_config")
        if ref_config_path is None:
            import nemo

            nemo_root = os.path.dirname(os.path.dirname(nemo.__file__))
            candidates = [
                os.path.join(nemo_root, "examples", "asr", "conf", "asr_streaming_inference", "cache_aware_rnnt.yaml"),
                os.path.join(os.path.dirname(__file__), "..", "..", "configs", "cache_aware_rnnt.yaml"),
            ]
            for c in candidates:
                if os.path.exists(c):
                    ref_config_path = c
                    break
            if ref_config_path is None:
                raise FileNotFoundError(
                    "Streaming pipeline config (cache_aware_rnnt.yaml) not found. "
                    "Set stream_model.pipeline_config in your serving config."
                )

        base_cfg = OmegaConf.load(ref_config_path)
        overrides = OmegaConf.create(
            {
                "asr": {
                    "model_name": config["name"],
                    "device": device_name,
                    "device_id": device_id,
                    "compute_dtype": "float16",
                    "use_amp": config.get("amp", True),
                },
                "enable_itn": False,
                "enable_nmt": False,
            }
        )

        source_lang = config.get("source_language", "English")
        overrides["source_language"] = source_lang

        max_streams = config.get("max_concurrent_streams", 256)
        overrides["streaming"] = {
            "att_context_size": None,
            "num_slots": max_streams,
        }
        cfg = OmegaConf.merge(base_cfg, overrides)

        pipeline = PipelineBuilder.build_pipeline(cfg)

        try:
            chunk_samples = int(pipeline.chunk_size_in_secs * pipeline.sample_rate)
        except AttributeError:
            chunk_samples = int(config.get("chunk_samples", 5120))
        log.info(
            f"Stream chunk target: {chunk_samples} samples "
            f"({chunk_samples / 16000 * 1000:.0f}ms)"
        )

        latency_mode = config.get("latency_mode", "480ms")
        right_ctx = self._LATENCY_MODE_TO_RIGHT_CONTEXT.get(latency_mode)
        if right_ctx is not None:
            left_ctx = pipeline.asr_model.get_att_context_size()[0]
            att_context = [left_ctx, right_ctx]
            pipeline.asr_model.set_default_att_context_size(att_context)
            log.info(f"Streaming latency mode: {latency_mode} (att_context_size={att_context})")

        pipeline.open_session()
        log.info("Streaming pipeline built and session opened")

        return pipeline, chunk_samples, source_lang


class EG2Backend:
    """EmbeddingGemma 2 backend for multimodal embeddings."""

    _VALID_DIMENSIONS = (768, 512, 256, 128)

    def __init__(self) -> None:
        self._model = None
        self._processor = None
        self._embedding_dim = 768

    @property
    def supports_streaming(self) -> bool:
        return False

    @property
    def supports_embeddings(self) -> bool:
        return True

    def load_batch_model(self, config: dict, device: str, pool_index: int = 0) -> Any:
        raise NotImplementedError("EG2Backend does not support ASR transcription")

    def transcribe_batch(self, model: Any, payload: dict) -> list:
        raise NotImplementedError("EG2Backend does not support ASR transcription")

    def build_stream_pipeline(self, config: dict) -> tuple[Any, int, str] | None:
        return None

    def load_embedding_model(self, config: dict, device: str) -> Any:
        from transformers import AutoModel, AutoProcessor

        name = config.get("name", "google/embeddinggemma-2")
        dtype_str = config.get("dtype", "bfloat16")
        if dtype_str == "float16":
            log.warning("EG2 produces NaN with float16 — forcing bfloat16")
            dtype_str = "bfloat16"
        dtype = getattr(torch, dtype_str, torch.bfloat16)

        log.info(f"Loading EG2 model: {name} (dtype={dtype_str}, device={device})")
        self._processor = AutoProcessor.from_pretrained(name, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            name, torch_dtype=dtype, trust_remote_code=True
        ).to(device).eval()
        self._embedding_dim = config.get("dimensions", 768)
        log.info(f"EG2 model loaded: {name}, output_dim={self._embedding_dim}")
        return model

    def _truncate(self, embeddings: torch.Tensor, dimensions: int) -> torch.Tensor:
        if dimensions and dimensions < embeddings.shape[-1] and dimensions in self._VALID_DIMENSIONS:
            return embeddings[..., :dimensions]
        return embeddings

    def _normalize(self, embeddings: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(embeddings, p=2, dim=-1)

    @staticmethod
    def _check_finite(embeddings: torch.Tensor) -> torch.Tensor:
        if not torch.isfinite(embeddings).all():
            nan_count = torch.isnan(embeddings).any(dim=-1).sum().item()
            inf_count = torch.isinf(embeddings).any(dim=-1).sum().item()
            raise RuntimeError(f"Non-finite embeddings detected: {nan_count} NaN, {inf_count} Inf vectors")
        return embeddings

    def embed_audio(self, model: Any, audio_paths: list[str], normalize: bool = True, dimensions: int = 0) -> list[list[float]]:
        import numpy as np
        import soundfile as sf

        audios = []
        for path in audio_paths:
            data, sr = sf.read(path, dtype="float32")
            if data.ndim > 1:
                data = data.mean(axis=1)
            if sr != 16000:
                import librosa
                data = librosa.resample(np.ascontiguousarray(data), orig_sr=sr, target_sr=16000)
            audios.append(data)

        device = next(model.parameters()).device
        inputs = self._processor(
            audios=audios,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        ).to(device)

        with torch.inference_mode():
            outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        embeddings = self._truncate(embeddings, dimensions)
        if normalize:
            embeddings = self._normalize(embeddings)
        return self._check_finite(embeddings).cpu().float().tolist()

    def embed_text(self, model: Any, texts: list[str], normalize: bool = True, dimensions: int = 0) -> list[list[float]]:
        device = next(model.parameters()).device
        inputs = self._processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=8192,
        ).to(device)

        with torch.inference_mode():
            outputs = model(**inputs)
        mask = inputs.get("attention_mask")
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()
            embeddings = (outputs.last_hidden_state * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)
        else:
            embeddings = outputs.last_hidden_state.mean(dim=1)
        embeddings = self._truncate(embeddings, dimensions)
        if normalize:
            embeddings = self._normalize(embeddings)
        return self._check_finite(embeddings).cpu().float().tolist()


def resolve_backend(backend_type: str = "nemo") -> ModelBackend:
    """Resolve a backend by name. Extensible for future backends."""
    if backend_type == "nemo":
        return NeMoBackend()
    if backend_type == "eg2":
        return EG2Backend()
    raise ValueError(f"Unknown backend type: {backend_type!r}. Available: nemo, eg2")
