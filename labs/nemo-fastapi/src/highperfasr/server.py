"""High-performance ASR inference server for NeMo models.

Supports three serving modes via config:
  mode: batch   — REST API only (offline transcription)
  mode: stream  — WebSocket only (real-time streaming)
  mode: both    — REST + WebSocket on one GPU
"""

import asyncio
import functools
import json
import logging
import os
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, File, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, PlainTextResponse

from highperfasr.batch_engine import BatchEngine, QueueFullError
from highperfasr.compat import apply_compat
from highperfasr.config import load_config
from highperfasr.gpu_worker import GPUWorker
from highperfasr.stream_engine import ChunkTooLargeError, StreamEngine, StreamExpiredError, TooManyStreamsError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
log = logging.getLogger("highperfasr")

_VALID_MODES = {"batch", "stream", "both"}
_VALID_ATTENTION_MODES = ("full", "local", "auto")
_VALID_LATENCY_MODES = ("80ms", "160ms", "480ms", "1040ms")

gpu_worker: Optional[GPUWorker] = None
batch_engine: Optional[BatchEngine] = None
stream_engine: Optional[StreamEngine] = None
config: dict = {}
serving_mode: str = "both"
start_time: float = 0


def _batch_enabled() -> bool:
    return serving_mode in ("batch", "both")


def _stream_enabled() -> bool:
    return serving_mode in ("stream", "both")


def _validate_startup_config() -> None:
    if serving_mode not in _VALID_MODES:
        raise RuntimeError("Invalid mode '%s' — must be one of: batch, stream, both" % serving_mode)

    if _batch_enabled():
        batch_cfg = config.get("batch_model")
        if not batch_cfg:
            raise RuntimeError(f"mode={serving_mode} requires batch_model config")

        attention_mode = batch_cfg.get("attention_mode", "full")
        if attention_mode not in _VALID_ATTENTION_MODES:
            raise RuntimeError(
                "Invalid batch_model.attention_mode '%s' — must be one of: %s"
                % (attention_mode, ", ".join(_VALID_ATTENTION_MODES))
            )

    if _stream_enabled():
        stream_cfg = config.get("stream_model")
        if not stream_cfg:
            raise RuntimeError(f"mode={serving_mode} requires stream_model config")

        latency_mode = stream_cfg.get("latency_mode", "480ms")
        if latency_mode not in _VALID_LATENCY_MODES:
            raise RuntimeError(
                "Invalid stream_model.latency_mode '%s' — must be one of: %s"
                % (latency_mode, ", ".join(_VALID_LATENCY_MODES))
            )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global gpu_worker, batch_engine, stream_engine, serving_mode, start_time
    start_time = time.monotonic()

    _validate_startup_config()

    log.info(f"Serving mode: {serving_mode}")

    batch_cfg = config.get("batch_model", {}) if _batch_enabled() else {}
    stream_cfg = config.get("stream_model", {}) if _stream_enabled() else {}
    if stream_cfg:
        stream_section = config.get("stream", {})
        stream_cfg["max_concurrent_streams"] = stream_section.get("max_concurrent_streams", 256)

    gpu_worker = GPUWorker()
    gpu_worker.start(batch_cfg=batch_cfg, stream_cfg=stream_cfg)
    log.info("Waiting for GPU models to load...")
    gpu_worker.wait_ready(timeout=600)

    if _batch_enabled() and not gpu_worker.has_batch:
        log.warning(f"mode={serving_mode} but batch model did not load — disabling batch endpoints")
        serving_mode = "stream" if serving_mode == "both" else serving_mode
    if _stream_enabled() and not gpu_worker.has_stream:
        log.warning(f"mode={serving_mode} but stream model did not load — disabling stream endpoints")
        serving_mode = "batch" if serving_mode == "both" else serving_mode

    if _batch_enabled():
        batcher_cfg = config.get("batcher", {})
        batch_engine = BatchEngine(
            gpu_worker=gpu_worker,
            max_batch_size=batcher_cfg.get("max_batch_size", 32),
            max_wait_seconds=batcher_cfg.get("max_wait_seconds", 0.002),
            max_queue_depth=batcher_cfg.get("max_queue_depth", 4096),
            vram_safety_factor=batcher_cfg.get("vram_safety_factor", 0.8),
            vram_bytes_per_t2=batcher_cfg.get("vram_bytes_per_t2", 136.6),
            starvation_timeout_sec=batcher_cfg.get("starvation_timeout_sec", 5.0),
            max_inflight=batcher_cfg.get("max_inflight", 2),
        )
        await batch_engine.start()

    if _stream_enabled():
        stream_settings = config.get("stream", {})
        stream_engine = StreamEngine(
            gpu_worker=gpu_worker,
            max_concurrent_streams=stream_settings.get("max_concurrent_streams", 128),
            chunk_duration_ms=stream_settings.get("chunk_duration_ms", 160),
            sample_rate=stream_settings.get("sample_rate", 16000),
            max_stream_duration=stream_settings.get("max_stream_duration", 0),
            idle_timeout=stream_settings.get("idle_timeout", 300),
            max_chunk_bytes=stream_settings.get("max_chunk_bytes", 512 * 1024),
        )
        await stream_engine.start()

    log.info(f"ASR server ready (mode={serving_mode})")
    yield

    log.info("Shutting down...")
    if batch_engine is not None:
        await batch_engine.stop()
    if stream_engine is not None:
        await stream_engine.stop()
    gpu_worker.stop()


app = FastAPI(
    title="highperfasr",
    description="Production ASR serving for NeMo models",
    lifespan=lifespan,
)


# --- Health & Metrics ---


@app.get("/health")
async def health():
    ready = gpu_worker.is_ready if gpu_worker else False
    return {
        "status": "ok" if ready else "loading",
        "ready": ready,
        "mode": serving_mode,
        "batch_model": (config.get("batch_model") or {}).get("name") if _batch_enabled() else None,
        "stream_model": (config.get("stream_model") or {}).get("name") if _stream_enabled() else None,
        "uptime_seconds": round(time.monotonic() - start_time, 1),
    }


@app.get("/metrics")
async def metrics():
    result = {
        "uptime_seconds": round(time.monotonic() - start_time, 1),
        "mode": serving_mode,
    }
    if batch_engine is not None:
        result["batch"] = batch_engine.metrics
    if stream_engine is not None:
        result["stream"] = stream_engine.metrics
    return result


@app.get("/metrics/prometheus")
async def metrics_prometheus():
    lines = [
        "# HELP highperfasr_up Server is running",
        "# TYPE highperfasr_up gauge",
        "highperfasr_up 1",
        "# HELP highperfasr_uptime_seconds Seconds since server start",
        "# TYPE highperfasr_uptime_seconds gauge",
        f"highperfasr_uptime_seconds {time.monotonic() - start_time:.1f}",
    ]
    if batch_engine is not None:
        bm = batch_engine.metrics
        for key in ("total_requests", "total_batches", "total_files", "rejected_requests", "vram_limited_batches"):
            lines.append(f"# TYPE highperfasr_batch_{key} counter")
            lines.append(f"highperfasr_batch_{key} {bm[key]}")
        lines.append("# TYPE highperfasr_batch_pending_requests gauge")
        lines.append(f"highperfasr_batch_pending_requests {bm['pending_requests']}")
    if stream_engine is not None:
        sm = stream_engine.metrics
        for key in ("total_streams_opened", "total_streams_closed", "total_chunks_processed", "total_streams_reaped"):
            lines.append(f"# TYPE highperfasr_stream_{key} counter")
            lines.append(f"highperfasr_stream_{key} {sm[key]}")
        lines.append("# TYPE highperfasr_stream_active_streams gauge")
        lines.append(f"highperfasr_stream_active_streams {sm['active_streams']}")
    return PlainTextResponse(
        "
".join(lines) + "
",
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


@app.get("/admin/config")
async def get_tuning():
    result = {"mode": serving_mode}
    if batch_engine is not None:
        result["max_batch_size"] = batch_engine._max_batch_size
        result["max_wait_seconds"] = batch_engine._max_wait_seconds
        result["max_queue_depth"] = batch_engine._max_queue_depth
    if gpu_worker is not None:
        result["gpu_poll_timeout"] = gpu_worker._batch_poll_timeout
    return result


@app.post("/admin/config")
async def set_tuning(
    max_batch_size: Optional[int] = Query(None, ge=1, le=256),
    max_wait_seconds: Optional[float] = Query(None, ge=0.001, le=5.0),
    max_queue_depth: Optional[int] = Query(None, ge=16, le=8192),
    gpu_poll_timeout: Optional[float] = Query(None, ge=0.001, le=1.0),
):
    changes = {}
    if batch_engine is not None:
        if max_batch_size is not None:
            batch_engine._max_batch_size = max_batch_size
            changes["max_batch_size"] = max_batch_size
        if max_wait_seconds is not None:
            batch_engine._max_wait_seconds = max_wait_seconds
            changes["max_wait_seconds"] = max_wait_seconds
        if max_queue_depth is not None:
            batch_engine._max_queue_depth = max_queue_depth
            changes["max_queue_depth"] = max_queue_depth
    if gpu_worker is not None and gpu_poll_timeout is not None:
        gpu_worker._batch_poll_timeout = gpu_poll_timeout
        changes["gpu_poll_timeout"] = gpu_poll_timeout
    log.info(f"Config updated: {changes}")
    return {"updated": changes}


# --- Batch Transcription ---


def _max_upload_bytes() -> int:
    return config.get("batcher", {}).get("max_upload_bytes", 100 * 1024 * 1024)


def _save_upload_sync(src_file, suffix: str, max_bytes: int) -> str:
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name
        total = 0
        while True:
            chunk = src_file.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                tmp.close()
                os.unlink(tmp_path)
                raise ValueError(f"File exceeds {max_bytes} byte limit")
            tmp.write(chunk)
    return tmp_path


def _require_batch():
    if batch_engine is None:
        raise HTTPException(
            status_code=404,
            detail=f"Batch transcription not available — server running in mode={serving_mode}",
        )


def _require_stream():
    if stream_engine is None:
        raise HTTPException(
            status_code=404,
            detail=f"Streaming not available — server running in mode={serving_mode}",
        )


@app.post("/v1/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    timestamps: bool = Query(False, description="Include word-level timestamps"),
):
    _require_batch()
    max_bytes = _max_upload_bytes()
    suffix = Path(file.filename).suffix if file.filename else ".wav"

    loop = asyncio.get_running_loop()
    try:
        tmp_path = await loop.run_in_executor(None, functools.partial(_save_upload_sync, file.file, suffix, max_bytes))
    except ValueError as exc:
        raise HTTPException(status_code=413, detail=str(exc))

    try:
        result = await batch_engine.submit(tmp_path, timestamps=timestamps, owns_file=True)
        return JSONResponse(content=result)
    except QueueFullError:
        raise HTTPException(status_code=503, detail="Server overloaded — try again later")
    except RuntimeError as exc:
        if "max_file_duration_sec" in str(exc):
            raise HTTPException(status_code=413, detail=str(exc))
        raise


_MAX_BATCH_FILES = 64


@app.post("/v1/transcriptions:batch")
async def transcribe_batch(
    files: list[UploadFile] = File(...),
    timestamps: bool = Query(False),
):
    _require_batch()
    if len(files) > _MAX_BATCH_FILES:
        raise HTTPException(status_code=400, detail=f"Too many files (max {_MAX_BATCH_FILES})")

    max_bytes = _max_upload_bytes()
    loop = asyncio.get_running_loop()
    tmp_paths = []
    submitted_paths = set()
    try:
        for f in files:
            suffix = Path(f.filename).suffix if f.filename else ".wav"
            try:
                path = await loop.run_in_executor(
                    None, functools.partial(_save_upload_sync, f.file, suffix, max_bytes)
                )
                tmp_paths.append(path)
            except ValueError:
                tmp_paths.append(None)

        tasks = []
        for i, p in enumerate(tmp_paths):
            if p is None:
                fut = loop.create_future()
                fut.set_exception(ValueError(f"File {files[i].filename} exceeds size limit"))
                tasks.append(fut)
            else:
                tasks.append(batch_engine.submit(p, timestamps=timestamps, owns_file=True))
                submitted_paths.add(p)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        output = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                output.append({"error": str(r), "file": files[i].filename})
            else:
                output.append(r)

        return JSONResponse(content={"results": output})
    finally:
        for p in tmp_paths:
            if p is not None and p not in submitted_paths:
                try:
                    os.unlink(p)
                except OSError:
                    pass


# --- Streaming Transcription ---


@app.websocket("/v1/stream")
async def stream_ws(websocket: WebSocket):
    if stream_engine is None:
        await websocket.accept()
        await websocket.send_json({"error": f"Streaming not available — server running in mode={serving_mode}"})
        await websocket.close(code=1008)
        return

    await websocket.accept()

    try:
        session = await stream_engine.open_stream()
        stream_id = session["stream_id"]
        await websocket.send_json(session)
    except TooManyStreamsError:
        await websocket.send_json({"error": "Too many active streams"})
        await websocket.close(code=1013)
        return
    except Exception as exc:
        await websocket.send_json({"error": str(exc)})
        await websocket.close(code=1011)
        return

    try:
        while True:
            message = await websocket.receive()

            if message.get("type") == "websocket.disconnect":
                break

            if "bytes" in message:
                try:
                    result = await stream_engine.process_chunk(stream_id, message["bytes"])
                except ValueError:
                    await websocket.send_json({"error": "Stream closed by server (idle timeout)"})
                    break
                await websocket.send_json(result)

            elif "text" in message:
                data = json.loads(message["text"])
                if data.get("action") == "close":
                    break

    except WebSocketDisconnect:
        pass
    except (StreamExpiredError, ChunkTooLargeError) as exc:
        await websocket.send_json({"error": str(exc)})
    except Exception as exc:
        log.error(f"Stream {stream_id} error: {exc}")
        try:
            await websocket.send_json({"error": str(exc)})
        except Exception:
            pass
    finally:
        final = await stream_engine.close_stream(stream_id)
        try:
            await websocket.send_json(final)
            await websocket.close()
        except Exception:
            pass


def run_server(
    config_path: str | None = None,
    mode: str | None = None,
    host: str | None = None,
    port: int | None = None,
    model: str | None = None,
):
    apply_compat()

    global config, serving_mode
    config = load_config(config_path)

    resolved_mode = mode or config.get("mode", "both")
    if resolved_mode not in _VALID_MODES:
        raise SystemExit(f"Invalid mode '{resolved_mode}' — must be one of: {', '.join(sorted(_VALID_MODES))}")
    serving_mode = resolved_mode

    if model:
        if _stream_enabled():
            config.setdefault("stream_model", {})["name"] = model
        if _batch_enabled():
            config.setdefault("batch_model", {})["name"] = model

    server_cfg = config.get("server", {})
    resolved_host = host or server_cfg.get("host", "0.0.0.0")
    resolved_port = port or server_cfg.get("port", 8000)

    loop_type = "auto"
    try:
        import uvloop  # noqa: F401

        loop_type = "uvloop"
        log.info("uvloop available — using for event loop")
    except ImportError:
        log.info("uvloop not installed — using default asyncio (pip install uvloop for better concurrency)")

    log.info(f"Starting highperfasr on {resolved_host}:{resolved_port} (mode={serving_mode}, loop={loop_type})")
    uvicorn.run(
        app,
        host=resolved_host,
        port=resolved_port,
        workers=server_cfg.get("workers", 1),
        ws="auto",
        ws_ping_interval=None,
        ws_ping_timeout=None,
        log_level="info",
        loop=loop_type,
    )
