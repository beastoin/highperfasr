# HighPerfASR Protocol Specification

**Version:** v1alpha1
**Status:** Draft

## 1. Overview

The HighPerfASR protocol defines a standard interface for automatic speech recognition (ASR) servers. Any server implementing this protocol MUST support at least one of the two transcription modes:

- **Batch** — REST API for file-based transcription
- **Streaming** — WebSocket API for real-time audio streams

The key words "MUST", "MUST NOT", "SHOULD", "SHOULD NOT", and "MAY" in this document are to be interpreted as described in RFC 2119.

## 2. Audio Format

All audio input MUST be one of:

| Format | MIME Type | Notes |
|--------|-----------|-------|
| WAV (PCM16) | audio/wav | Preferred for batch |
| Raw PCM16 | application/octet-stream | Preferred for streaming (binary WebSocket frames) |
| FLAC | audio/flac | Optional |
| MP3 | audio/mpeg | Optional |

For streaming, the server MUST accept raw PCM16 mono at the configured sample rate (default: 16000 Hz). Each binary WebSocket frame contains raw PCM16 samples in little-endian byte order.

## 3. Health Endpoint

### `GET /health`

MUST return 200 OK with a JSON body:

```json
{
  "status": "ok",
  "ready": true,
  "mode": "stream",
  "uptime_seconds": 42.5
}
```

Fields:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `status` | string | MUST | `"ok"` when healthy, `"loading"` during startup |
| `ready` | boolean | MUST | `true` when ready to accept requests |
| `mode` | string | MUST | One of: `"batch"`, `"stream"`, `"both"` |
| `uptime_seconds` | number | SHOULD | Seconds since server started |

The server MAY include additional fields (model names, version, etc.).

## 4. Capabilities Endpoint

### `GET /v1/capabilities`

SHOULD return 200 OK with:

```json
{
  "protocol_version": "v1alpha1",
  "modes": ["batch", "stream"],
  "audio_formats": ["wav", "pcm16"],
  "sample_rates": [16000],
  "languages": ["en"],
  "features": {
    "timestamps": true,
    "partial_transcripts": true,
    "batch_upload": true
  }
}
```

This endpoint is RECOMMENDED but not required.

## 5. Batch Transcription

### `POST /v1/transcriptions`

Upload a single audio file for transcription.

**Request:**
- Content-Type: `multipart/form-data`
- Field `file`: the audio file

**Query parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `timestamps` | boolean | false | Include word-level timestamps |

**Response (200 OK):**

```json
{
  "text": "the transcribed text"
}
```

With timestamps:

```json
{
  "text": "the transcribed text",
  "words": [
    {"word": "the", "start": 0.0, "end": 0.12},
    {"word": "transcribed", "start": 0.15, "end": 0.62}
  ]
}
```

**Error responses:**

| Code | Condition |
|------|-----------|
| 413 | File exceeds size limit |
| 503 | Server overloaded (queue full) |
| 415 | Unsupported audio format |

### `POST /v1/transcriptions:batch`

Upload multiple files for batch transcription.

**Request:**
- Content-Type: `multipart/form-data`
- Field `files`: multiple audio files

**Response (200 OK):**

```json
{
  "results": [
    {"text": "first file transcript"},
    {"text": "second file transcript"}
  ]
}
```

## 6. Streaming Transcription

### `WebSocket /v1/stream`

Real-time streaming transcription over WebSocket.

**Connection lifecycle:**

1. Client opens WebSocket connection to `/v1/stream`
2. Server accepts and sends an `opened` message
3. Client sends raw PCM16 audio as binary frames
4. Server sends transcript updates as JSON text frames
5. Client sends a `close` action to finalize
6. Server sends final transcript and closes the connection

**Server → Client messages:**

Session opened:
```json
{"stream_id": "uuid", "status": "opened"}
```

Transcript update:
```json
{
  "stream_id": "uuid",
  "partial_transcript": "the current partial",
  "final_transcript": "confirmed words",
  "is_final": false
}
```

Error:
```json
{"error": "description"}
```

Final transcript (on close):
```json
{
  "stream_id": "uuid",
  "final_text": "the complete transcript",
  "status": "closed"
}
```

**Client → Server messages:**

Binary frames: raw PCM16 audio data.

Close action (JSON text frame):
```json
{"action": "close"}
```

**Semantics:**

- `partial_transcript` — the current hypothesis for uncommitted audio. Changes between updates. Servers MUST send this field (empty string if no hypothesis).
- `final_transcript` — newly confirmed words since the last update. Once emitted, these words MUST NOT change. Empty string if no new confirmed words.
- `is_final` — `true` when `final_transcript` is non-empty.
- `final_text` — the complete transcript for the entire session, sent only on close.

**Error handling:**

| Close code | Condition |
|-----------|-----------|
| 1008 | Streaming not available in current mode |
| 1013 | Too many concurrent streams |
| 1011 | Internal server error |

## 7. Metrics Endpoint

### `GET /metrics`

MAY return server metrics in JSON:

```json
{
  "uptime_seconds": 600.0,
  "mode": "stream",
  "stream": {
    "active_streams": 42,
    "total_streams_opened": 1000,
    "total_chunks_processed": 50000
  }
}
```

The server MAY also expose Prometheus-format metrics.

## 8. Conformance Requirements

A conforming server MUST:

1. Implement `/health` returning the required fields
2. Implement at least one of `/v1/transcriptions` or `/v1/stream`
3. Return proper HTTP status codes for error conditions
4. Accept PCM16 audio at the advertised sample rate
5. Return UTF-8 text in all transcript fields

A conforming server SHOULD:

1. Implement `/v1/capabilities`
2. Implement `/metrics`
3. Support graceful shutdown (drain active streams)
4. Return partial transcripts during streaming
5. Support WebSocket close codes as specified

## 9. Versioning

The protocol version follows the pattern `v{major}alpha{n}` during development and `v{major}` for stable releases. Breaking changes increment the major version. The protocol version MUST be reported in `/v1/capabilities`.
