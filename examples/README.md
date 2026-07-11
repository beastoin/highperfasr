# Client Examples

Copy-paste examples for integrating with highperfasr.

## Prerequisites

Start the server:

```bash
docker compose up -d          # stream on :8001
# or for both batch + stream:
docker compose --profile full up -d   # batch :8000, stream :8001
```

## Python

Install dependencies:

```bash
pip install requests websockets
```

### Batch transcription (REST)

```bash
python examples/python/batch_client.py audio.wav
# With word timestamps:
python examples/python/batch_client.py audio.wav --timestamps
# Custom server:
python examples/python/batch_client.py audio.wav --url http://gpu-server:8000
```

Expected output:

```json
{
  "text": "the quick brown fox jumps over the lazy dog",
  "duration_seconds": 3.2,
  "processing_time_seconds": 0.15
}
```

### Streaming transcription (WebSocket)

```bash
python examples/python/stream_client.py audio.wav
# Custom endpoint:
python examples/python/stream_client.py audio.wav --url ws://gpu-server:8001/v1/stream
# Larger chunks (lower overhead, higher latency):
python examples/python/stream_client.py audio.wav --chunk-ms 480
```

Expected output:

```
Connecting to ws://localhost:8001/v1/stream ...
Stream opened: a1b2c3d4-...
  [1] the
  [2] the quick
  [3] the quick brown fox

Final transcript:
  the quick brown fox jumps over the lazy dog

Stats: 20 chunks sent (3.2s audio)
```

**Audio format**: 16-bit PCM mono WAV at 16kHz. The script warns if the format doesn't match.

## Node.js

Install dependencies:

```bash
npm install ws
```

### Streaming transcription (WebSocket)

```bash
node examples/js/stream_client.mjs audio.wav
# Custom endpoint:
node examples/js/stream_client.mjs audio.wav --url ws://gpu-server:8001/v1/stream
```

## Protocol Reference

### Batch (REST)

```
POST /v1/transcriptions
Content-Type: multipart/form-data

file=@audio.wav
timestamps=true  (optional)
```

### Streaming (WebSocket)

1. Connect to `ws://host:8001/v1/stream`
2. Server sends JSON ack: `{"stream_id": "uuid", ...}`
3. Send raw PCM16 bytes (16kHz mono, any chunk size — 160ms recommended)
4. Server responds with JSON after each chunk: `{"partial": "text so far"}`
5. Send `{"action": "close"}` when done
6. Server sends final transcript and closes the connection
