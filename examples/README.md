# Client Examples

Copy-paste examples for integrating with a highperfasr server.

## Prerequisites

A running highperfasr server:

```bash
docker compose up -d
# Batch endpoint: http://localhost:8000
# Streaming endpoint: ws://localhost:8001
```

A 16kHz mono WAV file for testing. Any short speech recording works.

## Python

### Batch transcription

```bash
pip install requests
python python/batch_client.py audio.wav
python python/batch_client.py audio.wav --timestamps
```

Expected output:

```json
{
  "text": "he hoped there would be stew for dinner turnips and carrots and bruised potatoes and fat mutton pieces"
}
```

### Streaming transcription

```bash
pip install websockets
python python/stream_client.py audio.wav
```

Expected output:

```
Connecting to ws://localhost:8001/v1/stream ...
Stream opened: abc123-...
  [final] he hoped there would be stew
  [final] for dinner turnips and carrots
Transcript: he hoped there would be stew for dinner turnips and carrots ...
Stats: 62 chunks, 5.0s audio
```

## JavaScript (Node.js)

### Streaming transcription

```bash
npm install ws
node js/stream_client.mjs audio.wav
```

Output format matches the Python streaming client.

## Protocol

- **Batch:** `POST /v1/transcriptions` with multipart file upload. Returns `{"text": "..."}`.
- **Streaming:** WebSocket to `/v1/stream`. Send raw PCM16 binary frames, receive JSON with `partial_transcript` and `final_transcript`. Send `{"action": "close"}` to finalize.

Full spec: [spec/protocol.md](../spec/protocol.md)
