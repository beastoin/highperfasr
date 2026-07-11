# Client Examples

Copy-paste examples for integrating with a HighPerfASR server.

## Prerequisites

- A running HighPerfASR server (batch on port 8000, streaming on port 8001)
- A test WAV file (16 kHz, mono, PCM16)

## Python

### Install dependencies

```bash
pip install requests websockets
```

### Batch transcription (REST)

Upload an audio file and get the full transcript:

```bash
python examples/python/batch_client.py audio.wav
```

With word-level timestamps:

```bash
python examples/python/batch_client.py --timestamps audio.wav
```

Custom server URL:

```bash
python examples/python/batch_client.py --url http://myserver:8000 audio.wav
```

**Expected output:**

```
Uploading audio.wav to http://localhost:8000 ...

Transcript: the transcribed text from the audio file
```

### Streaming transcription (WebSocket)

Stream a WAV file in real time and see partial transcripts:

```bash
python examples/python/stream_client.py audio.wav
```

Custom chunk size (200ms) and server URL:

```bash
python examples/python/stream_client.py --url ws://myserver:8001 --chunk-ms 200 audio.wav
```

**Expected output:**

```
Connecting to ws://localhost:8001/v1/stream ...
Stream opened (id: a1b2c3d4-...)
[partial] the current hypo
[partial] the current hypothesis
[final]   the current hypothesis is
[partial] updated
[partial] updated as more

Final transcript: the current hypothesis is updated as more audio arrives

Done.
```

## Node.js

### Install dependencies

```bash
npm install ws
```

### Streaming transcription (WebSocket)

```bash
node examples/js/stream_client.mjs audio.wav
```

Custom chunk size and server URL:

```bash
node examples/js/stream_client.mjs --url ws://myserver:8001 --chunk-ms 200 audio.wav
```

**Expected output:**

Same format as the Python streaming client above.

## Protocol Reference

See [spec/protocol.md](../spec/protocol.md) for the full protocol specification.

### Batch endpoint

```
POST /v1/transcriptions
Content-Type: multipart/form-data
Field: file (the audio file)

Response: {"text": "transcribed text"}
```

### Streaming endpoint

```
WebSocket /v1/stream

1. Connect
2. Server sends: {"stream_id": "uuid", "status": "opened"}
3. Client sends: raw PCM16 binary frames (3200 bytes = 100ms at 16kHz)
4. Server sends: {"partial_transcript": "...", "final_transcript": "...", "is_final": bool}
5. Client sends: {"action": "close"}
6. Server sends: {"final_text": "...", "status": "closed"}
```
