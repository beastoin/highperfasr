# Client Examples

Copy-paste examples for integrating with highperfasr.

## Python — Batch (REST)

```bash
pip install requests
python examples/python/batch_client.py audio.wav --server http://localhost:8000
```

Expected output:
```
the transcribed text from your audio file
```

With word timestamps:
```bash
python examples/python/batch_client.py audio.wav --timestamps
```

## Python — Streaming (WebSocket)

```bash
pip install websockets
python examples/python/stream_client.py audio.wav --server ws://localhost:8001
```

Expected output:
```
Stream opened: a1b2c3d4
  partial: the transcribed
  partial: the transcribed text
Final: the transcribed text from your audio file
```

## Node.js — Streaming (WebSocket)

```bash
npm install ws
node examples/js/stream_client.mjs audio.wav ws://localhost:8001
```

Expected output is the same as the Python streaming example.

## Audio Requirements

- **Batch:** WAV, FLAC, or MP3
- **Streaming:** WAV file must be 16-bit PCM, mono, 16 kHz

## Protocol

See [spec/protocol.md](../spec/protocol.md) for the full API specification.
