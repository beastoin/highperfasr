# Examples

## Evaluate on Google Cloud

Deploy a GPU server on your GCP project and benchmark it — server ready in
under 10 minutes.

### One-click (Cloud Shell)

[![Open in Cloud Shell](https://gstatic.com/cloudssh/images/open-btn.svg)](https://shell.cloud.google.com/cloudshell/open?git_repo=https://github.com/beastoin/highperfasr&tutorial=examples/cloud-shell-tutorial.md)

Opens a browser-based terminal, clones the repo, and walks you through
deployment. No local setup needed.

### CLI

```bash
# Deploy with current gcloud auth
./examples/launch-gce.sh

# Deploy with service account
./examples/launch-gce.sh --key service-account.json

# Batch mode on T4 GPU
./examples/launch-gce.sh --mode batch --gpu t4

# Teardown
./examples/launch-gce.sh teardown
```

Requirements: `gcloud` CLI, GPU quota in the target zone.

See `./examples/launch-gce.sh --help` for all options.

### Benchmark Dashboard

The [benchmark dashboard](web/benchmark-dashboard.html) is a self-contained
HTML tool for evaluating server performance:

- **Batch** — upload audio files, measure latency, RPS, and WER
- **Stream** — real-time streaming from file or microphone
- **Sweep** — concurrency sweep with heatmap visualization
- **Report** — quality gates, per-utterance WER diffs, export as ZIP

Open it directly or pass `?server=http://...` to auto-connect.

## Client Libraries

### Python — Batch (REST)

```bash
pip install requests
python examples/python/batch_client.py audio.wav --server http://localhost:8000
```

With word timestamps:
```bash
python examples/python/batch_client.py audio.wav --timestamps
```

### Python — Streaming (WebSocket)

```bash
pip install websockets
python examples/python/stream_client.py audio.wav --server ws://localhost:8001
```

### Node.js — Streaming (WebSocket)

```bash
npm install ws
node examples/js/stream_client.mjs audio.wav ws://localhost:8001
```

## Audio Requirements

- **Batch:** WAV, FLAC, or MP3
- **Streaming:** WAV file must be 16-bit PCM, mono, 16 kHz

## Protocol

See [spec/protocol.md](../spec/protocol.md) for the full API specification.
