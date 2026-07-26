# Examples

## Evaluate on Google Cloud

Deploy a GPU server on your GCP project and benchmark it — server ready in
under 10 minutes.

### One-click (Cloud Shell)

[![Open in Cloud Shell](https://gstatic.com/cloudssh/images/open-btn.svg)](https://shell.cloud.google.com/cloudshell/editor?cloudshell_git_repo=https://github.com/beastoin/highperfasr&cloudshell_tutorial=examples/cloud-shell-tutorial.md&cloudshell_workspace=.)

Opens a browser-based terminal, clones the repo, and walks you through
authentication, project selection, and deployment. No local setup needed.

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
HTML tool for exploratory server checks:

- **Batch** — upload audio files and inspect latency and RPS
- **Stream** — real-time streaming from file or microphone
- **Sweep** — concurrency sweep with heatmap visualization
- **Report** — request traces, timing summaries, and export as ZIP

Open it directly for same-origin/local checks, or pass `?server=https://...` to
auto-connect to an HTTPS/proxied endpoint. Browsers block the public GitHub Pages
dashboard from calling an HTTP VM endpoint. Use `benchmarks/scripts/` for
canonical WER and quality-gate results.

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
