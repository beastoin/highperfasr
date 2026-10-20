# highperfasr

Run speech recognition on your own GPU. No third-party API, no data leaving your infrastructure.

highperfasr turns open-source ASR models into production-ready servers — batch transcription for files, real-time streaming for live audio — on a single GPU you control.

| | Batch | Streaming |
|---|---|---|
| **What it does** | Transcribe audio files via REST | Live transcription via WebSocket |
| **Quality** | 1.92% WER | 3.51% WER |
| **Speed** | 129x realtime | 64 concurrent streams |
| **Model** | Parakeet TDT 0.6B | Nemotron 3.5 ASR 0.6B |
| **Languages** | 25 European | 40 locales |
| **Cost** | ~$0.70/hr on GKE L4 | ~$0.70/hr on GKE L4 |

All numbers measured on LibriSpeech test-clean (2,620 files). [Full benchmark reports →](https://beastoin.github.io/highperfasr/)

> "We use the highperfasr recipe for the Omi batch sync engine. It helped reduce cost from $7,830 to $614 — 12.7x savings — and there is still a lot of headroom in that $614 L4 GPU."
>
> — Thinh Nguyen, CTO at [Omi](https://omi.me)

## Get Started

Five commands to a running server:

```bash
mkdir highperfasr && cd highperfasr
curl -fsSLO https://raw.githubusercontent.com/beastoin/highperfasr/main/compose.yaml
docker compose pull stream
docker compose up -d
curl http://localhost:8001/health
```

### Deploy on Google Cloud

No GPU? No Docker? Deploy on your own GCP project in one click:

**[Launch Deploy Page](https://beastoin.github.io/highperfasr/deploy.html)** — sign in with Google, pick your project, click Deploy.

Or use Cloud Shell:

[![Open in Cloud Shell](https://gstatic.com/cloudssh/images/open-btn.svg)](https://shell.cloud.google.com/cloudshell/editor?cloudshell_git_repo=https://github.com/beastoin/highperfasr&cloudshell_tutorial=examples/cloud-shell-tutorial.md&cloudshell_workspace=.)

Or from any terminal:

```bash
git clone https://github.com/beastoin/highperfasr && cd highperfasr
./examples/launch-gce.sh
```

The script creates a GCE VM with an L4 GPU, starts the server, and prints
benchmark commands when ready. Auto-shuts down after 4 hours.

### Multi-Cloud Kubernetes

| Recipe | GPU | ~$/hr |
|--------|-----|-------|
| [gcp-t4](recipes/gcp-t4/) | NVIDIA T4 | $0.35 |
| [gcp-l4](recipes/gcp-l4/) | NVIDIA L4 | $0.70 |
| [aws-g6-l4](recipes/aws-g6-l4/) | NVIDIA L4 | $0.80 |
| [azure-a10](recipes/azure-a10/) | NVIDIA A10 | $0.91 |

```bash
kubectl apply -k recipes/gcp-l4    # pick your cloud
```

## How It Works

| Endpoint | Mode | Input |
|----------|------|-------|
| `POST /v1/transcriptions` | Batch | Upload an audio file |
| `WebSocket /v1/stream` | Streaming | Send live audio frames |
| `GET /health` | Health | Check readiness |

Full spec: [protocol.md](spec/protocol.md) | [OpenAPI](spec/openapi.yaml) | [AsyncAPI](spec/asyncapi.yaml)

See [examples/](examples/) for Python and Node.js integration.

## Performance

Benchmarked on GKE L4 GPU, full LibriSpeech test-clean corpus (2,620 files):

**Batch** — 129x realtime sustained at c=64, 17.5 requests/sec, WER 1.92%, zero failures across all concurrency levels.

**Streaming** — 64 concurrent streams with zero failures, 13.2x realtime at peak, first transcript token in 1.3s (TTFB p50 at c=4), WER 3.51%.

Every benchmark uses real speech (LibriSpeech, SHA256-verified), standard WER normalization
(Whisper English), and automated quality gates. Results are reproducible — run the same
scripts yourself: [benchmarks/scripts/](benchmarks/scripts/).

Full reports with detailed sweep tables: [GitHub Pages](https://beastoin.github.io/highperfasr/) |
[Batch JSON](benchmarks/results/l4-batch-20260725T055248/report.json) |
[Stream JSON](benchmarks/results/l4-stream-20261019T000000/report.json)

## Q&A

| Question | Batch | Streaming |
|----------|-------|-----------|
| Languages | 25 European, auto-detect | 40 locales, auto-detect |
| Punctuation? | Yes | Yes |
| Timestamps? | Word-level | Partial transcripts (real-time) |
| Max audio length | 24 min (full attention), 3 hr (local) | Indefinite (persistent WebSocket) |
| Diarization | Roadmap | Roadmap |

## Users & Sponsors

- **[Omi](https://omi.me)** uses highperfasr in production for an AI wearable
  workload with thousands of concurrent streams. Omi also sponsors the GPU
  benchmark work published in this repository.

## Mission

Speech recognition infrastructure should be something teams can run, measure,
and control — on their own GPUs, inside their own cloud account.

highperfasr exists to help companies keep audio inside their own pipeline,
talking directly to GPU providers like GCP, AWS, and Azure, without depending
on third-party ASR APIs. Your data never leaves your infrastructure.

## License

MIT
