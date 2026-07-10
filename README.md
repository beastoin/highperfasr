# highperfasr

Serving optimization for existing open-source ASR models.

highperfasr does not train models or change recognition quality. It tunes server configuration and patches framework bottlenecks to maximize throughput and concurrency while preserving the model's published WER.

Run batch and streaming transcription on a single GPU with a simple, framework-agnostic protocol:

- REST endpoint for file transcription
- WebSocket endpoint for real-time PCM16 streams
- Docker Compose for local GPU serving
- GKE L4 manifest for single-GPU deployment
- Published benchmarks with reproducible results

Measured on one GKE L4 GPU with model quality preserved:

| Workload | Result |
|----------|--------|
| Streaming concurrency | 512 WebSocket streams, 0 failures |
| Streaming quality | 3.21% WER on LibriSpeech test-clean |
| Streaming throughput | 297 sessions/min, 38.69x realtime |
| Batch throughput | 178x realtime, about 3 hours of audio per minute |
| Batch quality | 1.57% WER on LibriSpeech test-clean |
| Cost | About $0.70/hr on GKE L4 |

```bash
git clone https://github.com/beastoin/highperfasr
cd highperfasr
docker compose up -d
curl -F "file=@audio.wav" http://localhost:8000/v1/transcriptions
```

## Performance

### Streaming (Nemotron 3.5 ASR 0.6B)

![RTFx Scaling](benchmarks/results/2026-l4-nemo-512-streams/rtfx-scaling.png)

![Throughput Scaling](benchmarks/results/2026-l4-nemo-512-streams/throughput-scaling.png)

![Cost Efficiency](benchmarks/results/2026-l4-nemo-512-streams/cost-efficiency.png)

512 persistent WebSocket streams, 10-minute real-time soak, all 2,620 LibriSpeech
test-clean files. WER 3.21%, 297 sessions/min, 8672 MB VRAM (38%), 0 failures.

Full report: [benchmarks/results/2026-l4-nemo-512-streams/](benchmarks/results/2026-l4-nemo-512-streams/)

### Batch (Parakeet TDT 0.6B)

![Batch RTFx Scaling](benchmarks/results/2026-l4-nemo-batch/rtfx-scaling.png)

![Batch Throughput](benchmarks/results/2026-l4-nemo-batch/throughput-scaling.png)

![Batch Latency](benchmarks/results/2026-l4-nemo-batch/latency-scaling.png)

![Batch Cost Efficiency](benchmarks/results/2026-l4-nemo-batch/cost-efficiency.png)

REST concurrency sweep c=1..512, LibriSpeech test-clean (200 files).
WER 1.57%, peak 19.5 RPS (178x realtime), 0 failures at every level.

Full report: [benchmarks/results/2026-l4-nemo-batch/](benchmarks/results/2026-l4-nemo-batch/)

### Methodology

Quality rubric: real speech corpus, standard WER normalization (Whisper
EnglishTextNormalizer), sustained concurrent load, reproducible artifacts.
Verify: [report schema](benchmarks/report-schema.json),
[streaming result.json](benchmarks/results/2026-l4-nemo-512-streams/result.json),
[batch result.json](benchmarks/results/2026-l4-nemo-batch/result.json).

## Deploy

Prerequisites: Docker, NVIDIA Container Toolkit, and a CUDA-capable GPU. The
first run downloads the ASR models and caches them in Docker volumes.

| Command | What |
|---------|------|
| `docker compose up -d` | Start batch (:8000) + streaming (:8001) |
| `docker compose up -d stream` | Start streaming only |
| `docker compose up -d batch` | Start batch only |
| `make health` | Check server readiness |
| `make smoke` | Run a quick transcription test |
| `docker compose logs -f` | Tail server logs |

### GKE L4

```bash
docker build --target stream -t $REGISTRY/highperfasr-stream:v0.1.0 .
docker push $REGISTRY/highperfasr-stream:v0.1.0
kubectl apply -f gke-l4.yaml
```

## Protocol (v1alpha1)

highperfasr uses a framework-agnostic protocol: REST for files, WebSocket for
live audio, and health checks for orchestration.

| Endpoint | Mode | Input |
|----------|------|-------|
| `POST /v1/transcriptions` | Batch | Multipart file upload |
| `WebSocket /v1/stream` | Streaming | Raw PCM16 audio frames |
| `GET /health` | Health | Readiness and server mode |

Full spec: [spec/protocol.md](spec/protocol.md) | [OpenAPI](spec/openapi.yaml) | [AsyncAPI](spec/asyncapi.yaml)

## Structure

```text
Dockerfile           # multi-target image: batch + stream
compose.yaml         # docker compose up -d
gke-l4.yaml          # GKE L4 GPU deployment
labs/nemo-fastapi/   # NeMo serving + framework patches
spec/                # REST + WebSocket protocol
benchmarks/          # benchmark reports and tooling
```

## Q&A

**Does it support punctuation and capitalization?**
Yes. Built into the model output for both batch and streaming.

**Does it support word-level timestamps?**
Yes, in batch mode. Pass `?timestamps=true` on the REST endpoint.

**Does it support multiple languages?**
Streaming mode supports multilingual transcription via prompt-conditioned language selection. Batch mode is English-only with the current model.

**Does it support diarization (speaker identification)?**
Not yet. Diarization requires a separate model (e.g. pyannote). It is on the roadmap.

**Does it support long audio files?**
Yes, batch mode handles audio up to 1 hour on L4 via automatic local attention switching.

**Does it support inverse text normalization (ITN)?**
Not yet. The underlying models support it, but it is not exposed in the serving layer.

**What about partial/streaming transcripts?**
Yes. Streaming mode sends partial transcripts after each chunk, followed by a final transcript.

## Users & Sponsors

- **[Omi](https://omi.me)** uses highperfasr in production for an AI wearable
  workload with thousands of concurrent streams. Omi also sponsors the GPU
  benchmark work published in this repository.

## Mission

Speech recognition infrastructure should be something teams can run, measure,
and control.

highperfasr exists to help companies keep audio inside their own pipeline, on
their own infrastructure, without depending on third-party ASR APIs.

## License

MIT
