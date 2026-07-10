# HighPerfASR Benchmark Report: L4 Streaming 512 Concurrent

**Report ID:** 2026-l4-nemo-512-streams
**Protocol Version:** v1alpha1
**Date:** 2026-07-05

## System Under Test

| Field | Value |
|-------|-------|
| Implementation | labs/nemo-fastapi |
| Model | nvidia/nemotron-3.5-asr-streaming-0.6b (0.6B params) |
| GPU | NVIDIA L4 24GB |
| Container | nvcr.io/nvidia/nemo:26.02 |
| CUDA | 12.8, Driver 570.86.15 |
| PyTorch | 2.6.0 |

## Scenario

512 persistent WebSocket streams, real-time paced, 10-minute sustained soak. All 2620 LibriSpeech test-clean files used across streams (diverse audio, not repeated).

## Results

| Metric | Value |
|--------|-------|
| **WER** | 3.21% |
| **Max Concurrent Streams** | 512 |
| **Throughput** | 297 sessions/min (38.69x real-time) |
| **Failures** | 0 / 512 |
| **VRAM High-Water** | 8672 MB (38% of 24GB) |
| **VRAM Baseline** | 8250 MB (512 decoder slots preallocated) |

## Concurrency Sweep

| Concurrency | RTFx | Sessions/min | Failures |
|:-----------:|:----:|:------------:|:--------:|
| 32 | 10.32x | 79.2 | 0 |
| 64 | 13.05x | 100.2 | 0 |
| 128 | 18.76x | 144.0 | 0 |
| 256 | 27.55x | 211.4 | 0 |
| 512 | 38.69x | 297.0 | 0 |

Still scaling linearly at c=512 — no ceiling hit.

## VRAM Stability (10-minute soak)

| Checkpoint | Active Streams | VRAM |
|:----------:|:--------------:|:----:|
| 60s | 512/512 | 6456 MB |
| 182s | 512/512 | 8420 MB |
| 303s | 512/512 | 8672 MB |
| 604s | 512/512 | 8672 MB |

VRAM stabilized at 8672 MB after ~5 minutes. No memory leak detected.

## Reproduction

```bash
# Server config
highperfasr serve --config labs/nemo-fastapi/configs/serving-stream.yaml

# Benchmark
python3 bench_stream_longlive.py \
  --server ws://localhost:8000 \
  --concurrency 512 \
  --durations 600 \
  --persistent \
  --max-samples 0 \
  --vram-interval 10
```

## Notes

- WER measured with Whisper EnglishTextNormalizer (lowercase, expand contractions, strip punctuation)
- Each decoder slot costs ~3 MB VRAM (512 slots = ~1.5 GB)
- `max_stream_drain=16` — higher values cause VRAM explosion due to too many active decoder states per GPU batch
- `num_slots` auto-set from `max_concurrent_streams` config (default pipeline config only has 256)
