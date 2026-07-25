# Dockerfile Recipe Benchmark — Streaming

**GPU:** NVIDIA L4 24GB | **Date:** 2026-07-11
**Base image:** pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime + NeMo fork (beastoin/NeMo)
**Model:** nvidia/nemotron-3.5-asr-streaming-0.6b (0.6B params)
**Dataset:** LibriSpeech test-clean (200 samples)
**Scripts:** benchmarks/scripts/bench_stream.py, bench_stream_soak.py

## WER

| Metric | Value |
|--------|-------|
| Corpus WER | **3.21%** |
| Normalization | whisper_english |
| Samples | 200 |

## Concurrency Sweep

| c | RTFx | sess/min | p50 | p99 | Failures |
|---|------|----------|-----|-----|----------|
| 1 | 0.9x | 6.9 | 7.1s | 29.4s | 0 |
| 32 | 11.4x | 87.4 | 17.6s | 64.0s | 0 |
| 64 | 13.0x | 99.5 | 30.9s | 65.0s | 0 |
| 128 | 17.3x | 132.6 | 39.8s | 64.0s | 0 |
| 256 | 31.1x | 238.4 | 43.6s | 45.2s | 0 |
| 512 | 32.9x | 252.4 | 13.3s | 37.0s | 0 |

**Max clean concurrency: 512 (0 failures at all levels)**

## Sustained Load (c=64, 4 rounds)

| Metric | Value |
|--------|-------|
| RTFx | 12.9x |
| sess/min | 98.6 |
| Failures | 0 |

## Soak Test (c=64, 5 min, rotating)

| Metric | Value |
|--------|-------|
| Streams | 615 |
| Failures | 0 |
| RTFx | 13.0x |
| sess/min | 99.4 |
| WER | 7.5% |
| VRAM start | 8658 MB |
| VRAM end | 8658 MB |
| VRAM growth | **0 MB** |

## Comparison: Dockerfile vs NGC Container

| Metric | NGC (2026-07-10) | Dockerfile (2026-07-11) | Delta |
|--------|-------------------|--------------------------|-------|
| Base image | nvcr.io/nvidia/nemo:26.02 | pytorch:2.6.0-cuda12.6-cudnn9-runtime | -- |
| WER | 3.21% | 3.21% | **0.00%** |
| c=1 RTFx | 0.9x | 0.9x | 0% |
| c=64 RTFx | 12.9x | 13.0x | +1% |
| c=256 RTFx | 24.0x | 31.1x | **+30%** |
| c=512 RTFx | 24.7x | 32.9x | **+33%** |
| c=512 sess/min | 189.3 | 252.4 | **+33%** |
| Max clean c | 512 | 512 | same |
| Failures | 0 | 0 | same |
| VRAM | 8014 MB | 8658 MB | +644 MB |
| Soak VRAM growth | 0 MB | 0 MB | same |

## Verdict

**PASS — Dockerfile recipe matches or exceeds NGC container performance.**

- WER identical (3.21%)
- 0 failures at all concurrency levels up to c=512
- Throughput improved +33% at high concurrency (likely lighter base image overhead)
- VRAM stable (0 growth over 5 min soak)
- Ready for any company to deploy via `docker compose up`
