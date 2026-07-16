# highperfasr Benchmark Report: L4 Batch by Duration

**Report ID:** 2026-l4-batch-by-duration
**Date:** 2026-07-16

## System Under Test

| Field | Value |
|-------|-------|
| Implementation | labs/nemo-fastapi |
| Model | nvidia/parakeet-tdt-0.6b-v3 (0.6B params) |
| GPU | NVIDIA L4 24GB |
| Container | nvcr.io/nvidia/nemo:26.02 |
| Mode | both (batch + stream on single GPU) |
| Attention | rel_pos (manual, use_pytorch_sdpa=false) |
| Batch config | max_batch_size=32, vram_bytes_per_t2=136.6, safety=0.8 |

## Scenario

Duration-stratified batch concurrency sweep. 100 espeak-ng speech files per duration bracket, testing throughput and max safe concurrency as a function of audio duration.

WER measured separately on LibriSpeech test-clean: **1.57%** (batch-only) / **1.84%** (both mode).

## Summary

| Duration | Avg File | Max Safe c | Peak RPS | Peak RTFx | Sustained RPS | Sustained RTFx | Failures |
|----------|----------|-----------|----------|-----------|---------------|----------------|----------|
| 10s | 11.9s | c=256 | 14.91 | 235x | 13.34 | 211x | 0 |
| 30s | 34.7s | c=256 | 5.33 | 189x | 6.65 | 234x | 0 |
| 60s | 66.5s | c=24 | 2.45 | 161x | 3.11 | 205x | 0 |
| 120s | 126.9s | c=128 | 1.38 | 173x | 1.61 | 202x | 0 |

## Key Finding: 60s Concurrency Cliff

60s audio is hard-capped at c=24 due to VRAM estimator bug (issue #19):
- Coefficient 136.6 bytes/T² estimates 99.5 MB/file but actual is ~350 MB/file (3.5x under)
- Estimator allows batch=32 for 60s files → all 32 on GPU simultaneously → CUDA OOM
- 120s files are accidentally safe: estimator caps batch to 15, only 15 files on GPU at a time

Despite the c=24 cap, 60s RTFx (161x) is comparable to other durations because GPU utilization per batch is high.

## Concurrency Sweeps

### 10s Audio (11.9s avg)

| c | RPS | RTFx | RTF | p50 | p99 | Failures |
|---|-----|------|-----|-----|-----|----------|
| 1 | 2.05 | 32.5x | 0.031 | 0.5s | 1.4s | 0 |
| 8 | 7.72 | 122.2x | 0.008 | 1.0s | 1.7s | 0 |
| 32 | 8.34 | 132.1x | 0.008 | 3.9s | 5.0s | 0 |
| 64 | 9.74 | 154.2x | 0.006 | 6.6s | 10.3s | 0 |
| 128 | 12.35 | 192.1x | 0.005 | 5.0s | 10.4s | 0 |
| 256 | 14.91 | 235.2x | 0.004 | 9.5s | 17.2s | 0 |

Sustained (c=128, 4 rounds): 13.34 RPS / 211.3x RTFx / 0 failures

### 30s Audio (34.7s avg)

| c | RPS | RTFx | RTF | p50 | p99 | Failures |
|---|-----|------|-----|-----|-----|----------|
| 1 | 1.35 | 47.5x | 0.021 | 0.7s | 1.5s | 0 |
| 8 | 3.69 | 130.1x | 0.008 | 2.3s | 3.1s | 0 |
| 32 | 4.57 | 160.9x | 0.006 | 7.4s | 8.6s | 0 |
| 64 | 4.76 | 167.8x | 0.006 | 11.6s | 18.0s | 0 |
| 128 | 5.07 | 179.5x | 0.006 | 15.8s | 25.2s | 0 |
| 256 | 5.33 | 188.8x | 0.005 | 29.7s | 48.0s | 0 |

Sustained (c=128, 4 rounds): 6.65 RPS / 234.2x RTFx / 0 failures

### 60s Audio (66.5s avg) — VRAM-limited

| c | RPS | RTFx | RTF | p50 | p99 | Failures |
|---|-----|------|-----|-----|-----|----------|
| 1 | 0.83 | 54.8x | 0.018 | 1.2s | 2.1s | 0 |
| 4 | 1.78 | 117.3x | 0.009 | 2.2s | 3.1s | 0 |
| 8 | 1.89 | 124.1x | 0.008 | 4.5s | 5.5s | 0 |
| 16 | 2.24 | 147.2x | 0.007 | 7.4s | 9.1s | 0 |
| 24 | 2.45 | 161.3x | 0.006 | 9.7s | 12.2s | 0 |

Max safe: c=24. c=32 causes CUDA OOM (issue #19). Sustained (c=16, 4 rounds): 3.11 RPS / 204.5x RTFx / 0 failures

### 120s Audio (126.9s avg)

| c | RPS | RTFx | RTF | p50 | p99 | Failures |
|---|-----|------|-----|-----|-----|----------|
| 1 | 0.42 | 52.2x | 0.019 | 2.4s | 3.3s | 0 |
| 4 | 0.88 | 110.4x | 0.009 | 4.5s | 5.9s | 0 |
| 8 | 1.00 | 125.3x | 0.008 | 8.7s | 9.1s | 0 |
| 16 | 1.23 | 154.0x | 0.006 | 12.6s | 16.1s | 0 |
| 32 | 1.38 | 172.8x | 0.006 | 23.6s | 30.5s | 0 |
| 64 | 1.26 | 158.2x | 0.006 | 32.5s | 79.1s | 0 |
| 128 | 1.30 | 162.7x | 0.006 | 58.1s | 98.6s | 0 |

Peak at c=32. Throughput saturates beyond c=32 (estimator caps batch=15). Sustained (c=64, 4 rounds): 1.61 RPS / 201.8x RTFx / 0 failures

## Reproduction

```bash
python3 benchmarks/scripts/bench_batch_by_duration.py \
  --server http://localhost:8000 \
  --durations 10,30,60,120 \
  --files 100 \
  --sustained-rounds 4
```

## Notes

- Audio generated with espeak-ng (real speech, not silence/tones) — exercises full RNNT decoder path
- WER not measured per-duration (TTS audio); see LibriSpeech benchmarks for WER
- 60s concurrency hard-capped at c=24 to prevent OOM (issue #19)
- 120s throughput peaks at c=32 because VRAM estimator caps batch to 15; higher concurrency just increases queue depth
- RTFx is comparable across durations (150-235x) — GPU efficiency is high regardless of file length
- Zero failures across all 1,800+ requests in the sweep + 1,600 sustained load requests
