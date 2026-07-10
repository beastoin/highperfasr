# highperfasr Benchmark Report: L4 Batch Concurrency Sweep

**Report ID:** 2026-l4-nemo-batch
**Protocol Version:** v1alpha1
**Date:** 2026-07-10

## System Under Test

| Field | Value |
|-------|-------|
| Implementation | labs/nemo-fastapi |
| Model | nvidia/parakeet-tdt-0.6b-v3 (0.6B params) |
| GPU | NVIDIA L4 24GB |
| Container | nvcr.io/nvidia/nemo:26.02 |
| CUDA | 12.8, Driver 570.86.15 |
| PyTorch | 2.6.0 |

## Scenario

REST batch transcription concurrency sweep c=1..512. 200 LibriSpeech test-clean files per level.

## Results

| Metric | Value |
|--------|-------|
| **WER** | 1.57% |
| **Peak RTFx** | 178x (c=256) |
| **Peak RPS** | 19.5 req/s (c=256) |
| **Failures** | 0 across all levels |
| **VRAM High-Water** | 8500 MB (37% of 24GB) |

## Concurrency Sweep

| Concurrency | RTFx | RPS | p50 | p99 | Failures |
|:-----------:|:----:|:---:|:---:|:---:|:--------:|
| 1 | 13x | 1.4 | 711ms | 1462ms | 0 |
| 8 | 70x | 7.7 | 959ms | 2172ms | 0 |
| 32 | 118x | 13.5 | 2207ms | 2608ms | 0 |
| 64 | 124x | 14.1 | 4418ms | 5282ms | 0 |
| 128 | 131x | 14.4 | 8646ms | 11864ms | 0 |
| 256 | 178x | 19.5 | 10229ms | 13986ms | 0 |
| 512 | 175x | 19.2 | 10615ms | 20784ms | 0 |

Throughput peaks at c=256 (178x realtime). c=512 shows slight saturation (175x) with p99 rising to 20.8s.

## Reproduction

```bash
# Server config
highperfasr serve --config labs/nemo-fastapi/configs/serving-batch.yaml

# Benchmark
python3 benchmarks/scripts/bench_batch.py \
  --server http://localhost:8000 \
  --concurrency 1,8,32,64,128,256,512
```

## Notes

- WER measured with Whisper EnglishTextNormalizer (lowercase, expand contractions, strip punctuation)
- Parakeet TDT 0.6B with VRAM-aware batch sizing
- Throughput plateaus beyond c=256 as GPU compute saturates; latency continues rising
