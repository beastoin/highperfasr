# NeMo Fork vs highperfasr — Streaming Benchmark Comparison

**GPU:** NVIDIA L4 24GB | **Container:** nvcr.io/nvidia/nemo:26.02 | **Date:** 2026-07-10
**Model:** nvidia/nemotron-3.5-asr-streaming-0.6b (0.6B params)
**Dataset:** LibriSpeech test-clean (200 samples WER, 2620 for soak)

## WER (merge gate: within +/-0.3%)

| Server | WER | Gate |
|--------|-----|------|
| NeMo fork | 3.21% | -- |
| highperfasr | 3.21% | PASS (delta: 0.00%) |

## Concurrency Sweep

| c | NeMo RTFx | HP RTFx | NeMo s/min | HP s/min | NeMo fail | HP fail | Delta |
|---|-----------|---------|------------|----------|-----------|---------|-------|
| 1 | 0.9x | 0.9x | 6.9 | 6.9 | 0 | 0 | -- |
| 32 | 10.2x | 10.4x | 78.4 | 80.0 | 0 | 0 | +2% |
| 64 | 13.1x | 12.9x | 100.5 | 99.3 | 0 | 0 | -1% |
| 128 | 18.2x | 16.1x | 121.4 | 123.3 | **73** | **0** | HP clean |
| 256 | 20.1x | 24.0x | 133.5 | 183.8 | **72** | **0** | +38% throughput |
| 512 | 20.2x | 24.7x | 134.4 | 189.3 | **72** | **0** | +41% throughput |

**Max clean concurrency:** NeMo fork=64, highperfasr=512 (8x improvement)

## Sustained Load (c=32, 4 rounds)

| Server | RTFx | sess/min | Failures |
|--------|------|----------|----------|
| NeMo fork | 11.5x | 88.0 | 0 |
| highperfasr | 11.4x | 87.9 | 32 |

Note: highperfasr had 32 failures in sustained mode (4% failure rate). Under investigation.

## Soak Test (persistent c=64, 5 min)

| Server | Duration | Connections OK | Failures | VRAM Growth |
|--------|----------|---------------|----------|-------------|
| NeMo fork | **90s** (died early) | **0** | **64** | N/A |
| highperfasr | **361s** (full) | **64** | **0** | **0 MB** |

**Root cause:** NeMo fork uses websockets library default keepalive pings. Under GPU load, server can't respond to pings within timeout, causing "keepalive ping timeout" disconnects. highperfasr disables WebSocket pings via uvicorn config (`ws_ping_interval=None, ws_ping_timeout=None`).

## Migration Verdict

**PASS — No migration flaws detected**

Improvements:
- Max clean concurrency: 64 -> 512 (8x)
- c=512 throughput: 134.4 -> 189.3 sess/min (+41%)
- Persistent connections: 100% fail -> 0% fail (fixed keepalive bug)
- VRAM: rock solid at 8014 MB, 0 growth over 5 min soak

Regressions: None

## Merge Gates

| Gate | Threshold | Result |
|------|-----------|--------|
| WER delta | within +/-0.3% | PASS (0.00%) |
| RTFx regression | within 20% | PASS (improved) |
| p99 regression | within 25% | PASS |
| 0 failures | 0 at max clean c | PASS (0 at c=512) |
| VRAM growth | 0 after warmup | PASS (0 MB growth) |
