# Benchmark Metrics

Formal metric definitions for highperfasr benchmark reports. Every metric directly
proves the mission claim: **preserve model WER, maximize throughput and concurrency**.

Metrics are separated into **benchmark** (proof for published claims) and **tuning**
(guidance during config optimization). See [tuning.md](tuning.md) for tuning metrics.

## Batch Metrics (REST `/v1/transcriptions`)

| Metric | Key | Unit | Gate | Purpose |
|--------|-----|------|------|---------|
| Corpus WER | `wer_pct` | % | ≤ 2.5% | Quality preserved |
| WER delta vs reference | `wer_delta_pp` | pp | ≤ max(0.3, 0.05 × ref) | Server didn't degrade quality vs offline model |
| RTFx | `rtfx` | audio sec / wall sec | ≥ 1.0 | Primary throughput proof |
| RTF | `rtf` | wall sec / audio sec | — | Industry-standard inverse of RTFx |
| Requests per second | `rps` | req/s | — | API-level throughput |
| Max stable concurrency | `max_concurrency` | count | — | Saturation point where peak RTFx was achieved |
| Latency p50/p95/p99 | `p50_ms`, `p95_ms`, `p99_ms` | ms | — | Throughput wasn't achieved by making jobs slow |
| Failure rate | `failure_rate` | % | 0% | Throughput was stable |
| VRAM peak | `vram_peak_gb` | GB | ≤ GPU capacity | Config fits on target GPU |
| VRAM growth | `vram_growth_mb` | MB | < 100 MB | No memory leaks over sustained run |

### Batch proof line

> WER 1.57% (+0.00pp vs ref), 178x RTFx at c=64, 19.5 RPS, 0% failures, 8.6 GB VRAM on L4

## Streaming Metrics (WebSocket `/v1/stream`)

| Metric | Key | Unit | Gate | Purpose |
|--------|-----|------|------|---------|
| Corpus WER | `wer_pct` | % | ≤ 4.0% | Quality preserved |
| WER delta vs reference | `wer_delta_pp` | pp | ≤ max(0.3, 0.05 × ref) | Server didn't degrade quality vs offline model |
| Max concurrent streams | `max_streams` | count | — | Primary concurrency proof |
| Sessions per minute | `sessions_min` | sess/min | — | Throughput for short-session workloads |
| RTFx (aggregate) | `rtfx` | audio sec / wall sec | — | Aggregate streaming throughput |
| Realtime compliance | `rt_compliance_pct` | % | ≥ 95% | Streams keep up with realtime audio |
| Stream lag p95/p99 | `lag_p95_ms`, `lag_p99_ms` | ms | p95 ≤ 5000 | High concurrency is usable, not buffering |
| TTFB p50/p95 | `ttfb_p50_ms`, `ttfb_p95_ms` | ms | — | First partial responsiveness |
| EOS-to-final latency p50/p95 | `eos_final_p50_ms`, `eos_final_p95_ms` | ms | — | Final transcript speed after speech ends |
| Failure/disconnect rate | `failure_rate` | % | 0% | Load is sustainable |
| Sustained test duration | `sustained_duration_s` | s | ≥ 600 | Stability, not just a burst |
| VRAM peak | `vram_peak_gb` | GB | ≤ GPU capacity | Config fits on target GPU |
| VRAM growth | `vram_growth_mb` | MB | < 100 MB | No memory leaks |

### Streaming proof line

> WER 3.21% (+0.00pp vs ref), 512 streams, 297 sess/min, 100% RT compliance, 0% failures on L4

## WER Delta Gate

The WER delta gate proves the server preserves model quality vs running the model
directly (offline reference). Formula:

```
candidate_wer_pct <= reference_wer_pct + max(0.3, 0.05 * reference_wer_pct)
```

Requirements:
- Same text normalizer (Whisper EnglishTextNormalizer)
- Same evaluation manifest (corpus, file list, sample rate)
- Same model version (HuggingFace ID + revision)
- Corpus-level WER (not per-utterance average)
- Fail-closed: missing reference WER → gate fails

### Reference WER values

| Model | Dataset | Reference WER | Source |
|-------|---------|---------------|--------|
| nvidia/parakeet-tdt-0.6b-v3 | LS test-clean | 1.57% | highperfasr L4 batch c=1 |
| nvidia/nemotron-3.5-asr-streaming-0.6b | LS test-clean | 3.21% | highperfasr L4 streaming c=1 |

## Max Stable Concurrency Definition

A concurrency level is "stable" when ALL of these hold for ≥ 600 seconds:
- Zero failures
- WER gate passes
- Failure rate = 0%
- VRAM within GPU capacity
- (Streaming) Realtime compliance ≥ 95%
- (Streaming) Stream lag p95 ≤ 5000 ms

## WER Measurement Protocol

- Measure WER at c=1 (deterministic baseline) AND at max stable concurrency (under load)
- Both must pass the WER delta gate
- Use repo benchmark scripts only — no ad-hoc WER code
