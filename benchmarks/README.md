# Benchmarks

Published results and scripts for highperfasr.

## Scripts

All scripts are in [`scripts/`](scripts/). They are standalone — no NeMo imports,
just a server URL.

Dependencies: `pip install aiohttp websockets jiwer whisper-normalizer soundfile`

| Script | Purpose |
|--------|---------|
| `bench_batch.py` | Batch REST benchmark — concurrency sweep, sustained load, WER |
| `bench_stream.py` | Streaming WebSocket benchmark — concurrency sweep, WER |
| `bench_stream_soak.py` | Sustained streaming soak — persistent connections, VRAM tracking, TTFB |
| `bench_combined.py` | Simultaneous batch + streaming load, chaos testing |
| `eval_wer_detailed.py` | Per-utterance WER with S/I/D error breakdown — verification artifact |
| `wer_utils.py` | WER computation with Whisper EnglishTextNormalizer |

### Quick start

```bash
# Batch concurrency sweep (auto-downloads LibriSpeech test-clean)
python3 scripts/bench_batch.py --server http://localhost:8000

# Streaming concurrency sweep
python3 scripts/bench_stream.py --server ws://localhost:8001

# Sustained streaming soak at 512 streams for 10 minutes
python3 scripts/bench_stream_soak.py \
  --server ws://localhost:8001 \
  --concurrency 512 \
  --durations 600 \
  --persistent
```

## Published Results

- [`results/2026-l4-nemo-512-streams/`](results/2026-l4-nemo-512-streams/) — Streaming: L4 24GB, 512 streams, 3.21% WER, 297 sess/min
- [`results/2026-l4-nemo-batch/`](results/2026-l4-nemo-batch/) — Batch: L4 24GB, 178x RTFx, 1.57% WER, 19.5 RPS

## Report Schema

[`report-schema.json`](report-schema.json) — machine-readable format for all benchmark reports.

## Quality Gates

Automated pass/fail thresholds in [`config/quality-gates.json`](config/quality-gates.json):

| Scenario | Max WER | Max Failure Rate | Min RTFx | Max p99 |
|----------|---------|-----------------|----------|---------|
| batch | 3.0% | 0% | 1.0x | — |
| streaming-realtime | 5.0% | 0% | — | 60s |
| combined | 3.0% | 0% | — | — |

Evaluate gates programmatically:

```bash
python3 scripts/gates.py --report results/result.json --scenario batch
```

## Baseline Regression Detection

Baselines are registered in [`baselines/registry.json`](baselines/registry.json). Each defines a hardware/model/scenario triple with regression thresholds.

```bash
# Check one report against its matching baseline
python3 scripts/check_regression.py --report results/result.json --baseline-id l4-nemo-batch-2026

# Check all committed reports
python3 scripts/check_regression.py --all --registry baselines/registry.json
```

## Statistical Rigor

The [`scripts/stats.py`](scripts/stats.py) module provides mean, stddev, and 95% CI (Student's t-distribution for n≤30). Use `--trials 3` for publishable results.

## CI

The [`benchmark-validation`](.github/workflows/benchmark-validation.yml) workflow runs on every PR touching `benchmarks/`:

1. Unit tests (`pytest benchmarks/`)
2. Schema validation of all committed reports
3. Baseline regression checks

## Publishing

See [`docs/publishing.md`](docs/publishing.md) for the full workflow: run → validate → regression-check → commit → update registry.
