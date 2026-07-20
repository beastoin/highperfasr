# Benchmarks

Reproducible benchmark infrastructure for highperfasr. Every metric is backed by
a deterministic dataset, a fail-closed quality gate, and a regression baseline.

## Quick Start

All scripts are standalone — no NeMo imports, just a running server URL.

```bash
pip install aiohttp websockets jiwer whisper-normalizer soundfile

# Recommended: use the orchestrator (auto-detects mode, enforces correct order, runs gates)
python3 scripts/run_benchmark.py --server http://localhost:8000 --quick

# Full publishable run with 3 trials
python3 scripts/run_benchmark.py --server http://localhost:8000 --full --trials 3

# Individual scripts (the orchestrator calls these in the correct order)
python3 scripts/bench_batch.py --server http://localhost:8000
python3 scripts/bench_stream.py --server ws://localhost:8001  # compose: streaming on :8001
```

The orchestrator auto-detects the server mode (batch/streaming/both), resolves
the correct ports, runs batch before streaming (GPU contention causes streaming
failures if run simultaneously), evaluates quality gates, and prints a combined
pass/fail summary. Use `--quick` for fast validation (200 samples) and `--full`
for publishable results (all samples).

**Port mapping:** In Docker Compose, batch is on `:8000` and streaming on `:8001`
(separate services). In both-mode (single server), both share `:8000`. The
orchestrator auto-probes `:8001` when it detects batch-only on `:8000`.

## Scripts

| Script | Purpose |
|--------|---------|
| `run_benchmark.py` | **Orchestrator** — auto-detects mode, enforces batch-first ordering, runs gates |
| `preflight.py` | Server auto-detection, duration estimates, unbuffered output |
| `bench_batch.py` | Batch REST benchmark — concurrency sweep, sustained load, WER |
| `bench_stream.py` | Streaming WebSocket benchmark — concurrency sweep, TTFB, WER |
| `bench_stream_soak.py` | Sustained streaming soak — persistent connections, VRAM tracking |
| `bench_combined.py` | Simultaneous batch + streaming load |
| `bench_batch_by_duration.py` | Duration-stratified batch sweep (5s–120s brackets) |
| `profile_gpu.py` | VRAM profiler — measures per-batch memory across duration brackets |
| `tune_gpu.py` | Automated GPU config tuning — searches batch size × duration parameters |
| `eval_wer_detailed.py` | Per-utterance WER with S/I/D error breakdown |
| `gates.py` | Quality gate evaluation (fail-closed on missing data) |
| `check_regression.py` | Baseline regression detection with per-metric thresholds |
| `validate_report.py` | JSON schema validation for benchmark reports |
| `stats.py` | Statistical utilities — mean, stddev, 95% CI (Student's t) |
| `wer_utils.py` | WER computation with Whisper EnglishTextNormalizer |

## Published Results

| Report | GPU | Workload | WER | Throughput |
|--------|-----|----------|-----|------------|
| [L4 streaming 512](results/2026-l4-nemo-512-streams/) | L4 24GB | 512 WebSocket streams | 3.21% | 297 sess/min, 38.69x RTFx |
| [L4 batch](results/2026-l4-nemo-batch/) | L4 24GB | REST c=1..512 | 1.57% | 19.5 RPS, 178x RTFx |
| [T4 batch](results/2026-t4-nemo-batch/) | T4 16GB | REST c=1..32 | 1.86% | — |
| [L4 duration sweep](results/2026-l4-batch-by-duration/) | L4 24GB | 5s–120s audio brackets | — | Duration-stratified |

## Metrics & Datasets

- [**Metric definitions**](docs/metrics.md) — formal benchmark metrics (batch/streaming), WER delta gate, proof lines
- [**Dataset specification**](docs/datasets.md) — benchmark vs tuning dataset separation, sources, rules
- [**Tuning methodology**](docs/tuning.md) — knobs-vs-gauges, tuning workflow, common pitfalls

## Quality Gates

Fail-closed thresholds in [`config/quality-gates.json`](config/quality-gates.json).
Every gate returns `passed: false` when its metric data is missing — no silent passes.

| Scenario | Max WER | WER Delta | Max Failure Rate | Min RTFx | RT Compliance | Stream Lag p95 | Sustained | VRAM Growth |
|----------|---------|-----------|-----------------|----------|---------------|----------------|-----------|-------------|
| batch | 2.5% | ≤ max(0.3pp, 5% rel) | 0% | 1.0x | — | — | — | < 100 MB |
| streaming-realtime | 4.0% | ≤ max(0.3pp, 5% rel) | 0% | — | ≥ 95% | ≤ 5000 ms | ≥ 600s | < 100 MB |
| combined | 3.0% | — | 0% | — | — | — | — | — |

```bash
python3 scripts/gates.py --report results/2026-l4-nemo-batch/result.json --scenario batch
```

## Baseline Regression Detection

Baselines in [`baselines/registry.json`](baselines/registry.json) define hardware/model/scenario
triples with per-metric regression thresholds. Unknown baseline IDs fail explicitly.

```bash
# Compare a new report against a specific baseline
python3 scripts/check_regression.py --report result.json --baseline-id l4-nemo-batch-2026

# Validate all committed baselines exist and are parseable
python3 scripts/check_regression.py --all
```

## Dataset Infrastructure

The [`datasets/`](datasets/) package provides deterministic, cached dataset loading
with SHA256 integrity checks:

```python
from benchmarks.datasets.registry import load_dataset
manifest = load_dataset("librispeech-test-clean", cache_dir="/tmp/cache")
```

Datasets are registered in `datasets/registry.py` with download URLs, checksums,
and extraction logic. Round-robin loading handles concurrency > dataset size.

## CI Pipeline

[`benchmark-validation.yml`](../.github/workflows/benchmark-validation.yml) runs
on every PR touching `benchmarks/`:

1. Unit tests (`pytest benchmarks/` — 43 tests)
2. Schema validation of all committed reports
3. Baseline regression checks
4. Quality gate evaluation on current L4 batch and streaming proof reports

## Report Schema

[`report-schema.json`](report-schema.json) — v1alpha2 JSON schema for all reports.
Validate with:

```bash
python3 scripts/validate_report.py --report result.json
```

## Publishing

See [`docs/publishing.md`](docs/publishing.md) for the full workflow:
run → validate → regression-check → gate → commit → update registry.
