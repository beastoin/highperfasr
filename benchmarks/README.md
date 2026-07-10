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
