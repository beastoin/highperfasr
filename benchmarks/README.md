# Benchmarks

Published results, scripts, and tooling for highperfasr benchmarks.

## Scripts

All scripts are in [`scripts/`](scripts/). Dependencies: `aiohttp`, `jiwer`, `whisper-normalizer` (optional for WER).

| Script | Purpose |
|--------|---------|
| `bench_batch.py` | Batch concurrency sweep with WER — LibriSpeech auto-download, warmup, structured JSON output |
| `bench_stream.py` | Streaming WebSocket benchmark — realtime-paced audio, concurrency sweep |
| `bench_stream_longlive.py` | Sustained streaming soak test — persistent connections, VRAM monitoring, TTFB |
| `wer_utils.py` | WER computation with Whisper EnglishTextNormalizer |

### Quick start

```bash
# Batch benchmark (downloads LibriSpeech test-clean automatically)
python3 scripts/bench_batch.py --server http://localhost:8000

# Streaming soak at 512 concurrent streams for 10 minutes
python3 scripts/bench_stream_longlive.py \
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
