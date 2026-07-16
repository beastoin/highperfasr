# Publishing Benchmark Results

## Workflow

1. **Run benchmark** with quality gates:
   ```bash
   python3 benchmarks/scripts/bench_batch.py \
     --server http://localhost:8000 \
     --concurrency 1,8,16,32,64 \
     --sustained-rounds 4 \
     --output results/result.json
   ```

2. **Validate report** against schema:
   ```bash
   python3 benchmarks/scripts/validate_report.py results/result.json -v
   ```

3. **Check for regressions** against baselines:
   ```bash
   python3 benchmarks/scripts/check_regression.py \
     --report results/result.json \
     --registry benchmarks/baselines/registry.json
   ```

4. **Commit** the result.json and any raw artifacts.

5. **Update baseline registry** (only after review):
   Add a new entry to `benchmarks/baselines/registry.json` referencing the new result.

## Quality Gates

Quality gate thresholds are defined in `benchmarks/config/quality-gates.json`:

| Scenario | Max WER | Max Failure Rate | Min RTFx | Max p99 |
|----------|---------|-----------------|----------|---------|
| batch | 3.0% | 0% | 1.0x | -- |
| streaming-realtime | 5.0% | 0% | -- | 60s |
| combined | 3.0% | 0% | -- | -- |

## Schema

All committed `result.json` files must conform to `benchmarks/report-schema.json` (v1alpha2).

Required fields: `schema_version`, `report_id`, `sut`, `hardware`, `software`, `dataset`, `scenario`, `quality`, `performance`, `reliability`, `resources`, `reproduction`, `environment`.

## CI

The `benchmark-validation` workflow runs on every PR that touches `benchmarks/`:
- Unit tests (datasets + scripts)
- Schema validation of all committed reports
- Baseline regression check

## Statistical Rigor

For publishable results, use `--trials 3` (or more) to capture variance:

```bash
python3 benchmarks/scripts/bench_batch.py \
  --server http://localhost:8000 \
  --trials 3 \
  --output results/result.json
```

The `stats` module computes mean, stddev, and 95% confidence intervals using Student's t-distribution. Trial statistics are saved under the `trials` key in the report JSON.
