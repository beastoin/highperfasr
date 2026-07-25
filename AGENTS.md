# HighPerfASR — Agent Instructions

## Mission

HighPerfASR is "vLLM for ASR" — maximize serving performance of existing open-source NVIDIA ASR models without modifying model quality. Preserve WER, maximize throughput.

## Target Customer

B2B companies self-hosting ASR instead of paying per-minute API pricing. Our value: one L4 GPU serving 512 concurrent streams with published WER — reproducible, evidence-based.

## Agent Roles

### zen (lead)
- Primary workdir: `~/ossasr-zen/highperfasr`
- Owns: server code, framework patches, benchmark methodology, GHCR images
- Coordinates work across agents on this project

### aki (dev)
- Primary workdir: `~/ossasr-aki`
- Focus: benchmark runs, GPU tuning, quality gates, HTML dashboards
- Coordinate with zen via bridge messaging before architectural changes

## Coordination Rules

1. **Before changing benchmark methodology** (WER calculation, gate thresholds, sweep parameters) — discuss with zen first. These affect reproducibility of published results.
2. **Before changing server code** (`labs/nemo-fastapi/src/highperfasr/`) — coordinate with zen. Thread-safety and CUDA memory management are critical.
3. **HTML/dashboard changes** can proceed independently but cross-check hardcoded data against source JSON.
4. **New benchmark runs** can proceed independently — follow the workflow in CLAUDE.md.

## PR Workflow

Use the `highperfasr-pr-workflow` skill (10 checkpoints). Key steps:
1. Branch from main (`fix/` or `feat/`)
2. Implement + test (`python3 -m pytest benchmarks/scripts/tests/ -q`)
3. Create PR via REST API (not `gh pr edit` — token scope limitation)
4. Codex review cycle (max 8 iterations)
5. Benchmark gate (if serving/benchmark code changed)
6. Present PR link to manager — **never merge yourself**

## Communication

- Report results proactively — don't wait to be asked
- Share partial results as they come (iterative reporting)
- No real metrics on GitHub — percentages only
- Manager prefers root cause analysis over speculation — run the test first
