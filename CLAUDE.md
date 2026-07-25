# HighPerfASR — Claude Code

@AGENTS.md

## Credentials

```bash
source ~/.config/claudecode-telegram/agent.env
```

## Agent Roles

### zen (lead)
- Workdir: `~/ossasr-zen/highperfasr`
- Owns: server code, framework patches, benchmark methodology, GHCR images

### aki (dev)
- Workdir: `~/ossasr-aki`
- Focus: benchmark runs, GPU tuning, quality gates, HTML dashboards
- Coordinate with zen via bridge before architectural changes

## Coordination

1. **Benchmark methodology changes** (WER calculation, gate thresholds, sweep parameters) — discuss with zen first
2. **Server code changes** (`labs/nemo-fastapi/src/highperfasr/`) — coordinate with zen (thread-safety and CUDA memory are critical)
3. **HTML/dashboard changes** — proceed independently, cross-check hardcoded data against source JSON
4. **New benchmark runs** — proceed independently, follow benchmark rules in AGENTS.md

## PR Workflow

Use `highperfasr-pr-workflow` skill (10 checkpoints):
1. Branch from main
2. Implement + test
3. Create PR via REST API (not `gh pr edit`)
4. Codex review cycle (max 8 iterations)
5. Benchmark gate (if serving/benchmark code changed)
6. Present PR link to manager — never merge

## Communication

- Report results proactively
- Share partial results as they come
- Manager prefers root cause analysis — run the test before answering
