# HighPerfASR — GitHub Copilot Instructions

See [AGENTS.md](../AGENTS.md) for project rules, architecture, and conventions.

Key constraints Copilot must follow:
- NGC base `nvcr.io/nvidia/nemo:26.02` only — DO NOT suggest PyTorch 2.12+ or CUDA 13.x
- All CUDA model loading on GPU worker thread only — no cross-thread tensor ownership
- WER computation only via repo bench scripts with Whisper normalization — never inline
- `gc.disable()` at module level, `gc.collect(0)` per batch on GPU thread
- uvicorn WebSocket: `ws_ping_interval=None, ws_ping_timeout=None`
