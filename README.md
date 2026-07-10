# highperfasr

A language-neutral protocol, conformance suite, and benchmark harness for production ASR serving.

highperfasr defines **how** to serve speech recognition — not which framework or language to use. Any server that implements the protocol can be benchmarked, compared, and certified.

## What highperfasr IS

- **Protocol spec** — REST batch + WebSocket streaming APIs (OpenAPI 3.1 + AsyncAPI 3.1)
- **Benchmark suite** — MLPerf-inspired scenarios with reproducible reports
- **Conformance tests** — black-box verification against any server URL
- **Published reports** — hardware-specific, dataset-pinned, fully reproducible results

## What highperfasr is NOT

- A Python package or framework
- A NeMo abstraction layer
- A serving framework (use Triton, vLLM, FastAPI, Go, Rust — whatever fits)

## First Published Result

```
HighPerfASR OpenASR-Streaming v1alpha1
SUT:         labs/nemo-fastapi @ eb79ddf
Hardware:    NVIDIA L4 24GB
Dataset:     LibriSpeech test-clean (2620 files, SHA256 pinned)
Scenario:    512 real-time streams, 10 min soak
Quality:     3.21% WER (Whisper normalization)
Throughput:  297 sessions/min
Reliability: 512/512 completed, 0 failures
VRAM:        8672 MB high-water (38% of 24GB)
```

See [reports/2026-l4-nemo-512-streams/](reports/2026-l4-nemo-512-streams/) for full report.

## Protocol (v1alpha1)

### Batch — REST API

```bash
# Transcribe a file
curl -F "file=@audio.wav" http://localhost:8000/v1/transcriptions

# Health check
curl http://localhost:8000/health
```

### Streaming — WebSocket

```
ws://localhost:8000/v1/stream

1. Server sends: {"stream_id": "...", "status": "opened"}
2. Client sends: raw PCM16 audio (binary frames)
3. Server sends: {"partial_transcript": "...", "final_transcript": "...", "is_final": false}
4. Client sends: {"action": "close"}
5. Server sends: final transcript + closes
```

Full spec: [spec/protocol.md](spec/protocol.md)

## Repository Structure

```
highperfasr/
  spec/                    # The standard (protocol + API specs)
    protocol.md            # Normative RFC-style specification
    openapi.yaml           # REST batch API (OpenAPI 3.1)
    asyncapi.yaml          # WebSocket streaming API (AsyncAPI 3.1)
    schemas/               # Shared JSON Schema definitions

  reports/                 # Published benchmark results
    report-schema.json     # Machine-readable report format
    2026-l4-nemo-512-streams/

  benchmarks/              # Benchmark suite definitions
    openasr/               # OpenASR industrial benchmark scenarios

  conformance/             # Black-box protocol conformance tests

  labs/                    # Implementation examples + experiments
    nemo-fastapi/          # Production NeMo server (Python/FastAPI)
    nemo-patches/          # NeMo framework patches with evidence
```

## Implementing the Protocol

Any server in any language that implements the endpoints defined in `spec/` is a valid highperfasr server. The conformance suite verifies compliance, and the benchmark suite measures performance.

Example implementations:
- **Python/NeMo** — `labs/nemo-fastapi/` (production-tested, 512 concurrent streams on L4)
- **Faster-Whisper** — planned
- **Go** — planned
- **Rust** — planned

## License

Apache-2.0
