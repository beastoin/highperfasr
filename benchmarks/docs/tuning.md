# Tuning Methodology

Server config tuning for highperfasr. The goal: find the config that maximizes
throughput and concurrency while preserving model WER.

## Principle

Tuning is a knobs-and-gauges problem. Knobs are server config parameters.
Gauges are the metrics that tell you whether turning a knob helped or hurt.

**Benchmark metrics** answer: "Did it work?" (proof)
**Tuning metrics** answer: "Why or why not?" (guidance)

## Batch Tuning Gauges

Knobs: `batch_size`, `num_workers`, `max_file_duration_sec`, CUDA graphs,
`torch.compile`, attention mode.

| Gauge | Unit | What it guides |
|-------|------|---------------|
| GPU SM utilization | % | Low → CPU bottleneck, small batches, sync overhead |
| GPU memory headroom | GB, % | batch_size, max audio length, CUDA graph capture size |
| Memory allocation churn | allocs/sec | Dynamic shapes, poor buffer reuse, graph breaks |
| GPU kernel time | ms/batch | torch.compile, CUDA graphs, attention mode effects |
| GPU idle gaps | % timeline | Pipeline overlap, worker count, H2D transfer |
| Batch fill ratio | % (by audio seconds) | Whether configured batch_size is actually utilized |
| Batch wait time | ms | Dynamic batching timeout — too low hurts fill, too high hurts latency |
| Queue depth | requests | Overload, under-provisioned workers |
| Queue wait p50/p95 | ms | Separates serving delay from model compute |
| Preprocess time | ms/audio sec | Audio decode, resampling, feature extraction, CPU workers |
| H2D/D2H transfer time | ms, % | Pinned memory, batch collation, device preprocessing |
| Per-stage latency | ms | Identifies which stage changed after a knob turn |
| CPU utilization | % per worker | High CPU + low GPU → host bottleneck |
| Error/OOM count | count | Hard guardrail for aggressive settings |

### Batch tuning workflow

```
1. Baseline: RTFx at c=1 with default config
2. Sweep batch_size: [4, 8, 16, 32, 64] on tuning dataset (mixed durations)
3. For each: record RTFx, WER smoke, GPU util, VRAM, batch fill
4. Pick batch_size with best RTFx while VRAM < 90% capacity
5. Sweep torch.compile / CUDA graphs on/off
6. Sweep attention mode if model supports multiple
7. Verify final config: benchmark dataset at c=1 through max concurrency
```

## Streaming Tuning Gauges

Knobs: `num_slots`, `max_concurrent_streams`, `chunk_size_in_secs`,
`max_stream_drain`, buffer sizes, WebSocket settings.

| Gauge | Unit | What it guides |
|-------|------|---------------|
| Stream lag p50/p95/p99 | ms behind RT | **Most important** — is server keeping up? |
| Chunk processing time | ms/chunk | Must stay below chunk duration with headroom |
| Chunk queue depth | chunks | Backlog before GPU execution |
| Chunk queue wait p50/p95 | ms | Scheduling vs compute bottleneck |
| Active slot occupancy | % of num_slots | num_slots, stream scheduler capacity |
| Scheduler fairness (Jain's index) | 0–1 | Detects stream starvation under aggregate good numbers |
| Per-stream state memory | MB/stream | Predicts max concurrency, catches leaks |
| Real-time factor per stream | ratio | Values > 1.0 mean falling behind |
| EOS-to-final latency | ms | Endpointing, finalization, chunk size, decoder scheduling |
| Partial result latency | ms | Chunk size, emission cadence, buffering |
| GPU SM utilization | % | Low + high lag → CPU bottleneck; high + lag → GPU saturated |
| GPU memory headroom | GB, % | max_concurrent_streams, cache buffers, graph sizes |
| GPU idle gaps | % timeline | Chunk batching, scheduling cadence, CUDA graphs |
| CPU utilization | % per worker | WebSocket workers, audio ingestion, feature extraction |
| Backpressure events | count/sec, time blocked | Buffer sizes, flow control |
| Buffer occupancy | ms audio, % full | Input/output buffer tuning |
| WebSocket disconnect rate | % sessions | Overload, backpressure, timeout guardrail |

### Streaming tuning workflow

```
1. Baseline: WER + stream lag at c=1 with default config
2. Binary search max concurrent streams: c=32, 64, 128, 256, 512
3. For each level (sustained ≥ 90s):
   - Record: WER, stream lag, RT compliance, failures, VRAM
   - If lag p99 > chunk_duration or failures > 0: level is unstable
4. At max stable level:
   - Sweep num_slots (must match or exceed max_concurrent_streams)
   - Sweep max_stream_drain: [4, 8, 16] (>16 causes VRAM explosion)
   - Sweep chunk_size_in_secs if model supports variable
5. Verify final config: benchmark dataset, 600s sustained
```

## Common Pitfalls

- **Tuning on benchmark data**: Invalidates results. Always use tuning dataset.
- **Duration blindness**: A config that works for 5s clips may OOM on 60s. Always
  include mixed durations in tuning sweeps.
- **VRAM-optimal ≠ throughput-optimal**: Leaving 20% VRAM headroom often gives
  better throughput than filling to 95% (avoids allocator churn).
- **max_stream_drain > 16**: Causes VRAM explosion (5.3 → 11.3 GB in 3 min) on
  streaming models. The GPU processes too many distinct decoder states per batch.
- **num_slots < max_concurrent_streams**: Pipeline silently caps connections.
  Excess streams get "No free slots available" and disconnect.
- **torch.compile benefit at wrong concurrency**: Compile helps at c=16–32
  (batching efficiency), not c=1 (latency-bound) or c=64+ (GPU-saturated).
