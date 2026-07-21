# Dataset Specification

Benchmark and tuning use different datasets. Never tune on the benchmark set.
Pick configs on tuning data, then run the frozen benchmark once for proof.

## Benchmark Dataset (frozen, canonical)

Fixed evaluation sets for proving published claims. Same files every run.

| Slice | Corpus | Registry key | Files | Duration | Purpose |
|-------|--------|-------------|-------|----------|---------|
| Clean read speech | LibriSpeech test-clean | `librispeech-test-clean` | 2,620 | 1–35s each, 5.4h total | Canonical WER comparison |
| Hard read speech | LibriSpeech test-other | `librispeech-test-other` | 2,939 | 1–35s each, 5.3h total | Harder speakers/acoustics |
| Long-form batch | Earnings-22 full | `earnings22-full` | 125 | 17–94 min each, 119h total | REST behavior on long audio |
| Long-form streaming | AMI eval (headset mix) | `ami-eval-ihm` | 20 meetings | ~11h total | Streaming on multi-speaker meetings |

The combined frozen benchmark manifest is loadable as `benchmark`. Individual
benchmark corpora are pinned to immutable source revisions and SHA-256 verified
when downloaded.

### Why frozen matters

The benchmark must be immune to config search. If batch size, chunk size, VAD,
concurrency, or duration limits are selected using the benchmark set, the final
number is no longer proof — it's overfit. Freeze:
- File list and transcripts
- Sample rate conversion (16 kHz mono PCM16)
- WebSocket chunk size and real-time pacing
- Concurrency schedule
- Text normalization (Whisper EnglishTextNormalizer)
- Scoring script version

### Sources

- LibriSpeech: [openslr.org/12](https://www.openslr.org/12)
- Earnings-22: [HuggingFace](https://huggingface.co/datasets/distil-whisper/earnings22) (125 files, 119h)
- AMI: [groups.inf.ed.ac.uk/ami](https://groups.inf.ed.ac.uk/ami/corpus/) (scenario-only eval, close-talk headset mix)

## Tuning Dataset (varied, adversarial)

Deliberately varied data for finding configs that survive real traffic shape.
A config optimized on 5s clips may OOM on 60s clips. Duration variance is the
primary axis that breaks serving configs.

| Bucket | Duration | Sources | Sample count | What it catches |
|--------|----------|---------|-------------|-----------------|
| Very short | 1–5s | LibriSpeech train/dev, Common Voice EN | 400 | GPU underfill, per-request overhead, bad micro-batching |
| Short | 5–15s | Common Voice EN, SPGISpeech | 600 | Accents, MP3 decode cost, production clip shape |
| Medium | 15–60s | GigaSpeech, TED-LIUM 3 | 400 | Padding waste, configs that collapse at 45s |
| Long | 1–5 min | TED-LIUM 3, AMI train/dev | 300 | Streaming state growth, queue starvation |
| Very long | 17–94 min | Earnings-21 | 44 | REST timeouts, memory growth, decoder stability |
| Noisy/far-field | 30s–5 min | CHiME-6, AMI far-field | 200 | False VAD, silence, unstable streaming |

### Tuning manifests

Tuning data is loaded through prepared manifests under the dataset cache. Each
bucket has a registry alias:

| Bucket | Registry key |
|--------|--------------|
| Very short | `tuning-very-short` |
| Short | `tuning-short` |
| Medium | `tuning-medium` |
| Long | `tuning-long` |
| Very long | `tuning-very-long` |
| Noisy/far-field | `tuning-noisy` |

Use `tuning` to load every prepared tuning bucket. Each bucket expects a
`manifest.json` in `$HPFASR_DATASET_DIR/<registry-key>/` with entries containing
`utt_id`, `wav_path`, `duration_s`, and optional `reference`.

### Tuning dataset rules

- Use for performance stress, NOT for WER-sensitive knob selection
- Earnings-21 is same-domain as Earnings-22 benchmark — use only for throughput/memory testing
- Content variety (accents, noise, domain) matters less than duration variety for serving configs
- Tuning manifest is NOT canonical — it can be modified between tuning sessions

### Tuning sources

- Common Voice EN v26: 2.58M clips, 3,781h, avg 5.27s ([Mozilla](https://commonvoice.mozilla.org))
- SPGISpeech: 5,000h financial audio, 5–15s clips ([HuggingFace](https://huggingface.co/datasets/kensho/spgispeech))
- GigaSpeech: 10,000h audiobooks/podcasts/YouTube ([GitHub](https://github.com/SpeechColab/GigaSpeech))
- TED-LIUM 3: 2,351 talks, 452h ([OpenSLR](https://www.openslr.org/51/))
- Earnings-21: 44 calls, 39h ([HuggingFace](https://huggingface.co/datasets/revdotcom/earnings21))
- CHiME-6: far-field dinner-party speech, multi-room ([chimechallenge.github.io](https://chimechallenge.github.io/chime6/))

## Workflow

```
1. Select tuning dataset buckets relevant to target workload
2. Sweep configs on tuning data (see tuning.md)
3. Pick best config
4. Run frozen benchmark once for proof
5. Publish benchmark results
```
