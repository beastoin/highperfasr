# Deploy HighPerfASR on Google Cloud

## Overview

This tutorial deploys a GPU-accelerated ASR server on your GCP project. You will
have a running server in under 10 minutes, ready for benchmarking.

**Estimated time:** 10 minutes
**Cost:** L4 GPU usage in your GCP project; auto-shuts down after 4 hours

---

## Authenticate

Cloud Shell opens this repository in a temporary environment. Authenticate
explicitly before creating GCE resources:

```bash
gcloud auth login --no-launch-browser
```

Verify the active account:

```bash
gcloud auth list --filter=status:ACTIVE
```

## Set your project

<walkthrough-project-setup></walkthrough-project-setup>

Set the project in gcloud:

```bash
gcloud config set project <walkthrough-project-id/>
```

## Enable Compute Engine

```bash
gcloud services enable compute.googleapis.com
```

## Deploy the server

Launch a streaming ASR server on an L4 GPU:

```bash
./examples/launch-gce.sh --mode stream
```

The script creates a GCE VM with an NVIDIA L4 GPU, pulls the HighPerfASR
container image, and waits for the server to become healthy.

For batch transcription instead:

```bash
./examples/launch-gce.sh --mode batch
```

## Run a benchmark

Once the server is healthy, the script prints benchmark commands. Run those
commands to measure throughput, latency, and WER on LibriSpeech test-clean.

Use the printed benchmark commands for canonical WER and quality-gate results.
The static dashboard is for same-origin or HTTPS/proxied deployments; browsers
block the public GitHub Pages dashboard from calling an HTTP VM endpoint.

## Clean up

Delete the evaluation VM and firewall rule:

```bash
./examples/launch-gce.sh teardown
```

## Next steps

- [Benchmark scripts](../benchmarks/scripts/) — reproducible evaluation methodology
- [Deployment recipes](../recipes/) — Kubernetes recipes for GCP, AWS, Azure
- [Protocol spec](../spec/protocol.md) — REST and WebSocket API reference
