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

## Check GPU quota

The deploy script checks this automatically, but if you have never used GPUs in
this project you may need to request quota first:

```bash
gcloud compute regions describe us-central1 \
  --format='table(quotas.filter("metric:NVIDIA_L4_GPUS").limit,quotas.filter("metric:NVIDIA_L4_GPUS").usage)'
```

If the limit is 0, request L4 GPU quota at
[IAM & Admin > Quotas](https://console.cloud.google.com/iam-admin/quotas).
Quota approval is typically instant for small requests (1 GPU).

## Deploy the server

Launch a streaming ASR server on an L4 GPU:

```bash
./examples/launch-gce.sh --mode stream
```

The script creates a GCE VM with an NVIDIA L4 GPU, pulls the HighPerfASR
container image, and waits for the server to become healthy. You will see
progress updates as the VM installs drivers, pulls the image, and loads the
model.

For batch transcription instead:

```bash
./examples/launch-gce.sh --mode batch
```

For a T4 GPU (lower cost, lower throughput):

```bash
./examples/launch-gce.sh --gpu t4
```

## Run a benchmark

Once the server is healthy, the script prints ready-to-use benchmark commands
with the server URL pre-filled. Copy and run them directly.

For streaming:
```bash
python3 benchmarks/scripts/bench_stream.py \
  --server ws://SERVER_IP:8001 \
  --endpoint /v1/stream \
  --concurrency 1,16,32 \
  --image-tag 0.3.0 \
  --output /tmp/bench_stream.json
```

For batch:
```bash
python3 benchmarks/scripts/bench_batch.py \
  --server http://SERVER_IP:8000 \
  --concurrency 1,8,16,32,64 \
  --image-tag 0.3.0 \
  --output /tmp/bench_batch.json
```

Replace `SERVER_IP` with the IP printed by the deploy script.

## Clean up

Delete the evaluation VM and firewall rule:

```bash
./examples/launch-gce.sh teardown
```

The VM also auto-shuts down after 4 hours (configurable with `--ttl`).

## Troubleshooting

**GPU quota is 0:** Request quota at
[IAM & Admin > Quotas](https://console.cloud.google.com/iam-admin/quotas).
Filter for `NVIDIA_L4_GPUS` in `us-central1`.

**Server not healthy after 10 minutes:** The script prints a `docker logs`
command and a keep-waiting one-liner. Common causes: large model download on
first run (~2 GB), or GPU driver installation taking longer than usual.

**Billing not enabled:** GPU VMs require an active billing account.
Link one at [Billing](https://console.cloud.google.com/billing/linkedaccount).

**Zone has no GPU capacity:** Try a different zone with `--zone us-east1-b` or
`--zone us-west1-b`.

## Next steps

- [Benchmark scripts](../benchmarks/scripts/) — reproducible evaluation methodology
- [Deployment recipes](../recipes/) — Kubernetes recipes for GCP, AWS, Azure
- [Protocol spec](../spec/protocol.md) — REST and WebSocket API reference
