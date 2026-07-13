# GCP GKE — NVIDIA T4

Deploy highperfasr on GKE with T4 GPU nodes.

## Prerequisites

- GKE cluster with a T4 GPU node pool (`nvidia-tesla-t4` accelerator type)
- Two schedulable T4 GPUs for the full batch + streaming overlay
- NVIDIA GPU device plugin installed (included by default on GKE)
- `kubectl` configured for your cluster

```bash
gcloud container clusters get-credentials CLUSTER --region REGION --project PROJECT
kubectl get nodes -l cloud.google.com/gke-accelerator=nvidia-tesla-t4 --no-headers
```

## Deploy

```bash
kubectl apply -k recipes/gcp-t4
```

The overlay starts both services. On a one-GPU cluster, keep only one workload
running:

```bash
kubectl scale deployment/highperfasr-batch --replicas=0   # streaming only
kubectl scale deployment/highperfasr-stream --replicas=0  # batch only
```

## Verify

```bash
kubectl get pods -l app=highperfasr-stream
kubectl port-forward svc/highperfasr-stream 8001:8000
curl http://localhost:8001/health
```

## Instance

| Field | Value |
|-------|-------|
| Instance | n1-standard-4 |
| GPU | 1x NVIDIA T4 (16 GB) |
| Nodes for full overlay | 2 |
| vCPU | 4 |
| RAM | 15 GB |
| On-demand | ~$0.35/hr |
| Spot | ~$0.11/hr |
| Storage class | standard-rwo |

## Streaming Performance

T4 handles up to 256 concurrent streams with zero failures:

| Concurrency | RTFx | sess/min | p50 | p99 | Failures |
|-------------|------|----------|-----|-----|----------|
| 1 | 0.89 | 6.9 | 7.3s | 29.5s | 0 |
| 32 | 8.95 | 68.7 | 22.7s | 72.4s | 0 |
| 64 | 11.26 | 86.4 | 39.4s | 64.6s | 0 |
| 128 | 16.84 | 129.3 | 39.4s | 63.9s | 0 |
| 256 | 22.17 | 170.2 | 45.1s | 65.2s | 0 |

Sustained load (c=32, 4 rounds): 9.41x RTFx, 72.2 sess/min, 0 failures.

WER: 3.19% (LibriSpeech test-clean, Whisper English normalization).

## Notes

- T4 has 16 GB VRAM (vs 24 GB on L4) — lower peak throughput but viable for both batch and streaming
- `torch.compile` and CUDA graphs crash on T4 (Turing architecture, compute capability 7.5) — the recipe's ConfigMap override disables both
- T4 is the most cost-effective GPU on GCP for batch inference workloads
- Spot/preemptible T4 instances available at ~70% discount — suitable for batch workloads
- First pod startup downloads models from HuggingFace (~2 GB each, 2-3 min)
- Subsequent starts use the PVC-cached model
- No HPA is included because the shared ReadWriteOnce model-cache PVC is not safe for multi-node horizontal scaling
- Stream deployment uses Recreate strategy (single GPU, no rolling update)
