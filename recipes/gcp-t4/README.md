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

## Notes

- T4 has 16 GB VRAM (vs 24 GB on L4) — expect lower max concurrency for streaming
- `torch.compile` is not supported on T4 (Turing architecture, compute capability 7.5) — the server disables it automatically
- T4 is the most cost-effective GPU on GCP for inference workloads
- Spot/preemptible T4 instances available at ~70% discount — suitable for batch workloads
- First pod startup downloads models from HuggingFace (~2 GB each, 2-3 min)
- Subsequent starts use the PVC-cached model
- No HPA is included because the shared ReadWriteOnce model-cache PVC is not safe for multi-node horizontal scaling
- Stream deployment uses Recreate strategy (single GPU, no rolling update)
