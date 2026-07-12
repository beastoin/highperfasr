# GCP GKE — NVIDIA L4

Deploy highperfasr on GKE with L4 GPU nodes.

## Prerequisites

- GKE cluster with an L4 GPU node pool (`nvidia-l4` accelerator type)
- NVIDIA GPU device plugin installed (included by default on GKE)
- `kubectl` configured for your cluster

```bash
gcloud container clusters get-credentials CLUSTER --region REGION --project PROJECT
kubectl get nodes -l cloud.google.com/gke-accelerator=nvidia-l4 --no-headers
```

## Deploy

```bash
kubectl apply -k recipes/gcp-l4
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
| Instance | g2-standard-4 |
| GPU | 1x NVIDIA L4 (24 GB) |
| vCPU | 4 |
| RAM | 16 GB |
| On-demand | ~$0.70/hr |
| Storage class | standard-rwo |

## Notes

- First pod startup downloads models from HuggingFace (~2 GB each, 2-3 min)
- Subsequent starts use the PVC-cached model
- HPA is included for the batch deployment (CPU-based, 1-4 replicas)
- Stream deployment uses Recreate strategy (single GPU, no rolling update)
