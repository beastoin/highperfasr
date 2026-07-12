# GCP GKE — NVIDIA L4

Deploy highperfasr on GKE with L4 GPU nodes.

## Prerequisites

- GKE cluster with an L4 GPU node pool (`nvidia-l4` accelerator type)
- Two schedulable L4 GPUs for the full batch + streaming overlay
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
| Instance | g2-standard-4 |
| GPU | 1x NVIDIA L4 (24 GB) |
| Nodes for full overlay | 2 |
| vCPU | 4 |
| RAM | 16 GB |
| On-demand | ~$0.70/hr |
| Storage class | standard-rwo |

## Notes

- First pod startup downloads models from HuggingFace (~2 GB each, 2-3 min)
- Subsequent starts use the PVC-cached model
- No HPA is included because the shared ReadWriteOnce model-cache PVC is not safe for multi-node horizontal scaling
- Stream deployment uses Recreate strategy (single GPU, no rolling update)
