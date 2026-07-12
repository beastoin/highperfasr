# Deployment Recipes

Kustomize-based deployment recipes for running highperfasr on different cloud GPU instances. Each recipe is a kustomize overlay that adds provider-specific GPU scheduling and storage classes to a shared base.

## Quick Start

```bash
# GCP GKE with L4 GPU
kubectl apply -k recipes/gcp-l4

# AWS EKS with G6/L4 GPU
kubectl apply -k recipes/aws-g6-l4

# Azure AKS with A10 GPU
kubectl apply -k recipes/azure-a10
```

Each provider overlay deploys both streaming and batch services. Because each
service requests one GPU, the full overlay requires two one-GPU nodes or one node
with at least two schedulable GPUs. For a single-GPU cluster, scale one workload
to zero after applying the overlay.

## Available Recipes

| Recipe | GPU | Instance | Nodes for Full Overlay | ~$/hr per Node | Status |
|--------|-----|----------|------------------------|----------------|--------|
| [gcp-l4](gcp-l4/) | NVIDIA L4 | g2-standard-4 | 2 | $0.70 | Benchmarked |
| [aws-g6-l4](aws-g6-l4/) | NVIDIA L4 | g6.xlarge | 2 | $0.80 | Recipe ready |
| [azure-a10](azure-a10/) | NVIDIA A10 | Standard_NV36ads_A10_v5 | 2 | $0.91 | Recipe ready |

## Structure

```
recipes/
  base/              # Shared K8s resources (provider-agnostic)
    kustomization.yaml
    deployment-stream.yaml
    deployment-batch.yaml
    service.yaml
    pvc.yaml
  gcp-l4/            # GCP overlay
  aws-g6-l4/         # AWS overlay
  azure-a10/         # Azure overlay
```

Each overlay patches the base with:
- **GPU scheduling** — nodeSelector, tolerations, and node affinity for the target GPU
- **Storage class** — provider-specific PVC storage class

## Adding a Recipe

1. Create `recipes/<provider-gpu>/`
2. Add `kustomization.yaml` referencing `../base`
3. Add strategic merge patches for GPU scheduling and storage
4. Add a `README.md` with prerequisites, instance specs, and deploy instructions
5. Run benchmarks and publish results to `benchmarks/results/<provider-gpu>-on-demand-YYYYMMDD/`

## Common Operations

```bash
# Preview what will be applied
kubectl kustomize recipes/gcp-l4

# Run on a one-GPU cluster by disabling one workload after apply
kubectl apply -k recipes/gcp-l4
kubectl scale deployment/highperfasr-batch --replicas=0

# Check GPU allocation
kubectl describe node -l <gpu-label> | grep -A5 "Allocated resources"

# Port-forward for local testing
kubectl port-forward svc/highperfasr-stream 8001:8000
kubectl port-forward svc/highperfasr-batch 8000:8000
```

## Docker Compose

For local development without Kubernetes, use `compose.yaml` in the repo root:

```bash
docker compose up -d          # stream-only (1 GPU)
docker compose --profile full up -d   # batch + stream (2 GPUs)
```
