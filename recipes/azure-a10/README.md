# Azure AKS — NVIDIA A10

Deploy highperfasr on AKS with NVads A10 v5 instances (NVIDIA A10 GPU, 24 GB).

## Prerequisites

- AKS cluster with a GPU node pool using `Standard_NV36ads_A10_v5` VMs
- Two schedulable A10 GPUs for the full batch + streaming overlay
- Node pool created with `--node-taints sku=gpu:NoSchedule`
- NVIDIA device plugin installed (included in AKS GPU node pools by default)
- `kubectl` configured for your cluster

```bash
az aks get-credentials --resource-group RG --name CLUSTER
kubectl get nodes -l accelerator=nvidia --no-headers
```

## Deploy

```bash
kubectl apply -k recipes/azure-a10
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
| Instance | Standard_NV36ads_A10_v5 |
| GPU | 1x NVIDIA A10 (24 GB) |
| Nodes for full overlay | 2 |
| vCPU | 36 |
| RAM | 440 GB |
| On-demand | ~$0.91/hr |
| Storage class | managed-csi-premium |

## Notes

- The A10 is Ampere architecture (same generation as A100) with 24 GB VRAM — same as L4
- Smaller NVads sizes (NV6/12/18) use GPU partitions with less VRAM — not recommended
- Use `managed-csi-premium` storage class (modern CSI driver, not legacy `managed-premium`)
- First pod startup downloads models from HuggingFace (~2 GB each, 2-3 min)
- No HPA is included; use per-replica cache storage or RWX storage before horizontal scaling
- AKS managed GPU node pools are preview; use standard GPU node pools for production
