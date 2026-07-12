# AWS EKS — NVIDIA L4 (G6)

Deploy highperfasr on EKS with G6 instances (NVIDIA L4 GPU).

## Prerequisites

- EKS cluster with a G6 managed node group
- Two schedulable L4 GPUs for the full batch + streaming overlay
- NVIDIA device plugin installed (`kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.17.0/deployments/static/nvidia-device-plugin.yml`)
- EKS Auto Mode includes the device plugin automatically
- `kubectl` configured for your cluster

```bash
aws eks update-kubeconfig --name CLUSTER --region REGION
kubectl get nodes -l node.kubernetes.io/instance-type=g6.xlarge --no-headers
```

## Deploy

```bash
kubectl apply -k recipes/aws-g6-l4
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
| Instance | g6.xlarge |
| GPU | 1x NVIDIA L4 (24 GB) |
| Nodes for full overlay | 2 |
| vCPU | 4 |
| RAM | 16 GB |
| On-demand | ~$0.80/hr |
| Storage class | gp3 (EBS CSI driver required) |

## Notes

- EKS accelerated AMIs (AL2023) include NVIDIA drivers and container toolkit
- If using GPU Operator on EKS accelerated AMIs, disable driver/toolkit install to avoid conflicts
- The `gp3` storage class requires the EBS CSI driver add-on
- First pod startup downloads models from HuggingFace (~2 GB each, 2-3 min)
- No HPA is included; use per-replica cache storage or RWX storage before horizontal scaling
- For spot instances (batch workloads), create a separate node group with `capacity-type: SPOT`
