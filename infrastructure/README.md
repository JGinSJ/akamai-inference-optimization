# Infrastructure

Terraform configuration for the Akamai LKE cluster that runs the
Zero-Waste Inference on Akamai Cloud demo.

## Directory layout

```
infrastructure/
└── terraform/
    ├── main.tf                   # Terraform + Linode provider version pins
    ├── variables.tf              # All input variables with defaults
    ├── cluster.tf                # linode_lke_cluster + CPU node pool (inline)
    ├── node-pool-ada.tf          # RTX 4000 Ada GPU node pool (node_count=2)
    ├── node-pool-blackwell.tf    # RTX PRO 6000 Blackwell stub (commented out)
    ├── outputs.tf                # cluster_id, kubeconfig, pool IDs
    └── terraform.tfvars.example  # Safe-to-commit variable template
```

## Prerequisites

- `terraform >= 1.6.0`
- `linode-cli` authenticated (`linode-cli configure`)
- `kubectl` and `helm` installed
- A Linode personal access token with **Read/Write** on Linodes, LKE, and
  Block Storage scopes

## Provisioning the cluster

```bash
export LINODE_TOKEN="<your-token>"   # never commit this value

cd infrastructure/terraform
cp terraform.tfvars.example terraform.tfvars   # edit if needed
terraform init
terraform plan                                  # review before applying
terraform apply
```

Cluster reaches `ready` status in approximately 3–5 minutes.

Download the kubeconfig once the cluster is ready:

```bash
terraform output -raw kubeconfig | base64 --decode \
  > ~/.kube/akamai-inference-lke.yaml
export KUBECONFIG=~/.kube/akamai-inference-lke.yaml
kubectl get nodes
```

## Required post-cluster-creation steps

Complete these steps **in order** before deploying any workload manifests.

### 1. Install the NVIDIA device plugin

> **Important:** Akamai LKE does NOT pre-install the NVIDIA device plugin
> on GPU node pools. It must be installed manually. Without it,
> `nvidia.com/gpu` resource requests are ignored and GPU pods will not
> schedule onto GPU nodes.

```bash
kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.17.3/deployments/static/nvidia-device-plugin.yml
```

Verify the plugin is running and GPUs are advertised:

```bash
kubectl -n kube-system get daemonset nvidia-device-plugin-daemonset
kubectl describe node <gpu-node-name> | grep nvidia.com/gpu
# Expected output includes lines like:
#   nvidia.com/gpu:  1
#   nvidia.com/gpu:  1
```

### 2. Create the inference namespace

```bash
kubectl create namespace inference
```

### 3. Apply PVCs before deployments

Block storage volumes are provisioned on first `kubectl apply` of a PVC.
Apply PVCs before the deployments that reference them:

```bash
# Phase 2 — vLLM model cache (20 Gi)
kubectl apply -f phases/phase2-prefix-cache/vllm/pvc-model-cache.yaml

# Phase 3 — Qwen-Image HuggingFace cache (30 Gi)
kubectl apply -f phases/phase3-qwen-image/k8s/pvc-model-cache.yaml
```

Verify volumes are bound before proceeding:

```bash
kubectl -n inference get pvc
# Both PVCs should show STATUS = Bound before applying workload manifests.
```

### 4. Deploy workloads

```bash
# Phase 2
kubectl apply -f phases/phase2-prefix-cache/valkey/valkey.yaml
kubectl apply -f phases/phase2-prefix-cache/vllm/vllm.yaml

# Phase 3
kubectl apply -f phases/phase3-qwen-image/k8s/deployment.yaml
kubectl apply -f phases/phase3-qwen-image/k8s/service.yaml
```

## Running cost summary

| Pool | Plan | Nodes | Cost |
|------|------|-------|------|
| CPU (Valkey + services) | g6-dedicated-4 | 1 | $0.108/hr |
| RTX 4000 Ada | g2-gpu-rtx4000a1-l | 2 | $1.92/hr |
| **Total** | | | **$2.028/hr** |

Blackwell pool is not included — it is a commented-out stub pending plan slug confirmation.
Set `ada_node_count = 1` when not running data-parallel benchmarks to save ~$691/mo.

## Adding the Blackwell node pool

The RTX PRO 6000 Blackwell plan is not yet available in the Linode API.
When it becomes available, see `terraform/node-pool-blackwell.tf` for the
activation checklist.

## Teardown

```bash
cd infrastructure/terraform
terraform destroy   # destroys cluster, node pools, and associated resources
```

Block storage volumes provisioned by PVCs are **not** managed by Terraform
and must be deleted separately via the Akamai Cloud Manager or `linode-cli`
if no longer needed.
