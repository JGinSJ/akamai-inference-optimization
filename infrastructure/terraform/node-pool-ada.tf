# RTX 4000 Ada GPU node pool — Phase 2–3 workloads and Phase 4 single-GPU benchmarks.
#
# Plan:  g2-gpu-rtx4000a1-l — 1x RTX 4000 Ada, 16 vCPU, 64 GB RAM
# Cost:  $0.96/hr (~$691/mo) per node
# Count: 2 nodes — both carry gpu-type=rtx4000ada so workloads can land on either.
#
# Node label applied:
#   gpu-type = "rtx4000ada"
#
# This label is the canonical gpu-type value for all phase manifests.
# Workloads target this pool via nodeSelector:
#   nodeSelector:
#     gpu-type: rtx4000ada
#
# NOTE: A GPU taint is intentionally omitted here so that existing manifests
# (phase2/vllm, phase3/qwen-image, phase4/benchmark-jobs) schedule without
# requiring toleration changes.  Add a taint once all manifests are updated:
#
#   taints {
#     key    = "nvidia.com/gpu"
#     value  = "present"
#     effect = "NoSchedule"
#   }
#
# NOTE: The NVIDIA device plugin is NOT pre-installed on Akamai LKE GPU node
# pools — it must be installed manually after cluster creation:
#   kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.17.3/deployments/static/nvidia-device-plugin.yml
# See infrastructure/README.md for the full post-cluster-creation checklist.
# Verify after provisioning: kubectl describe node <gpu-node> | grep nvidia.com/gpu
#
# To add this pool to an existing cluster:
#   terraform plan   # confirm only this resource is created
#   terraform apply

resource "linode_lke_node_pool" "ada" {
  cluster_id = linode_lke_cluster.main.id
  type       = var.ada_node_type
  node_count = var.ada_node_count

  labels = {
    "gpu-type" = "rtx4000ada"
  }
}

output "ada_pool_id" {
  description = "Node pool ID for the RTX 4000 Ada pool."
  value       = linode_lke_node_pool.ada.id
}

output "ada_node_label" {
  description = "Node label applied to RTX 4000 Ada nodes. Use in nodeSelector."
  value       = "gpu-type: rtx4000ada"
}
