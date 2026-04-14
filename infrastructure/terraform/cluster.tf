# LKE cluster and CPU node pool.
#
# The Linode LKE API requires at least one node pool at cluster-creation
# time.  The CPU node pool (for Valkey and supporting services) is therefore
# declared inline in this resource rather than as a separate
# linode_lke_node_pool resource.
#
# GPU node pools are added after the cluster exists, each in its own file:
#   node-pool-ada.tf        — RTX 4000 Ada
#   node-pool-blackwell.tf  — RTX PRO 6000 Blackwell (stub, not yet available)
#
# To provision this cluster:
#   1. Copy terraform.tfvars.example -> terraform.tfvars and set any overrides.
#   2. export LINODE_TOKEN="<your-token>"   # never commit this value
#   3. terraform init
#   4. terraform plan                       # review before applying
#   5. terraform apply
#
# To download the kubeconfig after the cluster is Ready (~3–5 min):
#   terraform output -raw kubeconfig | base64 --decode \
#     > ~/.kube/akamai-inference-lke.yaml
#   export KUBECONFIG=~/.kube/akamai-inference-lke.yaml
#   kubectl get nodes

resource "linode_lke_cluster" "main" {
  label       = var.cluster_label
  k8s_version = var.k8s_version
  region      = var.region
  tags        = var.cluster_tags

  # CPU node pool — hosts Valkey (Phase 2) and other non-GPU services.
  # g6-dedicated-4: 4 vCPU, 8 GB RAM, dedicated cores.
  # Valkey requests 2.5 GB + kube-system overhead ~0.8 GB = ~3.3 GB total.
  # 8 GB gives comfortable headroom on dedicated (not shared) cores.
  pool {
    type  = var.cpu_node_type
    count = var.cpu_node_count

    labels = {
      "workload-type" = "cpu"
    }
  }
}
