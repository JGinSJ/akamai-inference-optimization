variable "region" {
  type        = string
  description = "Akamai Cloud region ID for the LKE cluster."
  default     = "us-ord"
}

variable "k8s_version" {
  type        = string
  description = "LKE Kubernetes version. Run: linode-cli lke versions-list"
  default     = "1.35"
}

variable "cluster_label" {
  type        = string
  description = "Display label for the cluster in Akamai Cloud Manager."
  default     = "akamai-inference-opt"
}

variable "cluster_tags" {
  type        = list(string)
  description = "Tags applied to the cluster resource."
  default     = ["inference-optimization"]
}

# ---------------------------------------------------------------------------
# CPU node pool (Valkey + supporting services)
# Declared here because Linode requires >= 1 pool at cluster-creation time.
# The pool is managed inline inside linode_lke_cluster in cluster.tf.
# ---------------------------------------------------------------------------

variable "cpu_node_type" {
  type        = string
  description = "Linode plan slug for the CPU node pool. Default: g6-dedicated-4 (4 vCPU, 8 GB, $0.108/hr)."
  default     = "g6-dedicated-4"
}

variable "cpu_node_count" {
  type        = number
  description = "Number of nodes in the CPU node pool."
  default     = 1
}

# ---------------------------------------------------------------------------
# RTX 4000 Ada GPU node pool (Phase 2–3 workloads, Phase 4 single-GPU bench)
# Managed as a separate linode_lke_node_pool in node-pool-ada.tf.
# ---------------------------------------------------------------------------

variable "ada_node_type" {
  type        = string
  description = "Linode plan slug for the RTX 4000 Ada node pool. Default: g2-gpu-rtx4000a1-l (1x Ada, 16 vCPU, 64 GB, $0.96/hr)."
  default     = "g2-gpu-rtx4000a1-l"
}

variable "ada_node_count" {
  type        = number
  description = "Number of nodes in the RTX 4000 Ada node pool. Set to 2 for Phase 4 data-parallel benchmarks."
  default     = 2
}

# ---------------------------------------------------------------------------
# RTX PRO 6000 Blackwell GPU node pool (Phase 4 benchmarks)
# NOT YET AVAILABLE in the Linode API as of 2026-04-11.
# These variables are unused until node-pool-blackwell.tf is uncommented.
# Steps to activate:
#   1. Confirm the plan slug with Akamai support or linode-cli linodes types.
#   2. Replace PLACEHOLDER in blackwell_node_type below.
#   3. Uncomment the resource block in node-pool-blackwell.tf.
# ---------------------------------------------------------------------------

variable "blackwell_node_type" {
  type        = string
  description = "Linode plan slug for the RTX PRO 6000 Blackwell node pool. PLACEHOLDER — plan not yet in API."
  default     = "PLACEHOLDER"
}

variable "blackwell_node_count" {
  type        = number
  description = "Number of nodes in the RTX PRO 6000 Blackwell node pool."
  default     = 1
}
