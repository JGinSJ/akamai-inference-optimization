# RTX PRO 6000 Blackwell GPU node pool — Phase 4 benchmarks.
#
# STATUS: NOT AVAILABLE — entire resource block is commented out.
#
# As of 2026-04-11 the RTX PRO 6000 Blackwell plan does not appear in the
# Linode API (linode-cli linodes types --json).  The old g1-gpu-rtx6000-*
# family in the API is the Turing-generation Quadro RTX 6000, not the
# Blackwell RTX PRO 6000, and must not be substituted.
#
# Before uncommenting this resource:
#   1. Confirm availability with Akamai support or your account team.
#   2. Retrieve the exact plan slug:
#        linode-cli linodes types --json | python3 -c \
#          "import json,sys; [print(t['id'],t['label']) \
#           for t in json.load(sys.stdin) if 'blackwell' in t['label'].lower()]"
#   3. Replace PLACEHOLDER in var.blackwell_node_type (variables.tf).
#   4. Uncomment the resource and output blocks below.
#   5. Run: terraform plan   # review cost before terraform apply
#
# Intended configuration when available:
#   Plan:  PLACEHOLDER (e.g. "g3-gpu-rtxpro6000-1" — confirm with API)
#   GPUs:  1x RTX PRO 6000 Blackwell
#   VRAM:  PLACEHOLDER GB GDDR7   (see docs/hardware.md)
#   vCPU:  PLACEHOLDER
#   RAM:   PLACEHOLDER GB
#   Cost:  PLACEHOLDER $/hr       (see docs/hardware.md; update when confirmed)
#
# Node label that will be applied:
#   gpu-type = "rtx6000blackwell"
#
# Workloads will target this pool via:
#   nodeSelector:
#     gpu-type: rtx6000blackwell

# resource "linode_lke_node_pool" "blackwell" {
#   cluster_id = linode_lke_cluster.main.id
#   type       = var.blackwell_node_type   # PLACEHOLDER — must be set before apply
#   node_count = var.blackwell_node_count
#
#   labels = {
#     "gpu-type" = "rtx6000blackwell"
#   }
#
#   # TODO: Add a GPU taint once all GPU workload manifests have tolerations:
#   # taints {
#   #   key    = "nvidia.com/gpu"
#   #   value  = "present"
#   #   effect = "NoSchedule"
#   # }
# }

# output "blackwell_pool_id" {
#   description = "Node pool ID for the RTX PRO 6000 Blackwell pool."
#   value       = linode_lke_node_pool.blackwell.id
# }

# output "blackwell_node_label" {
#   description = "Node label applied to RTX PRO 6000 Blackwell nodes. Use in nodeSelector."
#   value       = "gpu-type: rtx6000blackwell"
# }
