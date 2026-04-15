# This file is intentionally empty.
#
# The second RTX 4000 Ada node is managed by setting ada_node_count = 2
# in node-pool-ada.tf (linode_lke_node_pool.ada).
#
# A separate node pool resource is NOT needed — incrementing node_count
# on the existing pool is an in-place API update that avoids node recreation.
