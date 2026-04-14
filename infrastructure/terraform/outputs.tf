output "cluster_id" {
  description = "Numeric ID of the LKE cluster. Used in linode-cli and API calls."
  value       = linode_lke_cluster.main.id
}

output "cluster_label" {
  description = "Display label of the LKE cluster."
  value       = linode_lke_cluster.main.label
}

output "region" {
  description = "Region where the cluster is provisioned."
  value       = linode_lke_cluster.main.region
}

output "k8s_version" {
  description = "Kubernetes version running on the cluster."
  value       = linode_lke_cluster.main.k8s_version
}

output "api_endpoints" {
  description = "Kubernetes API server endpoints."
  value       = linode_lke_cluster.main.api_endpoints
}

output "status" {
  description = "Cluster provisioning status."
  value       = linode_lke_cluster.main.status
}

output "kubeconfig" {
  description = <<-EOT
    Base64-encoded kubeconfig for kubectl access.
    Decode and install with:
      terraform output -raw kubeconfig | base64 --decode \
        > ~/.kube/akamai-inference-lke.yaml
      export KUBECONFIG=~/.kube/akamai-inference-lke.yaml
  EOT
  value       = linode_lke_cluster.main.kubeconfig
  sensitive   = true
}

output "dashboard_url" {
  description = "Kubernetes dashboard URL (Akamai Cloud Manager)."
  value       = linode_lke_cluster.main.dashboard_url
  sensitive   = true
}
