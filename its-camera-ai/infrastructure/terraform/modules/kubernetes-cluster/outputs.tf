# Outputs for Kubernetes Cluster Module

output "cluster_id" {
  description = "EKS cluster ID"
  value       = aws_eks_cluster.main.id
}

output "cluster_arn" {
  description = "EKS cluster ARN"
  value       = aws_eks_cluster.main.arn
}

output "cluster_name" {
  description = "EKS cluster name"
  value       = aws_eks_cluster.main.name
}

output "cluster_endpoint" {
  description = "EKS cluster API server endpoint"
  value       = aws_eks_cluster.main.endpoint
}

output "cluster_version" {
  description = "EKS cluster Kubernetes version"
  value       = aws_eks_cluster.main.version
}

output "cluster_platform_version" {
  description = "EKS cluster platform version"
  value       = aws_eks_cluster.main.platform_version
}

output "cluster_status" {
  description = "EKS cluster status"
  value       = aws_eks_cluster.main.status
}

# Certificate Authority
output "cluster_certificate_authority_data" {
  description = "Base64 encoded certificate data required to communicate with the cluster"
  value       = aws_eks_cluster.main.certificate_authority[0].data
}

# Security Groups
output "cluster_security_group_id" {
  description = "EKS cluster security group ID"
  value       = aws_security_group.cluster.id
}

output "node_security_group_id" {
  description = "EKS node group security group ID"
  value       = aws_security_group.node.id
}

output "cluster_primary_security_group_id" {
  description = "EKS cluster primary security group ID"
  value       = aws_eks_cluster.main.vpc_config[0].cluster_security_group_id
}

# IAM
output "cluster_iam_role_arn" {
  description = "IAM role ARN of the EKS cluster"
  value       = aws_iam_role.cluster.arn
}

output "cluster_iam_role_name" {
  description = "IAM role name of the EKS cluster"
  value       = aws_iam_role.cluster.name
}

output "node_group_iam_role_arn" {
  description = "IAM role ARN of the EKS node group"
  value       = aws_iam_role.node.arn
}

output "node_group_iam_role_name" {
  description = "IAM role name of the EKS node group"
  value       = aws_iam_role.node.name
}

# OIDC
output "oidc_provider_arn" {
  description = "The ARN of the OIDC Identity Provider"
  value       = aws_iam_openid_connect_provider.cluster.arn
}

output "oidc_provider_url" {
  description = "The URL of the OIDC Identity Provider"
  value       = aws_iam_openid_connect_provider.cluster.url
}

# Node Groups
output "node_group_arn" {
  description = "EKS main node group ARN"
  value       = aws_eks_node_group.main.arn
}

output "node_group_status" {
  description = "EKS main node group status"
  value       = aws_eks_node_group.main.status
}

output "gpu_node_group_arn" {
  description = "EKS GPU node group ARN"
  value       = var.enable_gpu_nodes ? aws_eks_node_group.gpu[0].arn : null
}

output "gpu_node_group_status" {
  description = "EKS GPU node group status"
  value       = var.enable_gpu_nodes ? aws_eks_node_group.gpu[0].status : null
}

# Cluster Configuration
output "cluster_vpc_config" {
  description = "EKS cluster VPC configuration"
  value = {
    vpc_id                   = aws_eks_cluster.main.vpc_config[0].vpc_id
    subnet_ids              = aws_eks_cluster.main.vpc_config[0].subnet_ids
    endpoint_private_access = aws_eks_cluster.main.vpc_config[0].endpoint_private_access
    endpoint_public_access  = aws_eks_cluster.main.vpc_config[0].endpoint_public_access
    public_access_cidrs     = aws_eks_cluster.main.vpc_config[0].public_access_cidrs
  }
}

# Encryption
output "cluster_encryption_config" {
  description = "EKS cluster encryption configuration"
  value       = aws_eks_cluster.main.encryption_config
}

# Logging
output "cluster_logging" {
  description = "EKS cluster logging configuration"
  value       = aws_eks_cluster.main.enabled_cluster_log_types
}

output "cloudwatch_log_group_name" {
  description = "CloudWatch log group name for the cluster"
  value       = aws_cloudwatch_log_group.cluster.name
}

output "cloudwatch_log_group_arn" {
  description = "CloudWatch log group ARN for the cluster"
  value       = aws_cloudwatch_log_group.cluster.arn
}

# Add-ons
output "cluster_addons" {
  description = "Map of enabled cluster add-ons"
  value = {
    vpc_cni        = aws_eks_addon.vpc_cni.addon_version
    coredns        = aws_eks_addon.coredns.addon_version
    kube_proxy     = aws_eks_addon.kube_proxy.addon_version
    ebs_csi_driver = aws_eks_addon.ebs_csi_driver.addon_version
  }
}

# Launch Template
output "launch_template_id" {
  description = "Launch template ID for node groups"
  value       = aws_launch_template.node.id
}

output "launch_template_latest_version" {
  description = "Launch template latest version"
  value       = aws_launch_template.node.latest_version
}

# KMS
output "kms_key_id" {
  description = "KMS key ID for EKS cluster encryption"
  value       = aws_kms_key.eks.key_id
}

output "kms_key_arn" {
  description = "KMS key ARN for EKS cluster encryption"
  value       = aws_kms_key.eks.arn
}

# Kubeconfig
output "kubeconfig" {
  description = "kubectl configuration for accessing the cluster"
  value = {
    apiVersion      = "v1"
    kind           = "Config"
    current_context = aws_eks_cluster.main.name
    contexts = [{
      name = aws_eks_cluster.main.name
      context = {
        cluster = aws_eks_cluster.main.name
        user    = aws_eks_cluster.main.name
      }
    }]
    clusters = [{
      name = aws_eks_cluster.main.name
      cluster = {
        server                     = aws_eks_cluster.main.endpoint
        certificate_authority_data = aws_eks_cluster.main.certificate_authority[0].data
      }
    }]
    users = [{
      name = aws_eks_cluster.main.name
      user = {
        exec = {
          apiVersion = "client.authentication.k8s.io/v1beta1"
          command    = "aws"
          args       = ["eks", "get-token", "--cluster-name", aws_eks_cluster.main.name]
        }
      }
    }]
  }
  sensitive = true
}

# Cluster Information for GitOps
output "cluster_info" {
  description = "Consolidated cluster information for GitOps configuration"
  value = {
    name                    = aws_eks_cluster.main.name
    endpoint               = aws_eks_cluster.main.endpoint
    region                 = data.aws_region.current.name
    oidc_provider_url      = aws_iam_openid_connect_provider.cluster.url
    oidc_provider_arn      = aws_iam_openid_connect_provider.cluster.arn
    cluster_ca_data        = aws_eks_cluster.main.certificate_authority[0].data
    node_instance_types    = var.node_instance_types
    gpu_enabled           = var.enable_gpu_nodes
    encryption_enabled     = var.enable_cluster_encryption
  }
}

# Tags
output "cluster_tags" {
  description = "Tags applied to the cluster"
  value       = local.common_tags
}

# Data sources for outputs
data "aws_region" "current" {}
data "aws_caller_identity" "current" {}