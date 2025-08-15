# ITS Camera AI - Kubernetes Cluster Module Outputs

# Cluster Information
output "cluster_id" {
  description = "EKS cluster ID"
  value       = aws_eks_cluster.cluster.cluster_id
}

output "cluster_name" {
  description = "EKS cluster name"
  value       = aws_eks_cluster.cluster.name
}

output "cluster_endpoint" {
  description = "Endpoint for EKS control plane"
  value       = aws_eks_cluster.cluster.endpoint
}

output "cluster_version" {
  description = "The Kubernetes version for the cluster"
  value       = aws_eks_cluster.cluster.version
}

output "cluster_arn" {
  description = "The Amazon Resource Name (ARN) of the cluster"
  value       = aws_eks_cluster.cluster.arn
}

output "cluster_certificate_authority_data" {
  description = "Base64 encoded certificate data required to communicate with the cluster"
  value       = aws_eks_cluster.cluster.certificate_authority[0].data
}

output "cluster_platform_version" {
  description = "Platform version for the cluster"
  value       = aws_eks_cluster.cluster.platform_version
}

output "cluster_status" {
  description = "Status of the EKS cluster"
  value       = aws_eks_cluster.cluster.status
}

# OIDC Provider Information
output "cluster_oidc_issuer_url" {
  description = "The URL on the EKS cluster for the OpenID Connect identity provider"
  value       = aws_eks_cluster.cluster.identity[0].oidc[0].issuer
}

output "oidc_provider_arn" {
  description = "The ARN of the OIDC Provider"
  value       = aws_iam_openid_connect_provider.cluster_oidc.arn
}

# VPC Information
output "vpc_id" {
  description = "ID of the VPC where the cluster and workers are deployed"
  value       = aws_vpc.cluster_vpc.id
}

output "vpc_cidr_block" {
  description = "The CIDR block of the VPC"
  value       = aws_vpc.cluster_vpc.cidr_block
}

output "private_subnets" {
  description = "List of IDs of private subnets"
  value       = aws_subnet.private_subnets[*].id
}

output "public_subnets" {
  description = "List of IDs of public subnets"
  value       = aws_subnet.public_subnets[*].id
}

output "private_subnet_cidrs" {
  description = "List of CIDR blocks of private subnets"
  value       = aws_subnet.private_subnets[*].cidr_block
}

output "public_subnet_cidrs" {
  description = "List of CIDR blocks of public subnets"
  value       = aws_subnet.public_subnets[*].cidr_block
}

# Security Group Information
output "cluster_security_group_id" {
  description = "Security group ID attached to the EKS cluster"
  value       = aws_security_group.cluster_sg.id
}

output "node_security_group_id" {
  description = "Security group ID attached to the EKS node groups"
  value       = aws_security_group.node_sg.id
}

# IAM Role Information
output "cluster_iam_role_name" {
  description = "IAM role name associated with EKS cluster"
  value       = aws_iam_role.cluster_role.name
}

output "cluster_iam_role_arn" {
  description = "IAM role ARN associated with EKS cluster"
  value       = aws_iam_role.cluster_role.arn
}

output "node_iam_role_name" {
  description = "IAM role name associated with EKS node groups"
  value       = aws_iam_role.node_role.name
}

output "node_iam_role_arn" {
  description = "IAM role ARN associated with EKS node groups"
  value       = aws_iam_role.node_role.arn
}

# Node Group Information
output "gpu_node_group_arn" {
  description = "Amazon Resource Name (ARN) of the GPU EKS Node Group"
  value       = var.enable_gpu_nodes ? aws_eks_node_group.gpu_nodes[0].arn : null
}

output "gpu_node_group_status" {
  description = "Status of the GPU EKS Node Group"
  value       = var.enable_gpu_nodes ? aws_eks_node_group.gpu_nodes[0].status : null
}

output "general_node_group_arn" {
  description = "Amazon Resource Name (ARN) of the general EKS Node Group"
  value       = aws_eks_node_group.general_nodes.arn
}

output "general_node_group_status" {
  description = "Status of the general EKS Node Group"
  value       = aws_eks_node_group.general_nodes.status
}

output "memory_node_group_arn" {
  description = "Amazon Resource Name (ARN) of the memory-optimized EKS Node Group"
  value       = var.enable_memory_nodes ? aws_eks_node_group.memory_nodes[0].arn : null
}

output "memory_node_group_status" {
  description = "Status of the memory-optimized EKS Node Group"
  value       = var.enable_memory_nodes ? aws_eks_node_group.memory_nodes[0].status : null
}

# KMS Key Information
output "cluster_encryption_key_arn" {
  description = "The Amazon Resource Name (ARN) of the key used to encrypt cluster secrets"
  value       = aws_kms_key.cluster_encryption.arn
}

output "cluster_encryption_key_id" {
  description = "The globally unique identifier for the key used to encrypt cluster secrets"
  value       = aws_kms_key.cluster_encryption.key_id
}

# Networking Information
output "nat_gateway_ids" {
  description = "List of IDs of the NAT Gateways"
  value       = aws_nat_gateway.nat_gateways[*].id
}

output "internet_gateway_id" {
  description = "The ID of the Internet Gateway"
  value       = aws_internet_gateway.cluster_igw.id
}

output "route_table_ids" {
  description = "List of IDs of the route tables"
  value = {
    public  = aws_route_table.public_rt.id
    private = aws_route_table.private_rt[*].id
  }
}

# CloudWatch Information
output "cloudwatch_log_group_name" {
  description = "Name of the CloudWatch log group for cluster logs"
  value       = aws_cloudwatch_log_group.cluster_logs.name
}

output "cloudwatch_log_group_arn" {
  description = "ARN of the CloudWatch log group for cluster logs"
  value       = aws_cloudwatch_log_group.cluster_logs.arn
}

# Configuration for kubectl
output "kubectl_config" {
  description = "kubectl configuration for accessing the cluster"
  value = {
    cluster_name                 = aws_eks_cluster.cluster.name
    endpoint                     = aws_eks_cluster.cluster.endpoint
    certificate_authority_data   = aws_eks_cluster.cluster.certificate_authority[0].data
    region                      = data.aws_caller_identity.current.account_id
  }
  sensitive = true
}

# Kubeconfig command
output "kubeconfig_command" {
  description = "Command to update kubeconfig for the cluster"
  value       = "aws eks update-kubeconfig --region ${data.aws_availability_zones.available.id} --name ${aws_eks_cluster.cluster.name}"
}

# Node Group Scaling Configuration
output "node_groups_scaling_config" {
  description = "Scaling configuration for all node groups"
  value = {
    gpu_nodes = var.enable_gpu_nodes ? {
      desired_size = var.gpu_nodes_desired_size
      max_size     = var.gpu_nodes_max_size
      min_size     = var.gpu_nodes_min_size
    } : null
    general_nodes = {
      desired_size = var.general_nodes_desired_size
      max_size     = var.general_nodes_max_size
      min_size     = var.general_nodes_min_size
    }
    memory_nodes = var.enable_memory_nodes ? {
      desired_size = var.memory_nodes_desired_size
      max_size     = var.memory_nodes_max_size
      min_size     = var.memory_nodes_min_size
    } : null
  }
}

# Cluster add-ons information
output "cluster_addons" {
  description = "Map of cluster add-on names and their versions"
  value = {
    vpc-cni            = aws_eks_addon.vpc_cni.addon_version
    coredns            = aws_eks_addon.coredns.addon_version
    kube-proxy         = aws_eks_addon.kube_proxy.addon_version
    aws-ebs-csi-driver = aws_eks_addon.ebs_csi.addon_version
  }
}

# Tagging information
output "cluster_tags" {
  description = "Tags applied to the cluster"
  value       = var.common_tags
}

# Availability zones
output "availability_zones" {
  description = "List of availability zones used by the cluster"
  value       = data.aws_availability_zones.available.names
}

# Cluster capacity information
output "cluster_capacity" {
  description = "Total cluster capacity information"
  value = {
    total_nodes = {
      gpu_nodes     = var.enable_gpu_nodes ? var.gpu_nodes_desired_size : 0
      general_nodes = var.general_nodes_desired_size
      memory_nodes  = var.enable_memory_nodes ? var.memory_nodes_desired_size : 0
    }
    instance_types = {
      gpu_instances     = var.enable_gpu_nodes ? var.gpu_instance_types : []
      general_instances = var.general_instance_types
      memory_instances  = var.enable_memory_nodes ? var.memory_instance_types : []
    }
  }
}

# Cost optimization outputs
output "cost_optimization_features" {
  description = "Cost optimization features enabled"
  value = {
    spot_instances = var.enable_spot_instances
    auto_scaling   = var.enable_cluster_autoscaler
  }
}

# Security features
output "security_features" {
  description = "Security features enabled on the cluster"
  value = {
    encryption_at_rest   = var.enable_encryption_at_rest
    private_endpoint     = var.enable_private_endpoint
    logging_enabled      = var.enable_logging
    irsa_enabled        = var.enable_irsa
  }
}

# Monitoring features
output "monitoring_features" {
  description = "Monitoring features enabled"
  value = {
    container_insights      = var.enable_container_insights
    prometheus_metrics      = var.enable_prometheus_metrics
    performance_monitoring  = var.enable_performance_monitoring
  }
}

# Ready status
output "cluster_ready" {
  description = "Indicates if the cluster is ready for workloads"
  value       = aws_eks_cluster.cluster.status == "ACTIVE"
}