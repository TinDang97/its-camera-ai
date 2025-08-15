# ITS Camera AI - Kubernetes Cluster Module Variables

variable "cluster_name" {
  description = "Name of the EKS cluster"
  type        = string
  validation {
    condition     = can(regex("^[a-zA-Z0-9][a-zA-Z0-9-]*[a-zA-Z0-9]$", var.cluster_name))
    error_message = "Cluster name must be a valid EKS cluster name."
  }
}

variable "kubernetes_version" {
  description = "Kubernetes version for the cluster"
  type        = string
  default     = "1.28"
}

variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
  validation {
    condition     = can(cidrhost(var.vpc_cidr, 0))
    error_message = "VPC CIDR must be a valid IPv4 CIDR block."
  }
}

variable "public_subnet_cidrs" {
  description = "CIDR blocks for public subnets"
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  validation {
    condition     = length(var.public_subnet_cidrs) >= 2
    error_message = "At least 2 public subnets are required for high availability."
  }
}

variable "private_subnet_cidrs" {
  description = "CIDR blocks for private subnets"
  type        = list(string)
  default     = ["10.0.4.0/24", "10.0.5.0/24", "10.0.6.0/24"]
  validation {
    condition     = length(var.private_subnet_cidrs) >= 2
    error_message = "At least 2 private subnets are required for high availability."
  }
}

variable "endpoint_public_access" {
  description = "Enable public API server endpoint"
  type        = bool
  default     = true
}

variable "endpoint_public_access_cidrs" {
  description = "List of CIDR blocks that can access the public API server endpoint"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "cluster_log_types" {
  description = "List of cluster log types to enable"
  type        = list(string)
  default     = ["api", "audit", "authenticator", "controllerManager", "scheduler"]
}

variable "log_retention_days" {
  description = "Number of days to retain CloudWatch logs"
  type        = number
  default     = 30
}

variable "node_ssh_key" {
  description = "EC2 SSH key name for node access"
  type        = string
  default     = ""
}

# GPU Node Group Variables
variable "enable_gpu_nodes" {
  description = "Enable GPU node group for inference workloads"
  type        = bool
  default     = true
}

variable "gpu_instance_types" {
  description = "Instance types for GPU nodes"
  type        = list(string)
  default     = ["g5.2xlarge"]
  validation {
    condition = alltrue([
      for instance_type in var.gpu_instance_types :
      can(regex("^g[4-6]\\.", instance_type)) || can(regex("^p[3-4]\\.", instance_type))
    ])
    error_message = "GPU instance types must be from g4, g5, g6, p3, or p4 families."
  }
}

variable "gpu_nodes_desired_size" {
  description = "Desired number of GPU nodes"
  type        = number
  default     = 3
}

variable "gpu_nodes_min_size" {
  description = "Minimum number of GPU nodes"
  type        = number
  default     = 2
}

variable "gpu_nodes_max_size" {
  description = "Maximum number of GPU nodes"
  type        = number
  default     = 20
}

# General Node Group Variables
variable "general_instance_types" {
  description = "Instance types for general purpose nodes"
  type        = list(string)
  default     = ["c5.4xlarge"]
}

variable "general_nodes_desired_size" {
  description = "Desired number of general nodes"
  type        = number
  default     = 5
}

variable "general_nodes_min_size" {
  description = "Minimum number of general nodes"
  type        = number
  default     = 3
}

variable "general_nodes_max_size" {
  description = "Maximum number of general nodes"
  type        = number
  default     = 50
}

# Memory-optimized Node Group Variables
variable "enable_memory_nodes" {
  description = "Enable memory-optimized node group"
  type        = bool
  default     = true
}

variable "memory_instance_types" {
  description = "Instance types for memory-optimized nodes"
  type        = list(string)
  default     = ["r5.2xlarge"]
}

variable "memory_nodes_desired_size" {
  description = "Desired number of memory-optimized nodes"
  type        = number
  default     = 3
}

variable "memory_nodes_min_size" {
  description = "Minimum number of memory-optimized nodes"
  type        = number
  default     = 2
}

variable "memory_nodes_max_size" {
  description = "Maximum number of memory-optimized nodes"
  type        = number
  default     = 10
}

# EKS Addon Versions
variable "vpc_cni_version" {
  description = "Version of the vpc-cni addon"
  type        = string
  default     = "v1.15.1-eksbuild.1"
}

variable "coredns_version" {
  description = "Version of the coredns addon"
  type        = string
  default     = "v1.10.1-eksbuild.4"
}

variable "kube_proxy_version" {
  description = "Version of the kube-proxy addon"
  type        = string
  default     = "v1.28.2-eksbuild.2"
}

variable "ebs_csi_version" {
  description = "Version of the aws-ebs-csi-driver addon"
  type        = string
  default     = "v1.24.0-eksbuild.1"
}

# Common Tags
variable "common_tags" {
  description = "Common tags to apply to all resources"
  type        = map(string)
  default = {
    Project     = "its-camera-ai"
    Environment = "production"
    ManagedBy   = "terraform"
    Owner       = "platform-team"
  }
}

# Cost optimization variables
variable "enable_spot_instances" {
  description = "Enable spot instances for cost optimization"
  type        = bool
  default     = false
}

variable "spot_instance_types" {
  description = "Instance types for spot instances"
  type        = list(string)
  default     = ["c5.2xlarge", "c5.4xlarge", "m5.2xlarge", "m5.4xlarge"]
}

# Monitoring and observability
variable "enable_container_insights" {
  description = "Enable CloudWatch Container Insights"
  type        = bool
  default     = true
}

variable "enable_prometheus_metrics" {
  description = "Enable Prometheus metrics collection"
  type        = bool
  default     = true
}

# Security configurations
variable "enable_encryption_at_rest" {
  description = "Enable encryption at rest for EKS cluster"
  type        = bool
  default     = true
}

variable "enable_logging" {
  description = "Enable logging for EKS cluster"
  type        = bool
  default     = true
}

variable "enable_private_endpoint" {
  description = "Enable private endpoint access"
  type        = bool
  default     = true
}

# Networking configurations
variable "enable_irsa" {
  description = "Enable IAM Roles for Service Accounts"
  type        = bool
  default     = true
}

variable "cni_plugin" {
  description = "CNI plugin to use (aws-vpc-cni, calico)"
  type        = string
  default     = "aws-vpc-cni"
  validation {
    condition     = contains(["aws-vpc-cni", "calico"], var.cni_plugin)
    error_message = "CNI plugin must be either 'aws-vpc-cni' or 'calico'."
  }
}

# Auto-scaling configurations
variable "enable_cluster_autoscaler" {
  description = "Enable cluster autoscaler"
  type        = bool
  default     = true
}

variable "enable_metrics_server" {
  description = "Enable metrics server for HPA"
  type        = bool
  default     = true
}

# Storage configurations
variable "ebs_storage_class" {
  description = "Default storage class for EBS volumes"
  type        = string
  default     = "gp3"
}

variable "ebs_volume_size" {
  description = "Default EBS volume size for nodes (GB)"
  type        = number
  default     = 100
}

variable "ebs_volume_type" {
  description = "EBS volume type for nodes"
  type        = string
  default     = "gp3"
}

# GPU-specific configurations
variable "nvidia_device_plugin_version" {
  description = "Version of NVIDIA device plugin"
  type        = string
  default     = "v0.14.1"
}

variable "gpu_operator_version" {
  description = "Version of NVIDIA GPU operator"
  type        = string
  default     = "v23.9.0"
}

# ML workload optimizations
variable "enable_efa" {
  description = "Enable Elastic Fabric Adapter for multi-node training"
  type        = bool
  default     = false
}

variable "enable_nvme_ssd" {
  description = "Enable NVMe SSD for high-performance workloads"
  type        = bool
  default     = true
}

# Disaster recovery
variable "backup_retention_days" {
  description = "Number of days to retain backups"
  type        = number
  default     = 30
}

variable "enable_cross_region_backup" {
  description = "Enable cross-region backup"
  type        = bool
  default     = false
}

# Performance monitoring
variable "enable_performance_monitoring" {
  description = "Enable detailed performance monitoring"
  type        = bool
  default     = true
}

variable "monitoring_interval" {
  description = "Monitoring collection interval in seconds"
  type        = number
  default     = 60
}