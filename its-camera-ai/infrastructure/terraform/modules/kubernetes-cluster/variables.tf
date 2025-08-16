# Variables for Kubernetes Cluster Module

variable "environment" {
  description = "Environment name (e.g., dev, staging, production)"
  type        = string
  validation {
    condition = can(regex("^(development|staging|production)$", var.environment))
    error_message = "Environment must be development, staging, or production."
  }
}

variable "cluster_name" {
  description = "Name of the EKS cluster"
  type        = string
  default     = "its-camera-ai"
}

variable "kubernetes_version" {
  description = "Kubernetes version for the EKS cluster"
  type        = string
  default     = "1.31"
}

variable "vpc_id" {
  description = "VPC ID where the cluster will be created"
  type        = string
}

variable "subnet_ids" {
  description = "List of subnet IDs for the cluster"
  type        = list(string)
}

variable "allowed_cidr_blocks" {
  description = "List of CIDR blocks that can access the cluster API server"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

# Node Group Configuration
variable "node_instance_types" {
  description = "List of instance types for the main node group"
  type        = list(string)
  default     = ["m5.xlarge", "m5.2xlarge"]
}

variable "node_desired_size" {
  description = "Desired number of nodes in the main node group"
  type        = number
  default     = 3
}

variable "node_min_size" {
  description = "Minimum number of nodes in the main node group"
  type        = number
  default     = 1
}

variable "node_max_size" {
  description = "Maximum number of nodes in the main node group"
  type        = number
  default     = 10
}

variable "node_disk_size" {
  description = "Disk size for main nodes (GB)"
  type        = number
  default     = 100
}

variable "capacity_type" {
  description = "Type of capacity associated with the EKS Node Group (ON_DEMAND or SPOT)"
  type        = string
  default     = "ON_DEMAND"
}

variable "node_key_name" {
  description = "EC2 key pair name for node access"
  type        = string
  default     = null
}

# GPU Node Configuration
variable "enable_gpu_nodes" {
  description = "Enable GPU nodes for ML workloads"
  type        = bool
  default     = true
}

variable "gpu_node_desired_size" {
  description = "Desired number of GPU nodes"
  type        = number
  default     = 2
}

variable "gpu_node_min_size" {
  description = "Minimum number of GPU nodes"
  type        = number
  default     = 0
}

variable "gpu_node_max_size" {
  description = "Maximum number of GPU nodes"
  type        = number
  default     = 5
}

variable "gpu_node_disk_size" {
  description = "Disk size for GPU nodes (GB)"
  type        = number
  default     = 200
}

# Logging Configuration
variable "log_retention_days" {
  description = "CloudWatch log retention in days"
  type        = number
  default     = 7
}

# S3 Configuration
variable "s3_bucket_name" {
  description = "S3 bucket name for application data"
  type        = string
}

# Bootstrap Configuration
variable "bootstrap_arguments" {
  description = "Additional bootstrap arguments for nodes"
  type        = string
  default     = ""
}

# Networking Configuration
variable "enable_irsa" {
  description = "Enable IAM Roles for Service Accounts"
  type        = bool
  default     = true
}

variable "enable_cluster_encryption" {
  description = "Enable envelope encryption for Kubernetes secrets"
  type        = bool
  default     = true
}

# Monitoring Configuration
variable "enable_cloudwatch_logging" {
  description = "Enable CloudWatch logging for the cluster"
  type        = bool
  default     = true
}

variable "cluster_log_types" {
  description = "List of log types to enable for the cluster"
  type        = list(string)
  default     = ["api", "audit", "authenticator", "controllerManager", "scheduler"]
}

# Security Configuration
variable "cluster_endpoint_private_access" {
  description = "Enable private API server endpoint"
  type        = bool
  default     = true
}

variable "cluster_endpoint_public_access" {
  description = "Enable public API server endpoint"
  type        = bool
  default     = true
}

variable "cluster_endpoint_public_access_cidrs" {
  description = "List of CIDR blocks that can access the public API server endpoint"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

# Add-ons Configuration
variable "enable_vpc_cni" {
  description = "Enable VPC CNI add-on"
  type        = bool
  default     = true
}

variable "enable_coredns" {
  description = "Enable CoreDNS add-on"
  type        = bool
  default     = true
}

variable "enable_kube_proxy" {
  description = "Enable kube-proxy add-on"
  type        = bool
  default     = true
}

variable "enable_ebs_csi" {
  description = "Enable EBS CSI driver add-on"
  type        = bool
  default     = true
}

# Performance Configuration
variable "cluster_service_ipv4_cidr" {
  description = "The CIDR block to assign Kubernetes service IP addresses from"
  type        = string
  default     = null
}

variable "cluster_ip_family" {
  description = "The IP family used to assign Kubernetes service IP addresses"
  type        = string
  default     = "ipv4"
  validation {
    condition     = contains(["ipv4", "ipv6"], var.cluster_ip_family)
    error_message = "Cluster IP family must be ipv4 or ipv6."
  }
}

# Node Group AMI Configuration
variable "ami_type" {
  description = "Type of Amazon Machine Image (AMI) associated with the EKS Node Group"
  type        = string
  default     = "AL2_x86_64"
  validation {
    condition = contains([
      "AL2_x86_64",
      "AL2_x86_64_GPU",
      "AL2_ARM_64",
      "CUSTOM",
      "BOTTLEROCKET_ARM_64",
      "BOTTLEROCKET_x86_64"
    ], var.ami_type)
    error_message = "AMI type must be one of the supported types."
  }
}

# Tagging Configuration
variable "additional_tags" {
  description = "Additional tags to apply to resources"
  type        = map(string)
  default     = {}
}

# Network Security
variable "create_security_group" {
  description = "Create security group for the cluster"
  type        = bool
  default     = true
}

variable "security_group_ids" {
  description = "List of security group IDs to associate with the cluster"
  type        = list(string)
  default     = []
}

# Fargate Configuration
variable "enable_fargate" {
  description = "Enable Fargate profiles for serverless containers"
  type        = bool
  default     = false
}

variable "fargate_profiles" {
  description = "Map of Fargate profile configurations"
  type        = any
  default     = {}
}