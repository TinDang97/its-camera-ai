# ITS Camera AI - Production Environment Variables

# AWS Configuration
variable "aws_region" {
  description = "AWS region for deployment"
  type        = string
  default     = "us-west-2"
}

# Cluster Configuration
variable "cluster_name" {
  description = "Name of the EKS cluster"
  type        = string
  default     = "its-camera-ai-prod"
}

variable "kubernetes_version" {
  description = "Kubernetes version for the cluster"
  type        = string
  default     = "1.28"
}

# Networking Configuration
variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "public_subnet_cidrs" {
  description = "CIDR blocks for public subnets"
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
}

variable "private_subnet_cidrs" {
  description = "CIDR blocks for private subnets"
  type        = list(string)
  default     = ["10.0.4.0/24", "10.0.5.0/24", "10.0.6.0/24"]
}

variable "endpoint_public_access" {
  description = "Enable public API server endpoint"
  type        = bool
  default     = true
}

variable "endpoint_public_access_cidrs" {
  description = "List of CIDR blocks that can access the public API server endpoint"
  type        = list(string)
  default     = ["0.0.0.0/0"]  # Restrict this in production
}

# Logging Configuration
variable "cluster_log_types" {
  description = "List of cluster log types to enable"
  type        = list(string)
  default     = ["api", "audit", "authenticator", "controllerManager", "scheduler"]
}

variable "log_retention_days" {
  description = "Number of days to retain CloudWatch logs"
  type        = number
  default     = 90
}

# Node SSH Access
variable "node_ssh_key" {
  description = "EC2 SSH key name for node access"
  type        = string
  default     = ""  # Set this to your key pair name
}

# GPU Node Group Configuration
variable "gpu_instance_types" {
  description = "Instance types for GPU nodes"
  type        = list(string)
  default     = ["g5.2xlarge"]
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

# General Node Group Configuration
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

# Memory Node Group Configuration
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

# PostgreSQL Database Configuration
variable "postgres_instance_class" {
  description = "RDS instance class for PostgreSQL"
  type        = string
  default     = "db.r5.2xlarge"
}

variable "postgres_allocated_storage" {
  description = "Initial allocated storage for PostgreSQL (GB)"
  type        = number
  default     = 500
}

variable "postgres_max_allocated_storage" {
  description = "Maximum allocated storage for PostgreSQL (GB)"
  type        = number
  default     = 2000
}

variable "postgres_database_name" {
  description = "Name of the PostgreSQL database"
  type        = string
  default     = "its_camera_ai_prod"
}

variable "postgres_username" {
  description = "Username for PostgreSQL"
  type        = string
  default     = "its_admin"
}

variable "postgres_password" {
  description = "Password for PostgreSQL"
  type        = string
  sensitive   = true
  default     = ""  # Should be set via environment variable or AWS Secrets Manager
}

variable "postgres_backup_retention" {
  description = "Backup retention period for PostgreSQL (days)"
  type        = number
  default     = 30
}

variable "postgres_multi_az" {
  description = "Enable Multi-AZ deployment for PostgreSQL"
  type        = bool
  default     = true
}

variable "postgres_deletion_protection" {
  description = "Enable deletion protection for PostgreSQL"
  type        = bool
  default     = true
}

# Redis Configuration
variable "redis_node_type" {
  description = "ElastiCache node type for Redis"
  type        = string
  default     = "cache.r6g.2xlarge"
}

variable "redis_num_cache_clusters" {
  description = "Number of cache clusters for Redis"
  type        = number
  default     = 3
}

variable "redis_auth_token" {
  description = "Auth token for Redis"
  type        = string
  sensitive   = true
  default     = ""  # Should be set via environment variable
}

# S3 Configuration
variable "video_retention_days" {
  description = "Number of days to retain video files in S3"
  type        = number
  default     = 2555  # 7 years for compliance
}

# Load Balancer Configuration
variable "alb_deletion_protection" {
  description = "Enable deletion protection for Application Load Balancer"
  type        = bool
  default     = true
}

# CloudFront Configuration
variable "enable_cloudfront" {
  description = "Enable CloudFront distribution"
  type        = bool
  default     = true
}

variable "cloudfront_price_class" {
  description = "CloudFront price class"
  type        = string
  default     = "PriceClass_100"  # US, Canada, Europe
}

# Route53 Configuration
variable "enable_route53_health_check" {
  description = "Enable Route53 health check"
  type        = bool
  default     = true
}

# Common Tags
variable "common_tags" {
  description = "Common tags to apply to all resources"
  type        = map(string)
  default = {
    Project             = "its-camera-ai"
    Environment         = "production"
    ManagedBy          = "terraform"
    Owner              = "platform-team"
    CostCenter         = "engineering"
    Backup             = "required"
    CriticalityTier    = "tier-1"
    DataClassification = "confidential"
    Compliance         = "required"
  }
}

# Cost Optimization
variable "enable_spot_instances" {
  description = "Enable spot instances for non-critical workloads"
  type        = bool
  default     = false  # Disabled for production stability
}

# Security Configuration
variable "enable_waf" {
  description = "Enable AWS WAF for Application Load Balancer"
  type        = bool
  default     = true
}

variable "waf_rate_limit" {
  description = "Rate limit for WAF (requests per 5 minutes)"
  type        = number
  default     = 10000
}

# Monitoring Configuration
variable "enable_enhanced_monitoring" {
  description = "Enable enhanced monitoring for RDS"
  type        = bool
  default     = true
}

variable "monitoring_interval" {
  description = "Enhanced monitoring interval for RDS (seconds)"
  type        = number
  default     = 60
}

# Backup Configuration
variable "enable_automated_backups" {
  description = "Enable automated backups"
  type        = bool
  default     = true
}

variable "backup_retention_period" {
  description = "Backup retention period (days)"
  type        = number
  default     = 30
}

# Disaster Recovery Configuration
variable "enable_cross_region_backup" {
  description = "Enable cross-region backup"
  type        = bool
  default     = true
}

variable "dr_region" {
  description = "Disaster recovery region"
  type        = string
  default     = "us-east-1"
}

# Performance Configuration
variable "enable_performance_insights" {
  description = "Enable Performance Insights for RDS"
  type        = bool
  default     = true
}

variable "performance_insights_retention_period" {
  description = "Performance Insights retention period (days)"
  type        = number
  default     = 7
}

# Network Security
variable "enable_vpc_flow_logs" {
  description = "Enable VPC Flow Logs"
  type        = bool
  default     = true
}

variable "flow_logs_retention_days" {
  description = "VPC Flow Logs retention period (days)"
  type        = number
  default     = 30
}

# Compliance and Auditing
variable "enable_config" {
  description = "Enable AWS Config for compliance monitoring"
  type        = bool
  default     = true
}

variable "enable_cloudtrail" {
  description = "Enable AWS CloudTrail for audit logging"
  type        = bool
  default     = true
}

variable "cloudtrail_retention_days" {
  description = "CloudTrail log retention period (days)"
  type        = number
  default     = 365
}

# Auto-scaling Configuration
variable "enable_predictive_scaling" {
  description = "Enable predictive scaling for Auto Scaling Groups"
  type        = bool
  default     = true
}

variable "scaling_target_cpu" {
  description = "Target CPU utilization for auto-scaling"
  type        = number
  default     = 70
}

variable "scaling_target_memory" {
  description = "Target memory utilization for auto-scaling"
  type        = number
  default     = 80
}

# GPU Optimization
variable "enable_gpu_monitoring" {
  description = "Enable detailed GPU monitoring"
  type        = bool
  default     = true
}

variable "gpu_utilization_threshold" {
  description = "GPU utilization threshold for scaling"
  type        = number
  default     = 70
}

# ML Model Configuration
variable "model_storage_bucket_name" {
  description = "S3 bucket name for ML model storage"
  type        = string
  default     = ""  # Will be generated from cluster name
}

variable "enable_model_versioning" {
  description = "Enable model versioning in S3"
  type        = bool
  default     = true
}

# API Gateway Configuration
variable "api_throttle_rate" {
  description = "API throttle rate (requests per second)"
  type        = number
  default     = 1000
}

variable "api_burst_limit" {
  description = "API burst limit (requests)"
  type        = number
  default     = 2000
}

# Container Registry Configuration
variable "ecr_scan_on_push" {
  description = "Enable vulnerability scanning on ECR push"
  type        = bool
  default     = true
}

variable "ecr_lifecycle_policy" {
  description = "ECR lifecycle policy for image retention"
  type        = string
  default     = "keep_last_30_images"
}

# Secrets Management
variable "enable_secrets_manager" {
  description = "Enable AWS Secrets Manager for sensitive data"
  type        = bool
  default     = true
}

variable "secrets_rotation_days" {
  description = "Automatic secrets rotation period (days)"
  type        = number
  default     = 90
}

# Edge Computing Configuration
variable "enable_edge_locations" {
  description = "Enable edge computing locations"
  type        = bool
  default     = false  # Will be enabled in Q3
}

variable "edge_instance_types" {
  description = "Instance types for edge locations"
  type        = list(string)
  default     = ["t3.medium", "t3.large"]
}

# Multi-tenancy Configuration
variable "enable_multi_tenancy" {
  description = "Enable multi-tenancy support"
  type        = bool
  default     = false  # Will be enabled in Q2
}

variable "default_tenant_quota" {
  description = "Default resource quota per tenant"
  type        = map(string)
  default = {
    cpu_cores    = "10"
    memory_gb    = "32"
    storage_gb   = "100"
    gpu_units    = "1"
  }
}

# Feature Flags
variable "feature_flags" {
  description = "Feature flags for gradual rollout"
  type        = map(bool)
  default = {
    enable_vehicle_reid     = false  # Q3 feature
    enable_behavior_analysis = false  # Q3 feature
    enable_edge_inference   = false  # Q3 feature
    enable_federated_learning = false  # Q3 feature
    enable_api_v2          = false  # Q4 feature
    enable_mobile_sdk      = false  # Q4 feature
  }
}