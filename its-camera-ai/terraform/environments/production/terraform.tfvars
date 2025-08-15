# ITS Camera AI Production Configuration

# AWS Configuration
aws_region = "us-west-2"

# Cluster Configuration
cluster_name       = "its-camera-ai-prod"
kubernetes_version = "1.28"

# Networking Configuration
vpc_cidr             = "10.0.0.0/16"
public_subnet_cidrs  = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
private_subnet_cidrs = ["10.0.4.0/24", "10.0.5.0/24", "10.0.6.0/24"]

# API Server Access
endpoint_public_access       = true
endpoint_public_access_cidrs = ["0.0.0.0/0"]  # Restrict this for production

# Logging Configuration
cluster_log_types   = ["api", "audit", "authenticator", "controllerManager", "scheduler"]
log_retention_days  = 90

# GPU Node Group Configuration (Production Optimized)
gpu_instance_types     = ["g5.2xlarge"]
gpu_nodes_desired_size = 3
gpu_nodes_min_size     = 2
gpu_nodes_max_size     = 20

# General Node Group Configuration
general_instance_types     = ["c5.4xlarge"]
general_nodes_desired_size = 5
general_nodes_min_size     = 3
general_nodes_max_size     = 50

# Memory Node Group Configuration
memory_instance_types     = ["r5.2xlarge"]
memory_nodes_desired_size = 3
memory_nodes_min_size     = 2
memory_nodes_max_size     = 10

# Database Configuration
postgres_instance_class        = "db.r5.2xlarge"
postgres_allocated_storage     = 500
postgres_max_allocated_storage = 2000
postgres_database_name         = "its_camera_ai_prod"
postgres_username             = "its_admin"
postgres_backup_retention     = 30
postgres_multi_az             = true
postgres_deletion_protection  = true

# Redis Configuration
redis_node_type          = "cache.r6g.2xlarge"
redis_num_cache_clusters = 3

# S3 Configuration
video_retention_days = 2555  # 7 years for compliance

# Load Balancer Configuration
alb_deletion_protection = true

# CloudFront Configuration
enable_cloudfront      = true
cloudfront_price_class = "PriceClass_100"

# Route53 Configuration
enable_route53_health_check = true

# Security Configuration
enable_waf     = true
waf_rate_limit = 10000

# Monitoring Configuration
enable_enhanced_monitoring = true
monitoring_interval        = 60

# Backup Configuration
enable_automated_backups   = true
backup_retention_period    = 30
enable_cross_region_backup = true
dr_region                  = "us-east-1"

# Performance Configuration
enable_performance_insights             = true
performance_insights_retention_period   = 7

# Network Security
enable_vpc_flow_logs     = true
flow_logs_retention_days = 30

# Compliance and Auditing
enable_config               = true
enable_cloudtrail           = true
cloudtrail_retention_days   = 365

# Auto-scaling Configuration
enable_predictive_scaling = true
scaling_target_cpu        = 70
scaling_target_memory     = 80

# GPU Optimization
enable_gpu_monitoring       = true
gpu_utilization_threshold   = 70

# API Gateway Configuration
api_throttle_rate = 1000
api_burst_limit   = 2000

# Container Registry Configuration
ecr_scan_on_push       = true
ecr_lifecycle_policy   = "keep_last_30_images"

# Secrets Management
enable_secrets_manager = true
secrets_rotation_days  = 90

# Cost Optimization
enable_spot_instances = false  # Disabled for production stability

# Common Tags
common_tags = {
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