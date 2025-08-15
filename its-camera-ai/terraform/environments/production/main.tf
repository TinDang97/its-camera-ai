# ITS Camera AI - Production Environment
# This configuration deploys the production Kubernetes cluster and supporting infrastructure

terraform {
  required_version = ">= 1.6"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.23"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.11"
    }
  }

  # Backend configuration for state management
  backend "s3" {
    bucket         = "its-camera-ai-terraform-state"
    key            = "production/terraform.tfstate"
    region         = "us-west-2"
    encrypt        = true
    dynamodb_table = "its-camera-ai-terraform-locks"
  }
}

# Configure AWS Provider
provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "its-camera-ai"
      Environment = "production"
      ManagedBy   = "terraform"
      Owner       = "platform-team"
      CostCenter  = "engineering"
      Backup      = "required"
    }
  }
}

# Data sources
data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

# Production EKS Cluster
module "production_cluster" {
  source = "../../modules/kubernetes-cluster"

  # Basic Configuration
  cluster_name       = var.cluster_name
  kubernetes_version = var.kubernetes_version
  
  # Networking Configuration
  vpc_cidr             = var.vpc_cidr
  public_subnet_cidrs  = var.public_subnet_cidrs
  private_subnet_cidrs = var.private_subnet_cidrs
  
  # API Server Configuration
  endpoint_public_access       = var.endpoint_public_access
  endpoint_public_access_cidrs = var.endpoint_public_access_cidrs
  
  # Logging and Monitoring
  cluster_log_types   = var.cluster_log_types
  log_retention_days  = var.log_retention_days
  
  # Node SSH Access
  node_ssh_key = var.node_ssh_key
  
  # GPU Node Group for AI Inference
  enable_gpu_nodes        = true
  gpu_instance_types      = var.gpu_instance_types
  gpu_nodes_desired_size  = var.gpu_nodes_desired_size
  gpu_nodes_min_size      = var.gpu_nodes_min_size
  gpu_nodes_max_size      = var.gpu_nodes_max_size
  
  # General Purpose Node Group
  general_instance_types      = var.general_instance_types
  general_nodes_desired_size  = var.general_nodes_desired_size
  general_nodes_min_size      = var.general_nodes_min_size
  general_nodes_max_size      = var.general_nodes_max_size
  
  # Memory-Optimized Node Group for Redis/Kafka
  enable_memory_nodes        = true
  memory_instance_types      = var.memory_instance_types
  memory_nodes_desired_size  = var.memory_nodes_desired_size
  memory_nodes_min_size      = var.memory_nodes_min_size
  memory_nodes_max_size      = var.memory_nodes_max_size
  
  # Security Features
  enable_encryption_at_rest = true
  enable_logging           = true
  enable_private_endpoint  = true
  enable_irsa             = true
  
  # Monitoring Features
  enable_container_insights     = true
  enable_prometheus_metrics     = true
  enable_performance_monitoring = true
  
  # Auto-scaling Features
  enable_cluster_autoscaler = true
  enable_metrics_server     = true
  
  # Storage Configuration
  ebs_storage_class = "gp3"
  ebs_volume_size   = 200
  ebs_volume_type   = "gp3"
  
  # GPU Optimizations
  nvidia_device_plugin_version = "v0.14.1"
  gpu_operator_version         = "v23.9.0"
  enable_nvme_ssd             = true
  
  # Common Tags
  common_tags = merge(var.common_tags, {
    Environment     = "production"
    CriticalityTier = "tier-1"
    DataClassification = "confidential"
  })
}

# S3 Bucket for ML Model Storage
resource "aws_s3_bucket" "ml_models" {
  bucket = "${var.cluster_name}-ml-models"
}

resource "aws_s3_bucket_versioning" "ml_models" {
  bucket = aws_s3_bucket.ml_models.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_encryption" "ml_models" {
  bucket = aws_s3_bucket.ml_models.id

  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        sse_algorithm = "AES256"
      }
    }
  }
}

resource "aws_s3_bucket_public_access_block" "ml_models" {
  bucket = aws_s3_bucket.ml_models.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# S3 Bucket for Video Storage
resource "aws_s3_bucket" "video_storage" {
  bucket = "${var.cluster_name}-video-storage"
}

resource "aws_s3_bucket_versioning" "video_storage" {
  bucket = aws_s3_bucket.video_storage.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_encryption" "video_storage" {
  bucket = aws_s3_bucket.video_storage.id

  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        sse_algorithm = "AES256"
      }
    }
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "video_storage" {
  bucket = aws_s3_bucket.video_storage.id

  rule {
    id     = "video_lifecycle"
    status = "Enabled"

    transition {
      days          = 30
      storage_class = "STANDARD_IA"
    }

    transition {
      days          = 90
      storage_class = "GLACIER"
    }

    transition {
      days          = 365
      storage_class = "DEEP_ARCHIVE"
    }

    expiration {
      days = var.video_retention_days
    }
  }
}

# RDS for PostgreSQL Primary Database
resource "aws_db_subnet_group" "postgres" {
  name       = "${var.cluster_name}-postgres-subnet-group"
  subnet_ids = module.production_cluster.private_subnets

  tags = merge(var.common_tags, {
    Name = "${var.cluster_name}-postgres-subnet-group"
  })
}

resource "aws_security_group" "postgres" {
  name_prefix = "${var.cluster_name}-postgres-"
  vpc_id      = module.production_cluster.vpc_id

  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [module.production_cluster.node_security_group_id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(var.common_tags, {
    Name = "${var.cluster_name}-postgres-sg"
  })
}

resource "aws_db_instance" "postgres_primary" {
  identifier = "${var.cluster_name}-postgres-primary"

  # Database Configuration
  engine         = "postgres"
  engine_version = "15.4"
  instance_class = var.postgres_instance_class
  
  # Storage Configuration
  allocated_storage     = var.postgres_allocated_storage
  max_allocated_storage = var.postgres_max_allocated_storage
  storage_type          = "gp3"
  storage_encrypted     = true
  
  # Database Settings
  db_name  = var.postgres_database_name
  username = var.postgres_username
  password = var.postgres_password  # Should use AWS Secrets Manager in production
  
  # Networking
  db_subnet_group_name   = aws_db_subnet_group.postgres.name
  vpc_security_group_ids = [aws_security_group.postgres.id]
  publicly_accessible    = false
  
  # Backup and Maintenance
  backup_retention_period = var.postgres_backup_retention
  backup_window          = "07:00-08:00"
  maintenance_window     = "sun:08:00-sun:09:00"
  
  # Performance Insights
  performance_insights_enabled = true
  performance_insights_retention_period = 7
  
  # Monitoring
  monitoring_interval = 60
  monitoring_role_arn = aws_iam_role.rds_monitoring.arn
  
  # High Availability
  multi_az = var.postgres_multi_az
  
  # Delete Protection
  deletion_protection = var.postgres_deletion_protection
  
  tags = merge(var.common_tags, {
    Name = "${var.cluster_name}-postgres-primary"
    Type = "primary-database"
  })
}

# IAM Role for RDS Enhanced Monitoring
resource "aws_iam_role" "rds_monitoring" {
  name = "${var.cluster_name}-rds-monitoring-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "monitoring.rds.amazonaws.com"
        }
      }
    ]
  })

  tags = var.common_tags
}

resource "aws_iam_role_policy_attachment" "rds_monitoring" {
  role       = aws_iam_role.rds_monitoring.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonRDSEnhancedMonitoringRole"
}

# ElastiCache Redis Cluster for Caching
resource "aws_elasticache_subnet_group" "redis" {
  name       = "${var.cluster_name}-redis-subnet-group"
  subnet_ids = module.production_cluster.private_subnets
}

resource "aws_security_group" "redis" {
  name_prefix = "${var.cluster_name}-redis-"
  vpc_id      = module.production_cluster.vpc_id

  ingress {
    from_port       = 6379
    to_port         = 6379
    protocol        = "tcp"
    security_groups = [module.production_cluster.node_security_group_id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(var.common_tags, {
    Name = "${var.cluster_name}-redis-sg"
  })
}

resource "aws_elasticache_replication_group" "redis" {
  replication_group_id       = "${var.cluster_name}-redis"
  description                = "Redis cluster for ITS Camera AI"
  
  node_type                 = var.redis_node_type
  port                      = 6379
  parameter_group_name      = "default.redis7"
  
  num_cache_clusters        = var.redis_num_cache_clusters
  
  # Security
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  auth_token                 = var.redis_auth_token
  
  # Networking
  subnet_group_name = aws_elasticache_subnet_group.redis.name
  security_group_ids = [aws_security_group.redis.id]
  
  # Backup
  snapshot_retention_limit = 7
  snapshot_window         = "07:00-09:00"
  
  # Maintenance
  maintenance_window = "sun:10:00-sun:12:00"
  
  tags = merge(var.common_tags, {
    Name = "${var.cluster_name}-redis"
  })
}

# Application Load Balancer for external access
resource "aws_security_group" "alb" {
  name_prefix = "${var.cluster_name}-alb-"
  vpc_id      = module.production_cluster.vpc_id

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(var.common_tags, {
    Name = "${var.cluster_name}-alb-sg"
  })
}

resource "aws_lb" "main" {
  name               = "${var.cluster_name}-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = module.production_cluster.public_subnets

  enable_deletion_protection = var.alb_deletion_protection

  tags = merge(var.common_tags, {
    Name = "${var.cluster_name}-alb"
  })
}

# WAF for Application Load Balancer
resource "aws_wafv2_web_acl" "main" {
  name  = "${var.cluster_name}-waf"
  scope = "REGIONAL"

  default_action {
    allow {}
  }

  # Rate limiting rule
  rule {
    name     = "rate-limit"
    priority = 1

    action {
      block {}
    }

    statement {
      rate_based_statement {
        limit              = 10000
        aggregate_key_type = "IP"
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "${var.cluster_name}-rate-limit"
      sampled_requests_enabled   = true
    }
  }

  # AWS Managed Core Rule Set
  rule {
    name     = "aws-core-rule-set"
    priority = 2

    statement {
      managed_rule_group_statement {
        name        = "AWSManagedRulesCore​RuleSet"
        vendor_name = "AWS"
      }
    }

    override_action {
      none {}
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "${var.cluster_name}-core-rules"
      sampled_requests_enabled   = true
    }
  }

  tags = var.common_tags

  visibility_config {
    cloudwatch_metrics_enabled = true
    metric_name                = "${var.cluster_name}-waf"
    sampled_requests_enabled   = true
  }
}

# Associate WAF with ALB
resource "aws_wafv2_web_acl_association" "main" {
  resource_arn = aws_lb.main.arn
  web_acl_arn  = aws_wafv2_web_acl.main.arn
}

# CloudFront Distribution for global content delivery
resource "aws_cloudfront_distribution" "main" {
  count = var.enable_cloudfront ? 1 : 0

  origin {
    domain_name = aws_lb.main.dns_name
    origin_id   = "${var.cluster_name}-alb"

    custom_origin_config {
      http_port              = 80
      https_port             = 443
      origin_protocol_policy = "https-only"
      origin_ssl_protocols   = ["TLSv1.2"]
    }
  }

  enabled = true
  comment = "ITS Camera AI Production Distribution"

  default_cache_behavior {
    allowed_methods        = ["DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"]
    cached_methods         = ["GET", "HEAD"]
    target_origin_id       = "${var.cluster_name}-alb"
    compress               = true
    viewer_protocol_policy = "redirect-to-https"

    forwarded_values {
      query_string = true
      headers      = ["Host", "Authorization", "CloudFront-Forwarded-Proto"]

      cookies {
        forward = "none"
      }
    }

    min_ttl     = 0
    default_ttl = 3600
    max_ttl     = 86400
  }

  # Cache behavior for API endpoints
  ordered_cache_behavior {
    path_pattern           = "/api/*"
    allowed_methods        = ["DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"]
    cached_methods         = ["GET", "HEAD"]
    target_origin_id       = "${var.cluster_name}-alb"
    compress               = true
    viewer_protocol_policy = "https-only"

    forwarded_values {
      query_string = true
      headers      = ["*"]

      cookies {
        forward = "all"
      }
    }

    min_ttl     = 0
    default_ttl = 0
    max_ttl     = 0
  }

  # Cache behavior for streaming content
  ordered_cache_behavior {
    path_pattern           = "/streaming/*"
    allowed_methods        = ["GET", "HEAD", "OPTIONS"]
    cached_methods         = ["GET", "HEAD"]
    target_origin_id       = "${var.cluster_name}-alb"
    compress               = false
    viewer_protocol_policy = "https-only"

    forwarded_values {
      query_string = true
      headers      = ["Range"]

      cookies {
        forward = "none"
      }
    }

    min_ttl     = 0
    default_ttl = 86400
    max_ttl     = 31536000
  }

  price_class = var.cloudfront_price_class

  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }

  viewer_certificate {
    cloudfront_default_certificate = true
  }

  tags = merge(var.common_tags, {
    Name = "${var.cluster_name}-cloudfront"
  })
}

# Route53 Health Check for monitoring
resource "aws_route53_health_check" "main" {
  count = var.enable_route53_health_check ? 1 : 0

  fqdn                            = aws_lb.main.dns_name
  port                            = 443
  type                            = "HTTPS"
  resource_path                   = "/health"
  failure_threshold               = 3
  request_interval                = 30
  cloudwatch_alarm_region         = var.aws_region
  cloudwatch_alarm_name           = "${var.cluster_name}-health-check"
  insufficient_data_health_status = "Failure"

  tags = merge(var.common_tags, {
    Name = "${var.cluster_name}-health-check"
  })
}