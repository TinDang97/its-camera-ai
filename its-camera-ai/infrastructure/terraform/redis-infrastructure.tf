# Redis Infrastructure for ITS Camera AI
# High-performance Redis Streams deployment with monitoring and backup

terraform {
  required_version = ">= 1.0"
  required_providers {
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.23"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.11"
    }
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

# Variables
variable "namespace" {
  description = "Kubernetes namespace for Redis deployment"
  type        = string
  default     = "its-camera-ai"
}

variable "redis_password" {
  description = "Redis authentication password"
  type        = string
  sensitive   = true
  default     = "ItsCarneraAIRedisPassword2024"
}

variable "storage_class" {
  description = "Storage class for Redis persistence"
  type        = string
  default     = "fast-ssd"
}

variable "redis_memory_limit" {
  description = "Redis memory limit per instance"
  type        = string
  default     = "16Gi"
}

variable "backup_retention_days" {
  description = "Number of days to retain Redis backups"
  type        = number
  default     = 30
}

variable "enable_monitoring" {
  description = "Enable Prometheus monitoring for Redis"
  type        = bool
  default     = true
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "prod"
}

# Local values
locals {
  common_labels = {
    "app.kubernetes.io/name"       = "redis-streams"
    "app.kubernetes.io/component"  = "queue-engine"
    "app.kubernetes.io/version"    = "7.2.0"
    "app.kubernetes.io/part-of"    = "its-camera-ai"
    "app.kubernetes.io/managed-by" = "terraform"
    "environment"                  = var.environment
  }
  
  redis_config = {
    maxmemory               = "12gb"
    maxmemory_policy       = "volatile-lru"
    tcp_backlog            = "1024"
    timeout                = "300"
    tcp_keepalive          = "300"
    save                   = "60 1000 300 100 900 10"
    appendonly             = "yes"
    appendfsync            = "everysec"
    stream_node_max_bytes  = "4096"
    stream_node_max_entries = "100"
    io_threads             = "4"
    io_threads_do_reads    = "yes"
  }
}

# Redis authentication secret
resource "kubernetes_secret" "redis_auth" {
  metadata {
    name      = "redis-auth"
    namespace = var.namespace
    labels    = local.common_labels
  }

  data = {
    password = base64encode(var.redis_password)
  }

  type = "Opaque"
}

# Redis configuration
resource "kubernetes_config_map" "redis_config" {
  metadata {
    name      = "redis-streams-config"
    namespace = var.namespace
    labels    = local.common_labels
  }

  data = {
    "redis.conf" = templatefile("${path.module}/templates/redis.conf.tpl", {
      redis_config = local.redis_config
    })
  }
}

# Redis StatefulSet
resource "kubernetes_stateful_set" "redis_streams" {
  metadata {
    name      = "redis-streams"
    namespace = var.namespace
    labels    = local.common_labels
  }

  spec {
    service_name = "redis-streams-headless"
    replicas     = 3

    selector {
      match_labels = {
        "app.kubernetes.io/name"      = "redis-streams"
        "app.kubernetes.io/component" = "queue-engine"
      }
    }

    template {
      metadata {
        labels = merge(local.common_labels, {
          "version" = "7.2.0"
        })
        annotations = {
          "prometheus.io/scrape" = "true"
          "prometheus.io/port"   = "9121"
          "prometheus.io/path"   = "/metrics"
          "config.checksum"      = sha256(kubernetes_config_map.redis_config.data["redis.conf"])
        }
      }

      spec {
        service_account_name = kubernetes_service_account.redis_streams.metadata[0].name

        security_context {
          run_as_non_root = true
          run_as_user     = 999
          run_as_group    = 999
          fs_group        = 999
        }

        container {
          name  = "redis-streams"
          image = "redis:7.2.0-alpine"

          command = ["redis-server"]
          args    = ["/etc/redis/redis.conf", "--port", "6379"]

          port {
            name           = "redis"
            container_port = 6379
            protocol       = "TCP"
          }

          env {
            name = "REDIS_PASSWORD"
            value_from {
              secret_key_ref {
                name = kubernetes_secret.redis_auth.metadata[0].name
                key  = "password"
              }
            }
          }

          resources {
            requests = {
              cpu    = "2"
              memory = "8Gi"
            }
            limits = {
              cpu    = "4"
              memory = var.redis_memory_limit
            }
          }

          volume_mount {
            name       = "redis-config"
            mount_path = "/etc/redis"
            read_only  = true
          }

          volume_mount {
            name       = "redis-data"
            mount_path = "/data"
          }

          volume_mount {
            name       = "tmp"
            mount_path = "/tmp"
          }

          liveness_probe {
            tcp_socket {
              port = "redis"
            }
            initial_delay_seconds = 30
            period_seconds        = 10
            timeout_seconds       = 5
            failure_threshold     = 3
          }

          readiness_probe {
            exec {
              command = ["redis-cli", "ping"]
            }
            initial_delay_seconds = 5
            period_seconds        = 5
            timeout_seconds       = 3
            failure_threshold     = 2
          }

          security_context {
            allow_privilege_escalation = false
            read_only_root_filesystem  = true
            capabilities {
              drop = ["ALL"]
            }
          }
        }

        # Redis Exporter for monitoring
        container {
          name  = "redis-exporter"
          image = "oliver006/redis_exporter:v1.56.0"

          port {
            name           = "metrics"
            container_port = 9121
            protocol       = "TCP"
          }

          env {
            name  = "REDIS_ADDR"
            value = "redis://localhost:6379"
          }

          env {
            name = "REDIS_PASSWORD"
            value_from {
              secret_key_ref {
                name = kubernetes_secret.redis_auth.metadata[0].name
                key  = "password"
              }
            }
          }

          env {
            name  = "REDIS_EXPORTER_INCL_SYSTEM_METRICS"
            value = "true"
          }

          env {
            name  = "REDIS_EXPORTER_CHECK_KEYS"
            value = "camera_frames*,processed_frames*,inference_queue*"
          }

          resources {
            requests = {
              cpu    = "100m"
              memory = "128Mi"
            }
            limits = {
              cpu    = "200m"
              memory = "256Mi"
            }
          }

          security_context {
            allow_privilege_escalation = false
            read_only_root_filesystem  = true
            capabilities {
              drop = ["ALL"]
            }
          }
        }

        volume {
          name = "redis-config"
          config_map {
            name = kubernetes_config_map.redis_config.metadata[0].name
          }
        }

        volume {
          name = "tmp"
          empty_dir {}
        }

        affinity {
          pod_anti_affinity {
            preferred_during_scheduling_ignored_during_execution {
              weight = 100
              pod_affinity_term {
                label_selector {
                  match_expressions {
                    key      = "app.kubernetes.io/name"
                    operator = "In"
                    values   = ["redis-streams"]
                  }
                }
                topology_key = "kubernetes.io/hostname"
              }
            }
          }
        }

        toleration {
          key      = "workload-type"
          operator = "Equal"
          value    = "streaming"
          effect   = "NoSchedule"
        }
      }
    }

    volume_claim_template {
      metadata {
        name = "redis-data"
        labels = merge(local.common_labels, {
          "app.kubernetes.io/component" = "storage"
        })
      }
      spec {
        access_modes       = ["ReadWriteOnce"]
        storage_class_name = var.storage_class
        resources {
          requests = {
            storage = "200Gi"
          }
        }
      }
    }
  }

  depends_on = [
    kubernetes_config_map.redis_config,
    kubernetes_secret.redis_auth
  ]
}

# Service Account
resource "kubernetes_service_account" "redis_streams" {
  metadata {
    name      = "redis-streams"
    namespace = var.namespace
    labels    = local.common_labels
  }
}

# Services
resource "kubernetes_service" "redis_streams" {
  metadata {
    name      = "redis-streams"
    namespace = var.namespace
    labels    = local.common_labels
    annotations = {
      "prometheus.io/scrape" = "true"
      "prometheus.io/port"   = "9121"
      "prometheus.io/path"   = "/metrics"
    }
  }

  spec {
    type             = "ClusterIP"
    session_affinity = "ClientIP"

    session_affinity_config {
      client_ip {
        timeout_seconds = 300
      }
    }

    selector = {
      "app.kubernetes.io/name"      = "redis-streams"
      "app.kubernetes.io/component" = "queue-engine"
    }

    port {
      name        = "redis"
      port        = 6379
      target_port = 6379
      protocol    = "TCP"
    }

    port {
      name        = "metrics"
      port        = 9121
      target_port = 9121
      protocol    = "TCP"
    }
  }
}

resource "kubernetes_service" "redis_streams_headless" {
  metadata {
    name      = "redis-streams-headless"
    namespace = var.namespace
    labels    = local.common_labels
  }

  spec {
    type       = "ClusterIP"
    cluster_ip = "None"

    selector = {
      "app.kubernetes.io/name"      = "redis-streams"
      "app.kubernetes.io/component" = "queue-engine"
    }

    port {
      name        = "redis"
      port        = 6379
      target_port = 6379
      protocol    = "TCP"
    }
  }
}

# Pod Disruption Budget
resource "kubernetes_pod_disruption_budget_v1" "redis_streams" {
  metadata {
    name      = "redis-streams-pdb"
    namespace = var.namespace
    labels    = local.common_labels
  }

  spec {
    min_available = "2"
    selector {
      match_labels = {
        "app.kubernetes.io/name"      = "redis-streams"
        "app.kubernetes.io/component" = "queue-engine"
      }
    }
  }
}

# Monitoring resources (conditional)
resource "kubernetes_manifest" "service_monitor" {
  count = var.enable_monitoring ? 1 : 0

  manifest = {
    apiVersion = "monitoring.coreos.com/v1"
    kind       = "ServiceMonitor"
    
    metadata = {
      name      = "redis-streams-monitor"
      namespace = var.namespace
      labels = merge(local.common_labels, {
        "team" = "its-camera-ai"
      })
    }

    spec = {
      selector = {
        matchLabels = {
          "app.kubernetes.io/name"      = "redis-streams"
          "app.kubernetes.io/component" = "queue-engine"
        }
      }
      endpoints = [
        {
          port           = "metrics"
          interval       = "15s"
          scrapeTimeout  = "10s"
          path          = "/metrics"
        }
      ]
    }
  }
}

# AWS S3 Bucket for Redis backups
resource "aws_s3_bucket" "redis_backups" {
  bucket = "its-camera-ai-redis-backups-${var.environment}"

  tags = {
    Name        = "Redis Backups"
    Environment = var.environment
    Component   = "redis-streams"
  }
}

resource "aws_s3_bucket_versioning" "redis_backups" {
  bucket = aws_s3_bucket.redis_backups.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "redis_backups" {
  bucket = aws_s3_bucket.redis_backups.id

  rule {
    id     = "backup_retention"
    status = "Enabled"

    expiration {
      days = var.backup_retention_days
    }

    noncurrent_version_expiration {
      noncurrent_days = 7
    }
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "redis_backups" {
  bucket = aws_s3_bucket.redis_backups.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# CronJob for Redis backups
resource "kubernetes_cron_job_v1" "redis_backup" {
  metadata {
    name      = "redis-backup"
    namespace = var.namespace
    labels    = local.common_labels
  }

  spec {
    schedule = "0 3 * * *"  # Daily at 3 AM

    job_template {
      metadata {
        labels = local.common_labels
      }

      spec {
        template {
          metadata {
            labels = local.common_labels
          }

          spec {
            restart_policy = "OnFailure"

            container {
              name  = "redis-backup"
              image = "redis:7.2.0-alpine"

              command = ["/bin/sh"]
              args = [
                "-c",
                <<-EOT
                #!/bin/sh
                set -e
                
                echo "Starting Redis backup..."
                DATE=$(date +"%Y%m%d_%H%M%S")
                
                # Connect to each Redis instance and create backup
                for i in 0 1 2; do
                  echo "Backing up redis-streams-$i..."
                  redis-cli -h redis-streams-$i.redis-streams-headless -p 6379 -a "$REDIS_PASSWORD" BGSAVE
                  
                  # Wait for backup to complete
                  while [ "$(redis-cli -h redis-streams-$i.redis-streams-headless -p 6379 -a "$REDIS_PASSWORD" PING)" != "PONG" ]; do
                    echo "Waiting for backup to complete..."
                    sleep 10
                  done
                  
                  echo "Backup completed for redis-streams-$i"
                done
                
                echo "Redis backup job completed at $DATE"
                EOT
              ]

              env {
                name = "REDIS_PASSWORD"
                value_from {
                  secret_key_ref {
                    name = kubernetes_secret.redis_auth.metadata[0].name
                    key  = "password"
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

# Outputs
output "redis_service_name" {
  description = "Redis service name for application configuration"
  value       = kubernetes_service.redis_streams.metadata[0].name
}

output "redis_service_port" {
  description = "Redis service port"
  value       = 6379
}

output "redis_connection_string" {
  description = "Redis connection string for applications"
  value       = "redis://${kubernetes_service.redis_streams.metadata[0].name}.${var.namespace}.svc.cluster.local:6379"
  sensitive   = false
}

output "backup_bucket_name" {
  description = "S3 bucket name for Redis backups"
  value       = aws_s3_bucket.redis_backups.bucket
}

output "monitoring_enabled" {
  description = "Whether monitoring is enabled"
  value       = var.enable_monitoring
}