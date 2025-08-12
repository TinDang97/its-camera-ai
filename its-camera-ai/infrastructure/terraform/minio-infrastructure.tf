# MinIO Infrastructure for ITS Camera AI
# High-performance distributed object storage with monitoring and backup

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
    random = {
      source  = "hashicorp/random"
      version = "~> 3.5"
    }
  }
}

# Variables
variable "namespace" {
  description = "Kubernetes namespace for MinIO deployment"
  type        = string
  default     = "its-camera-ai"
}

variable "minio_root_password" {
  description = "MinIO root password"
  type        = string
  sensitive   = true
  default     = "ItsCAI2024SecureMinIOPass!"
}

variable "storage_class" {
  description = "Storage class for MinIO persistence"
  type        = string
  default     = "fast-ssd"
}

variable "replica_count" {
  description = "Number of MinIO replicas for distributed mode"
  type        = number
  default     = 4
  validation {
    condition     = var.replica_count >= 4 && var.replica_count <= 32
    error_message = "MinIO replica count must be between 4 and 32 for distributed mode."
  }
}

variable "storage_size_per_node" {
  description = "Storage size per MinIO node"
  type        = string
  default     = "500Gi"
}

variable "drives_per_node" {
  description = "Number of drives per MinIO node"
  type        = number
  default     = 4
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "prod"
}

variable "enable_monitoring" {
  description = "Enable Prometheus monitoring for MinIO"
  type        = bool
  default     = true
}

variable "enable_backup" {
  description = "Enable S3 backup for MinIO"
  type        = bool
  default     = true
}

variable "backup_retention_days" {
  description = "Number of days to retain MinIO backups"
  type        = number
  default     = 30
}

variable "ingress_hostname" {
  description = "Hostname for MinIO ingress"
  type        = string
  default     = "minio.its-camera-ai.local"
}

variable "console_hostname" {
  description = "Hostname for MinIO console ingress"
  type        = string
  default     = "minio-console.its-camera-ai.local"
}

# Local values
locals {
  common_labels = {
    "app.kubernetes.io/name"       = "minio"
    "app.kubernetes.io/component"  = "object-storage"
    "app.kubernetes.io/version"    = "RELEASE.2024-08-17T01-24-54Z"
    "app.kubernetes.io/part-of"    = "its-camera-ai"
    "app.kubernetes.io/managed-by" = "terraform"
    "environment"                  = var.environment
  }

  # Calculate total storage across all nodes
  total_storage_bytes = tonumber(regex("(\\d+)", var.storage_size_per_node)[0]) * (
    regex("Ti", var.storage_size_per_node) != null ? 1099511627776 : 
    regex("Gi", var.storage_size_per_node) != null ? 1073741824 : 1
  ) * var.replica_count * var.drives_per_node

  # MinIO distributed endpoints
  minio_distributed_endpoints = join(" ", [
    for i in range(var.replica_count) : 
    "http://minio-${i}.minio-headless.${var.namespace}.svc.cluster.local:9000/data{1...${var.drives_per_node}}"
  ])
}

# Random password generation for service accounts
resource "random_password" "minio_service_accounts" {
  for_each = toset(["stream-processor", "ml-service", "analytics-service", "backup-service"])
  length   = 32
  special  = true
}

# MinIO namespace
resource "kubernetes_namespace" "minio" {
  metadata {
    name = var.namespace
    labels = merge(local.common_labels, {
      "pod-security.kubernetes.io/enforce" = "restricted"
      "pod-security.kubernetes.io/audit"   = "restricted"
      "pod-security.kubernetes.io/warn"    = "restricted"
    })
  }
}

# MinIO credentials secret
resource "kubernetes_secret" "minio_credentials" {
  metadata {
    name      = "minio-credentials"
    namespace = kubernetes_namespace.minio.metadata[0].name
    labels    = local.common_labels
  }

  data = {
    root-user                = "its_camera_ai_admin"
    root-password           = var.minio_root_password
    service-account-key     = random_password.minio_service_accounts["stream-processor"].result
    ml-service-key          = random_password.minio_service_accounts["ml-service"].result
    analytics-service-key   = random_password.minio_service_accounts["analytics-service"].result
    backup-service-key      = random_password.minio_service_accounts["backup-service"].result
  }

  type = "Opaque"
}

# MinIO configuration
resource "kubernetes_config_map" "minio_config" {
  metadata {
    name      = "minio-config"
    namespace = kubernetes_namespace.minio.metadata[0].name
    labels    = local.common_labels
  }

  data = {
    # Performance and optimization settings
    MINIO_STORAGE_CLASS_STANDARD = "EC:2"
    MINIO_CACHE_QUOTA           = "80"
    MINIO_CACHE_AFTER           = "3"
    MINIO_CACHE_WATERMARK_LOW   = "70"
    MINIO_CACHE_WATERMARK_HIGH  = "90"
    MINIO_API_REQUESTS_MAX      = "10000"
    MINIO_API_REQUESTS_DEADLINE = "10s"
    MINIO_API_CLUSTER_DEADLINE  = "10s"
    
    # Distributed mode settings
    MINIO_DISTRIBUTED_MODE_ENABLED = "yes"
    MINIO_DISTRIBUTED_NODES        = local.minio_distributed_endpoints
    
    # Security and audit
    MINIO_BROWSER_REDIRECT_URL                = "https://${var.console_hostname}"
    MINIO_SERVER_URL                         = "https://${var.ingress_hostname}"
    MINIO_AUDIT_WEBHOOK_ENABLE_target1       = "on"
    MINIO_AUDIT_WEBHOOK_ENDPOINT_target1     = "http://api-service.${var.namespace}.svc.cluster.local:8000/api/v1/audit/minio"
    MINIO_AUDIT_WEBHOOK_AUTH_TOKEN_target1   = "minio-audit-token"
    
    # Default buckets configuration
    MINIO_DEFAULT_BUCKETS = "camera-streams,ml-models,analytics,backups,temp"
  }
}

# MinIO StatefulSet
resource "kubernetes_stateful_set" "minio" {
  metadata {
    name      = "minio"
    namespace = kubernetes_namespace.minio.metadata[0].name
    labels    = local.common_labels
  }

  spec {
    service_name = "minio-headless"
    replicas     = var.replica_count

    update_strategy {
      type = "RollingUpdate"
      rolling_update {
        partition = 0
      }
    }

    selector {
      match_labels = {
        "app.kubernetes.io/name"      = "minio"
        "app.kubernetes.io/component" = "object-storage"
      }
    }

    template {
      metadata {
        labels = merge(local.common_labels, {
          "version" = "RELEASE.2024-08-17T01-24-54Z"
        })
        annotations = {
          "prometheus.io/scrape" = "true"
          "prometheus.io/port"   = "9000"
          "prometheus.io/path"   = "/minio/v2/metrics/cluster"
          "config.checksum"      = sha256(jsonencode(kubernetes_config_map.minio_config.data))
        }
      }

      spec {
        service_account_name = kubernetes_service_account.minio.metadata[0].name

        security_context {
          run_as_non_root        = true
          run_as_user           = 1000
          run_as_group          = 1000
          fs_group              = 1000
          fs_group_change_policy = "OnRootMismatch"
        }

        container {
          name  = "minio"
          image = "quay.io/minio/minio:RELEASE.2024-08-17T01-24-54Z"

          args = [
            "server",
            "/data{1...${var.drives_per_node}}",
            "--console-address", ":9001",
            "--address", ":9000"
          ]

          port {
            name           = "http-api"
            container_port = 9000
            protocol       = "TCP"
          }

          port {
            name           = "http-console"
            container_port = 9001
            protocol       = "TCP"
          }

          env_from {
            config_map_ref {
              name = kubernetes_config_map.minio_config.metadata[0].name
            }
          }

          env_from {
            secret_ref {
              name = kubernetes_secret.minio_credentials.metadata[0].name
            }
          }

          env {
            name  = "MINIO_ROOT_USER"
            value_from {
              secret_key_ref {
                name = kubernetes_secret.minio_credentials.metadata[0].name
                key  = "root-user"
              }
            }
          }

          env {
            name  = "MINIO_ROOT_PASSWORD"
            value_from {
              secret_key_ref {
                name = kubernetes_secret.minio_credentials.metadata[0].name
                key  = "root-password"
              }
            }
          }

          resources {
            requests = {
              cpu               = "1000m"
              memory            = "2Gi"
              ephemeral-storage = "1Gi"
            }
            limits = {
              cpu               = "2000m"
              memory            = "4Gi"
              ephemeral-storage = "10Gi"
            }
          }

          volume_mount {
            name       = "data1"
            mount_path = "/data1"
          }

          volume_mount {
            name       = "data2"
            mount_path = "/data2"
          }

          volume_mount {
            name       = "data3"
            mount_path = "/data3"
          }

          volume_mount {
            name       = "data4"
            mount_path = "/data4"
          }

          volume_mount {
            name       = "tmp"
            mount_path = "/tmp"
          }

          liveness_probe {
            http_get {
              path   = "/minio/health/live"
              port   = 9000
              scheme = "HTTP"
            }
            initial_delay_seconds = 120
            period_seconds        = 15
            timeout_seconds       = 10
            failure_threshold     = 3
            success_threshold     = 1
          }

          readiness_probe {
            http_get {
              path   = "/minio/health/ready"
              port   = 9000
              scheme = "HTTP"
            }
            initial_delay_seconds = 30
            period_seconds        = 5
            timeout_seconds       = 5
            failure_threshold     = 3
            success_threshold     = 1
          }

          startup_probe {
            http_get {
              path   = "/minio/health/live"
              port   = 9000
              scheme = "HTTP"
            }
            initial_delay_seconds = 30
            period_seconds        = 10
            timeout_seconds       = 5
            failure_threshold     = 30
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
          name = "tmp"
          empty_dir {
            size_limit = "1Gi"
          }
        }

        affinity {
          pod_anti_affinity {
            required_during_scheduling_ignored_during_execution {
              label_selector {
                match_labels = {
                  "app.kubernetes.io/name"      = "minio"
                  "app.kubernetes.io/component" = "object-storage"
                }
              }
              topology_key = "kubernetes.io/hostname"
            }
          }

          node_affinity {
            preferred_during_scheduling_ignored_during_execution {
              weight = 100
              preference {
                match_expressions {
                  key      = "workload-type"
                  operator = "In"
                  values   = ["storage"]
                }
              }
            }
          }
        }

        toleration {
          key      = "workload-type"
          operator = "Equal"
          value    = "storage"
          effect   = "NoSchedule"
        }

        toleration {
          key      = "storage-tier"
          operator = "Equal"
          value    = "high-performance"
          effect   = "NoSchedule"
        }

        topology_spread_constraint {
          max_skew           = 1
          topology_key       = "topology.kubernetes.io/zone"
          when_unsatisfiable = "DoNotSchedule"
          label_selector {
            match_labels = {
              "app.kubernetes.io/name"      = "minio"
              "app.kubernetes.io/component" = "object-storage"
            }
          }
        }
      }
    }

    dynamic "volume_claim_template" {
      for_each = range(var.drives_per_node)
      content {
        metadata {
          name = "data${volume_claim_template.value + 1}"
          labels = merge(local.common_labels, {
            "app.kubernetes.io/component" = "storage"
            "drive-index"                 = tostring(volume_claim_template.value + 1)
          })
        }
        spec {
          access_modes       = ["ReadWriteOnce"]
          storage_class_name = var.storage_class
          resources {
            requests = {
              storage = var.storage_size_per_node
            }
          }
        }
      }
    }
  }

  depends_on = [
    kubernetes_config_map.minio_config,
    kubernetes_secret.minio_credentials
  ]
}

# Service Account
resource "kubernetes_service_account" "minio" {
  metadata {
    name      = "minio-service-account"
    namespace = kubernetes_namespace.minio.metadata[0].name
    labels    = local.common_labels
  }
  automount_service_account_token = false
}

# Services
resource "kubernetes_service" "minio" {
  metadata {
    name      = "minio-service"
    namespace = kubernetes_namespace.minio.metadata[0].name
    labels    = local.common_labels
    annotations = {
      "prometheus.io/scrape"                      = "true"
      "prometheus.io/port"                        = "9000"
      "prometheus.io/path"                        = "/minio/v2/metrics/cluster"
      "service.beta.kubernetes.io/aws-load-balancer-type" = "nlb"
    }
  }

  spec {
    type             = "LoadBalancer"
    session_affinity = "ClientIP"

    session_affinity_config {
      client_ip {
        timeout_seconds = 300
      }
    }

    selector = {
      "app.kubernetes.io/name"      = "minio"
      "app.kubernetes.io/component" = "object-storage"
    }

    port {
      name        = "http-api"
      port        = 9000
      target_port = 9000
      protocol    = "TCP"
    }

    port {
      name        = "http-console"
      port        = 9001
      target_port = 9001
      protocol    = "TCP"
    }
  }
}

resource "kubernetes_service" "minio_headless" {
  metadata {
    name      = "minio-headless"
    namespace = kubernetes_namespace.minio.metadata[0].name
    labels    = local.common_labels
  }

  spec {
    type       = "ClusterIP"
    cluster_ip = "None"

    selector = {
      "app.kubernetes.io/name"      = "minio"
      "app.kubernetes.io/component" = "object-storage"
    }

    port {
      name        = "http-api"
      port        = 9000
      target_port = 9000
      protocol    = "TCP"
    }

    port {
      name        = "http-console"
      port        = 9001
      target_port = 9001
      protocol    = "TCP"
    }
  }
}

# Pod Disruption Budget
resource "kubernetes_pod_disruption_budget_v1" "minio" {
  metadata {
    name      = "minio-pdb"
    namespace = kubernetes_namespace.minio.metadata[0].name
    labels    = local.common_labels
  }

  spec {
    min_available = "50%"
    selector {
      match_labels = {
        "app.kubernetes.io/name"      = "minio"
        "app.kubernetes.io/component" = "object-storage"
      }
    }
  }
}

# Horizontal Pod Autoscaler
resource "kubernetes_horizontal_pod_autoscaler_v2" "minio" {
  metadata {
    name      = "minio-hpa"
    namespace = kubernetes_namespace.minio.metadata[0].name
    labels    = local.common_labels
  }

  spec {
    scale_target_ref {
      api_version = "apps/v1"
      kind        = "StatefulSet"
      name        = kubernetes_stateful_set.minio.metadata[0].name
    }

    min_replicas = var.replica_count
    max_replicas = var.replica_count * 2

    behavior {
      scale_down {
        stabilization_window_seconds = 300
        policy {
          type   = "Percent"
          value  = 25
          period_seconds = 60
        }
      }
      scale_up {
        stabilization_window_seconds = 60
        policy {
          type   = "Percent"
          value  = 50
          period_seconds = 60
        }
        policy {
          type   = "Pods"
          value  = 2
          period_seconds = 60
        }
        select_policy = "Max"
      }
    }

    metric {
      type = "Resource"
      resource {
        name = "cpu"
        target {
          type                = "Utilization"
          average_utilization = 70
        }
      }
    }

    metric {
      type = "Resource"
      resource {
        name = "memory"
        target {
          type                = "Utilization"
          average_utilization = 80
        }
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
      name      = "minio-metrics"
      namespace = kubernetes_namespace.minio.metadata[0].name
      labels = merge(local.common_labels, {
        "team" = "its-camera-ai"
      })
    }

    spec = {
      selector = {
        matchLabels = {
          "app.kubernetes.io/name"      = "minio"
          "app.kubernetes.io/component" = "object-storage"
        }
      }
      endpoints = [
        {
          port           = "http-api"
          path           = "/minio/v2/metrics/cluster"
          interval       = "30s"
          scrapeTimeout  = "15s"
          scheme         = "http"
          metricRelabelings = [
            {
              sourceLabels = ["__name__"]
              regex        = "minio_.*"
              action       = "keep"
            }
          ]
        }
      ]
    }
  }
}

# AWS S3 Bucket for MinIO backups (conditional)
resource "aws_s3_bucket" "minio_backups" {
  count  = var.enable_backup ? 1 : 0
  bucket = "its-camera-ai-minio-backups-${var.environment}"

  tags = {
    Name        = "MinIO Backups"
    Environment = var.environment
    Component   = "object-storage"
  }
}

resource "aws_s3_bucket_versioning" "minio_backups" {
  count  = var.enable_backup ? 1 : 0
  bucket = aws_s3_bucket.minio_backups[0].id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "minio_backups" {
  count  = var.enable_backup ? 1 : 0
  bucket = aws_s3_bucket.minio_backups[0].id

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

resource "aws_s3_bucket_server_side_encryption_configuration" "minio_backups" {
  count  = var.enable_backup ? 1 : 0
  bucket = aws_s3_bucket.minio_backups[0].id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# Outputs
output "minio_service_name" {
  description = "MinIO service name for application configuration"
  value       = kubernetes_service.minio.metadata[0].name
}

output "minio_api_port" {
  description = "MinIO API port"
  value       = 9000
}

output "minio_console_port" {
  description = "MinIO Console port"
  value       = 9001
}

output "minio_connection_string" {
  description = "MinIO connection string for applications"
  value       = "http://${kubernetes_service.minio.metadata[0].name}.${kubernetes_namespace.minio.metadata[0].name}.svc.cluster.local:9000"
  sensitive   = false
}

output "minio_console_url" {
  description = "MinIO Console URL"
  value       = "http://${kubernetes_service.minio.metadata[0].name}.${kubernetes_namespace.minio.metadata[0].name}.svc.cluster.local:9001"
  sensitive   = false
}

output "total_storage_capacity" {
  description = "Total storage capacity across all MinIO nodes"
  value       = "${var.replica_count * var.drives_per_node * tonumber(regex("(\\d+)", var.storage_size_per_node)[0])}${regex("[A-Za-z]+", var.storage_size_per_node)[0]}"
}

output "backup_bucket_name" {
  description = "S3 bucket name for MinIO backups"
  value       = var.enable_backup ? aws_s3_bucket.minio_backups[0].bucket : "N/A - Backup disabled"
}

output "monitoring_enabled" {
  description = "Whether monitoring is enabled"
  value       = var.enable_monitoring
}

output "service_account_credentials" {
  description = "Service account credentials for MinIO access"
  value = {
    for k, v in random_password.minio_service_accounts : k => {
      username = k
      password = v.result
    }
  }
  sensitive = true
}