# Staging Environment Configuration for MinIO
environment         = "staging"
namespace           = "its-camera-ai-staging"
replica_count       = 6
storage_size_per_node = "250Gi"
drives_per_node     = 3
storage_class       = "fast-ssd"
enable_monitoring   = true
enable_backup       = true
backup_retention_days = 14
ingress_hostname    = "minio-staging.its-camera-ai.local"
console_hostname    = "minio-console-staging.its-camera-ai.local"