# Development Environment Configuration for MinIO
environment         = "development"
namespace           = "its-camera-ai-dev"
replica_count       = 4
storage_size_per_node = "100Gi"
drives_per_node     = 2
storage_class       = "standard"
enable_monitoring   = true
enable_backup       = false
backup_retention_days = 7
ingress_hostname    = "minio-dev.its-camera-ai.local"
console_hostname    = "minio-console-dev.its-camera-ai.local"