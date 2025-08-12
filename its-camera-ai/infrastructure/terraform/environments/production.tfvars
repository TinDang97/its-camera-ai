# Production Environment Configuration for MinIO
environment         = "production"
namespace           = "its-camera-ai"
replica_count       = 8
storage_size_per_node = "500Gi"
drives_per_node     = 4
storage_class       = "fast-ssd"
enable_monitoring   = true
enable_backup       = true
backup_retention_days = 30
ingress_hostname    = "minio.its-camera-ai.com"
console_hostname    = "minio-console.its-camera-ai.com"