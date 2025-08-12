"""Storage module for ITS Camera AI.

Provides MinIO S3-compatible object storage integration for:
- Video frame storage and retrieval
- ML model artifact management
- Training dataset versioning
- Backup and archival operations

Features:
- Async MinIO client with connection pooling
- Multi-part upload for large files
- Presigned URL generation
- Server-side encryption (SSE-S3)
- Object lifecycle management
- Redis caching layer
- Comprehensive error handling
"""

from .factory import StorageServiceFactory, get_model_registry, get_storage_service
from .minio_service import MinIOService
from .model_registry import MinIOModelRegistry
from .models import (
    BucketConfig,
    BucketInfo,
    CompressionType,
    DownloadRequest,
    EncryptionType,
    ListObjectsRequest,
    ObjectListItem,
    ObjectMetadata,
    ObjectType,
    StorageBackend,
    StorageConfig,
    StorageStats,
    UploadProgress,
    UploadRequest,
    UploadResponse,
)

__all__ = [
    # Services
    "MinIOService",
    "MinIOModelRegistry",
    "StorageServiceFactory",
    # Factory functions
    "get_storage_service",
    "get_model_registry",
    # Models and configurations
    "BucketConfig",
    "BucketInfo",
    "ObjectMetadata",
    "ObjectListItem",
    "StorageConfig",
    "StorageStats",
    "UploadProgress",
    "UploadRequest",
    "UploadResponse",
    "DownloadRequest",
    "ListObjectsRequest",
    # Enums
    "CompressionType",
    "EncryptionType",
    "ObjectType",
    "StorageBackend",
]
