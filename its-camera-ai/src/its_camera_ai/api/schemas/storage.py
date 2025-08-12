"""Pydantic schemas for storage API endpoints.

Defines request/response models for media upload, retrieval, and management
operations with comprehensive validation and documentation.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from ...storage.models import ObjectType


class MediaUploadResponse(BaseModel):
    """Response for media upload operations."""

    success: bool = Field(..., description="Upload success status")
    upload_id: str = Field(..., description="Unique upload identifier")
    object_id: str = Field(..., description="Object key/identifier")
    bucket: str = Field(..., description="Storage bucket")
    key: str = Field(..., description="Object key")
    size: int = Field(..., description="Object size in bytes")
    content_type: str = Field(..., description="Content type")
    upload_time: float = Field(..., description="Upload duration in seconds")
    etag: str | None = Field(None, description="Object ETag")
    version_id: str | None = Field(None, description="Object version ID")
    message: str = Field(..., description="Status message")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "upload_id": "550e8400-e29b-41d4-a716-446655440000",
                "object_id": "videos/cam_001/20240812_143000_abc12345.mp4",
                "bucket": "its-video",
                "key": "videos/cam_001/20240812_143000_abc12345.mp4",
                "size": 15728640,
                "content_type": "video/mp4",
                "upload_time": 2.45,
                "etag": "d41d8cd98f00b204e9800998ecf8427e",
                "version_id": "v1.0",
                "message": "Video uploaded successfully"
            }
        }


class PresignedUrlRequest(BaseModel):
    """Request for generating presigned URLs."""

    object_id: str = Field(..., description="Object key/identifier")
    method: str = Field(default="GET", description="HTTP method (GET, PUT, DELETE)")
    expires_in_seconds: int = Field(
        default=3600,
        ge=300,
        le=86400,
        description="URL expiration time in seconds"
    )
    bucket: str | None = Field(None, description="Storage bucket (optional)")

    class Config:
        json_schema_extra = {
            "example": {
                "object_id": "videos/cam_001/20240812_143000_abc12345.mp4",
                "method": "GET",
                "expires_in_seconds": 3600,
                "bucket": "its-video"
            }
        }


class PresignedUrlResponse(BaseModel):
    """Response for presigned URL generation."""

    url: str = Field(..., description="Presigned URL")
    expires_at: datetime = Field(..., description="URL expiration time")
    method: str = Field(..., description="HTTP method")
    object_id: str = Field(..., description="Object identifier")
    bucket: str = Field(..., description="Storage bucket")

    class Config:
        json_schema_extra = {
            "example": {
                "url": "https://minio.example.com/its-video/videos/cam_001/video.mp4?X-Amz-Algorithm=AWS4-HMAC-SHA256&...",
                "expires_at": "2024-08-12T15:30:00Z",
                "method": "GET",
                "object_id": "videos/cam_001/20240812_143000_abc12345.mp4",
                "bucket": "its-video"
            }
        }


class UploadProgressResponse(BaseModel):
    """Response for upload progress tracking."""

    upload_id: str = Field(..., description="Upload identifier")
    key: str = Field(..., description="Object key")
    bucket: str = Field(..., description="Storage bucket")
    total_size: int = Field(..., description="Total file size")
    uploaded_size: int = Field(..., description="Uploaded size")
    progress_percent: float = Field(..., description="Progress percentage")
    upload_speed_bps: float = Field(..., description="Upload speed in bytes/sec")
    status: str = Field(..., description="Upload status")
    started_at: datetime = Field(..., description="Upload start time")
    estimated_completion: datetime | None = Field(None, description="Estimated completion")
    error_message: str | None = Field(None, description="Error message if failed")

    class Config:
        json_schema_extra = {
            "example": {
                "upload_id": "550e8400-e29b-41d4-a716-446655440000",
                "key": "videos/cam_001/20240812_143000_abc12345.mp4",
                "bucket": "its-video",
                "total_size": 15728640,
                "uploaded_size": 7864320,
                "progress_percent": 50.0,
                "upload_speed_bps": 1048576,
                "status": "uploading",
                "started_at": "2024-08-12T14:30:00Z",
                "estimated_completion": "2024-08-12T14:30:15Z",
                "error_message": None
            }
        }


class MediaObjectItem(BaseModel):
    """Individual media object in list responses."""

    key: str = Field(..., description="Object key")
    size: int = Field(..., description="Object size in bytes")
    content_type: str | None = Field(None, description="Content type")
    last_modified: datetime = Field(..., description="Last modification time")
    etag: str = Field(..., description="Object ETag")
    storage_class: str = Field(..., description="Storage class")
    object_type: ObjectType | None = Field(None, description="Object type classification")
    metadata: dict[str, Any] | None = Field(None, description="Object metadata")

    class Config:
        json_schema_extra = {
            "example": {
                "key": "videos/cam_001/20240812_143000_abc12345.mp4",
                "size": 15728640,
                "content_type": "video/mp4",
                "last_modified": "2024-08-12T14:30:00Z",
                "etag": "d41d8cd98f00b204e9800998ecf8427e",
                "storage_class": "STANDARD",
                "object_type": "video_frame",
                "metadata": {
                    "camera_id": "cam_001",
                    "frame_timestamp": "2024-08-12T14:30:00Z"
                }
            }
        }


class MediaListResponse(BaseModel):
    """Response for media listing operations."""

    objects: list[MediaObjectItem] = Field(..., description="List of media objects")
    total_count: int = Field(..., description="Total number of objects")
    bucket: str = Field(..., description="Storage bucket")
    prefix: str | None = Field(None, description="Key prefix filter")
    has_more: bool = Field(..., description="Whether more results are available")
    next_cursor: str | None = Field(None, description="Cursor for next page")

    class Config:
        json_schema_extra = {
            "example": {
                "objects": [
                    {
                        "key": "videos/cam_001/20240812_143000_abc12345.mp4",
                        "size": 15728640,
                        "content_type": "video/mp4",
                        "last_modified": "2024-08-12T14:30:00Z",
                        "etag": "d41d8cd98f00b204e9800998ecf8427e",
                        "storage_class": "STANDARD",
                        "object_type": "video_frame",
                        "metadata": {"camera_id": "cam_001"}
                    }
                ],
                "total_count": 1,
                "bucket": "its-video",
                "prefix": "videos/cam_001/",
                "has_more": False,
                "next_cursor": None
            }
        }


class StorageStatsResponse(BaseModel):
    """Response for storage statistics."""

    total_objects: int = Field(..., description="Total number of objects")
    total_size_mb: float = Field(..., description="Total storage size in MB")
    total_uploads: int = Field(..., description="Total upload operations")
    total_downloads: int = Field(..., description="Total download operations")
    success_rate: float = Field(..., description="Operation success rate percentage")
    average_upload_speed_mbps: float = Field(..., description="Average upload speed in MB/s")
    average_download_speed_mbps: float = Field(..., description="Average download speed in MB/s")
    cache_hit_rate: float = Field(..., description="Cache hit rate percentage")
    active_uploads: int = Field(..., description="Currently active uploads")

    # Breakdown by object type
    object_counts_by_type: dict[str, int] = Field(
        default_factory=dict,
        description="Object counts by type"
    )
    storage_by_bucket: dict[str, float] = Field(
        default_factory=dict,
        description="Storage usage by bucket in MB"
    )

    # Recent activity
    uploads_last_24h: int = Field(..., description="Uploads in last 24 hours")
    downloads_last_24h: int = Field(..., description="Downloads in last 24 hours")
    errors_last_24h: int = Field(..., description="Errors in last 24 hours")

    last_updated: datetime = Field(..., description="Statistics last updated time")

    class Config:
        json_schema_extra = {
            "example": {
                "total_objects": 15420,
                "total_size_mb": 245760.5,
                "total_uploads": 18650,
                "total_downloads": 7230,
                "success_rate": 99.2,
                "average_upload_speed_mbps": 12.5,
                "average_download_speed_mbps": 18.7,
                "cache_hit_rate": 85.3,
                "active_uploads": 3,
                "object_counts_by_type": {
                    "video_frame": 12400,
                    "model_artifact": 85,
                    "inference_result": 2935
                },
                "storage_by_bucket": {
                    "its-video": 198450.2,
                    "its-models": 15680.8,
                    "its-results": 31629.5
                },
                "uploads_last_24h": 1250,
                "downloads_last_24h": 680,
                "errors_last_24h": 12,
                "last_updated": "2024-08-12T14:30:00Z"
            }
        }


class BucketInfoResponse(BaseModel):
    """Response for bucket information."""

    name: str = Field(..., description="Bucket name")
    creation_date: datetime = Field(..., description="Bucket creation date")
    region: str = Field(..., description="Bucket region")
    versioning_enabled: bool = Field(..., description="Versioning status")
    encryption_enabled: bool = Field(..., description="Encryption status")
    object_count: int = Field(..., description="Total objects in bucket")
    total_size_mb: float = Field(..., description="Total bucket size in MB")
    lifecycle_rules: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Lifecycle management rules"
    )
    tags: dict[str, str] = Field(default_factory=dict, description="Bucket tags")

    # Access statistics
    uploads_last_24h: int = Field(..., description="Uploads in last 24 hours")
    downloads_last_24h: int = Field(..., description="Downloads in last 24 hours")
    storage_class_distribution: dict[str, int] = Field(
        default_factory=dict,
        description="Object count by storage class"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "name": "its-video",
                "creation_date": "2024-01-01T00:00:00Z",
                "region": "us-east-1",
                "versioning_enabled": True,
                "encryption_enabled": True,
                "object_count": 12400,
                "total_size_mb": 198450.2,
                "lifecycle_rules": [
                    {
                        "id": "delete_old_frames",
                        "status": "Enabled",
                        "expiration_days": 30
                    }
                ],
                "tags": {
                    "project": "its-camera-ai",
                    "environment": "production"
                },
                "uploads_last_24h": 850,
                "downloads_last_24h": 420,
                "storage_class_distribution": {
                    "STANDARD": 11500,
                    "IA": 900
                }
            }
        }


class HealthCheckResponse(BaseModel):
    """Response for storage health check."""

    status: str = Field(..., description="Health status (healthy, degraded, unhealthy)")
    timestamp: datetime = Field(..., description="Health check timestamp")

    # Connection status
    minio_connected: bool = Field(..., description="MinIO connection status")
    redis_connected: bool = Field(..., description="Redis cache connection status")

    # Performance metrics
    response_time_ms: float = Field(..., description="Average response time")
    error_rate: float = Field(..., description="Error rate percentage")

    # Resource utilization
    active_connections: int = Field(..., description="Active connections")
    memory_usage_mb: float = Field(..., description="Memory usage in MB")

    # Recent activity
    operations_last_minute: int = Field(..., description="Operations in last minute")
    errors_last_minute: int = Field(..., description="Errors in last minute")

    # Detailed status
    components: dict[str, str] = Field(
        default_factory=dict,
        description="Component-level health status"
    )

    message: str | None = Field(None, description="Additional status information")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2024-08-12T14:30:00Z",
                "minio_connected": True,
                "redis_connected": True,
                "response_time_ms": 45.2,
                "error_rate": 0.8,
                "active_connections": 15,
                "memory_usage_mb": 256.5,
                "operations_last_minute": 42,
                "errors_last_minute": 0,
                "components": {
                    "minio_service": "healthy",
                    "redis_cache": "healthy",
                    "upload_manager": "healthy"
                },
                "message": "All systems operational"
            }
        }
