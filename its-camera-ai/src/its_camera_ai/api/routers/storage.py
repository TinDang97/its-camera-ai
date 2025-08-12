"""FastAPI endpoints for media upload/retrieval and storage operations.

Provides comprehensive media management functionality with MinIO integration,
including file uploads, downloads, presigned URLs, and metadata management.
"""

import tempfile
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import aiofiles
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from ...core.exceptions import ResourceNotFoundError, ValidationError
from ...core.logging import get_logger
from ...storage.minio_service import MinIOError, MinIOService
from ...storage.models import (
    CompressionType,
    DownloadRequest,
    ListObjectsRequest,
    ObjectType,
    UploadProgress,
    UploadRequest,
)
from ..dependencies import (
    get_current_user,
    get_minio_service,
    get_settings,
    validate_api_key,
)
from ..schemas.storage import (
    MediaListResponse,
    MediaUploadResponse,
    PresignedUrlRequest,
    PresignedUrlResponse,
    UploadProgressResponse,
)

logger = get_logger(__name__)

# Create router
router = APIRouter(
    prefix="/storage",
    tags=["storage"],
    dependencies=[Depends(validate_api_key)],
)

# WebSocket connection manager for upload progress
class UploadConnectionManager:
    """Manages WebSocket connections for upload progress."""

    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}
        self.upload_sessions: dict[str, str] = {}  # upload_id -> connection_id

    async def connect(self, websocket: WebSocket, connection_id: str):
        """Accept WebSocket connection."""
        await websocket.accept()
        self.active_connections[connection_id] = websocket
        logger.info(f"WebSocket connection established: {connection_id}")

    def disconnect(self, connection_id: str):
        """Remove WebSocket connection."""
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]

        # Remove associated upload sessions
        sessions_to_remove = [
            upload_id for upload_id, conn_id in self.upload_sessions.items()
            if conn_id == connection_id
        ]
        for upload_id in sessions_to_remove:
            del self.upload_sessions[upload_id]

        logger.info(f"WebSocket connection closed: {connection_id}")

    def register_upload(self, upload_id: str, connection_id: str):
        """Register upload session with connection."""
        self.upload_sessions[upload_id] = connection_id

    async def send_progress_update(self, upload_id: str, progress: UploadProgress):
        """Send progress update to associated connection."""
        connection_id = self.upload_sessions.get(upload_id)
        if not connection_id or connection_id not in self.active_connections:
            return

        try:
            websocket = self.active_connections[connection_id]
            await websocket.send_json({
                "type": "progress_update",
                "upload_id": upload_id,
                "progress": progress.model_dump(),
            })
        except Exception as e:
            logger.error(f"Failed to send progress update for {upload_id}: {e}")
            self.disconnect(connection_id)

upload_manager = UploadConnectionManager()


# Request/Response Models
class ChunkedUploadRequest(BaseModel):
    """Request for chunked file upload initialization."""

    bucket: str = Field(..., description="Target bucket")
    key: str = Field(..., description="Object key")
    total_size: int = Field(..., description="Total file size")
    chunk_size: int = Field(default=16*1024*1024, description="Chunk size in bytes")
    content_type: str | None = Field(None, description="Content type")
    object_type: ObjectType = Field(default=ObjectType.VIDEO_FRAME)
    metadata: dict[str, str] = Field(default_factory=dict)
    tags: dict[str, str] = Field(default_factory=dict)


class ChunkedUploadResponse(BaseModel):
    """Response for chunked upload initialization."""

    upload_id: str = Field(..., description="Upload session ID")
    chunk_urls: list[str] = Field(..., description="Presigned URLs for chunks")
    total_chunks: int = Field(..., description="Total number of chunks")
    expires_at: datetime = Field(..., description="URL expiration time")


class MediaDeleteRequest(BaseModel):
    """Request for deleting media objects."""

    keys: list[str] = Field(..., description="Object keys to delete")
    bucket: str | None = Field(None, description="Bucket name (optional)")


# Media Upload Endpoints
@router.post(
    "/upload/video",
    response_model=MediaUploadResponse,
    summary="Upload video files from cameras",
    description="Upload video files with metadata and automatic compression",
)
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Video file to upload"),
    camera_id: str = Form(..., description="Source camera ID"),
    timestamp: datetime | None = Form(None, description="Video timestamp"),
    bucket: str | None = Form(None, description="Target bucket"),
    compression: CompressionType = Form(CompressionType.GZIP, description="Compression type"),
    metadata: str | None = Form(None, description="Additional metadata as JSON"),
    tags: str | None = Form(None, description="Tags as JSON"),
    minio_service: MinIOService = Depends(get_minio_service),
    current_user = Depends(get_current_user),
    settings = Depends(get_settings),
) -> MediaUploadResponse:
    """Upload video file with automatic processing."""

    try:
        # Validate file
        if not file.filename:
            raise ValidationError("Filename is required")

        # Validate file type
        if not file.content_type or not file.content_type.startswith('video/'):
            raise ValidationError("File must be a video")

        # Generate unique key
        file_extension = Path(file.filename).suffix
        timestamp_str = (timestamp or datetime.now(UTC)).strftime("%Y%m%d_%H%M%S")
        object_key = f"videos/{camera_id}/{timestamp_str}_{uuid4().hex[:8]}{file_extension}"

        # Parse metadata and tags
        import json
        parsed_metadata = json.loads(metadata) if metadata else {}
        parsed_tags = json.loads(tags) if tags else {}

        # Add standard metadata
        parsed_metadata.update({
            "camera_id": camera_id,
            "upload_user": current_user.username if hasattr(current_user, 'username') else 'system',
            "upload_timestamp": datetime.now(UTC).isoformat(),
            "original_filename": file.filename,
        })

        # Add standard tags
        parsed_tags.update({
            "camera_id": camera_id,
            "object_type": "video",
            "source": "camera_upload",
        })

        # Create temporary file for upload
        temp_file = None
        try:
            # Save uploaded file to temporary location
            temp_file = Path(tempfile.mktemp(suffix=file_extension))

            async with aiofiles.open(temp_file, 'wb') as f:
                content = await file.read()
                await f.write(content)

            # Create upload request
            upload_request = UploadRequest(
                bucket=bucket or settings.minio.video_bucket,
                key=object_key,
                file_path=temp_file,
                content_type=file.content_type,
                object_type=ObjectType.VIDEO_FRAME,
                metadata=parsed_metadata,
                tags=parsed_tags,
                compression=compression,
                storage_class="STANDARD",
            )

            # Upload to MinIO
            upload_response = await minio_service.upload_object(upload_request)

            if not upload_response.success:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Upload failed: {upload_response.error_message}",
                )

            # Schedule cleanup
            background_tasks.add_task(cleanup_temp_file, temp_file)

            return MediaUploadResponse(
                success=True,
                upload_id=upload_response.upload_id,
                object_id=object_key,
                bucket=upload_response.bucket,
                key=upload_response.key,
                size=upload_response.total_size,
                content_type=upload_response.content_type,
                upload_time=upload_response.upload_time_seconds,
                etag=upload_response.etag,
                version_id=upload_response.version_id,
                message="Video uploaded successfully",
            )

        except Exception:
            if temp_file and temp_file.exists():
                background_tasks.add_task(cleanup_temp_file, temp_file)
            raise

    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except MinIOError as e:
        logger.error(f"MinIO error during video upload: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Storage error: {e.message}",
        )
    except Exception as e:
        logger.error(f"Unexpected error during video upload: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.post(
    "/upload/frame",
    response_model=MediaUploadResponse,
    summary="Upload individual video frames",
    description="Upload individual video frames for processing",
)
async def upload_frame(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Frame image file"),
    camera_id: str = Form(..., description="Source camera ID"),
    frame_number: int = Form(..., description="Frame sequence number"),
    timestamp: datetime | None = Form(None, description="Frame timestamp"),
    bucket: str | None = Form(None, description="Target bucket"),
    compression: CompressionType = Form(CompressionType.JPEG, description="Image compression"),
    quality: int = Form(85, description="Compression quality (1-100)"),
    metadata: str | None = Form(None, description="Additional metadata as JSON"),
    minio_service: MinIOService = Depends(get_minio_service),
    current_user = Depends(get_current_user),
    settings = Depends(get_settings),
) -> MediaUploadResponse:
    """Upload individual video frame."""

    try:
        # Validate file
        if not file.filename:
            raise ValidationError("Filename is required")

        # Validate image file
        if not file.content_type or not file.content_type.startswith('image/'):
            raise ValidationError("File must be an image")

        # Generate frame key
        file_extension = Path(file.filename).suffix or '.jpg'
        timestamp_str = (timestamp or datetime.now(UTC)).strftime("%Y%m%d_%H%M%S")
        object_key = f"frames/{camera_id}/{timestamp_str}/frame_{frame_number:06d}{file_extension}"

        # Parse metadata
        import json
        parsed_metadata = json.loads(metadata) if metadata else {}

        # Add frame-specific metadata
        parsed_metadata.update({
            "camera_id": camera_id,
            "frame_number": str(frame_number),
            "frame_timestamp": (timestamp or datetime.now(UTC)).isoformat(),
            "upload_user": current_user.username if hasattr(current_user, 'username') else 'system',
            "compression_quality": str(quality),
            "original_filename": file.filename,
        })

        # Create upload request directly from file content
        content = await file.read()

        upload_request = UploadRequest(
            bucket=bucket or settings.minio.video_bucket,
            key=object_key,
            data=content,
            content_type=file.content_type,
            object_type=ObjectType.VIDEO_FRAME,
            metadata=parsed_metadata,
            tags={
                "camera_id": camera_id,
                "frame_number": str(frame_number),
                "object_type": "frame",
                "source": "frame_upload",
            },
            compression=compression,
            storage_class="STANDARD",
        )

        # Upload to MinIO
        upload_response = await minio_service.upload_object(upload_request)

        if not upload_response.success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Upload failed: {upload_response.error_message}",
            )

        return MediaUploadResponse(
            success=True,
            upload_id=upload_response.upload_id,
            object_id=object_key,
            bucket=upload_response.bucket,
            key=upload_response.key,
            size=upload_response.total_size,
            content_type=upload_response.content_type,
            upload_time=upload_response.upload_time_seconds,
            etag=upload_response.etag,
            version_id=upload_response.version_id,
            message="Frame uploaded successfully",
        )

    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except MinIOError as e:
        logger.error(f"MinIO error during frame upload: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Storage error: {e.message}",
        )
    except Exception as e:
        logger.error(f"Unexpected error during frame upload: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.post(
    "/upload/chunked/init",
    response_model=ChunkedUploadResponse,
    summary="Initialize chunked upload for large files",
    description="Initialize chunked upload session for large video files",
)
async def init_chunked_upload(
    request: ChunkedUploadRequest,
    minio_service: MinIOService = Depends(get_minio_service),
    current_user = Depends(get_current_user),
) -> ChunkedUploadResponse:
    """Initialize chunked upload for large files."""

    try:
        # Calculate number of chunks
        total_chunks = (request.total_size + request.chunk_size - 1) // request.chunk_size

        # Generate upload ID
        upload_id = str(uuid4())

        # Generate presigned URLs for each chunk (simplified implementation)
        # In a real implementation, you'd use MinIO's multipart upload
        chunk_urls = []
        for i in range(total_chunks):
            chunk_key = f"{request.key}.part{i:04d}"
            url = await minio_service.generate_presigned_url(
                request.bucket,
                chunk_key,
                expires_in_seconds=3600,
                method="PUT"
            )
            chunk_urls.append(url)

        return ChunkedUploadResponse(
            upload_id=upload_id,
            chunk_urls=chunk_urls,
            total_chunks=total_chunks,
            expires_at=datetime.now(UTC).replace(hour=datetime.now().hour + 1),
        )

    except Exception as e:
        logger.error(f"Failed to initialize chunked upload: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to initialize chunked upload: {str(e)}",
        )


# Media Retrieval Endpoints
@router.get(
    "/media/{object_id:path}",
    summary="Retrieve media by ID",
    description="Download media object by its ID with optional range support",
)
async def get_media(
    object_id: str,
    bucket: str | None = Query(None, description="Source bucket"),
    version_id: str | None = Query(None, description="Object version ID"),
    range_header: str | None = Query(None, alias="range", description="Byte range"),
    download: bool = Query(False, description="Force download vs inline display"),
    minio_service: MinIOService = Depends(get_minio_service),
    settings = Depends(get_settings),
):
    """Retrieve media object by ID."""

    try:
        # Default bucket if not specified
        target_bucket = bucket or settings.minio.video_bucket

        # Check if object exists
        if not await minio_service.object_exists(target_bucket, object_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Media object not found: {object_id}",
            )

        # Get object metadata
        metadata = await minio_service.get_object_metadata(
            target_bucket, object_id, version_id
        )

        # Parse range header if provided
        byte_range = None
        if range_header:
            try:
                # Parse range header: "bytes=0-1023"
                range_match = range_header.replace('bytes=', '')
                start, end = map(int, range_match.split('-'))
                byte_range = (start, end)
            except (ValueError, AttributeError):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid range header format",
                )

        # Create download request
        download_request = DownloadRequest(
            bucket=target_bucket,
            key=object_id,
            version_id=version_id,
            byte_range=byte_range,
            validate_checksum=True,
            decompress=True,
        )

        # Download object
        data = await minio_service.download_object(download_request)

        # Determine content disposition
        content_disposition = "attachment" if download else "inline"
        if download:
            filename = Path(object_id).name
            content_disposition += f'; filename="{filename}"'

        # Create response headers
        headers = {
            "Content-Type": metadata.content_type,
            "Content-Length": str(len(data)),
            "Content-Disposition": content_disposition,
            "ETag": metadata.etag or '',
            "Last-Modified": metadata.modified_at.strftime('%a, %d %b %Y %H:%M:%S GMT') if metadata.modified_at else '',
        }

        # Add range response headers if applicable
        if byte_range:
            start, end = byte_range
            headers["Content-Range"] = f"bytes {start}-{end}/{metadata.size}"
            headers["Accept-Ranges"] = "bytes"
            return StreamingResponse(
                iter([data]),
                status_code=206,  # Partial Content
                headers=headers,
                media_type=metadata.content_type,
            )

        return StreamingResponse(
            iter([data]),
            headers=headers,
            media_type=metadata.content_type,
        )

    except ResourceNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Media object not found: {object_id}",
        )
    except MinIOError as e:
        logger.error(f"MinIO error retrieving media {object_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Storage error: {e.message}",
        )
    except Exception as e:
        logger.error(f"Unexpected error retrieving media {object_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.get(
    "/media/presigned-url",
    response_model=PresignedUrlResponse,
    summary="Generate presigned URL for media access",
    description="Generate presigned URL for direct media access",
)
async def get_presigned_url(
    request: PresignedUrlRequest = Depends(),
    minio_service: MinIOService = Depends(get_minio_service),
    settings = Depends(get_settings),
) -> PresignedUrlResponse:
    """Generate presigned URL for media access."""

    try:
        # Default bucket if not specified
        target_bucket = request.bucket or settings.minio.video_bucket

        # Check if object exists for GET requests
        if request.method.upper() == "GET":
            if not await minio_service.object_exists(target_bucket, request.object_id):
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Media object not found: {request.object_id}",
                )

        # Generate presigned URL
        url = await minio_service.generate_presigned_url(
            target_bucket,
            request.object_id,
            request.expires_in_seconds,
            request.method
        )

        return PresignedUrlResponse(
            url=url,
            expires_at=datetime.now(UTC).replace(
                second=datetime.now().second + request.expires_in_seconds
            ),
            method=request.method,
            object_id=request.object_id,
            bucket=target_bucket,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to generate presigned URL: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate presigned URL: {str(e)}",
        )


@router.delete(
    "/media/{object_id:path}",
    summary="Delete media object",
    description="Delete media object from storage",
)
async def delete_media(
    object_id: str,
    bucket: str | None = Query(None, description="Source bucket"),
    version_id: str | None = Query(None, description="Specific version to delete"),
    minio_service: MinIOService = Depends(get_minio_service),
    current_user = Depends(get_current_user),
    settings = Depends(get_settings),
):
    """Delete media object from storage."""

    try:
        # Default bucket if not specified
        target_bucket = bucket or settings.minio.video_bucket

        # Check if object exists
        if not await minio_service.object_exists(target_bucket, object_id, version_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Media object not found: {object_id}",
            )

        # Delete object
        success = await minio_service.delete_object(target_bucket, object_id, version_id)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete media object",
            )

        return JSONResponse(
            content={
                "success": True,
                "message": f"Media object deleted: {object_id}",
                "object_id": object_id,
                "bucket": target_bucket,
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete media {object_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete media object: {str(e)}",
        )


@router.post(
    "/media/batch-delete",
    summary="Delete multiple media objects",
    description="Delete multiple media objects in a single request",
)
async def batch_delete_media(
    request: MediaDeleteRequest,
    minio_service: MinIOService = Depends(get_minio_service),
    current_user = Depends(get_current_user),
    settings = Depends(get_settings),
):
    """Delete multiple media objects."""

    try:
        target_bucket = request.bucket or settings.minio.video_bucket

        deleted_objects = []
        failed_objects = []

        # Delete each object
        for object_key in request.keys:
            try:
                if await minio_service.object_exists(target_bucket, object_key):
                    success = await minio_service.delete_object(target_bucket, object_key)
                    if success:
                        deleted_objects.append(object_key)
                    else:
                        failed_objects.append(object_key)
                else:
                    failed_objects.append(object_key)
            except Exception as e:
                logger.error(f"Failed to delete {object_key}: {e}")
                failed_objects.append(object_key)

        return JSONResponse(
            content={
                "success": len(failed_objects) == 0,
                "deleted_count": len(deleted_objects),
                "failed_count": len(failed_objects),
                "deleted_objects": deleted_objects,
                "failed_objects": failed_objects,
                "message": f"Deleted {len(deleted_objects)} objects, {len(failed_objects)} failed",
            }
        )

    except Exception as e:
        logger.error(f"Batch delete operation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch delete failed: {str(e)}",
        )


@router.get(
    "/media/list",
    response_model=MediaListResponse,
    summary="List media objects",
    description="List media objects with filtering and pagination",
)
async def list_media(
    bucket: str | None = Query(None, description="Source bucket"),
    prefix: str | None = Query(None, description="Key prefix filter"),
    camera_id: str | None = Query(None, description="Filter by camera ID"),
    object_type: ObjectType | None = Query(None, description="Filter by object type"),
    start_date: datetime | None = Query(None, description="Start date filter"),
    end_date: datetime | None = Query(None, description="End date filter"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum results"),
    start_after: str | None = Query(None, description="Pagination cursor"),
    include_metadata: bool = Query(False, description="Include full metadata"),
    minio_service: MinIOService = Depends(get_minio_service),
    settings = Depends(get_settings),
) -> MediaListResponse:
    """List media objects with filtering."""

    try:
        # Default bucket if not specified
        target_bucket = bucket or settings.minio.video_bucket

        # Build prefix filter
        search_prefix = prefix or ""
        if camera_id:
            if search_prefix:
                search_prefix = f"{search_prefix}/{camera_id}"
            else:
                search_prefix = f"frames/{camera_id}" if object_type == ObjectType.VIDEO_FRAME else f"videos/{camera_id}"

        # Create list request
        list_request = ListObjectsRequest(
            bucket=target_bucket,
            prefix=search_prefix,
            max_keys=limit,
            start_after=start_after,
            object_types=[object_type] if object_type else None,
            date_range=(start_date, end_date) if start_date and end_date else None,
            include_metadata=include_metadata,
            recursive=True,
        )

        # List objects
        objects = await minio_service.list_objects(list_request)

        return MediaListResponse(
            objects=[
                {
                    "key": obj.key,
                    "size": obj.size,
                    "content_type": obj.content_type,
                    "last_modified": obj.last_modified,
                    "etag": obj.etag,
                    "storage_class": obj.storage_class,
                    "object_type": obj.object_type,
                    "metadata": obj.metadata.model_dump() if obj.metadata else None,
                }
                for obj in objects
            ],
            total_count=len(objects),
            bucket=target_bucket,
            prefix=search_prefix,
            has_more=len(objects) >= limit,
            next_cursor=objects[-1].key if objects and len(objects) >= limit else None,
        )

    except Exception as e:
        logger.error(f"Failed to list media objects: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list media objects: {str(e)}",
        )


# WebSocket endpoint for real-time upload progress
@router.websocket("/upload/progress/{connection_id}")
async def websocket_upload_progress(
    websocket: WebSocket,
    connection_id: str,
):
    """WebSocket endpoint for real-time upload progress updates."""

    await upload_manager.connect(websocket, connection_id)

    try:
        while True:
            # Keep connection alive and handle any client messages
            message = await websocket.receive_json()

            if message.get("type") == "register_upload":
                upload_id = message.get("upload_id")
                if upload_id:
                    upload_manager.register_upload(upload_id, connection_id)
                    await websocket.send_json({
                        "type": "registration_success",
                        "upload_id": upload_id,
                    })

    except WebSocketDisconnect:
        upload_manager.disconnect(connection_id)
    except Exception as e:
        logger.error(f"WebSocket error for connection {connection_id}: {e}")
        upload_manager.disconnect(connection_id)


# Upload progress tracking endpoint
@router.get(
    "/upload/progress/{upload_id}",
    response_model=UploadProgressResponse,
    summary="Get upload progress",
    description="Get current progress for an active upload",
)
async def get_upload_progress(
    upload_id: str,
    minio_service: MinIOService = Depends(get_minio_service),
) -> UploadProgressResponse:
    """Get upload progress for specific upload ID."""

    try:
        progress = minio_service.get_upload_progress(upload_id)

        if not progress:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Upload not found: {upload_id}",
            )

        return UploadProgressResponse(
            upload_id=progress.upload_id,
            key=progress.key,
            bucket=progress.bucket,
            total_size=progress.total_size,
            uploaded_size=progress.uploaded_size,
            progress_percent=progress.progress_percent,
            upload_speed_bps=progress.upload_speed_bps,
            status=progress.status,
            started_at=progress.started_at,
            estimated_completion=progress.estimated_completion,
            error_message=progress.error_message,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get upload progress: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get upload progress",
        )


# Utility functions
async def cleanup_temp_file(file_path: Path) -> None:
    """Clean up temporary file."""
    try:
        if file_path.exists():
            file_path.unlink()
            logger.info(f"Cleaned up temporary file: {file_path}")
    except Exception as e:
        logger.error(f"Failed to cleanup temp file {file_path}: {e}")


# Progress callback for uploads with WebSocket integration
async def websocket_progress_callback(upload_id: str, progress: UploadProgress) -> None:
    """Progress callback that sends updates via WebSocket."""
    await upload_manager.send_progress_update(upload_id, progress)
