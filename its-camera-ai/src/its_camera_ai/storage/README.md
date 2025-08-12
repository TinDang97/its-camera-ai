# MinIO Storage Integration

Comprehensive MinIO object storage integration for the ITS Camera AI system, providing scalable storage for video frames, ML model artifacts, and system metadata.

## Features

### Core Storage Capabilities
- **Async MinIO Client**: High-performance async operations with connection pooling
- **Multi-part Uploads**: Efficient handling of large files (>64MB) with parallel uploads
- **Server-side Encryption**: SSE-S3 encryption for data security
- **Object Versioning**: Version control for all stored objects
- **Lifecycle Management**: Automatic cleanup and archival policies
- **Compression**: Optional data compression (GZIP, LZ4, ZSTD)
- **Redis Caching**: Fast metadata retrieval with configurable TTL

### Advanced Features
- **Presigned URLs**: Secure temporary access for client-side uploads/downloads
- **Batch Operations**: Efficient bulk operations for large datasets
- **Performance Monitoring**: Real-time metrics and health monitoring
- **Error Handling**: Comprehensive retry logic and error recovery
- **Model Registry**: Specialized storage for ML model artifacts with metadata

## Architecture

### Components

1. **MinIOService** (`minio_service.py`)
   - Core storage service with async operations
   - Connection pooling and resource management
   - Upload/download with progress tracking
   - Bucket management and lifecycle policies

2. **MinIOModelRegistry** (`model_registry.py`)
   - Specialized registry for ML model artifacts
   - Version-controlled model storage
   - Metadata tracking and search capabilities
   - Integration with training pipeline

3. **Storage Models** (`models.py`)
   - Pydantic models for all storage operations
   - Type-safe configurations and requests
   - Comprehensive validation and serialization

4. **Storage Factory** (`factory.py`)
   - Centralized service creation and configuration
   - Singleton pattern for resource efficiency
   - Dependency injection and lifecycle management

### Data Flow

```
Application Layer
       ↓
Storage Factory
       ↓
┌─────────────────────────────────┐
│        MinIO Service            │
├─────────────────────────────────┤
│  • Async Operations            │
│  • Connection Pooling          │
│  • Progress Tracking           │
│  • Error Handling              │
└─────────────────────────────────┘
       ↓
┌─────────────────────────────────┐
│      MinIO Server               │
├─────────────────────────────────┤
│  • Object Storage              │
│  • Bucket Management           │
│  • Encryption                  │
│  • Versioning                  │
└─────────────────────────────────┘
```

## Usage

### Basic Storage Operations

```python
from its_camera_ai.storage import get_storage_service, UploadRequest, ObjectType

# Get storage service
storage = await get_storage_service()

# Upload video frame
upload_request = UploadRequest(
    bucket="its-video",
    key="camera_001/frame_20240812_120000.jpg",
    file_path=Path("frame.jpg"),
    object_type=ObjectType.VIDEO_FRAME,
    metadata={"camera_id": "camera_001", "timestamp": "2024-08-12T12:00:00Z"},
    tags={"camera": "camera_001", "type": "video-frame"},
)

response = await storage.upload_object(upload_request)
print(f"Uploaded: {response.key} ({response.total_size} bytes)")

# Download object
from its_camera_ai.storage import DownloadRequest

download_request = DownloadRequest(
    bucket="its-video",
    key="camera_001/frame_20240812_120000.jpg",
    file_path=Path("downloaded_frame.jpg"),
)

result = await storage.download_object(download_request)
print(f"Downloaded to: {result}")
```

### Model Registry Operations

```python
from its_camera_ai.storage import get_model_registry

# Get model registry
registry = await get_model_registry()

# Register new model
model_version = await registry.register_model(
    model_path=Path("yolo11_traffic_v2.pt"),
    model_name="yolo11_traffic",
    version="2.0.0",
    metrics={"accuracy": 0.94, "latency_p95_ms": 42.0},
    training_config={"epochs": 100, "batch_size": 32},
    tags={"dataset": "traffic_v2", "environment": "production"}
)

# List models
models = await registry.list_models(limit=10)
for model in models:
    print(f"{model['model_name']}:{model['version']} - Acc: {model['metrics']['accuracy']:.2f}")

# Retrieve model
model = await registry.get_model(
    "yolo11_traffic", "2.0.0", 
    download_path=Path("models/current_model.pt")
)

# Generate download URL
url = await registry.generate_download_url(
    "yolo11_traffic", "2.0.0", 
    expires_in_seconds=3600
)
```

### Batch Operations

```python
from its_camera_ai.storage import ListObjectsRequest

# List objects with filters
list_request = ListObjectsRequest(
    bucket="its-video",
    prefix="camera_001/",
    max_keys=100,
    object_types=[ObjectType.VIDEO_FRAME],
    size_range=(1024, 10*1024*1024),  # 1KB to 10MB
    include_metadata=True
)

objects = await storage.list_objects(list_request)
print(f"Found {len(objects)} video frames from camera_001")

# Batch delete (example)
for obj in objects[:10]:  # Delete first 10
    await storage.delete_object(obj.bucket, obj.key)
```

### Presigned URLs

```python
# Generate upload URL for client-side uploads
upload_url = await storage.generate_presigned_url(
    bucket="its-video",
    key="camera_002/new_frame.jpg",
    expires_in_seconds=300,  # 5 minutes
    method="PUT"
)

# Generate download URL for sharing
download_url = await storage.generate_presigned_url(
    bucket="its-models",
    key="yolo11_traffic/2.0.0/model.pt",
    expires_in_seconds=1800,  # 30 minutes
    method="GET"
)
```

## Configuration

### Environment Variables

```bash
# MinIO Connection
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin123
MINIO_SECURE=false
MINIO_REGION=us-east-1

# Buckets
MINIO_DEFAULT_BUCKET=its-camera-ai
MINIO_MODELS_BUCKET=its-models
MINIO_VIDEO_BUCKET=its-video
MINIO_LOGS_BUCKET=its-logs

# Performance
MINIO_MULTIPART_THRESHOLD=67108864  # 64MB
MINIO_MAX_POOL_CONNECTIONS=20
MINIO_ENABLE_CACHING=true
MINIO_CACHE_TTL=3600

# Security
MINIO_ENABLE_VERSIONING=true
MINIO_ENABLE_ENCRYPTION=true
MINIO_ENABLE_COMPRESSION=true
```

### Docker Compose

MinIO service is already configured in `docker-compose.yml`:

```yaml
minio:
  image: minio/minio:latest
  ports:
    - "9000:9000"  # API
    - "9001:9001"  # Console
  environment:
    MINIO_ROOT_USER: minioadmin
    MINIO_ROOT_PASSWORD: minioadmin123
  command: server /data --console-address ":9001"
  volumes:
    - minio_data:/data
  healthcheck:
    test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
```

## Performance Optimization

### Upload Optimization
- **Multi-part uploads**: Automatically used for files >64MB
- **Connection pooling**: Reuse connections for multiple operations
- **Compression**: Optional compression for text and JSON data
- **Parallel uploads**: Concurrent part uploads for large files

### Download Optimization
- **Redis caching**: Frequently accessed metadata cached in Redis
- **Range downloads**: Partial content downloads for large files
- **Streaming**: Direct streaming for large objects

### Monitoring
- **Real-time metrics**: Upload/download speeds, success rates
- **Health checks**: Service availability and performance
- **Error tracking**: Comprehensive error logging and analysis

## Security

### Encryption
- **Server-side encryption** (SSE-S3) for all stored objects
- **TLS encryption** for data in transit (when `secure=true`)
- **Access key management** with secure credential handling

### Access Control
- **Presigned URLs** for temporary, limited access
- **Bucket policies** for fine-grained access control
- **Object tagging** for classification and filtering

### Compliance
- **Audit logging** for all storage operations
- **Data retention** policies with lifecycle management
- **GDPR compliance** with data deletion capabilities

## Testing

Run the comprehensive test suite:

```bash
# Run all storage tests
pytest tests/test_storage_integration.py -v

# Run with coverage
pytest tests/test_storage_integration.py --cov=src/its_camera_ai/storage --cov-report=html

# Run integration tests (requires MinIO server)
pytest tests/test_storage_integration.py -m "integration" -v
```

## Examples

See `examples/minio_integration_example.py` for a comprehensive demonstration:

```bash
# Run the complete example
python examples/minio_integration_example.py
```

This example demonstrates:
- Service initialization and configuration
- Basic storage operations (upload, download, list, delete)
- Video frame storage with metadata
- Model registry operations
- Presigned URL generation
- Performance monitoring
- Error handling and recovery

## Troubleshooting

### Common Issues

1. **Connection Timeout**
   - Check MinIO server is running: `docker-compose ps minio`
   - Verify endpoint configuration
   - Check network connectivity

2. **Access Denied**
   - Verify access key and secret key
   - Check bucket permissions
   - Ensure bucket exists (or `create_buckets=true`)

3. **Large File Upload Fails**
   - Increase `multipart_threshold` and `multipart_chunksize`
   - Check available disk space
   - Verify network stability for large transfers

4. **Cache Issues**
   - Check Redis connection
   - Verify cache TTL settings
   - Clear cache if needed: `redis-cli FLUSHDB`

### Debug Mode

Enable debug logging:

```python
import logging
logging.getLogger("its_camera_ai.storage").setLevel(logging.DEBUG)
```

### Health Checks

```python
from its_camera_ai.storage import get_storage_service

storage = await get_storage_service()
health = storage.get_health_status()
print(f"Storage Health: {health['status']}")

stats = storage.get_stats()
print(f"Success Rate: {stats.get_success_rate():.1f}%")
print(f"Cache Hit Rate: {stats.get_cache_hit_rate():.1f}%")
```

## Development

### Adding New Storage Backends

1. Implement the storage interface in a new service class
2. Add configuration models in `models.py`
3. Update the factory to support the new backend
4. Add comprehensive tests

### Performance Tuning

1. **Connection Pool Size**: Adjust based on concurrent operations
2. **Multipart Settings**: Tune for your file sizes and network
3. **Cache Settings**: Balance memory usage vs. performance
4. **Compression**: Enable for text data, disable for media files

### Monitoring Integration

```python
# Custom metrics collection
from its_camera_ai.storage import get_storage_service

storage = await get_storage_service()
stats = storage.get_stats()

# Export to Prometheus, TimescaleDB, etc.
metrics = {
    "storage_upload_speed_mbps": stats.average_upload_speed_bps / 1024 / 1024,
    "storage_success_rate": stats.get_success_rate(),
    "storage_cache_hit_rate": stats.get_cache_hit_rate(),
}
```