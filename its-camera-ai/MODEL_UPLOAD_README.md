# Model Registry Upload Service

This document describes the comprehensive model registry upload service implemented for the ITS Camera AI system.

## Overview

The model upload service allows users to upload ML models (primarily YOLO11 models) with proper validation, security scanning, and metadata management. The service follows ML best practices for model versioning, performance benchmarking, and deployment pipeline integration.

## Features

### 1. File Upload Support
- **Model files**: `.pt`, `.pth`, `.onnx`, `.engine`, `.pb`, `.tflite`
- **Configuration files**: `.yaml`, `.yml`, `.json`, `.txt` 
- **Requirements files**: `.txt`, `.yaml`, `.yml`
- **File size limit**: 500MB per file
- **Async processing** for large files

### 2. Comprehensive File Validation
- **Extension validation**: Only allows safe model formats
- **File size limits**: Prevents oversized uploads
- **Model structure validation**: Framework-specific validation
  - PyTorch: `torch.load()` with `weights_only=True` for security
  - ONNX: Model structure and opset validation
  - TensorRT: Engine file validation
- **Security scanning**: Detects potentially malicious patterns
- **Checksum generation**: SHA256 for integrity verification

### 3. Metadata Management
- **Semantic versioning**: Enforced version format (e.g., 1.2.3)
- **Model metadata**: Name, description, type, framework
- **Performance tracking**: Input/output shapes, classes supported
- **Tagging system**: Flexible tag-based organization
- **Training configuration**: Optional training metadata storage

### 4. Security Features
- **Permission-based access**: Requires `models:upload` permission
- **Rate limiting**: 5 uploads per hour per user
- **Malicious content scanning**: Checks for dangerous code patterns
- **File isolation**: Models stored in isolated directories
- **Authentication**: JWT-based user authentication

### 5. Storage Management
- **Organized structure**: `./models/{model_id}/{version}/`
- **File integrity**: Checksum verification
- **Version control**: Complete version history tracking
- **Cleanup operations**: Automated file cleanup on deletion

### 6. Performance Benchmarking
- **Background processing**: Async model benchmarking
- **Performance metrics**: Latency, throughput, memory usage
- **Framework integration**: Compatible with existing deployment pipeline
- **Health monitoring**: Model status tracking

## API Endpoints

### Upload Model
```http
POST /api/v1/models/upload
Content-Type: multipart/form-data
Authorization: Bearer <token>
```

**Form Fields:**
- `name` (string): Model name
- `version` (string): Semantic version (e.g., "1.0.0")
- `model_type` (enum): Type of model (detection, classification, etc.)
- `framework` (enum): ML framework (pytorch, onnx, tensorrt, etc.)
- `description` (string, optional): Model description
- `classes` (JSON string): Array of supported object classes
- `input_shape` (JSON string): Array of input dimensions
- `config` (JSON string): Model configuration object
- `tags` (JSON string, optional): Array of tags
- `benchmark_dataset` (string, optional): Dataset used for evaluation
- `training_config` (JSON string, optional): Training configuration

**File Fields:**
- `model_file` (file): Main model file (required)
- `config_file` (file, optional): Configuration file
- `requirements_file` (file, optional): Requirements/dependencies file

**Response:**
```json
{
  "success": true,
  "message": "Model uploaded successfully",
  "data": {
    "model_id": "yolo11n-1.0.0",
    "name": "YOLO11 Nano",
    "version": "1.0.0", 
    "framework": "pytorch",
    "file_size": 5242880,
    "checksum": "sha256:abc123...",
    "validation_warnings": [],
    "storage_path": "/models/yolo11n-1.0.0/1.0.0"
  }
}
```

### Delete Model
```http
DELETE /api/v1/models/{model_id}
Authorization: Bearer <token>
```

**Requirements:**
- `models:delete` permission
- Model must not be in production stage

### List Models
```http
GET /api/v1/models/
Authorization: Bearer <token>
```

**Query Parameters:**
- `page`: Page number (default: 1)
- `size`: Page size (default: 20, max: 100)
- `model_type`: Filter by model type
- `framework`: Filter by framework
- `status`: Filter by status
- `stage`: Filter by deployment stage
- `search`: Search in name and description

## Usage Examples

### Upload YOLO11 Model

```bash
curl -X POST "http://localhost:8000/api/v1/models/upload" \
  -H "Authorization: Bearer your-jwt-token" \
  -F "name=YOLO11 Nano Traffic" \
  -F "version=1.0.0" \
  -F "model_type=detection" \
  -F "framework=pytorch" \
  -F "description=Lightweight YOLO11 for traffic monitoring" \
  -F "classes=[\"car\", \"truck\", \"bus\", \"motorcycle\", \"bicycle\", \"person\"]" \
  -F "input_shape=[1, 3, 640, 640]" \
  -F "config={\"confidence_threshold\": 0.5, \"iou_threshold\": 0.45}" \
  -F "tags=[\"traffic\", \"real-time\", \"edge\"]" \
  -F "model_file=@yolo11n.pt" \
  -F "config_file=@config.yaml"
```

### Python Client Example

```python
import httpx

async def upload_model():
    files = {
        'model_file': ('yolo11n.pt', open('yolo11n.pt', 'rb'), 'application/octet-stream'),
        'config_file': ('config.yaml', open('config.yaml', 'rb'), 'text/yaml'),
    }
    
    data = {
        'name': 'YOLO11 Nano Traffic',
        'version': '1.0.0',
        'model_type': 'detection',
        'framework': 'pytorch',
        'description': 'Lightweight YOLO11 for traffic monitoring',
        'classes': '["car", "truck", "bus", "motorcycle", "bicycle", "person"]',
        'input_shape': '[1, 3, 640, 640]',
        'config': '{"confidence_threshold": 0.5, "iou_threshold": 0.45}',
        'tags': '["traffic", "real-time", "edge"]'
    }
    
    headers = {'Authorization': 'Bearer your-jwt-token'}
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            'http://localhost:8000/api/v1/models/upload',
            files=files,
            data=data,
            headers=headers
        )
        return response.json()
```

## Storage Structure

```
./models/
├── yolo11n-1.0.0/
│   └── 1.0.0/
│       ├── model.pt
│       ├── config.yaml
│       └── requirements.txt
├── yolo11s-1.1.0/
│   └── 1.1.0/
│       ├── model.pt
│       └── config.json
└── custom-model-2.0.0/
    ├── 1.0.0/
    │   └── model.onnx
    └── 2.0.0/
        ├── model.onnx
        └── config.yaml
```

## Security Considerations

### File Upload Security
- **Extension allowlist**: Only specific file types allowed
- **Content scanning**: Checks for malicious code patterns
- **Size limits**: Prevents resource exhaustion
- **Isolated storage**: Models stored in separate directories
- **Secure loading**: PyTorch models loaded with `weights_only=True`

### Access Control
- **Permission-based**: Requires specific permissions for operations
- **Rate limiting**: Prevents abuse through upload limits
- **Authentication**: JWT token validation required
- **Audit logging**: All operations logged with user context

### Model Validation
- **Framework validation**: Ensures model compatibility
- **Structure verification**: Validates model architecture
- **Checksum verification**: Ensures file integrity
- **Metadata validation**: Enforces proper versioning and format

## Integration with Deployment Pipeline

The upload service integrates seamlessly with the existing deployment pipeline:

1. **Model Upload** → Model stored and validated
2. **Background Benchmarking** → Performance metrics generated
3. **Status Update** → Model marked as available for deployment
4. **Deployment** → Model can be deployed to development/staging/production
5. **A/B Testing** → Models can participate in A/B tests
6. **Optimization** → Models can be optimized for specific platforms

## Performance Considerations

- **Async processing**: Large file uploads don't block the API
- **Background benchmarking**: Performance metrics calculated asynchronously  
- **Caching**: Model metadata cached for fast retrieval
- **Rate limiting**: Prevents resource exhaustion
- **File streaming**: Efficient handling of large model files

## Monitoring and Observability

- **Upload tracking**: Complete audit trail of uploads
- **Performance metrics**: Model benchmarking results
- **Error logging**: Detailed error reporting and debugging
- **Health checks**: Model status monitoring
- **Usage analytics**: Upload patterns and usage statistics

## Error Handling

The service provides comprehensive error handling:

- **400 Bad Request**: Invalid file format, JSON parsing errors, validation failures
- **401 Unauthorized**: Missing or invalid authentication
- **403 Forbidden**: Insufficient permissions
- **409 Conflict**: Model already exists
- **413 Payload Too Large**: File size exceeds limits
- **429 Too Many Requests**: Rate limit exceeded
- **500 Internal Server Error**: Server-side processing errors

All errors include detailed messages for debugging and user guidance.

## Configuration

Environment variables for customization:

- `MODEL_STORAGE_PATH`: Base path for model storage (default: `./models`)
- `MAX_UPLOAD_SIZE`: Maximum file size in bytes (default: 500MB)
- `UPLOAD_RATE_LIMIT`: Uploads per hour per user (default: 5)
- `BENCHMARK_TIMEOUT`: Benchmarking timeout in seconds (default: 300)

## Implementation Summary

### Completed Features ✅

#### Core Upload Functionality
- ✅ **Multi-file upload support**: Model files, config files, requirements files
- ✅ **File type validation**: Extension-based validation with allowlists
- ✅ **File size limits**: Configurable maximum file size (default: 500MB)
- ✅ **Security scanning**: Pattern-based malicious content detection
- ✅ **Checksum verification**: SHA256 for file integrity
- ✅ **Async processing**: Non-blocking file operations

#### Model Validation
- ✅ **Framework-specific validation**: PyTorch, ONNX, TensorRT support
- ✅ **Safe model loading**: PyTorch models loaded with `weights_only=True`
- ✅ **Structure validation**: Basic model format verification
- ✅ **Metadata extraction**: Automatic inference of model properties
- ✅ **Warning system**: Non-fatal validation warnings

#### Storage Management
- ✅ **Organized storage**: Hierarchical directory structure
- ✅ **Version isolation**: Separate directories per model version
- ✅ **Atomic operations**: Rollback on failure
- ✅ **File cleanup**: Automatic cleanup on model deletion
- ✅ **Path management**: Configurable storage locations

#### Security & Access Control
- ✅ **Permission-based access**: Granular permission system
- ✅ **Rate limiting**: Configurable upload limits per user
- ✅ **JWT authentication**: Token-based user authentication
- ✅ **Content scanning**: Malicious pattern detection
- ✅ **Audit logging**: Complete operation logging

#### API Integration
- ✅ **REST endpoints**: Upload, delete, list models
- ✅ **OpenAPI documentation**: Auto-generated API docs
- ✅ **Error handling**: Comprehensive error responses
- ✅ **Response schemas**: Structured API responses
- ✅ **Cache integration**: Redis-based caching

#### Background Processing
- ✅ **Async benchmarking**: Background performance evaluation
- ✅ **Status tracking**: Model lifecycle status management
- ✅ **Metrics generation**: Automated performance metrics
- ✅ **Progress monitoring**: Upload and processing status

#### Utility Functions
- ✅ **File size formatting**: Human-readable file sizes
- ✅ **Version validation**: Semantic version enforcement
- ✅ **Model ID generation**: Standardized naming
- ✅ **Metadata extraction**: Filename-based metadata parsing
- ✅ **Performance scoring**: Model quality assessment

### File Structure

```
src/its_camera_ai/api/
├── routers/
│   └── models.py              # Main upload endpoints and logic
├── schemas/
│   └── models.py              # Pydantic schemas (ModelUpload, etc.)
├── utils.py                   # Utility functions
└── dependencies.py            # Rate limiting and auth

tests/unit/
└── test_model_upload.py       # Comprehensive test suite

./models/                      # Model storage directory
├── {model-id}/
│   └── {version}/
│       ├── model.{ext}        # Main model file
│       ├── config.{ext}       # Optional config file
│       └── requirements.{ext} # Optional requirements
```

### Key Implementation Files

1. **`src/its_camera_ai/api/routers/models.py`**
   - Main upload endpoint (`POST /upload`)
   - Model deletion endpoint (`DELETE /{model_id}`)
   - File validation and security functions
   - Storage management functions
   - Background benchmarking tasks

2. **`src/its_camera_ai/api/utils.py`**
   - File size formatting utilities
   - Version validation functions
   - Model metadata extraction
   - Performance scoring algorithms

3. **`src/its_camera_ai/api/schemas/models.py`**
   - `ModelUpload` schema for validation
   - Enhanced existing schemas for upload support

4. **`tests/unit/test_model_upload.py`**
   - Comprehensive test suite
   - File validation tests
   - Security scanning tests
   - Storage management tests
   - Integration workflow tests

### Configuration Options

```python
# Environment variables
MODEL_STORAGE_PATH = "./models"          # Base storage path
MAX_UPLOAD_SIZE = 500 * 1024 * 1024      # 500MB file size limit
UPLOAD_RATE_LIMIT = 5                    # Uploads per hour per user

# Allowed file extensions
ALLOWED_MODEL_EXTENSIONS = {".pt", ".pth", ".onnx", ".engine", ".pb", ".tflite"}
ALLOWED_CONFIG_EXTENSIONS = {".yaml", ".yml", ".json", ".txt"}

# Security patterns
MALICIOUS_PATTERNS = ["exec(", "eval(", "__import__", "subprocess", "os.system"]
```

## Future Enhancements

- **Model versioning API**: Advanced version management
- **Batch uploads**: Multiple model upload support  
- **Model conversion**: Automatic format conversion
- **Advanced validation**: ML-specific model validation
- **Performance baselines**: Automated performance comparison
- **Model marketplace**: Public model sharing
- **Integration testing**: Automated model testing pipeline