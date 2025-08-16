# ML Pipeline Integration Documentation

## Overview

This documentation covers the complete integration of the ML pipeline with streaming endpoints for the ITS Camera AI system. The integration provides:

- **Real ML Inference**: YOLO11 models connected to gRPC streaming endpoints
- **Sub-100ms Latency**: Optimized pipeline for real-time traffic monitoring  
- **1000+ Concurrent Streams**: Scalable architecture with GPU optimization
- **Event-Driven Analytics**: Kafka streaming for real-time dashboard updates
- **Production-Ready**: Comprehensive error handling and monitoring

## Architecture

```
┌─────────────────┐    ┌──────────────────────┐    ┌─────────────────────┐
│   Camera Feed   │───▶│  gRPC Streaming      │───▶│   ML Pipeline       │
│   (RTSP/WebRTC) │    │  Servicers           │    │   Integration       │
└─────────────────┘    └──────────────────────┘    └─────────────────────┘
                                │                           │
                                ▼                           ▼
┌─────────────────┐    ┌──────────────────────┐    ┌─────────────────────┐
│  Real-time      │◀───│  Kafka Event         │◀───│  Core Vision Engine │
│  Dashboard/SSE  │    │  Streaming           │    │  (YOLO11 Inference) │
└─────────────────┘    └──────────────────────┘    └─────────────────────┘
                                │                           │
                                ▼                           ▼
┌─────────────────┐    ┌──────────────────────┐    ┌─────────────────────┐
│   External      │◀───│  Analytics Service   │◀───│  Unified Analytics  │
│   Systems       │    │  (Traffic Metrics)   │    │  Engine             │
└─────────────────┘    └──────────────────────┘    └─────────────────────┘
```

## Key Components

### 1. ML Streaming Integration Service (`ml_streaming_integration_service.py`)

**Purpose**: Central orchestrator connecting ML inference with streaming endpoints.

**Features**:
- Initializes and manages CoreVisionEngine (YOLO11)
- Coordinates with UnifiedVisionAnalyticsEngine
- Provides unified interface for frame processing
- Tracks performance metrics and health status

**Key Methods**:
```python
# Process single frame with complete ML pipeline
async def process_frame(
    frame: np.ndarray,
    camera_id: str,
    frame_id: str = None,
    priority: RequestPriority = RequestPriority.NORMAL,
    include_analytics: bool = True,
) -> UnifiedResult

# Process batch for optimal GPU utilization
async def process_batch(
    frames: list[np.ndarray],
    camera_ids: list[str],
    frame_ids: list[str] = None,
) -> list[UnifiedResult]
```

### 2. Unified Vision Analytics Engine (Enhanced)

**Purpose**: High-performance engine combining ML inference with real-time analytics.

**Enhancements**:
- **Real ML Integration**: Connects to CoreVisionEngine for actual YOLO11 inference
- **Kafka Event Streaming**: Publishes detection/analytics events in real-time
- **GPU Memory Management**: Optimized CUDA streams and memory pooling
- **Circuit Breaker**: Fault tolerance with graceful degradation

**Critical Methods Added**:
```python
# Run actual YOLO11 inference through CoreVisionEngine
async def _run_vision_inference(
    frames: list[np.ndarray], 
    frame_ids: list[str],
    camera_ids: list[str],
    device_id: int
) -> list[DetectionResult]

# Stream detection events to Kafka for real-time processing
async def _stream_detection_event(
    detections: list[DetectionResultDTO], 
    camera_id: str, 
    frame_id: str
)
```

### 3. Production ML gRPC Server (`production_ml_grpc_server.py`)

**Purpose**: Production-ready gRPC server with complete ML pipeline integration.

**Features**:
- **VisionCoreService**: Complete ML inference + analytics in single RPC
- **StreamingService**: Real-time frame processing with batching
- **AnalyticsService**: Traffic analytics and metrics
- **Health Monitoring**: Comprehensive health checks and performance tracking

**Server Configuration**:
```python
# Optimized for production workloads
server = grpc.aio.server(
    futures.ThreadPoolExecutor(max_workers=20),
    options=[
        ("grpc.max_send_message_length", 100 * 1024 * 1024),  # 100MB
        ("grpc.max_receive_message_length", 100 * 1024 * 1024),
        ("grpc.keepalive_time_ms", 10000),
        ("grpc.keepalive_timeout_ms", 2000),
    ],
)
```

### 4. Kafka Event Processor (`kafka_event_processor.py`)

**Purpose**: Real-time event streaming for detection results and analytics.

**Event Types**:
- **Detection Events**: ML inference results with bounding boxes
- **Analytics Events**: Traffic metrics, violations, anomalies
- **Metrics Events**: System performance and health data
- **Alert Events**: Incidents and notifications

**Usage**:
```python
# Publish detection event for real-time dashboard updates
await event_processor.publish_detection_event(
    camera_id="cam_001",
    frame_id="frame_123", 
    detections=detection_data,
    metadata={"quality_score": 0.85}
)
```

## Performance Optimizations

### 1. GPU Memory Management

**Unified Memory Manager**:
- Zero-copy operations between CPU and GPU
- Predictive allocation based on usage patterns
- Automatic memory cleanup and defragmentation

**CUDA Streams Manager**:
- Parallel processing across multiple GPU streams
- Device affinity for consistent camera routing
- Load balancing across multiple GPUs

### 2. Adaptive Batching

**Dynamic Batch Sizing**:
```python
# Batch size adapts based on system load
current_metrics = self._get_current_performance_metrics()
target_batch_size = self.batch_sizer.adapt_batch_size(
    current_load=current_metrics["avg_latency_ms"],
    target_latency=50.0,  # Target 50ms latency
    queue_depth=queue_depth,
    gpu_utilization=current_metrics["gpu_utilization"]
)
```

**Priority Queuing**:
- Emergency requests bypass normal queue
- High-priority streams get preferential processing
- Adaptive timeout management

### 3. Circuit Breaker Pattern

**GPU-Optimized Fault Tolerance**:
```python
# Separate failure tracking for different error types
failure_types = {
    "cuda_error": 0,
    "memory_error": 0, 
    "timeout_error": 0,
    "general_error": 0,
}

# Faster circuit opening for persistent GPU memory issues
if gpu_memory_failures >= gpu_memory_failure_threshold:
    self.state = "open"
```

## Real-Time Data Flow

### 1. Detection Event Flow

```mermaid
graph LR
    A[Camera Frame] --> B[gRPC Servicer]
    B --> C[Unified Engine]
    C --> D[Core Vision Engine]
    D --> E[YOLO11 Inference]
    E --> F[Detection Results]
    F --> G[Analytics Processing]
    G --> H[Kafka Event Stream]
    H --> I[Dashboard/SSE]
```

### 2. Event Streaming Architecture

**Topics**:
- `detection_events`: Real-time ML inference results
- `analytics_events`: Traffic metrics and analysis
- `metrics_events`: System performance data
- `alert_events`: Incidents and notifications

**Event Format**:
```json
{
  "event_type": "detection",
  "timestamp": "2024-01-15T10:30:45.123Z",
  "camera_id": "cam_001",
  "frame_id": "frame_123",
  "data": {
    "detections": [
      {
        "class_name": "car",
        "confidence": 0.85,
        "bbox": {"x_min": 100, "y_min": 150, "x_max": 300, "y_max": 400},
        "tracking_id": 1,
        "vehicle_type": "car",
        "speed": 45.0,
        "direction": "north",
        "is_vehicle": true
      }
    ],
    "vehicle_count": 2
  }
}
```

## Running the System

### 1. Quick Start

```bash
# Start with default configuration
python scripts/run_ml_streaming_pipeline.py

# Start with custom model and GPU settings
python scripts/run_ml_streaming_pipeline.py \
    --model models/yolo11m.pt \
    --gpus 0,1 \
    --port 50051

# Start with Kafka streaming enabled
python scripts/run_ml_streaming_pipeline.py \
    --kafka \
    --kafka-servers localhost:9092,localhost:9093

# Production mode with optimizations
python scripts/run_ml_streaming_pipeline.py \
    --prod \
    --monitoring-port 8080
```

### 2. Configuration Options

**Model Configuration**:
- `--model`: Path to YOLO11 model file
- `--confidence`: Detection confidence threshold (0.0-1.0)
- `--iou`: IoU threshold for NMS (0.0-1.0)

**Performance Configuration**:
- `--batch-size`: Inference batch size (default: 8)
- `--max-batch-size`: Maximum batch size (default: 32)
- `--target-fps`: Target FPS per camera (default: 30)

**GPU Configuration**:
- `--gpus`: GPU device IDs (comma-separated)
- `--no-tensorrt`: Disable TensorRT optimization
- `--precision`: Inference precision (fp32/fp16/int8)

### 3. Docker Deployment

```yaml
# docker-compose.yml
version: '3.8'
services:
  ml-pipeline:
    build: .
    command: python scripts/run_ml_streaming_pipeline.py --prod --kafka
    ports:
      - "50051:50051"
    volumes:
      - ./models:/app/models
    environment:
      - CUDA_VISIBLE_DEVICES=0,1
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 2
              capabilities: [gpu]
```

## Testing

### 1. Integration Tests

```bash
# Run complete ML pipeline integration tests
pytest tests/integration/test_ml_pipeline_integration.py -v

# Test specific components
pytest tests/integration/test_ml_pipeline_integration.py::TestMLPipelineIntegration -v
pytest tests/integration/test_ml_pipeline_integration.py::TestKafkaEventStreaming -v
pytest tests/integration/test_ml_pipeline_integration.py::TestEndToEndPipelineFlow -v
```

### 2. Performance Testing

```bash
# Test inference latency (should be <100ms)
pytest tests/integration/test_ml_pipeline_integration.py::test_complete_detection_to_analytics_flow -v

# Test concurrent stream processing  
pytest tests/integration/test_ml_pipeline_integration.py::test_concurrent_stream_processing -v

# Test batch processing optimization
pytest tests/integration/test_ml_pipeline_integration.py::test_batch_processing_optimization -v
```

### 3. Load Testing

```bash
# Generate synthetic camera streams for load testing
python scripts/generate_test_streams.py \
    --cameras 100 \
    --fps 30 \
    --duration 300 \
    --target-server localhost:50051
```

## Monitoring

### 1. Health Checks

```bash
# Check overall system health
curl http://localhost:8080/health

# Check ML service health
curl http://localhost:8080/ml/health

# Check gRPC server health
grpc_health_probe -addr=localhost:50051
```

### 2. Performance Metrics

**Key Metrics**:
- `inference_latency_ms`: ML inference time (target: <50ms)
- `total_processing_time_ms`: End-to-end processing (target: <100ms)
- `throughput_fps`: Frames processed per second
- `gpu_utilization`: GPU usage percentage (target: >80%)
- `queue_depth`: Number of pending requests
- `error_rate`: Percentage of failed requests

**Prometheus Metrics**:
```
# HELP ml_inference_latency_seconds ML inference latency
# TYPE ml_inference_latency_seconds histogram
ml_inference_latency_seconds_bucket{le="0.05"} 1250
ml_inference_latency_seconds_bucket{le="0.1"} 1340
ml_inference_latency_seconds_bucket{le="+Inf"} 1400

# HELP ml_throughput_fps Current ML processing throughput
# TYPE ml_throughput_fps gauge
ml_throughput_fps{camera_id="cam_001"} 28.5
```

### 3. Alerting

**Critical Alerts**:
- Inference latency > 100ms for >5 minutes
- GPU utilization < 50% (underutilization)
- Error rate > 5%
- Queue depth > 1000 requests

**Warning Alerts**:
- Inference latency > 75ms
- Throughput < target FPS
- Memory usage > 90%

## Troubleshooting

### 1. Common Issues

**High Latency**:
```bash
# Check GPU utilization
nvidia-smi

# Check queue depth
curl http://localhost:8080/metrics | grep queue_depth

# Reduce batch size for lower latency
python scripts/run_ml_streaming_pipeline.py --batch-size 4
```

**Memory Issues**:
```bash
# Monitor GPU memory
watch -n 1 nvidia-smi

# Enable memory debugging
export CUDA_LAUNCH_BLOCKING=1
python scripts/run_ml_streaming_pipeline.py --log-level DEBUG
```

**Connection Issues**:
```bash
# Test gRPC connectivity
grpcurl -plaintext localhost:50051 list

# Check Kafka connectivity
kafka-topics --bootstrap-server localhost:9092 --list
```

### 2. Performance Tuning

**For Low Latency**:
- Reduce batch size (--batch-size 4)
- Use fp16 precision (--precision fp16)
- Enable TensorRT optimization (default)
- Increase GPU memory allocation

**For High Throughput**:
- Increase batch size (--batch-size 16)
- Use multiple GPUs (--gpus 0,1,2,3)
- Enable Kafka streaming (--kafka)
- Optimize network settings

### 3. Debug Mode

```bash
# Run with comprehensive debugging
python scripts/run_ml_streaming_pipeline.py \
    --log-level DEBUG \
    --no-tensorrt \
    --batch-size 1 \
    --monitoring-port 8080
```

## Production Deployment

### 1. Prerequisites

**Hardware**:
- NVIDIA GPU with Compute Capability ≥ 7.5
- 16GB+ GPU memory for optimal performance
- 32GB+ system RAM
- NVMe SSD for model storage

**Software**:
- CUDA 11.8+ and cuDNN 8.6+
- Python 3.11+
- Docker and Docker Compose
- Kafka cluster (if streaming enabled)

### 2. Deployment Steps

1. **Model Preparation**:
```bash
# Download and optimize YOLO11 models
python scripts/prepare_models.py --optimize --precision fp16
```

2. **Configuration**:
```bash
# Create production configuration
cp config/production.example.yaml config/production.yaml
# Edit configuration for your environment
```

3. **Database Setup**:
```bash
# Run database migrations
alembic upgrade head
# Seed initial data
python scripts/seed_data.py --env production
```

4. **Start Services**:
```bash
# Start with production configuration
python scripts/run_ml_streaming_pipeline.py \
    --prod \
    --config config/production.yaml \
    --kafka \
    --monitoring-port 8080
```

### 3. Production Checklist

- [ ] GPU drivers and CUDA installed
- [ ] Models downloaded and optimized
- [ ] Database initialized and migrated
- [ ] Kafka cluster configured (if enabled)
- [ ] Monitoring and alerting setup
- [ ] Load balancer configured
- [ ] SSL/TLS certificates installed
- [ ] Backup and disaster recovery tested

## API Reference

### gRPC Services

**VisionCoreService**:
- `ProcessFrame`: Single frame with ML inference + analytics
- `ProcessFrameBatch`: Batch processing for efficiency
- `ProcessCameraStream`: Bidirectional streaming
- `GetEngineHealth`: System health status
- `GetEngineMetrics`: Performance metrics

**StreamingService**:
- `StreamFrames`: Real-time frame streaming
- `ProcessFrameBatch`: Batch frame processing
- `GetQueueMetrics`: Queue status and metrics
- `RegisterStream`: Camera stream registration

**AnalyticsService**:
- `GetTrafficMetrics`: Real-time traffic analysis
- `GetAggregatedMetrics`: Historical data aggregation
- `GetCameraHealth`: Per-camera health status

### Event Streaming

**Topics**:
- `detection_events`: ML inference results
- `analytics_events`: Traffic analytics data
- `metrics_events`: System performance metrics
- `alert_events`: Incidents and notifications

## Contributing

1. **Development Setup**:
```bash
# Install development dependencies
uv sync --group dev --group ml --group gpu

# Run tests
pytest tests/integration/test_ml_pipeline_integration.py -v

# Run linting
ruff check src/ tests/
mypy src/
```

2. **Adding New Features**:
- Follow the established patterns in `ml_streaming_integration_service.py`
- Add comprehensive tests in `tests/integration/`
- Update documentation and configuration examples
- Ensure performance requirements are met (<100ms latency)

3. **Performance Testing**:
- Test with realistic camera stream loads
- Verify GPU utilization and memory usage
- Measure end-to-end latency under load
- Test error handling and recovery scenarios

---

This integration provides the crucial missing link between the ML pipeline and streaming endpoints, enabling real-time traffic monitoring with production-grade performance and reliability.