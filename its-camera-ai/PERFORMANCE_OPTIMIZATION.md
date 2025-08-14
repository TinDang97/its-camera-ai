# Performance Optimization System - ITS Camera AI

This document provides comprehensive guidance on the performance optimization system for ITS Camera AI, designed to achieve **sub-100ms end-to-end latency** with support for **100+ concurrent streams**.

## Overview

The performance optimization system consists of five integrated components:

1. **GPU Memory Optimizer** - TensorRT optimization and GPU memory pooling
2. **Streaming Cache Manager** - Multi-level caching with predictive algorithms
3. **Connection Pool Optimizer** - Redis, database, and FFMPEG process pool optimization
4. **Latency Monitor** - Real-time SLA monitoring with automated alerting
5. **Adaptive Quality Manager** - Dynamic quality adjustment based on system load

## Quick Start

### 1. Enable Performance Optimization

```bash
# Check current performance status
its-camera-ai performance status

# Initialize performance optimization with default settings
its-camera-ai performance optimize --strategy latency_optimized

# Generate configuration file
its-camera-ai performance config --strategy balanced --output perf_config.json
```

### 2. Programmatic Usage

```python
from its_camera_ai.performance import (
    create_performance_optimizer,
    create_production_optimization_config,
    OptimizationStrategy,
    PipelineStage
)

# Create latency-optimized configuration
config = create_production_optimization_config(
    max_concurrent_streams=100,
    target_latency_ms=80.0,
    strategy=OptimizationStrategy.LATENCY_OPTIMIZED
)

# Initialize optimizer
optimizer = await create_performance_optimizer(config)

# Use in streaming context
async with optimizer.optimize_stream_processing("camera_001") as context:
    # Your stream processing code here
    optimized_model = context.get("optimized_model")
    cache_manager = context.get("cache_manager")
    quality_profile = context.get("quality_profile")
```

### 3. Integration with Streaming Service

```python
from its_camera_ai.services.streaming_service import StreamingService

# Create streaming service
streaming_service = StreamingService()

# Initialize with performance optimization
await streaming_service.initialize_performance_optimization(
    optimization_config={
        "strategy": "latency_optimized",
        "target_latency_ms": 90.0,
        "gpu_optimization_enabled": True,
        "caching_enabled": True
    },
    redis_url="redis://localhost:6379",
    database_url="postgresql://localhost/its_camera_ai"
)
```

## Architecture Deep Dive

### GPU Memory Optimizer

Optimizes GPU inference for YOLO11 models with production-ready performance:

**Key Features:**
- **TensorRT Optimization**: Automatic model optimization with FP16/INT8 precision
- **Dynamic Batching**: Adaptive batch sizing based on load (1-32 batch size)
- **Memory Pooling**: Efficient GPU memory allocation with 8GB default pool
- **Model Quantization**: INT8 quantization with >90% accuracy retention

**Configuration:**
```python
gpu_config = GPUOptimizationConfig(
    enable_tensorrt=True,
    tensorrt_precision="fp16",  # fp32, fp16, int8
    tensorrt_max_batch_size=32,
    gpu_memory_pool_size_gb=8.0,
    enable_dynamic_batching=True,
    batch_timeout_ms=10.0,
    enable_quantization=True,
    quantization_mode="dynamic"  # dynamic, static
)
```

**Performance Targets:**
- GPU Utilization: >85%
- Batch Processing Latency: <50ms for YOLO11
- Memory Efficiency: >90%

### Streaming Cache Manager

Multi-level caching system with predictive algorithms:

**Cache Hierarchy:**
- **L1 Cache**: In-memory, 512MB, 5-second TTL
- **L2 Cache**: Redis distributed, 2GB, 5-minute TTL
- **L3 Cache**: CDN-ready with compression, 1-hour TTL

**Features:**
- **Predictive Caching**: ML-based access pattern prediction
- **Cache Warming**: Automatic pre-caching of popular streams
- **Compression**: LZ4/gzip compression for storage optimization
- **Hit Rate Monitoring**: Target >85% cache hit ratio

**Configuration:**
```python
caching_config = CachingConfig(
    l1_cache_size_mb=512,
    l1_cache_ttl_seconds=5,
    l2_cache_enabled=True,
    l2_cache_ttl_seconds=300,
    l3_cache_enabled=True,
    l3_compression_enabled=True,
    enable_predictive_caching=True
)
```

### Connection Pool Optimizer

Optimizes all external connections for maximum throughput:

**Pool Types:**
- **Redis Pool**: 50 connections, keepalive enabled
- **Database Pool**: 20 connections with overflow, async support
- **FFMPEG Pool**: 10 concurrent processes, automatic restart

**Configuration:**
```python
pool_config = ConnectionPoolConfig(
    redis_pool_size=50,
    redis_max_connections=100,
    db_pool_size=20,
    db_max_overflow=30,
    ffmpeg_pool_size=10,
    ffmpeg_max_concurrent=20
)
```

### Latency Monitor

Real-time SLA monitoring with comprehensive metrics:

**Pipeline Stages Tracked:**
- Camera Capture: <20ms
- Preprocessing: <10ms
- ML Inference: <50ms
- Postprocessing: <5ms
- Encoding: <10ms
- Network Transmission: <20ms
- **End-to-End**: <100ms (SLA target)

**Features:**
- **SLA Violation Detection**: Automated alerts for violations
- **Performance Regression**: Statistical regression detection
- **Prometheus Integration**: Metrics export for monitoring
- **Distributed Tracing**: End-to-end request tracking

**Configuration:**
```python
latency_config = LatencyMonitoringConfig(
    latency_sla_ms=100.0,
    latency_p95_target_ms=80.0,
    enable_sla_alerts=True,
    regression_detection_enabled=True,
    enable_distributed_tracing=True
)
```

### Adaptive Quality Manager

Dynamic quality adjustment based on system performance:

**Quality Levels:**
- **Ultra High**: 4K, 30fps, enhanced ML
- **High**: 1080p, 30fps, full ML (default)
- **Medium**: 720p, 25fps, standard ML
- **Low**: 360p, 20fps, simplified ML
- **Ultra Low**: 240p, 15fps, minimal ML

**Load-based Adjustment:**
- **Optimal**: High quality maintained
- **Elevated**: Medium quality adjustment
- **High**: Low quality for performance
- **Critical**: Ultra low quality (emergency mode)

**Configuration:**
```python
quality_config = AdaptiveQualityConfig(
    enable_adaptive_quality=True,
    default_quality="high",
    quality_adjustment_interval_seconds=30,
    enable_priority_streaming=True,
    priority_camera_ids=["critical_camera_001"],
    quality_recovery_enabled=True
)
```

## Optimization Strategies

### 1. Latency Optimized (Sub-100ms Target)

**Optimizations:**
- Batch timeout: 5ms (vs 10ms default)
- TensorRT batch size: 4 (vs 8 default)
- L1 cache TTL: 3s (vs 5s default)
- SLA target: 80ms (vs 100ms default)

**Use Case:** Real-time traffic monitoring, emergency response

```python
config = create_production_optimization_config(
    strategy=OptimizationStrategy.LATENCY_OPTIMIZED,
    target_latency_ms=80.0
)
```

### 2. Memory Optimized (Resource Constrained)

**Optimizations:**
- GPU memory pool: 4GB (vs 8GB default)
- L1 cache: 256MB (vs 512MB default)
- Batch size: 16 (vs 32 default)
- Connection pools: 50% smaller

**Use Case:** Edge deployments, resource-constrained environments

```python
config = create_production_optimization_config(
    strategy=OptimizationStrategy.MEMORY_OPTIMIZED,
    max_concurrent_streams=200
)
```

### 3. Balanced Production (Default)

**Optimizations:**
- Balanced CPU/GPU/memory usage
- Standard cache sizes and TTLs
- Moderate batch processing
- Adaptive quality enabled

**Use Case:** General production deployments

```python
config = create_production_optimization_config(
    strategy=OptimizationStrategy.BALANCED,
    max_concurrent_streams=150
)
```

## Performance Monitoring

### CLI Monitoring Commands

```bash
# Real-time performance monitoring
its-camera-ai performance monitor --interval 5 --duration 300

# Performance benchmarking
its-camera-ai performance benchmark --streams 50 --duration 60

# Detailed status with metrics
its-camera-ai performance status --detailed --format json
```

### Programmatic Monitoring

```python
# Get comprehensive metrics
metrics = optimizer.get_comprehensive_metrics()

# Check optimization status
status = optimizer.get_optimization_status()

# Record custom measurements
await optimizer.record_pipeline_measurement(
    camera_id="camera_001",
    stage=PipelineStage.ML_INFERENCE,
    latency_ms=45.0,
    metadata={"batch_size": 8}
)
```

### Key Performance Indicators (KPIs)

**Latency Metrics:**
- End-to-end P95 latency: <80ms
- SLA violation rate: <5%
- Pipeline stage breakdown available

**Throughput Metrics:**
- Frames per second: 30,000+ aggregate
- Concurrent streams: 100+
- GPU utilization: >85%

**Efficiency Metrics:**
- Cache hit rate: >85%
- Memory efficiency: >90%
- Connection pool utilization: <80%

## Production Deployment

### Environment Configuration

```bash
# Environment variables for production
export PERFORMANCE__ENABLED=true
export PERFORMANCE__STRATEGY=latency_optimized
export PERFORMANCE__TARGET_LATENCY_MS=90
export PERFORMANCE__GPU_OPTIMIZATION_ENABLED=true
export PERFORMANCE__CACHING_ENABLED=true
export PERFORMANCE__L1_CACHE_SIZE_MB=1024
export PERFORMANCE__REDIS_POOL_SIZE=100
export PERFORMANCE__ENABLE_SLA_ALERTS=true
```

### Docker Configuration

```dockerfile
# Dockerfile optimizations for performance
FROM nvidia/cuda:12.0-runtime-ubuntu22.04

# Install performance dependencies
RUN apt-get update && apt-get install -y \
    libnvidia-ml1 \
    nvidia-utils-520 \
    && rm -rf /var/lib/apt/lists/*

# Set GPU memory growth
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Performance environment
ENV PERFORMANCE__ENABLED=true
ENV PERFORMANCE__STRATEGY=balanced
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: its-camera-ai-performance
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: its-camera-ai
        image: its-camera-ai:latest
        resources:
          requests:
            memory: "4Gi"
            cpu: "2000m"
            nvidia.com/gpu: 1
          limits:
            memory: "8Gi" 
            cpu: "4000m"
            nvidia.com/gpu: 1
        env:
        - name: PERFORMANCE__ENABLED
          value: "true"
        - name: PERFORMANCE__STRATEGY
          value: "balanced"
        - name: PERFORMANCE__GPU_OPTIMIZATION_ENABLED
          value: "true"
        livenessProbe:
          httpGet:
            path: /api/v1/health/performance
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
```

## Troubleshooting

### Common Performance Issues

**1. High Latency (>100ms)**
```bash
# Check pipeline breakdown
its-camera-ai performance status --detailed

# Identify bottleneck stages
its-camera-ai performance monitor --interval 1
```

**Solutions:**
- Enable TensorRT optimization
- Reduce batch timeout
- Increase cache sizes
- Check GPU utilization

**2. Low Cache Hit Rate (<70%)**
```bash
# Check cache metrics
its-camera-ai performance status --format json | jq '.performance_summary.cache_hit_rate_percent'
```

**Solutions:**
- Increase cache sizes
- Enable predictive caching
- Adjust TTL settings
- Check access patterns

**3. GPU Underutilization (<60%)**
```bash
# Check GPU metrics
nvidia-smi
its-camera-ai performance monitor
```

**Solutions:**
- Increase batch sizes
- Enable dynamic batching
- Check model optimization
- Verify TensorRT setup

**4. Memory Issues**
```bash
# Check memory usage
its-camera-ai performance status --detailed
```

**Solutions:**
- Use memory-optimized strategy
- Reduce cache sizes
- Enable model quantization
- Check for memory leaks

### Performance Tuning Guide

**Step 1: Profile Current Performance**
```python
# Run comprehensive benchmark
config = create_production_optimization_config(
    strategy=OptimizationStrategy.BALANCED
)
optimizer = await create_performance_optimizer(config)

# Monitor for baseline metrics
await optimizer.start()
# ... run your workload ...
metrics = optimizer.get_comprehensive_metrics()
```

**Step 2: Identify Bottlenecks**
- Check pipeline stage latencies
- Analyze GPU utilization
- Review cache hit rates
- Monitor connection pools

**Step 3: Apply Targeted Optimizations**
- GPU: Enable TensorRT, adjust batch sizes
- Cache: Increase sizes, enable prediction
- Pools: Scale connection limits
- Quality: Enable adaptive management

**Step 4: Validate Improvements**
- Measure latency improvements
- Confirm SLA compliance
- Check resource utilization
- Monitor stability

## Best Practices

### Development

1. **Always Profile First**: Use benchmarking tools before optimizing
2. **Incremental Optimization**: Apply one optimization at a time
3. **Monitor Regressions**: Set up automated performance testing
4. **Test Under Load**: Validate with realistic concurrent loads

### Production

1. **Gradual Rollout**: Deploy optimizations gradually
2. **Monitor Closely**: Watch KPIs during and after deployment
3. **Have Rollback Plans**: Prepare rollback procedures
4. **Capacity Planning**: Monitor resource usage trends

### Configuration

1. **Environment-Specific**: Use different configs for dev/prod
2. **Start Conservative**: Begin with balanced strategy
3. **Tune Incrementally**: Adjust parameters based on metrics
4. **Document Changes**: Track optimization changes

## API Reference

### Core Classes

- `PerformanceOptimizer`: Main coordinator class
- `OptimizationConfig`: Configuration management
- `GPUMemoryOptimizer`: GPU optimization
- `StreamingCacheManager`: Multi-level caching
- `LatencyMonitor`: SLA monitoring
- `AdaptiveQualityManager`: Quality management

### Factory Functions

- `create_performance_optimizer()`: Create optimizer instance
- `create_production_optimization_config()`: Create production config
- `create_latency_optimized_system()`: Create latency-focused system
- `create_balanced_performance_system()`: Create balanced system

### CLI Commands

- `its-camera-ai performance status`: Show performance status
- `its-camera-ai performance benchmark`: Run benchmarks
- `its-camera-ai performance monitor`: Real-time monitoring
- `its-camera-ai performance optimize`: Initialize optimization
- `its-camera-ai performance config`: Generate config files

## Support and Contributing

For questions about performance optimization:

1. Check this documentation
2. Review example configurations
3. Run diagnostic commands
4. Check monitoring dashboards

For performance issues:

1. Collect metrics with `performance status --detailed`
2. Run benchmarks to establish baseline
3. Check logs for errors or warnings
4. Report issues with full context

---

**Performance Optimization System v1.0**  
*Designed for sub-100ms latency with 100+ concurrent streams*