# Docker Deployment Guide for ITS Camera AI

This guide covers the comprehensive Docker deployment strategy for the ITS Camera AI system, featuring multi-stage builds optimized for different deployment scenarios.

## Quick Start

### Development Environment
```bash
# Build and run development environment
./docker-build.sh development
docker-compose up app

# With GPU support
./docker-build.sh gpu-dev
docker-compose --profile gpu up app-gpu
```

### Production Deployment
```bash
# Build production image
./docker-build.sh production

# Run production stack
docker-compose --profile prod up -d
```

## Docker Build Targets

### 1. Development (`development`)
**Purpose**: Full development environment with hot reload and debugging capabilities.

**Features**:
- Hot code reloading
- Debug port (5678) exposed  
- Full development dependencies
- Testing tools included
- Interactive debugging support

**Usage**:
```bash
./docker-build.sh development
docker run -p 8000:8000 -p 5678:5678 -v $(pwd)/src:/app/src its-camera-ai:dev
```

### 2. GPU Development (`gpu-development`) 
**Purpose**: GPU-enabled development with CUDA support.

**Features**:
- NVIDIA CUDA 12.6 runtime
- GPU memory optimization
- PyTorch GPU acceleration  
- Development tools included
- GPU monitoring capabilities

**Requirements**:
- NVIDIA Docker runtime
- CUDA-compatible GPU
- nvidia-container-toolkit

**Usage**:
```bash
./docker-build.sh gpu-dev
docker run --gpus all -p 8000:8000 its-camera-ai:gpu-dev
```

### 3. Production (`production`)
**Purpose**: Optimized production runtime with minimal attack surface.

**Features**:
- Minimal system dependencies
- Non-root execution (user ID 1001)
- Security hardening
- Memory optimizations
- Multi-platform support (AMD64, ARM64)
- Sub-100ms inference latency
- Comprehensive health checks

**Usage**:
```bash
./docker-build.sh production
docker run -p 8000:8000 its-camera-ai:production
```

### 4. GPU Production (`gpu-production`)
**Purpose**: GPU-accelerated production deployment.

**Features**:
- Production-optimized CUDA environment
- GPU memory management
- Hardware acceleration for inference
- Security hardening
- Performance monitoring

**Usage**:
```bash
./docker-build.sh gpu-prod  
docker run --gpus all -p 8000:8000 its-camera-ai:gpu-prod
```

### 5. Edge Deployment (`edge`)
**Purpose**: Lightweight deployment for resource-constrained environments.

**Features**:
- Minimal footprint
- ARM64 and x86_64 support
- Optimized for edge devices
- Reduced memory usage
- Platform-specific optimizations

**Usage**:
```bash
./docker-build.sh edge
docker run -p 8000:8000 its-camera-ai:edge
```

### 6. Triton Inference (`triton-inference`)
**Purpose**: High-performance batch inference using NVIDIA Triton.

**Features**:
- NVIDIA Triton Inference Server
- High-throughput batch processing
- Model ensemble support
- gRPC and HTTP interfaces
- Advanced model management

**Usage**:
```bash
./docker-build.sh triton
docker run --gpus all -p 8000:8000 -p 8001:8001 -p 8002:8002 its-camera-ai:triton
```

### 7. Testing (`testing`)
**Purpose**: Comprehensive testing environment with full test suite.

**Features**:
- Complete test dependencies
- Coverage reporting
- Security scanning
- Code quality checks
- CI/CD integration

**Usage**:
```bash
./docker-build.sh test
```

## Build Script Usage

The `docker-build.sh` script provides a comprehensive build interface:

```bash
# Basic commands
./docker-build.sh development    # Build dev environment
./docker-build.sh production     # Build production image
./docker-build.sh gpu-prod       # Build GPU production
./docker-build.sh edge          # Build edge deployment
./docker-build.sh triton        # Build Triton inference

# Advanced commands
./docker-build.sh test          # Run test suite
./docker-build.sh all           # Build all targets
./docker-build.sh push          # Build and push to registry
./docker-build.sh scan          # Security scan with Trivy
./docker-build.sh clean         # Clean build cache

# Environment variables
VERSION=1.0.0 ./docker-build.sh production
PLATFORM=linux/arm64 ./docker-build.sh edge
REGISTRY=myregistry.com/org ./docker-build.sh push
```

## Docker Compose Profiles

### Development Profile (`dev`)
```bash
docker-compose --profile dev up
```
- Application with development features
- Hot reload enabled
- Debug ports exposed
- Full logging

### GPU Profile (`gpu`) 
```bash
docker-compose --profile gpu up
```
- GPU-enabled application
- NVIDIA runtime configuration
- GPU resource reservations
- CUDA environment variables

### Production Profile (`prod`)
```bash
docker-compose --profile prod up -d
```
- Production-optimized configuration
- Resource limits
- Health checks
- Restart policies

### Monitoring Profile (`monitoring`)
```bash
docker-compose --profile monitoring up
```
- Prometheus metrics collection
- Grafana visualization
- Performance monitoring
- Alert management

## Performance Optimizations

### Memory Management
- **MALLOC_ARENA_MAX=2**: Reduces memory fragmentation
- **MALLOC_MMAP_THRESHOLD_=131072**: Optimizes large allocations
- **PYTORCH_CUDA_ALLOC_CONF**: GPU memory management

### CPU Optimization  
- **OMP_NUM_THREADS**: Controls OpenMP parallelism
- **OPENBLAS_NUM_THREADS**: BLAS threading
- Multi-stage builds reduce final image size

### Inference Performance
- Model pre-compilation and caching
- GPU memory pre-allocation
- Optimized batch processing
- Sub-100ms latency targets

## Security Features

### Container Security
- Non-root user execution (UID 1001)
- Minimal system dependencies
- Security scanning with Trivy
- Vulnerability management

### Image Hardening
- Distroless-style approach
- Removed unnecessary packages
- Security labels and annotations
- Regular base image updates

### Network Security
- Minimal port exposure
- TLS/SSL ready
- Security headers configured
- Network policies supported

## Monitoring and Observability

### Health Checks
All images include comprehensive health checks:
```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
  interval: 30s
  timeout: 10s
  retries: 3
```

### Metrics Collection
- Prometheus metrics endpoint
- Application performance monitoring
- Resource utilization tracking
- Custom business metrics

### Logging
- Structured JSON logging
- Log level configuration
- Centralized log aggregation ready
- Error tracking integration

## Deployment Examples

### Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: its-camera-ai
spec:
  replicas: 3
  selector:
    matchLabels:
      app: its-camera-ai
  template:
    metadata:
      labels:
        app: its-camera-ai
    spec:
      containers:
      - name: its-camera-ai
        image: ghcr.io/your-org/its-camera-ai:production
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi" 
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
```

### Docker Swarm Stack
```yaml
version: '3.8'
services:
  app:
    image: ghcr.io/your-org/its-camera-ai:production
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 2G
        reservations:
          memory: 1G
      restart_policy:
        condition: on-failure
      update_config:
        parallelism: 1
        delay: 10s
```

### Edge Deployment (ARM64)
```bash
# Build for ARM64
PLATFORM=linux/arm64 ./docker-build.sh edge

# Deploy to edge device
docker run -d \
  --name its-camera-ai-edge \
  --restart unless-stopped \
  -p 8000:8000 \
  -v /opt/its-data:/app/data \
  -v /opt/its-models:/app/models \
  --memory=512m \
  --cpus=1.0 \
  its-camera-ai:edge
```

## Troubleshooting

### Common Issues

#### GPU Not Detected
```bash
# Verify NVIDIA runtime
docker info | grep nvidia

# Check GPU availability
nvidia-smi

# Install nvidia-container-toolkit
sudo apt-get install nvidia-container-toolkit
sudo systemctl restart docker
```

#### Memory Issues
```bash
# Check container memory usage
docker stats

# Adjust memory limits
docker run --memory=2g its-camera-ai:production

# Monitor memory patterns
docker run -it its-camera-ai:production \
  python -c "import psutil; print(f'Memory: {psutil.virtual_memory()}')"
```

#### Performance Issues
```bash
# Profile application
docker run --rm -it its-camera-ai:development \
  python -m cProfile -s cumulative -m its_camera_ai.main

# Monitor resource usage
docker run --rm -it its-camera-ai:production \
  top -b -n1 | head -20
```

### Debugging

#### Development Debugging
```bash
# Run with debugger
docker run -p 8000:8000 -p 5678:5678 \
  -v $(pwd)/src:/app/src \
  its-camera-ai:development

# Attach VS Code debugger to port 5678
```

#### Production Debugging
```bash
# Access container shell
docker exec -it container_name bash

# View logs
docker logs -f container_name

# Health check manually
docker exec container_name curl http://localhost:8000/health
```

## Best Practices

### Development
- Use volume mounts for source code hot reload
- Enable debug ports for interactive debugging
- Use development-specific environment variables
- Regular dependency updates

### Production
- Always use specific image tags (not `latest`)
- Implement proper health checks
- Configure resource limits
- Use non-root users
- Regular security scanning
- Monitor performance metrics

### CI/CD Integration
```yaml
# GitHub Actions example
- name: Build and test
  run: |
    ./docker-build.sh test
    ./docker-build.sh production
    ./docker-build.sh scan its-camera-ai:production
    
- name: Deploy to production
  if: github.ref == 'refs/heads/main'
  run: |
    VERSION=${{ github.sha }} ./docker-build.sh push
```

## Support

For issues and questions:
- Check container logs: `docker logs <container_name>`
- Review health check status: `docker inspect <container_name>`
- Monitor resource usage: `docker stats`
- Security scanning: `./docker-build.sh scan`

This Docker setup provides a comprehensive, production-ready deployment strategy optimized for the ITS Camera AI system's specific requirements including sub-100ms inference latency, GPU acceleration, and multi-platform support.