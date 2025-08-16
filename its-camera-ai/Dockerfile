# Multi-stage Dockerfile for ITS Camera AI System
# Supports multiple deployment targets with optimization for each environment
# Build targets: development, gpu-development, production, gpu-production, edge, triton-inference, testing

ARG PYTHON_VERSION=3.12
ARG UV_VERSION=0.5.8
ARG NVIDIA_CUDA_VERSION=12.1

# =============================================================================
# BASE STAGE - Common dependencies and user setup
# =============================================================================
FROM python:${PYTHON_VERSION}-slim as base

# Build metadata
ARG BUILD_DATE
ARG VCS_REF  
ARG VERSION
LABEL org.opencontainers.image.title="ITS Camera AI"
LABEL org.opencontainers.image.description="AI-powered camera traffic monitoring system"
LABEL org.opencontainers.image.created=${BUILD_DATE}
LABEL org.opencontainers.image.revision=${VCS_REF}
LABEL org.opencontainers.image.version=${VERSION}
LABEL org.opencontainers.image.authors="ITS Camera AI Team"
LABEL org.opencontainers.image.vendor="ITS Camera AI"
LABEL org.opencontainers.image.licenses="MIT"

# Set environment variables for Python
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libpq-dev \
    libffi-dev \
    libssl-dev \
    # OpenCV dependencies
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgstreamer1.0-0 \
    libgtk-3-0 \
    # Additional ML dependencies
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /bin/bash appuser
WORKDIR /app

# Install uv for fast Python package management
ARG UV_VERSION
RUN pip install uv==${UV_VERSION}

# =============================================================================
# DEPENDENCIES STAGE - Install Python dependencies
# =============================================================================
FROM base as dependencies

# Copy dependency files
COPY --chown=appuser:appuser pyproject.toml uv.lock ./

# Install Python dependencies with uv
RUN uv sync --frozen --no-dev --no-cache
ENV PATH="/app/.venv/bin:$PATH"

# =============================================================================
# DEVELOPMENT DEPENDENCIES STAGE - Add development tools
# =============================================================================
FROM dependencies as dev-dependencies

# Install development dependencies
RUN uv sync --frozen --group dev --group test --no-cache

# =============================================================================
# GPU DEPENDENCIES STAGE - Add GPU/ML dependencies
# =============================================================================
FROM dependencies as gpu-dependencies

# Install ML and GPU dependencies
RUN uv sync --frozen --group ml --group gpu --no-cache

# =============================================================================
# APPLICATION CODE STAGE - Add application source code
# =============================================================================
FROM dependencies as app-code

# Copy application source code
COPY --chown=appuser:appuser src/ src/
COPY --chown=appuser:appuser alembic.ini ./
COPY --chown=appuser:appuser alembic/ alembic/

# Create necessary directories
RUN mkdir -p logs models data && \
    chown -R appuser:appuser logs models data

# Switch to non-root user
USER appuser

# =============================================================================
# DEVELOPMENT TARGET - Hot-reload development environment
# =============================================================================
FROM dev-dependencies as development

# Copy application code
COPY --chown=appuser:appuser src/ src/
COPY --chown=appuser:appuser alembic.ini ./
COPY --chown=appuser:appuser alembic/ alembic/
COPY --chown=appuser:appuser tests/ tests/

# Create directories and switch to non-root user
RUN mkdir -p logs models data test-results coverage && \
    chown -R appuser:appuser logs models data test-results coverage
USER appuser

# Expose port for development server
EXPOSE 8000

# Health check for development
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Development command with hot reload
CMD ["uvicorn", "its_camera_ai.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# =============================================================================
# GPU DEVELOPMENT TARGET - GPU-enabled development environment
# =============================================================================
FROM nvidia/cuda:${NVIDIA_CUDA_VERSION}-devel-ubuntu22.04 as gpu-base

# Install Python and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3-pip \
    build-essential \
    curl \
    git \
    libpq-dev \
    libffi-dev \
    libssl-dev \
    # OpenCV and ML dependencies
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgstreamer1.0-0 \
    libgtk-3-0 \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    # CUDA ML libraries
    libcurand-dev-12-1 \
    libcublas-dev-12-1 \
    libcudnn8-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set Python environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH="/usr/bin:$PATH"

# Create symlink for python
RUN ln -s /usr/bin/python3.12 /usr/bin/python

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /bin/bash appuser
WORKDIR /app

# Install uv
ARG UV_VERSION
RUN pip install uv==${UV_VERSION}

FROM gpu-base as gpu-development

# Copy dependency files and install with GPU support
COPY --chown=appuser:appuser pyproject.toml uv.lock ./
RUN uv sync --frozen --group dev --group test --group ml --group gpu --no-cache

# Copy application code
COPY --chown=appuser:appuser src/ src/
COPY --chown=appuser:appuser alembic.ini ./
COPY --chown=appuser:appuser alembic/ alembic/
COPY --chown=appuser:appuser tests/ tests/

# Create directories and switch to non-root user
RUN mkdir -p logs models data test-results coverage && \
    chown -R appuser:appuser logs models data test-results coverage
USER appuser

# Set environment for GPU
ENV CUDA_VISIBLE_DEVICES=all
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Expose port for development server
EXPOSE 8000

# Health check for GPU development
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# GPU development command
CMD ["uvicorn", "its_camera_ai.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# =============================================================================
# TESTING TARGET - Full test suite with coverage
# =============================================================================
FROM dev-dependencies as testing

# Copy application and test code
COPY --chown=appuser:appuser src/ src/
COPY --chown=appuser:appuser tests/ tests/
COPY --chown=appuser:appuser alembic.ini ./
COPY --chown=appuser:appuser alembic/ alembic/
COPY --chown=appuser:appuser pytest.ini ./

# Create test directories
RUN mkdir -p test-results coverage logs && \
    chown -R appuser:appuser test-results coverage logs
USER appuser

# Default command runs tests
CMD ["pytest", "-v", "--cov=its_camera_ai", "--cov-report=html", "--cov-report=term-missing", "--cov-fail-under=90"]

# =============================================================================
# PRODUCTION BASE - Optimized runtime environment
# =============================================================================
FROM app-code as production-base

# Remove development tools and clean up
USER root
RUN apt-get autoremove -y build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

USER appuser

# Set production environment
ENV ENVIRONMENT=production
ENV PYTHONOPTIMIZE=1

# =============================================================================
# PRODUCTION TARGET - Optimized production runtime
# =============================================================================  
FROM production-base as production

# Expose application port
EXPOSE 8000

# Health check for production
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Production command with multiple workers
CMD ["uvicorn", "its_camera_ai.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

# =============================================================================
# GPU PRODUCTION TARGET - GPU-optimized production runtime
# =============================================================================
FROM gpu-base as gpu-production

# Copy dependency files and install production + GPU dependencies
COPY --chown=appuser:appuser pyproject.toml uv.lock ./
RUN uv sync --frozen --group ml --group gpu --no-cache

# Copy application code
COPY --chown=appuser:appuser src/ src/
COPY --chown=appuser:appuser alembic.ini ./
COPY --chown=appuser:appuser alembic/ alembic/

# Create directories and switch to non-root user
RUN mkdir -p logs models data && \
    chown -R appuser:appuser logs models data
USER appuser

# Set production environment with GPU
ENV ENVIRONMENT=production
ENV PYTHONOPTIMIZE=1
ENV CUDA_VISIBLE_DEVICES=all
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Expose application port
EXPOSE 8000

# Health check for GPU production
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# GPU production command
CMD ["uvicorn", "its_camera_ai.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]

# =============================================================================
# EDGE TARGET - Lightweight edge deployment
# =============================================================================
FROM python:${PYTHON_VERSION}-alpine as edge

# Install minimal system dependencies
RUN apk add --no-cache \
    build-base \
    curl \
    libpq-dev \
    libffi-dev \
    openssl-dev \
    jpeg-dev \
    openblas-dev

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Create non-root user
RUN addgroup -g 1000 appuser && adduser -u 1000 -G appuser -s /bin/sh -D appuser
WORKDIR /app

# Install uv and dependencies
ARG UV_VERSION
RUN pip install uv==${UV_VERSION}

# Copy dependency files and install minimal dependencies
COPY --chown=appuser:appuser pyproject.toml uv.lock ./
RUN uv sync --frozen --group edge --no-cache

# Copy application code
COPY --chown=appuser:appuser src/ src/
COPY --chown=appuser:appuser alembic.ini ./
COPY --chown=appuser:appuser alembic/ alembic/

USER appuser

# Expose port
EXPOSE 8000

# Health check for edge
HEALTHCHECK --interval=60s --timeout=5s --start-period=30s --retries=2 \
    CMD curl -f http://localhost:8000/health || exit 1

# Edge deployment command
CMD ["uvicorn", "its_camera_ai.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

# =============================================================================
# TRITON INFERENCE TARGET - High-performance GPU inference server
# =============================================================================
FROM nvcr.io/nvidia/tritonserver:24.01-py3 as triton-inference

# Install Python dependencies for model conversion
RUN pip install torch torchvision ultralytics onnx

# Create model repository structure
RUN mkdir -p /models/yolo11/1 && \
    mkdir -p /models/yolo11_trt/1

# Copy model conversion scripts
COPY --chown=triton-server:triton-server scripts/optimize_models.py /workspace/
COPY --chown=triton-server:triton-server src/its_camera_ai/ml/ /workspace/ml/

# Create Triton model config
RUN cat > /models/yolo11/config.pbtxt << 'EOF'
name: "yolo11"
platform: "pytorch_libtorch"
max_batch_size: 8
input [
  {
    name: "input"
    data_type: TYPE_FP32
    dims: [ 3, 640, 640 ]
  }
]
output [
  {
    name: "output"
    data_type: TYPE_FP32
    dims: [ 25200, 85 ]
  }
]
EOF

# Expose Triton ports
EXPOSE 8000 8001 8002

# Health check for Triton
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/v2/health/ready || exit 1

# Triton server command
CMD ["tritonserver", "--model-repository=/models", "--strict-model-config=false", "--log-verbose=1"]

# =============================================================================
# WORKER TARGET - Background worker processes
# =============================================================================
FROM production-base as worker

# Worker-specific environment
ENV WORKER_TYPE=""
ENV WORKER_CONCURRENCY=2
ENV WORKER_QUEUE="default"

# Worker health check script
RUN cat > /app/worker_health.py << 'EOF'
#!/usr/bin/env python3
import sys
import redis
import os
from src.its_camera_ai.core.config import get_settings

def check_worker_health():
    try:
        settings = get_settings()
        r = redis.Redis.from_url(settings.redis_url)
        # Check if Redis is accessible
        r.ping()
        # Check if worker queues exist
        worker_type = os.getenv('WORKER_TYPE', 'analytics')
        queue_name = f"{worker_type}_queue"
        # Simple queue existence check
        r.llen(queue_name)
        return True
    except Exception as e:
        print(f"Worker health check failed: {e}")
        return False

if __name__ == "__main__":
    sys.exit(0 if check_worker_health() else 1)
EOF

RUN chmod +x /app/worker_health.py

# Health check for workers
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python /app/worker_health.py || exit 1

# Worker startup script
RUN cat > /app/start_worker.sh << 'EOF'
#!/bin/bash
set -e

# Get worker type from environment
WORKER_TYPE=${WORKER_TYPE:-analytics}
CONCURRENCY=${WORKER_CONCURRENCY:-2}
QUEUE=${WORKER_QUEUE:-default}

echo "Starting $WORKER_TYPE worker with concurrency $CONCURRENCY on queue $QUEUE"

# Start the appropriate worker
case $WORKER_TYPE in
  "analytics")
    exec python -m celery worker -A src.its_camera_ai.workers.analytics_worker:app --loglevel=info --concurrency=$CONCURRENCY -Q $QUEUE
    ;;
  "aggregation")
    exec python -m celery worker -A src.its_camera_ai.workers.aggregation_worker:app --loglevel=info --concurrency=$CONCURRENCY -Q $QUEUE
    ;;
  "event")
    exec python -m celery worker -A src.its_camera_ai.workers.event_worker:app --loglevel=info --concurrency=$CONCURRENCY -Q $QUEUE
    ;;
  "maintenance")
    exec python -m celery worker -A src.its_camera_ai.workers.maintenance_worker:app --loglevel=info --concurrency=$CONCURRENCY -Q $QUEUE
    ;;
  *)
    echo "Unknown worker type: $WORKER_TYPE"
    exit 1
    ;;
esac
EOF

RUN chmod +x /app/start_worker.sh

# Worker command
CMD ["/app/start_worker.sh"]