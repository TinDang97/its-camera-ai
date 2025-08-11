# Multi-stage Docker build for ITS Camera AI
# Cross-platform optimized for ML/Computer Vision workloads
# Supports AMD64, ARM64, with GPU acceleration and inference optimization

# Build arguments for platform detection
ARG TARGETPLATFORM
ARG BUILDPLATFORM
ARG TARGETOS
ARG TARGETARCH

# Stage 1: Base Python with system dependencies
FROM --platform=$TARGETPLATFORM python:3.12-slim-bookworm AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies for ML/CV libraries
RUN apt-get update && apt-get install -y \
    # Build tools
    build-essential \
    cmake \
    pkg-config \
    # OpenCV dependencies
    libopencv-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # Image processing
    libgdal-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    # Audio/Video processing
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    # Math libraries
    libatlas-base-dev \
    liblapack-dev \
    # Network tools
    curl \
    wget \
    git \
    # Clean up
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install uv for fast Python package management
RUN pip install --no-cache-dir uv==0.4.29

# Stage 2: Development environment
FROM base AS development

WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install all dependencies including dev and ML groups
RUN uv sync --group dev --group ml --group gpu

# Copy source code
COPY . .

# Install the application in editable mode
RUN uv pip install --no-deps -e .

# Expose ports
EXPOSE 8000 8001

CMD ["uvicorn", "its_camera_ai.app.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Stage 3: GPU-enabled base for training/inference
FROM nvidia/cuda:12.1-devel-ubuntu22.04 AS gpu-base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3.12-distutils \
    build-essential \
    cmake \
    pkg-config \
    libopencv-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgdal-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libatlas-base-dev \
    liblapack-dev \
    curl \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create symlinks for python
RUN ln -sf /usr/bin/python3.12 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.12 /usr/bin/python

# Install pip and uv
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3 \
    && pip install uv==0.4.29

# Stage 4: GPU development environment
FROM gpu-base AS gpu-development

WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install GPU dependencies
RUN uv sync --group dev --group ml --group gpu

# Copy source code
COPY . .

# Install the application in editable mode
RUN uv pip install --no-deps -e .

# Expose ports
EXPOSE 8000 8001

CMD ["uvicorn", "its_camera_ai.app.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Stage 5: Production build
FROM base AS builder

WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install production dependencies only
RUN uv sync --no-dev

# Copy source code
COPY src/ ./src/
COPY main.py ./

# Install the application (production)
RUN uv pip install --no-deps --no-cache .

# Stage 6: Production runtime
FROM python:3.12-slim-bookworm AS production

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH=/app/.venv/bin:$PATH

# Install only runtime system dependencies
RUN apt-get update && apt-get install -y \
    libopencv-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    libgdal-dev \
    libjpeg-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN groupadd -r app && useradd -r -g app -d /app -s /sbin/nologin app

# Copy virtual environment from builder
COPY --from=builder --chown=app:app /app/.venv /app/.venv

# Copy application code
COPY --from=builder --chown=app:app /app/src /app/src
COPY --from=builder --chown=app:app /app/main.py /app/

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/models /app/temp \
    && chown -R app:app /app

# Switch to non-root user
USER app
WORKDIR /app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Security labels
LABEL maintainer="ITS Camera AI Team <team@its-camera-ai.com>" \
      version="0.1.0" \
      description="AI-powered camera traffic monitoring system" \
      security.scan="enabled"

# Expose port
EXPOSE 8000

# Default command
CMD ["python", "main.py"]

# Stage 7: Triton Inference Server (NVIDIA optimized)
FROM nvcr.io/nvidia/tritonserver:24.02-py3 AS triton-base

# Install Python dependencies in Triton container
RUN pip install fastapi uvicorn pydantic

WORKDIR /workspace

# Copy model repository structure
COPY models/ models/
COPY infrastructure/triton/ .

# Expose Triton ports
EXPOSE 8000 8001 8002

# Start Triton Inference Server
CMD ["tritonserver", "--model-repository=/workspace/models", "--allow-http=true", "--allow-grpc=true", "--allow-metrics=true"]

# Stage 8: BentoML Inference Service
FROM python:3.12-slim-bookworm AS bentoml-service

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libopencv-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv and BentoML
RUN pip install uv==0.4.29

WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install inference dependencies
RUN uv sync --group inference --no-dev

# Copy BentoML service
COPY src/ ./src/
COPY infrastructure/bentoml/ ./

# Build BentoML service
RUN uv run bentoml build

# Expose BentoML port
EXPOSE 3000

CMD ["bentoml", "serve", "its_camera_ai_service:latest", "--host", "0.0.0.0", "--port", "3000"]

# Stage 9: Edge deployment (optimized for ARM64/edge devices)
FROM --platform=$TARGETPLATFORM python:3.12-slim-bookworm AS edge

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TARGETPLATFORM=${TARGETPLATFORM} \
    TARGETARCH=${TARGETARCH}

# Install platform-specific system dependencies
RUN apt-get update && apt-get install -y \
    libopencv-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    # Platform-specific optimizations
    $([ "$TARGETARCH" = "arm64" ] && echo "libblas-dev liblapack-dev gfortran" || echo "") \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv==0.4.29

WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install edge-optimized dependencies based on platform
RUN if [ "$TARGETARCH" = "arm64" ]; then \
        uv sync --group edge --group inference --no-dev; \
    else \
        uv sync --group edge --no-dev; \
    fi

# Copy optimized models and source
COPY src/ ./src/
COPY main.py ./

# Install application
RUN uv pip install --no-deps .

# Create app user
RUN useradd -r -s /bin/false app && chown -R app:app /app
USER app

# Expose port
EXPOSE 8000

CMD ["python", "main.py"]

# Stage 10: Multi-platform production with inference optimization
FROM base AS production-optimized

# Set platform-aware environment
ENV TARGETPLATFORM=${TARGETPLATFORM} \
    TARGETARCH=${TARGETARCH}

WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install production dependencies with platform optimizations
RUN if [ "$TARGETARCH" = "arm64" ]; then \
        uv sync --group inference --no-dev; \
    else \
        uv sync --no-dev; \
    fi

# Copy source code
COPY src/ ./src/
COPY main.py ./

# Install the application
RUN uv pip install --no-deps .

# Create necessary directories and non-root user
RUN mkdir -p /app/data /app/logs /app/models /app/temp \
    && groupadd -r app && useradd -r -g app -d /app -s /sbin/nologin app \
    && chown -R app:app /app

# Switch to non-root user
USER app

# Health check with platform awareness
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Default command
CMD ["python", "main.py"]