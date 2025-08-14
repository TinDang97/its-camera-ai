"""
Ultra-fast YOLO11 deployment configuration for different hardware environments.

This module provides optimized deployment configurations for the complete ultra-fast
pipeline across various edge devices and cloud environments, ensuring optimal
performance for each hardware configuration.

Supported Deployment Targets:
- NVIDIA Jetson (Xavier NX, AGX Orin, Nano)
- Intel NCS2 (Neural Compute Stick 2)
- Cloud GPUs (T4, V100, A100, RTX 4090)
- CPU-only environments
- Hybrid edge-cloud deployments

Performance Optimization Strategies:
- Device-specific model quantization (INT8, FP16)
- Hardware-accelerated preprocessing
- Memory optimization for edge devices
- Dynamic batch sizing based on compute capacity
- Power management for battery-powered devices
"""

import json
import logging
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class HardwareType(Enum):
    """Supported hardware deployment targets."""
    JETSON_NANO = "jetson_nano"
    JETSON_XAVIER_NX = "jetson_xavier_nx"
    JETSON_AGX_ORIN = "jetson_agx_orin"
    INTEL_NCS2 = "intel_ncs2"
    CLOUD_T4 = "cloud_t4"
    CLOUD_V100 = "cloud_v100"
    CLOUD_A100 = "cloud_a100"
    RTX_4090 = "rtx_4090"
    RTX_3080 = "rtx_3080"
    CPU_ONLY = "cpu_only"
    CUSTOM = "custom"


class DeploymentMode(Enum):
    """Deployment modes for different use cases."""
    EDGE_ONLY = "edge_only"                 # All processing on edge
    HYBRID_EDGE_CLOUD = "hybrid_edge_cloud" # Smart routing between edge/cloud
    CLOUD_ONLY = "cloud_only"               # All processing in cloud
    MOBILE_DEVICE = "mobile_device"         # Mobile/battery optimized


@dataclass
class HardwareSpecs:
    """Hardware specifications and capabilities."""

    # Compute capabilities
    cuda_cores: int = 0
    tensor_cores: int = 0
    cpu_cores: int = 4
    gpu_memory_gb: float = 0.0
    system_memory_gb: float = 4.0

    # Performance characteristics
    gpu_tflops_fp32: float = 0.0
    gpu_tflops_fp16: float = 0.0
    gpu_tops_int8: float = 0.0
    memory_bandwidth_gbps: float = 100.0

    # Power constraints
    max_power_watts: float = 200.0
    thermal_design_power: float = 150.0
    battery_powered: bool = False

    # I/O capabilities
    max_camera_streams: int = 1
    usb_ports: int = 4
    ethernet_ports: int = 1
    wifi_capable: bool = True

    # Architecture details
    architecture: str = "x86_64"
    cuda_compute_capability: str = "0.0"
    supports_tensorrt: bool = False
    supports_openvino: bool = False
    supports_coreml: bool = False


@dataclass
class ModelConfiguration:
    """Model-specific configuration for deployment."""

    # Model selection
    model_variant: str = "yolo11n"  # nano, small, medium, large, extra_large
    input_resolution: tuple[int, int] = (640, 640)
    num_classes: int = 80

    # Quantization settings
    quantization: str = "fp16"  # fp32, fp16, int8, dynamic
    calibration_dataset_size: int = 1000
    quantization_aware_training: bool = False

    # Optimization settings
    enable_tensorrt: bool = False
    tensorrt_workspace_mb: int = 1024
    enable_cuda_graphs: bool = False
    enable_torch_script: bool = True

    # Inference settings
    batch_size_range: tuple[int, int, int] = (1, 8, 32)  # min, opt, max
    confidence_threshold: float = 0.25
    nms_threshold: float = 0.45
    max_detections: int = 1000

    # Memory optimization
    enable_memory_pooling: bool = True
    preallocate_tensors: bool = True
    max_tensor_cache_mb: int = 512


@dataclass
class ProcessingPipelineConfig:
    """Processing pipeline configuration."""

    # Preprocessing
    preprocessing_backend: str = "opencv"  # opencv, dali, custom
    normalize_inputs: bool = True
    resize_algorithm: str = "bilinear"  # bilinear, bicubic, nearest
    color_space: str = "RGB"  # RGB, BGR, YUV

    # Quality validation
    enable_quality_validation: bool = True
    min_quality_score: float = 0.5
    max_blur_threshold: float = 100.0
    enable_async_validation: bool = True

    # Batching strategy
    adaptive_batching: bool = True
    base_timeout_ms: int = 10
    enable_priority_lane: bool = True
    priority_timeout_ms: int = 1
    max_queue_depth: int = 100

    # Post-processing
    enable_tracking: bool = False
    tracking_algorithm: str = "sort"  # sort, deepsort, bytetrack
    enable_analytics: bool = True

    # Error handling
    enable_frame_skipping: bool = True
    max_consecutive_errors: int = 10
    fallback_processing: bool = True


@dataclass
class DeploymentConfiguration:
    """Complete deployment configuration."""

    # Deployment metadata
    config_name: str = "ultra_fast_yolo11"
    version: str = "1.0.0"
    target_hardware: HardwareType = HardwareType.CLOUD_T4
    deployment_mode: DeploymentMode = DeploymentMode.EDGE_ONLY

    # Performance targets
    target_latency_p99_ms: float = 50.0
    target_throughput_fps: float = 30.0
    max_concurrent_streams: int = 10

    # Component configurations
    hardware_specs: HardwareSpecs = field(default_factory=HardwareSpecs)
    model_config: ModelConfiguration = field(default_factory=ModelConfiguration)
    pipeline_config: ProcessingPipelineConfig = field(default_factory=ProcessingPipelineConfig)

    # Resource limits
    max_cpu_usage_percent: float = 80.0
    max_memory_usage_percent: float = 85.0
    max_gpu_usage_percent: float = 90.0

    # Monitoring and alerting
    enable_performance_monitoring: bool = True
    monitoring_interval_seconds: int = 1
    enable_automated_scaling: bool = False

    # Storage and networking
    model_cache_path: str = "/tmp/yolo11_models"
    log_level: str = "INFO"
    enable_remote_logging: bool = False

    # Backup and failover
    enable_model_fallback: bool = True
    fallback_model_path: str | None = None
    enable_cloud_fallback: bool = False
    cloud_endpoint: str | None = None


class UltraFastDeploymentConfigGenerator:
    """Generates optimized deployment configurations for different hardware."""

    def __init__(self):
        self.hardware_profiles = self._initialize_hardware_profiles()
        logger.info("Deployment config generator initialized with hardware profiles")

    def _initialize_hardware_profiles(self) -> dict[HardwareType, HardwareSpecs]:
        """Initialize hardware specification profiles."""
        return {
            # NVIDIA Jetson Family
            HardwareType.JETSON_NANO: HardwareSpecs(
                cuda_cores=128,
                cpu_cores=4,
                gpu_memory_gb=2.0,
                system_memory_gb=4.0,
                gpu_tflops_fp16=0.472,
                gpu_tops_int8=0.5,
                max_power_watts=10.0,
                thermal_design_power=5.0,
                max_camera_streams=2,
                architecture="aarch64",
                cuda_compute_capability="5.3",
                supports_tensorrt=True
            ),

            HardwareType.JETSON_XAVIER_NX: HardwareSpecs(
                cuda_cores=384,
                tensor_cores=48,
                cpu_cores=6,
                gpu_memory_gb=8.0,
                system_memory_gb=8.0,
                gpu_tflops_fp32=1.37,
                gpu_tflops_fp16=2.74,
                gpu_tops_int8=21.0,
                memory_bandwidth_gbps=136.5,
                max_power_watts=20.0,
                thermal_design_power=15.0,
                max_camera_streams=8,
                architecture="aarch64",
                cuda_compute_capability="7.2",
                supports_tensorrt=True
            ),

            HardwareType.JETSON_AGX_ORIN: HardwareSpecs(
                cuda_cores=2048,
                tensor_cores=64,
                cpu_cores=12,
                gpu_memory_gb=32.0,
                system_memory_gb=32.0,
                gpu_tflops_fp32=5.3,
                gpu_tflops_fp16=10.6,
                gpu_tops_int8=275.0,
                memory_bandwidth_gbps=204.8,
                max_power_watts=60.0,
                thermal_design_power=40.0,
                max_camera_streams=32,
                architecture="aarch64",
                cuda_compute_capability="8.7",
                supports_tensorrt=True
            ),

            # Intel NCS2
            HardwareType.INTEL_NCS2: HardwareSpecs(
                cuda_cores=0,
                cpu_cores=1,  # VPU
                gpu_memory_gb=0.0,
                system_memory_gb=0.0,  # Uses host memory
                gpu_tops_int8=1.0,
                max_power_watts=1.2,
                thermal_design_power=1.0,
                max_camera_streams=1,
                architecture="x86_64",
                supports_openvino=True
            ),

            # Cloud GPUs
            HardwareType.CLOUD_T4: HardwareSpecs(
                cuda_cores=2560,
                tensor_cores=320,
                cpu_cores=8,  # Typical cloud instance
                gpu_memory_gb=16.0,
                system_memory_gb=32.0,
                gpu_tflops_fp32=8.1,
                gpu_tflops_fp16=65.0,
                gpu_tops_int8=130.0,
                memory_bandwidth_gbps=300.0,
                max_power_watts=70.0,
                max_camera_streams=100,
                architecture="x86_64",
                cuda_compute_capability="7.5",
                supports_tensorrt=True
            ),

            HardwareType.CLOUD_V100: HardwareSpecs(
                cuda_cores=5120,
                tensor_cores=640,
                cpu_cores=16,
                gpu_memory_gb=32.0,
                system_memory_gb=64.0,
                gpu_tflops_fp32=15.7,
                gpu_tflops_fp16=31.4,
                gpu_tops_int8=125.0,
                memory_bandwidth_gbps=900.0,
                max_power_watts=300.0,
                max_camera_streams=200,
                architecture="x86_64",
                cuda_compute_capability="7.0",
                supports_tensorrt=True
            ),

            HardwareType.CLOUD_A100: HardwareSpecs(
                cuda_cores=6912,
                tensor_cores=432,
                cpu_cores=32,
                gpu_memory_gb=80.0,
                system_memory_gb=128.0,
                gpu_tflops_fp32=19.5,
                gpu_tflops_fp16=78.0,
                gpu_tops_int8=624.0,
                memory_bandwidth_gbps=2039.0,
                max_power_watts=400.0,
                max_camera_streams=500,
                architecture="x86_64",
                cuda_compute_capability="8.0",
                supports_tensorrt=True
            ),

            # Consumer GPUs
            HardwareType.RTX_4090: HardwareSpecs(
                cuda_cores=16384,
                tensor_cores=512,
                cpu_cores=16,
                gpu_memory_gb=24.0,
                system_memory_gb=32.0,
                gpu_tflops_fp32=35.6,
                gpu_tflops_fp16=71.0,
                gpu_tops_int8=330.0,
                memory_bandwidth_gbps=1008.0,
                max_power_watts=450.0,
                max_camera_streams=300,
                architecture="x86_64",
                cuda_compute_capability="8.9",
                supports_tensorrt=True
            ),

            # CPU Only
            HardwareType.CPU_ONLY: HardwareSpecs(
                cuda_cores=0,
                cpu_cores=8,
                gpu_memory_gb=0.0,
                system_memory_gb=16.0,
                max_power_watts=65.0,
                max_camera_streams=4,
                architecture="x86_64",
                supports_openvino=True
            )
        }

    def generate_optimal_config(
        self,
        hardware_type: HardwareType,
        target_latency_ms: float = 50.0,
        max_concurrent_streams: int = None,
        deployment_mode: DeploymentMode = DeploymentMode.EDGE_ONLY
    ) -> DeploymentConfiguration:
        """Generate optimal configuration for specified hardware."""

        hardware_specs = self.hardware_profiles.get(hardware_type)
        if not hardware_specs:
            raise ValueError(f"Unsupported hardware type: {hardware_type}")

        # Determine optimal configuration based on hardware capabilities
        config = DeploymentConfiguration(
            config_name=f"ultra_fast_{hardware_type.value}",
            target_hardware=hardware_type,
            deployment_mode=deployment_mode,
            target_latency_p99_ms=target_latency_ms,
            hardware_specs=hardware_specs
        )

        # Configure concurrent streams based on hardware
        if max_concurrent_streams is None:
            max_concurrent_streams = self._calculate_optimal_streams(hardware_specs, target_latency_ms)
        config.max_concurrent_streams = max_concurrent_streams

        # Configure model based on hardware capabilities
        config.model_config = self._configure_model_for_hardware(hardware_specs, target_latency_ms)

        # Configure processing pipeline
        config.pipeline_config = self._configure_pipeline_for_hardware(hardware_specs, target_latency_ms)

        # Adjust resource limits
        config = self._adjust_resource_limits(config, hardware_specs)

        logger.info(f"Generated optimal configuration for {hardware_type.value}: "
                   f"streams={max_concurrent_streams}, latency_target={target_latency_ms}ms")

        return config

    def _calculate_optimal_streams(self, hardware_specs: HardwareSpecs, target_latency_ms: float) -> int:
        """Calculate optimal number of concurrent streams for hardware."""

        # Base calculation on GPU memory and compute capability
        if hardware_specs.gpu_memory_gb > 0:
            # GPU-based calculation
            memory_factor = hardware_specs.gpu_memory_gb / 4.0  # 4GB baseline
            compute_factor = hardware_specs.gpu_tflops_fp16 / 10.0  # 10 TFLOPS baseline

            base_streams = int(min(memory_factor, compute_factor) * 20)  # 20 streams baseline

            # Adjust for latency requirements
            if target_latency_ms <= 25:  # Ultra-low latency
                base_streams = int(base_streams * 0.6)
            elif target_latency_ms <= 50:  # Low latency
                base_streams = int(base_streams * 0.8)

        else:
            # CPU-only calculation
            base_streams = max(1, hardware_specs.cpu_cores // 2)

        # Hardware-specific limits
        return min(base_streams, hardware_specs.max_camera_streams)

    def _configure_model_for_hardware(self, hardware_specs: HardwareSpecs, target_latency_ms: float) -> ModelConfiguration:
        """Configure model settings for specific hardware."""

        model_config = ModelConfiguration()

        # Select model variant based on hardware capability
        if hardware_specs.gpu_tflops_fp16 >= 50:  # High-end GPUs
            if target_latency_ms >= 100:
                model_config.model_variant = "yolo11l"
            elif target_latency_ms >= 50:
                model_config.model_variant = "yolo11m"
            else:
                model_config.model_variant = "yolo11s"
        elif hardware_specs.gpu_tflops_fp16 >= 10:  # Mid-range GPUs
            if target_latency_ms >= 50:
                model_config.model_variant = "yolo11s"
            else:
                model_config.model_variant = "yolo11n"
        else:  # Edge devices or CPU
            model_config.model_variant = "yolo11n"

        # Configure quantization
        if hardware_specs.supports_tensorrt:
            if hardware_specs.gpu_tops_int8 >= 100:
                model_config.quantization = "int8"
                model_config.enable_tensorrt = True
                model_config.tensorrt_workspace_mb = min(2048, int(hardware_specs.gpu_memory_gb * 1024 * 0.3))
            else:
                model_config.quantization = "fp16"
                model_config.enable_tensorrt = True
                model_config.tensorrt_workspace_mb = 1024
        elif hardware_specs.supports_openvino:
            model_config.quantization = "int8"
        else:
            model_config.quantization = "fp16" if hardware_specs.gpu_memory_gb > 4 else "dynamic"

        # Configure batch sizes based on memory and latency requirements
        if target_latency_ms <= 25:  # Ultra-low latency
            model_config.batch_size_range = (1, 2, 4)
        elif target_latency_ms <= 50:  # Low latency
            model_config.batch_size_range = (1, 4, 8)
        else:  # Balanced latency/throughput
            max_batch = min(32, int(hardware_specs.gpu_memory_gb // 2))
            model_config.batch_size_range = (1, max_batch // 2, max_batch)

        # Enable advanced features based on hardware
        if hardware_specs.cuda_compute_capability >= "7.0":
            model_config.enable_cuda_graphs = True

        # Memory optimization for edge devices
        if hardware_specs.gpu_memory_gb <= 8:
            model_config.max_tensor_cache_mb = int(hardware_specs.gpu_memory_gb * 1024 * 0.1)
            model_config.preallocate_tensors = False  # Dynamic allocation for limited memory

        return model_config

    def _configure_pipeline_for_hardware(self, hardware_specs: HardwareSpecs, target_latency_ms: float) -> ProcessingPipelineConfig:
        """Configure processing pipeline for specific hardware."""

        pipeline_config = ProcessingPipelineConfig()

        # Configure preprocessing backend
        if hardware_specs.supports_tensorrt and hardware_specs.gpu_memory_gb >= 8:
            pipeline_config.preprocessing_backend = "dali"  # NVIDIA DALI for high-end GPUs
        else:
            pipeline_config.preprocessing_backend = "opencv"

        # Configure batching strategy
        if target_latency_ms <= 25:
            pipeline_config.base_timeout_ms = 1
            pipeline_config.priority_timeout_ms = 0.5
        elif target_latency_ms <= 50:
            pipeline_config.base_timeout_ms = 3
            pipeline_config.priority_timeout_ms = 1
        else:
            pipeline_config.base_timeout_ms = 10
            pipeline_config.priority_timeout_ms = 2

        # Enable async processing for capable hardware
        if hardware_specs.cpu_cores >= 4:
            pipeline_config.enable_async_validation = True

        # Configure queue depth based on memory
        if hardware_specs.system_memory_gb >= 16:
            pipeline_config.max_queue_depth = 200
        elif hardware_specs.system_memory_gb >= 8:
            pipeline_config.max_queue_depth = 100
        else:
            pipeline_config.max_queue_depth = 50

        # Enable tracking and analytics for powerful hardware
        if hardware_specs.gpu_tflops_fp16 >= 20:
            pipeline_config.enable_tracking = True
            pipeline_config.enable_analytics = True

        return pipeline_config

    def _adjust_resource_limits(self, config: DeploymentConfiguration, hardware_specs: HardwareSpecs) -> DeploymentConfiguration:
        """Adjust resource usage limits based on hardware."""

        # Conservative limits for edge devices
        if hardware_specs.max_power_watts <= 20:  # Edge devices
            config.max_cpu_usage_percent = 70.0
            config.max_memory_usage_percent = 75.0
            config.max_gpu_usage_percent = 85.0
        elif hardware_specs.max_power_watts <= 100:  # Mid-range devices
            config.max_cpu_usage_percent = 80.0
            config.max_memory_usage_percent = 80.0
            config.max_gpu_usage_percent = 90.0
        else:  # High-end devices
            config.max_cpu_usage_percent = 90.0
            config.max_memory_usage_percent = 85.0
            config.max_gpu_usage_percent = 95.0

        # Enable scaling for cloud deployments
        if config.target_hardware.value.startswith("cloud_"):
            config.enable_automated_scaling = True

        return config

    def generate_deployment_package(
        self,
        config: DeploymentConfiguration,
        output_dir: Path,
        include_docker: bool = True,
        include_kubernetes: bool = True
    ) -> dict[str, Path]:
        """Generate complete deployment package."""

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        generated_files = {}

        # 1. Save configuration
        config_path = output_dir / "deployment_config.json"
        with open(config_path, 'w') as f:
            json.dump(asdict(config), f, indent=2)
        generated_files['config'] = config_path

        # 2. Generate Docker configuration
        if include_docker:
            docker_path = self._generate_dockerfile(config, output_dir)
            generated_files['dockerfile'] = docker_path

            compose_path = self._generate_docker_compose(config, output_dir)
            generated_files['docker_compose'] = compose_path

        # 3. Generate Kubernetes manifests
        if include_kubernetes:
            k8s_dir = output_dir / "kubernetes"
            k8s_dir.mkdir(exist_ok=True)

            deployment_path = self._generate_k8s_deployment(config, k8s_dir)
            service_path = self._generate_k8s_service(config, k8s_dir)

            generated_files['k8s_deployment'] = deployment_path
            generated_files['k8s_service'] = service_path

        # 4. Generate performance monitoring config
        monitoring_path = self._generate_monitoring_config(config, output_dir)
        generated_files['monitoring'] = monitoring_path

        # 5. Generate deployment script
        deploy_script_path = self._generate_deployment_script(config, output_dir)
        generated_files['deploy_script'] = deploy_script_path

        logger.info(f"Deployment package generated in {output_dir}")
        return generated_files

    def _generate_dockerfile(self, config: DeploymentConfiguration, output_dir: Path) -> Path:
        """Generate optimized Dockerfile for deployment."""

        dockerfile_path = output_dir / "Dockerfile"

        # Select base image based on hardware
        if config.target_hardware.value.startswith("jetson"):
            base_image = "nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3"
        elif config.hardware_specs.supports_tensorrt:
            base_image = "nvcr.io/nvidia/pytorch:23.10-py3"
        elif config.hardware_specs.supports_openvino:
            base_image = "openvino/ubuntu20_dev:latest"
        else:
            base_image = "python:3.11-slim"

        dockerfile_content = f'''FROM {base_image}

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    libopencv-dev \\
    libglib2.0-0 \\
    libsm6 \\
    libxext6 \\
    libxrender-dev \\
    libgomp1 \\
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install hardware-specific optimizations
'''

        if config.hardware_specs.supports_tensorrt:
            dockerfile_content += '''RUN pip install --no-cache-dir tensorrt nvidia-tensorrt
'''

        if config.hardware_specs.supports_openvino:
            dockerfile_content += '''RUN pip install --no-cache-dir openvino-dev
'''

        dockerfile_content += f'''
# Copy application code
COPY src/ ./src/
COPY config/ ./config/

# Set environment variables for optimization
ENV CUDA_VISIBLE_DEVICES=0
ENV OMP_NUM_THREADS={config.hardware_specs.cpu_cores}
ENV OPENCV_NUM_THREADS={config.hardware_specs.cpu_cores}
ENV TARGET_LATENCY_MS={config.target_latency_p99_ms}
ENV MAX_CONCURRENT_STREAMS={config.max_concurrent_streams}

# Create model cache directory
RUN mkdir -p {config.model_cache_path}

# Expose application port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Start application
CMD ["python", "src/its_camera_ai/api/app.py", "--config", "config/deployment_config.json"]
'''

        with open(dockerfile_path, 'w') as f:
            f.write(dockerfile_content)

        return dockerfile_path

    def _generate_docker_compose(self, config: DeploymentConfiguration, output_dir: Path) -> Path:
        """Generate Docker Compose configuration."""

        compose_path = output_dir / "docker-compose.yml"

        services = {
            'ultra-fast-yolo11': {
                'build': '.',
                'ports': ['8000:8000'],
                'environment': {
                    'TARGET_LATENCY_MS': config.target_latency_p99_ms,
                    'MAX_CONCURRENT_STREAMS': config.max_concurrent_streams,
                    'LOG_LEVEL': config.log_level
                },
                'volumes': [
                    f"{config.model_cache_path}:/app/models",
                    "/dev/video0:/dev/video0"  # Camera access
                ],
                'deploy': {
                    'resources': {
                        'limits': {
                            'cpus': str(config.hardware_specs.cpu_cores * config.max_cpu_usage_percent / 100),
                            'memory': f"{int(config.hardware_specs.system_memory_gb * config.max_memory_usage_percent / 100)}G"
                        }
                    }
                }
            }
        }

        # Add GPU support if available
        if config.hardware_specs.gpu_memory_gb > 0:
            services['ultra-fast-yolo11']['deploy']['resources']['reservations'] = {
                'devices': [{
                    'driver': 'nvidia',
                    'count': 1,
                    'capabilities': ['gpu']
                }]
            }

        # Add Redis for caching
        services['redis'] = {
            'image': 'redis:alpine',
            'ports': ['6379:6379'],
            'deploy': {
                'resources': {
                    'limits': {
                        'memory': '512M'
                    }
                }
            }
        }

        compose_content = {
            'version': '3.8',
            'services': services,
            'networks': {
                'yolo11-network': {
                    'driver': 'bridge'
                }
            }
        }

        with open(compose_path, 'w') as f:
            json.dump(compose_content, f, indent=2)

        return compose_path

    def _generate_k8s_deployment(self, config: DeploymentConfiguration, k8s_dir: Path) -> Path:
        """Generate Kubernetes deployment manifest."""

        deployment_path = k8s_dir / "deployment.yaml"

        deployment_yaml = f'''apiVersion: apps/v1
kind: Deployment
metadata:
  name: ultra-fast-yolo11
  labels:
    app: ultra-fast-yolo11
    version: {config.version}
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ultra-fast-yolo11
  template:
    metadata:
      labels:
        app: ultra-fast-yolo11
    spec:
      containers:
      - name: yolo11-inference
        image: ultra-fast-yolo11:latest
        ports:
        - containerPort: 8000
        env:
        - name: TARGET_LATENCY_MS
          value: "{config.target_latency_p99_ms}"
        - name: MAX_CONCURRENT_STREAMS
          value: "{config.max_concurrent_streams}"
        - name: LOG_LEVEL
          value: "{config.log_level}"
        resources:
          limits:
            cpu: "{int(config.hardware_specs.cpu_cores * config.max_cpu_usage_percent / 100)}"
            memory: "{int(config.hardware_specs.system_memory_gb * config.max_memory_usage_percent / 100)}Gi"
'''

        if config.hardware_specs.gpu_memory_gb > 0:
            deployment_yaml += '''            nvidia.com/gpu: 1
          requests:
            nvidia.com/gpu: 1
'''

        deployment_yaml += '''        volumeMounts:
        - name: model-cache
          mountPath: /app/models
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: model-cache
        persistentVolumeClaim:
          claimName: model-cache-pvc
'''

        with open(deployment_path, 'w') as f:
            f.write(deployment_yaml)

        return deployment_path

    def _generate_k8s_service(self, config: DeploymentConfiguration, k8s_dir: Path) -> Path:
        """Generate Kubernetes service manifest."""

        service_path = k8s_dir / "service.yaml"

        service_yaml = '''apiVersion: v1
kind: Service
metadata:
  name: ultra-fast-yolo11-service
spec:
  selector:
    app: ultra-fast-yolo11
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-cache-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
'''

        with open(service_path, 'w') as f:
            f.write(service_yaml)

        return service_path

    def _generate_monitoring_config(self, config: DeploymentConfiguration, output_dir: Path) -> Path:
        """Generate monitoring configuration."""

        monitoring_path = output_dir / "monitoring_config.json"

        monitoring_config = {
            "performance_targets": {
                "latency_p99_ms": config.target_latency_p99_ms,
                "throughput_fps": config.target_throughput_fps,
                "error_rate_max": 0.01,
                "cpu_usage_max": config.max_cpu_usage_percent,
                "memory_usage_max": config.max_memory_usage_percent,
                "gpu_usage_max": config.max_gpu_usage_percent
            },
            "monitoring_interval_seconds": config.monitoring_interval_seconds,
            "alert_thresholds": {
                "latency_warning_ms": config.target_latency_p99_ms * 0.8,
                "latency_critical_ms": config.target_latency_p99_ms * 1.2,
                "error_rate_warning": 0.005,
                "error_rate_critical": 0.02,
                "queue_depth_warning": 50,
                "queue_depth_critical": 100
            },
            "metrics_collection": {
                "enable_detailed_tracing": True,
                "enable_gpu_metrics": config.hardware_specs.gpu_memory_gb > 0,
                "enable_custom_metrics": True,
                "retention_days": 7
            }
        }

        with open(monitoring_path, 'w') as f:
            json.dump(monitoring_config, f, indent=2)

        return monitoring_path

    def _generate_deployment_script(self, config: DeploymentConfiguration, output_dir: Path) -> Path:
        """Generate deployment automation script."""

        script_path = output_dir / "deploy.sh"

        script_content = f'''#!/bin/bash
set -e

echo "🚀 Deploying Ultra-Fast YOLO11 Pipeline for {config.target_hardware.value}"
echo "Target Latency: {config.target_latency_p99_ms}ms P99"
echo "Max Streams: {config.max_concurrent_streams}"
echo ""

# Check system requirements
echo "📋 Checking system requirements..."
'''

        if config.hardware_specs.supports_tensorrt:
            script_content += '''
if ! nvidia-smi &> /dev/null; then
    echo "❌ NVIDIA GPU not detected or drivers not installed"
    exit 1
fi
echo "✅ NVIDIA GPU detected"
'''

        script_content += f'''
# Create model cache directory
echo "📁 Setting up model cache..."
mkdir -p {config.model_cache_path}

# Build and deploy
echo "🔨 Building Docker image..."
docker build -t ultra-fast-yolo11:latest .

echo "🚀 Starting services..."
docker-compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 30

# Health check
echo "🏥 Performing health check..."
if curl -f http://localhost:8000/health; then
    echo "✅ Deployment successful!"
    echo ""
    echo "🌐 Service endpoints:"
    echo "  - Health: http://localhost:8000/health"
    echo "  - Inference: http://localhost:8000/inference"
    echo "  - Metrics: http://localhost:8000/metrics"
    echo ""
    echo "📊 Performance targets:"
    echo "  - P99 Latency: <{config.target_latency_p99_ms}ms"
    echo "  - Throughput: >{config.target_throughput_fps} FPS"
    echo "  - Max Streams: {config.max_concurrent_streams}"
else
    echo "❌ Deployment failed - health check failed"
    exit 1
fi
'''

        with open(script_path, 'w') as f:
            f.write(script_content)

        # Make script executable
        script_path.chmod(0o755)

        return script_path

    def auto_detect_hardware(self) -> HardwareType:
        """Auto-detect the current hardware environment."""

        try:
            # Check for NVIDIA GPU
            try:
                result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    gpu_info = result.stdout.strip()

                    if "T4" in gpu_info:
                        return HardwareType.CLOUD_T4
                    elif "V100" in gpu_info:
                        return HardwareType.CLOUD_V100
                    elif "A100" in gpu_info:
                        return HardwareType.CLOUD_A100
                    elif "RTX 4090" in gpu_info or "RTX4090" in gpu_info:
                        return HardwareType.RTX_4090
                    elif "RTX 3080" in gpu_info or "RTX3080" in gpu_info:
                        return HardwareType.RTX_3080
            except FileNotFoundError:
                pass

            # Check for Jetson
            if platform.machine() == "aarch64":
                try:
                    with open("/proc/device-tree/model") as f:
                        model = f.read().strip()

                    if "Xavier NX" in model:
                        return HardwareType.JETSON_XAVIER_NX
                    elif "AGX Orin" in model:
                        return HardwareType.JETSON_AGX_ORIN
                    elif "Nano" in model:
                        return HardwareType.JETSON_NANO
                except FileNotFoundError:
                    pass

            # Check for Intel NCS2
            try:
                result = subprocess.run(['lsusb'], capture_output=True, text=True)
                if "03e7:2485" in result.stdout:  # Intel NCS2 USB ID
                    return HardwareType.INTEL_NCS2
            except FileNotFoundError:
                pass

            # Default to CPU-only
            return HardwareType.CPU_ONLY

        except Exception as e:
            logger.warning(f"Hardware detection failed: {e}")
            return HardwareType.CPU_ONLY


# Factory functions
def create_config_for_hardware(
    hardware_type: HardwareType,
    target_latency_ms: float = 50.0,
    max_concurrent_streams: int | None = None
) -> DeploymentConfiguration:
    """Create deployment configuration for specific hardware."""

    generator = UltraFastDeploymentConfigGenerator()
    return generator.generate_optimal_config(
        hardware_type=hardware_type,
        target_latency_ms=target_latency_ms,
        max_concurrent_streams=max_concurrent_streams
    )


def auto_generate_config(target_latency_ms: float = 50.0) -> DeploymentConfiguration:
    """Auto-detect hardware and generate optimal configuration."""

    generator = UltraFastDeploymentConfigGenerator()
    detected_hardware = generator.auto_detect_hardware()

    logger.info(f"Auto-detected hardware: {detected_hardware.value}")

    return generator.generate_optimal_config(
        hardware_type=detected_hardware,
        target_latency_ms=target_latency_ms
    )


# CLI interface
def main():
    """CLI for deployment configuration generation."""
    import argparse

    parser = argparse.ArgumentParser(description="Ultra-fast YOLO11 deployment configuration generator")
    parser.add_argument("--hardware", type=str, choices=[h.value for h in HardwareType],
                       help="Target hardware type (auto-detect if not specified)")
    parser.add_argument("--target-latency", type=float, default=50.0,
                       help="Target P99 latency in milliseconds")
    parser.add_argument("--max-streams", type=int,
                       help="Maximum concurrent streams (auto-calculate if not specified)")
    parser.add_argument("--output-dir", type=Path, default="./deployment",
                       help="Output directory for deployment package")
    parser.add_argument("--include-docker", action="store_true", default=True,
                       help="Include Docker configuration")
    parser.add_argument("--include-k8s", action="store_true", default=True,
                       help="Include Kubernetes manifests")

    args = parser.parse_args()

    # Create configuration
    generator = UltraFastDeploymentConfigGenerator()

    if args.hardware:
        hardware_type = HardwareType(args.hardware)
        config = generator.generate_optimal_config(
            hardware_type=hardware_type,
            target_latency_ms=args.target_latency,
            max_concurrent_streams=args.max_streams
        )
    else:
        config = auto_generate_config(target_latency_ms=args.target_latency)

    # Generate deployment package
    generated_files = generator.generate_deployment_package(
        config=config,
        output_dir=args.output_dir,
        include_docker=args.include_docker,
        include_kubernetes=args.include_k8s
    )

    print(f"\n🎯 Deployment configuration generated for {config.target_hardware.value}")
    print(f"Target P99 Latency: {config.target_latency_p99_ms}ms")
    print(f"Max Concurrent Streams: {config.max_concurrent_streams}")
    print(f"Model Variant: {config.model_config.model_variant}")
    print(f"Quantization: {config.model_config.quantization}")
    print("\n📁 Generated files:")
    for name, path in generated_files.items():
        print(f"  - {name}: {path}")
    print(f"\n🚀 Run deployment with: cd {args.output_dir} && ./deploy.sh")


if __name__ == "__main__":
    main()
