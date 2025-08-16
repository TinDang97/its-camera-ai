"""
Production-optimized inference engine for ITS Camera AI Traffic Monitoring System.

This module implements high-performance YOLO11 inference with TensorRT optimization,
dynamic batching, and memory management specifically designed for real-time traffic
monitoring with sub-100ms latency requirements.

Performance Targets:
- Sub-100ms inference latency (critical for real-time traffic)
- 95%+ vehicle detection accuracy
- 30 FPS processing per camera
- Support for 1000+ concurrent camera streams

Key Optimizations:
1. TensorRT model compilation with FP16/INT8 quantization
2. Dynamic batching with intelligent timeout management
3. GPU memory pooling and zero-copy operations
4. Multi-GPU load balancing with device affinity
5. Async processing pipeline with queue management
"""

import asyncio
import contextlib
import logging
import os
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Any

import cv2
import numpy as np
import torch
from torch.cuda.amp import autocast
from ultralytics import YOLO

# Import new optimization modules (with fallbacks)
try:
    from .tensorrt_production_optimizer import (
        ProductionTensorRTEngine,
        ProductionTensorRTOptimizer,
        TensorRTConfig,
    )
    ENHANCED_TRT_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Enhanced TensorRT optimizer not available: {e}")
    ENHANCED_TRT_AVAILABLE = False
    ProductionTensorRTEngine = None
    ProductionTensorRTOptimizer = None
    TensorRTConfig = None

try:
    from .enhanced_memory_manager import MultiGPUMemoryManager, TensorPoolConfig
    ENHANCED_MEMORY_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Enhanced memory manager not available: {e}")
    ENHANCED_MEMORY_AVAILABLE = False
    MultiGPUMemoryManager = None
    TensorPoolConfig = None

try:
    from .advanced_batching_system import (
        AdvancedDynamicBatcher as NewAdvancedDynamicBatcher,
    )
    from .advanced_batching_system import BatchConfiguration
    ENHANCED_BATCHING_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Enhanced batching system not available: {e}")
    ENHANCED_BATCHING_AVAILABLE = False
    NewAdvancedDynamicBatcher = None
    BatchConfiguration = None

try:
    import pycuda.driver as cuda
    import tensorrt as trt
    TRT_AVAILABLE = True
except ImportError:
    trt = None
    cuda = None
    TRT_AVAILABLE = False

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ort = None
    ONNX_AVAILABLE = False

try:
    import openvino as ov
    OPENVINO_AVAILABLE = True
except ImportError:
    ov = None
    OPENVINO_AVAILABLE = False

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    pynvml = None
    PYNVML_AVAILABLE = False


logger = logging.getLogger(__name__)


class TensorRTModel:
    """Wrapper for TensorRT inference engine."""

    def __init__(self, engine: Any, context: Any, device: str) -> None:
        self.engine = engine
        self.context = context
        self.device = device

        # Pre-allocate GPU memory for inputs and outputs
        self.bindings: list[int] = []
        self.inputs: list[dict[str, Any]] = []
        self.outputs: list[dict[str, Any]] = []

        for i in range(engine.num_bindings):
            binding_name = engine.get_binding_name(i)
            dtype = engine.get_binding_dtype(i)
            shape = engine.get_binding_shape(i)

            # Convert TensorRT dtype to numpy dtype
            if dtype == engine.float32:
                np_dtype = np.float32
            elif dtype == engine.float16:
                np_dtype = np.float16
            elif dtype == engine.int8:
                np_dtype = np.int8
            else:
                np_dtype = np.float32

            # Allocate GPU memory
            size = int(np.prod(shape) * np_dtype().itemsize)
            if torch.cuda.is_available():
                mem_gpu = torch.cuda.empty(size, dtype=torch.uint8).data_ptr()
            else:
                mem_gpu = None

            self.bindings.append(int(mem_gpu))

            if engine.binding_is_input(i):
                self.inputs.append(
                    {
                        "name": binding_name,
                        "mem_gpu": mem_gpu,
                        "shape": shape,
                        "dtype": np_dtype,
                    }
                )
            else:
                self.outputs.append(
                    {
                        "name": binding_name,
                        "mem_gpu": mem_gpu,
                        "shape": shape,
                        "dtype": np_dtype,
                    }
                )

    def __call__(
        self, input_tensor: torch.Tensor
    ) -> torch.Tensor | list[torch.Tensor] | None:
        """Run inference using TensorRT engine."""
        if not isinstance(input_tensor, torch.Tensor):
            input_tensor = (
                torch.from_numpy(input_tensor)
                if isinstance(input_tensor, np.ndarray)
                else input_tensor
            )

        # Copy input to GPU memory
        input_data = input_tensor.contiguous().cpu().numpy()

        try:
            import pycuda.driver as cuda

            # Copy input data to GPU
            cuda.memcpy_htod(self.inputs[0]["mem_gpu"], input_data)

            # Run inference
            self.context.execute_v2(self.bindings)

            # Copy output back to host
            outputs = []
            for output_info in self.outputs:
                output_shape = output_info["shape"]
                output_dtype = output_info["dtype"]

                # Handle dynamic batch size
                if output_shape[0] == -1:
                    output_shape = (input_tensor.shape[0],) + output_shape[1:]

                output_data = np.empty(output_shape, dtype=output_dtype)
                cuda.memcpy_dtoh(output_data, output_info["mem_gpu"])

                outputs.append(torch.from_numpy(output_data))

            return outputs[0] if len(outputs) == 1 else outputs

        except ImportError:
            logger.error("PyCUDA not available for TensorRT inference")
            return None


class ModelType(Enum):
    """Supported YOLO11 model variants for different deployment scenarios."""

    NANO = "yolo11n.pt"  # Ultra-fast edge deployment, 2.6M params
    SMALL = "yolo11s.pt"  # Balanced performance, 9.4M params
    MEDIUM = "yolo11m.pt"  # High accuracy cloud deployment, 20.1M params
    LARGE = "yolo11l.pt"  # Maximum accuracy, 25.3M params
    XLARGE = "yolo11x.pt"  # Research grade, 56.9M params


class OptimizationBackend(Enum):
    """Inference optimization backends."""

    PYTORCH = "pytorch"
    TENSORRT = "tensorrt"
    ONNX = "onnx"
    OPENVINO = "openvino"


@dataclass
class InferenceConfig:
    """Configuration for optimized inference pipeline."""

    # Model Configuration
    model_type: ModelType = ModelType.SMALL
    backend: OptimizationBackend = OptimizationBackend.TENSORRT
    precision: str = "fp16"  # fp32, fp16, int8

    # Performance Settings
    batch_size: int = 8
    max_batch_size: int = 32
    batch_timeout_ms: int = 10
    input_size: tuple[int, int] = (640, 640)

    # GPU Settings
    device_ids: list[int] | None = None
    memory_fraction: float = 0.8
    enable_cudnn_benchmark: bool = True

    # Quality Settings
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    max_detections: int = 300

    # Edge Deployment
    enable_edge_optimization: bool = False
    target_fps: int = 30
    max_latency_ms: int = 100

    def __post_init__(self) -> None:
        if self.device_ids is None:
            self.device_ids = [0] if torch.cuda.is_available() else []
        if not self.device_ids:
            logger.warning("No GPU devices available, falling back to CPU")


@dataclass
class DetectionResult:
    """Structured detection result with performance metrics."""

    # Detection Data
    boxes: np.ndarray[Any, np.dtype[np.float32]]  # [N, 4] - xyxy format
    scores: np.ndarray[Any, np.dtype[np.float32]]  # [N] - confidence scores
    classes: np.ndarray[Any, np.dtype[np.int32]]  # [N] - class indices
    class_names: list[str]  # Human-readable class names

    # Metadata
    frame_id: str
    camera_id: str
    timestamp: float

    # Performance Metrics
    inference_time_ms: float
    preprocessing_time_ms: float
    postprocessing_time_ms: float
    total_time_ms: float

    # Quality Metrics
    detection_count: int
    avg_confidence: float
    gpu_memory_used_mb: float


class CUDAStreamManager:
    """Advanced CUDA stream management for parallel inference processing."""

    def __init__(self, device_ids: list[int], streams_per_device: int = 4) -> None:
        self.device_ids = device_ids
        self.streams_per_device = streams_per_device
        self.streams: dict[int, list[torch.cuda.Stream]] = {}
        self.stream_idx: dict[int, int] = {}
        self.stream_locks: dict[int, list[Lock]] = {}

        self._initialize_streams()

    def _initialize_streams(self) -> None:
        """Initialize CUDA streams for each device."""
        for device_id in self.device_ids:
            with torch.cuda.device(device_id):
                device_streams = []
                device_locks = []

                for _ in range(self.streams_per_device):
                    stream = torch.cuda.Stream(device=device_id)
                    device_streams.append(stream)
                    device_locks.append(Lock())

                self.streams[device_id] = device_streams
                self.stream_locks[device_id] = device_locks
                self.stream_idx[device_id] = 0

                logger.info(f"Initialized {self.streams_per_device} CUDA streams for GPU {device_id}")

    def get_next_stream(self, device_id: int) -> tuple[torch.cuda.Stream, Lock]:
        """Get next available stream with round-robin scheduling."""
        idx = self.stream_idx[device_id]
        self.stream_idx[device_id] = (idx + 1) % self.streams_per_device

        return self.streams[device_id][idx], self.stream_locks[device_id][idx]

    def synchronize_all(self) -> None:
        """Synchronize all streams across all devices."""
        for device_id in self.device_ids:
            for stream in self.streams[device_id]:
                stream.synchronize()

    def cleanup(self) -> None:
        """Clean up streams."""
        self.synchronize_all()


class GPUMemoryManager:
    """Advanced GPU memory management for high-throughput inference."""

    def __init__(self, device_ids: list[int], memory_fraction: float = 0.8) -> None:
        self.device_ids = device_ids
        self.memory_fraction = memory_fraction
        self.memory_pools: dict[int, dict[tuple[int, ...], deque[torch.Tensor]]] = {}
        self.allocated_tensors: dict[int, list[torch.Tensor]] = {}
        self.pool_locks: dict[int, Lock] = {}

        self._initialize_memory_pools()

    def _initialize_memory_pools(self) -> None:
        """Initialize memory pools for each GPU device."""
        for device_id in self.device_ids:
            torch.cuda.set_device(device_id)

            # Set memory fraction to prevent OOM
            torch.cuda.set_per_process_memory_fraction(self.memory_fraction, device_id)

            # Pre-allocate common tensor shapes with multiple instances
            self.memory_pools[device_id] = {}
            self.pool_locks[device_id] = Lock()

            common_shapes = [
                (1, 3, 640, 640),  # Single frame
                (2, 3, 640, 640),  # Pair batch
                (4, 3, 640, 640),  # Small batch
                (8, 3, 640, 640),  # Standard batch
                (16, 3, 640, 640),  # Large batch
                (32, 3, 640, 640),  # Max batch
                (1, 3, 1280, 1280),  # High-res single
                (4, 3, 1280, 1280),  # High-res batch
            ]

            for shape in common_shapes:
                # Pre-allocate multiple tensors per shape for concurrent use
                tensor_pool = deque()
                for _ in range(4):  # 4 tensors per shape
                    tensor = torch.zeros(
                        shape, dtype=torch.float16, device=f"cuda:{device_id}"
                    ).to(memory_format=torch.channels_last)
                    tensor_pool.append(tensor)

                self.memory_pools[device_id][shape] = tensor_pool

            logger.info(f"Initialized memory pool for GPU {device_id} with {len(common_shapes)} shapes")

    def get_tensor(self, shape: tuple[int, ...], device_id: int) -> torch.Tensor:
        """Get pre-allocated tensor or create new one with memory pooling."""
        with self.pool_locks[device_id]:
            if shape in self.memory_pools[device_id] and self.memory_pools[device_id][shape]:
                # Reuse existing tensor from pool
                tensor = self.memory_pools[device_id][shape].popleft()
                tensor.zero_()  # Clear previous data
                return tensor

        # Create new tensor if pool is empty
        return torch.zeros(
            shape,
            dtype=torch.float16,
            device=f"cuda:{device_id}"
        ).to(memory_format=torch.channels_last)

    def return_tensor(self, tensor: torch.Tensor, device_id: int) -> None:
        """Return tensor to memory pool for reuse."""
        shape = tuple(tensor.shape)

        with self.pool_locks[device_id]:
            if shape in self.memory_pools[device_id] and len(self.memory_pools[device_id][shape]) < 8:
                # Return to pool if not full (max 8 tensors per shape)
                self.memory_pools[device_id][shape].append(tensor)
            # Otherwise let it be garbage collected

    def cleanup(self) -> None:
        """Clean up GPU memory and pools."""
        for device_id in self.device_ids:
            torch.cuda.set_device(device_id)

            # Clear memory pools
            with self.pool_locks[device_id]:
                for shape_pool in self.memory_pools[device_id].values():
                    shape_pool.clear()

            torch.cuda.empty_cache()
            torch.cuda.synchronize(device_id)


class AdvancedTensorRTOptimizer:
    """TensorRT model optimization for maximum inference performance."""

    def __init__(self, config: InferenceConfig):
        self.config = config
        self.trt_engines: dict[str, Any] = {}

        if not TRT_AVAILABLE:
            logger.warning("TensorRT not available, falling back to PyTorch")
            return

    def compile_model(self, model_path: Path, output_path: Path) -> Path:
        """Compile YOLO11 model to optimized TensorRT engine."""
        if not TRT_AVAILABLE:
            return model_path

        logger.info(f"Compiling {model_path} to TensorRT engine...")

        # Load PyTorch model
        model = YOLO(str(model_path))

        # Export to ONNX first
        onnx_path = output_path.with_suffix(".onnx")
        model.export(
            format="onnx",
            imgsz=self.config.input_size,
            dynamic=True,
            batch_size=1,
            opset=16,
            half=self.config.precision == "fp16",
        )

        # Compile ONNX to TensorRT
        engine_path = output_path.with_suffix(".trt")
        self._compile_onnx_to_tensorrt(onnx_path, engine_path)

        return engine_path

    def optimize_onnx_model(self, model_path: Path, output_path: Path) -> Path:
        """Optimize ONNX model for production deployment."""
        if not ONNX_AVAILABLE:
            logger.warning("ONNX Runtime not available")
            return model_path

        logger.info(f"Optimizing ONNX model {model_path}...")

        try:
            import onnxruntime as ort

            # Load original model
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )

            # Enable optimizations
            session_options.optimized_model_filepath = str(output_path)

            # Create session to trigger optimization
            providers = (
                ["CUDAExecutionProvider"]
                if torch.cuda.is_available()
                else ["CPUExecutionProvider"]
            )
            session = ort.InferenceSession(
                str(model_path), session_options, providers=providers
            )

            # Test run to ensure optimization
            dummy_input = np.random.randn(1, 3, *self.config.input_size).astype(
                np.float32
            )
            session.run(None, {"images": dummy_input})

            logger.info(f"ONNX model optimized and saved to {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"ONNX optimization failed: {e}")
            return model_path

    def quantize_model_int8(
        self, model_path: Path, calibration_data: list[np.ndarray[Any, np.dtype[Any]]]
    ) -> Path:
        """Quantize model to INT8 for edge deployment."""
        logger.info(f"Quantizing model to INT8: {model_path}")

        try:
            import torch.quantization as quant
            from ultralytics import YOLO

            # Load model
            model = YOLO(str(model_path))

            # Prepare model for quantization
            model.model.eval()
            model.model.qconfig = quant.get_default_qconfig("fbgemm")

            # Prepare for quantization
            quant.prepare(model.model, inplace=True)

            # Calibration with sample data
            with torch.no_grad():
                for calib_data in calibration_data[:50]:  # Use subset for calibration
                    if isinstance(calib_data, np.ndarray):
                        calib_tensor = torch.from_numpy(calib_data)
                    else:
                        calib_tensor = calib_data

                    if len(calib_tensor.shape) == 3:
                        calib_tensor = calib_tensor.unsqueeze(0)

                    model.model(calib_tensor)

            # Convert to quantized model
            quant.convert(model.model, inplace=True)

            # Save quantized model
            quantized_path = model_path.with_suffix(".int8.pt")
            torch.save(model.model.state_dict(), quantized_path)

            logger.info(f"INT8 quantized model saved to {quantized_path}")
            return quantized_path

        except Exception as e:
            logger.error(f"INT8 quantization failed: {e}")
            return model_path

    def _compile_onnx_to_tensorrt(self, onnx_path: Path, engine_path: Path) -> None:
        """Compile ONNX model to TensorRT engine."""
        import tensorrt as trt

        logger.info("Building TensorRT engine...")

        # Initialize TensorRT
        trt_logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(trt_logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, trt_logger)

        # Parse ONNX model
        with open(onnx_path, "rb") as model:
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    logger.error(parser.get_error(error))
                raise RuntimeError("Failed to parse ONNX model")

        # Configure builder
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)  # 4GB

        # Set precision
        if self.config.precision == "fp16":
            config.set_flag(trt.BuilderFlag.FP16)
        elif self.config.precision == "int8":
            config.set_flag(trt.BuilderFlag.INT8)
            # INT8 calibration using sample dataset
            config.set_flag(trt.BuilderFlag.INT8)
            calibrator = self._create_int8_calibrator()
            if calibrator:
                config.int8_calibrator = calibrator

        # Set dynamic shapes for batching
        profile = builder.create_optimization_profile()
        input_name = network.get_input(0).name

        profile.set_shape(
            input_name,
            (1, 3, *self.config.input_size),  # min
            (self.config.batch_size, 3, *self.config.input_size),  # opt
            (self.config.max_batch_size, 3, *self.config.input_size),  # max
        )
        config.add_optimization_profile(profile)

        # Build engine
        engine = builder.build_serialized_network(network, config)
        if engine is None:
            raise RuntimeError("Failed to build TensorRT engine")

        # Save engine
        with open(engine_path, "wb") as f:
            f.write(engine)

        logger.info(f"TensorRT engine saved to {engine_path}")

    def _create_int8_calibrator(self) -> Any | None:
        """Create INT8 calibrator for quantization."""
        try:
            import tensorrt as trt

            class TrafficCalibrator(trt.IInt8EntropyCalibrator2):
                def __init__(self, calibration_files, cache_file, batch_size=1):
                    trt.IInt8EntropyCalibrator2.__init__(self)
                    self.calibration_files = calibration_files
                    self.cache_file = cache_file
                    self.batch_size = batch_size
                    self.current_index = 0

                    # Allocate device memory
                    self.device_input = None
                    self.input_shape = (batch_size, 3, *self.config.input_size)

                def get_batch_size(self):
                    return self.batch_size

                def get_batch(self, _names):
                    if self.current_index + self.batch_size > len(
                        self.calibration_files
                    ):
                        return None

                    # Load and preprocess batch
                    batch_data = self._load_calibration_batch()

                    if self.device_input is None:
                        import pycuda.driver as cuda

                        self.device_input = cuda.mem_alloc(batch_data.nbytes)

                    # Copy to device
                    import pycuda.driver as cuda

                    cuda.memcpy_htod(self.device_input, batch_data)

                    self.current_index += self.batch_size
                    return [int(self.device_input)]

                def _load_calibration_batch(self):
                    """Load and preprocess calibration images."""
                    batch_images = []

                    for i in range(self.batch_size):
                        if self.current_index + i >= len(self.calibration_files):
                            break

                        # For demo, create synthetic calibration data
                        # In production, use real traffic images
                        img = np.random.randint(
                            0,
                            255,
                            (self.config.input_size[0], self.config.input_size[1], 3),
                            dtype=np.uint8,
                        )

                        # Preprocess like real inference
                        img_tensor = (
                            torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                        )
                        batch_images.append(img_tensor)

                    if batch_images:
                        batch_tensor = torch.stack(batch_images)
                        return batch_tensor.numpy().ascontiguousarray()
                    return None

                def read_calibration_cache(self):
                    if os.path.exists(self.cache_file):
                        with open(self.cache_file, "rb") as f:
                            return f.read()
                    return None

                def write_calibration_cache(self, cache):
                    with open(self.cache_file, "wb") as f:
                        f.write(cache)

            # Create calibrator with synthetic data for demo
            # In production, use real traffic images
            calibration_files = [f"calib_{i}.jpg" for i in range(100)]
            cache_file = "int8_calibration.cache"

            return TrafficCalibrator(calibration_files, cache_file, batch_size=1)

        except ImportError:
            logger.warning("PyCUDA not available for INT8 calibration")
            return None


class EdgeOptimizer:
    """Edge deployment optimizations for NVIDIA Jetson and Intel NUC."""

    def __init__(self, device_type: str = "jetson"):
        self.device_type = device_type.lower()
        self.optimization_config = self._get_edge_optimization_config()

    def _get_edge_optimization_config(self) -> dict[str, Any]:
        """Get device-specific optimization configuration."""
        configs = {
            "jetson": {
                "use_dla": True,  # Deep Learning Accelerator
                "dla_core": 0,
                "max_batch_size": 4,
                "precision": "int8",
                "workspace_size_gb": 1,
                "enable_gpu_fallback": True,
                "cpu_threads": 2,
                "memory_optimization": "aggressive",
            },
            "nuc": {
                "use_dla": False,
                "max_batch_size": 2,
                "precision": "fp16",
                "workspace_size_gb": 2,
                "enable_gpu_fallback": False,
                "cpu_threads": 4,
                "memory_optimization": "balanced",
            },
            "xavier": {
                "use_dla": True,
                "dla_core": 0,
                "max_batch_size": 8,
                "precision": "fp16",
                "workspace_size_gb": 2,
                "enable_gpu_fallback": True,
                "cpu_threads": 4,
                "memory_optimization": "performance",
            },
            "orin": {
                "use_dla": True,
                "dla_core": 0,
                "max_batch_size": 16,
                "precision": "fp16",
                "workspace_size_gb": 4,
                "enable_gpu_fallback": True,
                "cpu_threads": 8,
                "memory_optimization": "performance",
            },
        }

        return configs.get(self.device_type, configs["jetson"])

    def optimize_for_edge(self, config: InferenceConfig) -> InferenceConfig:
        """Apply edge-specific optimizations to inference config."""
        edge_config = self.optimization_config

        # Update config for edge deployment
        config.batch_size = min(config.batch_size, edge_config["max_batch_size"])
        config.max_batch_size = edge_config["max_batch_size"]
        config.precision = edge_config["precision"]
        config.batch_timeout_ms = 5  # Aggressive batching for edge

        # Memory optimization
        if edge_config["memory_optimization"] == "aggressive":
            config.memory_fraction = 0.6
        elif edge_config["memory_optimization"] == "balanced":
            config.memory_fraction = 0.7
        else:
            config.memory_fraction = 0.8

        # Edge-specific settings
        config.enable_edge_optimization = True
        config.target_fps = 15 if self.device_type in ["jetson", "nuc"] else 30
        config.max_latency_ms = 150 if self.device_type == "jetson" else 100

        logger.info(f"Applied {self.device_type} edge optimizations")
        return config

    def compile_edge_model(self, model_path: Path, output_path: Path, config: InferenceConfig) -> Path:
        """Compile model specifically for edge deployment."""

        if self.device_type.startswith("jetson") and TRT_AVAILABLE:
            return self._compile_jetson_model(model_path, output_path, config)
        elif self.device_type == "nuc":
            return self._compile_nuc_model(model_path, output_path, config)
        else:
            logger.warning(f"No specific optimization for {self.device_type}")
            return model_path

    def _compile_jetson_model(self, model_path: Path, output_path: Path, config: InferenceConfig) -> Path:
        """Compile model optimized for NVIDIA Jetson devices."""
        try:

            logger.info(f"Compiling model for Jetson {self.device_type}...")

            # Load and export model to ONNX first
            model = YOLO(str(model_path))
            onnx_path = output_path.with_suffix(".onnx")

            model.export(
                format="onnx",
                imgsz=config.input_size,
                dynamic=False,  # Static shapes for DLA
                batch_size=1,
                opset=16,
                half=config.precision == "fp16",
                simplify=True,
            )

            # Build TensorRT engine with DLA support
            engine_path = output_path.with_suffix(".trt")
            self._build_jetson_tensorrt_engine(onnx_path, engine_path, config)

            return engine_path

        except Exception as e:
            logger.error(f"Jetson model compilation failed: {e}")
            return model_path

    def _build_jetson_tensorrt_engine(self, onnx_path: Path, engine_path: Path, config: InferenceConfig) -> None:
        """Build TensorRT engine optimized for Jetson with DLA support."""
        import tensorrt as trt

        # Initialize TensorRT with DLA support
        trt_logger = trt.Logger(trt.Logger.INFO)
        builder = trt.Builder(trt_logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, trt_logger)

        # Parse ONNX model
        with open(onnx_path, "rb") as model:
            if not parser.parse(model.read()):
                raise RuntimeError("Failed to parse ONNX model for Jetson")

        # Configure for Jetson with DLA
        config_trt = builder.create_builder_config()
        edge_config = self.optimization_config

        # Set workspace size (smaller for edge devices)
        workspace_size = edge_config["workspace_size_gb"] << 30
        config_trt.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size)

        # Enable DLA if supported
        if edge_config["use_dla"] and builder.num_DLA_cores > 0:
            config_trt.set_flag(trt.BuilderFlag.GPU_FALLBACK)
            config_trt.DLA_core = edge_config["dla_core"]
            logger.info(f"Enabled DLA core {edge_config['dla_core']}")

        # Set precision
        if config.precision == "fp16":
            config_trt.set_flag(trt.BuilderFlag.FP16)
        elif config.precision == "int8":
            config_trt.set_flag(trt.BuilderFlag.INT8)
            config_trt.set_flag(trt.BuilderFlag.FP16)  # Fallback

        # Static shapes for DLA compatibility
        profile = builder.create_optimization_profile()
        input_name = network.get_input(0).name
        static_shape = (1, 3, *config.input_size)
        profile.set_shape(input_name, static_shape, static_shape, static_shape)
        config_trt.add_optimization_profile(profile)

        # Build engine
        serialized_engine = builder.build_serialized_network(network, config_trt)
        if serialized_engine is None:
            raise RuntimeError("Failed to build Jetson TensorRT engine")

        # Save engine
        with open(engine_path, "wb") as f:
            f.write(serialized_engine)

        logger.info(f"Jetson-optimized TensorRT engine saved: {engine_path}")

    def _compile_nuc_model(self, model_path: Path, output_path: Path, config: InferenceConfig) -> Path:
        """Compile model optimized for Intel NUC with OpenVINO."""
        try:
            # Try OpenVINO optimization
            return self._compile_openvino_model(model_path, output_path, config)
        except ImportError:
            logger.warning("OpenVINO not available, using ONNX Runtime")
            return self._compile_onnx_optimized(model_path, output_path, config)

    def _compile_openvino_model(self, model_path: Path, output_path: Path, config: InferenceConfig) -> Path:
        """Compile model with OpenVINO for Intel hardware."""
        try:
            import openvino as ov

            logger.info("Compiling model with OpenVINO...")

            # Export to ONNX first
            model = YOLO(str(model_path))
            onnx_path = output_path.with_suffix(".onnx")

            model.export(
                format="onnx",
                imgsz=config.input_size,
                dynamic=False,
                batch_size=config.batch_size,
                opset=16,
                half=False,  # OpenVINO handles precision optimization
            )

            # Convert to OpenVINO IR
            core = ov.Core()
            ov_model = core.read_model(onnx_path)

            # Apply optimizations
            if config.precision == "fp16":
                from openvino.tools import mo
                ov_model = mo.compress_to_fp16(ov_model)

            # Save optimized model
            ir_path = output_path.with_suffix(".xml")
            ov.save_model(ov_model, ir_path)

            logger.info(f"OpenVINO model saved: {ir_path}")
            return ir_path

        except ImportError:
            raise ImportError("OpenVINO not available")
        except Exception as e:
            logger.error(f"OpenVINO compilation failed: {e}")
            raise

    def _compile_onnx_optimized(self, model_path: Path, output_path: Path, config: InferenceConfig) -> Path:
        """Compile optimized ONNX model for CPU inference."""
        logger.info("Creating CPU-optimized ONNX model...")

        model = YOLO(str(model_path))
        onnx_path = output_path.with_suffix(".onnx")

        # Export with CPU optimizations
        model.export(
            format="onnx",
            imgsz=config.input_size,
            dynamic=True,
            batch_size=config.batch_size,
            opset=17,
            half=False,  # CPU doesn't benefit from FP16
            simplify=True,
        )

        return onnx_path

    def get_edge_inference_config(self) -> dict[str, Any]:
        """Get optimal inference configuration for edge device."""
        edge_config = self.optimization_config

        return {
            "device_type": self.device_type,
            "max_batch_size": edge_config["max_batch_size"],
            "precision": edge_config["precision"],
            "use_dla": edge_config.get("use_dla", False),
            "memory_optimization": edge_config["memory_optimization"],
            "recommended_input_size": (640, 640) if self.device_type == "jetson" else (416, 416),
            "target_fps": 15 if self.device_type in ["jetson", "nuc"] else 30,
        }


class CPUFallbackEngine:
    """CPU fallback inference engine using ONNX Runtime for high availability."""

    def __init__(self, config: InferenceConfig):
        self.config = config
        self.onnx_session = None
        self.model_input_name = None
        self.model_output_names = None
        self.inference_times: deque[float] = deque(maxlen=100)

    def initialize(self, model_path: Path) -> None:
        """Initialize CPU fallback engine with ONNX Runtime."""
        if not ONNX_AVAILABLE:
            raise RuntimeError("ONNX Runtime not available for CPU fallback")

        try:
            import onnxruntime as ort

            # Export YOLO11 to ONNX if needed
            onnx_path = self._ensure_onnx_model(model_path)

            # Configure ONNX Runtime for CPU optimization
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
            session_options.inter_op_num_threads = 4
            session_options.intra_op_num_threads = 8

            # Enable CPU-specific optimizations
            providers = [
                ('CPUExecutionProvider', {
                    'use_arena': True,
                    'arena_extend_strategy': 'kSameAsRequested',
                })
            ]

            # Create inference session
            self.onnx_session = ort.InferenceSession(
                str(onnx_path),
                session_options,
                providers=providers
            )

            # Get input/output metadata
            self.model_input_name = self.onnx_session.get_inputs()[0].name
            self.model_output_names = [output.name for output in self.onnx_session.get_outputs()]

            logger.info("CPU fallback engine initialized with ONNX Runtime")

        except Exception as e:
            logger.error(f"Failed to initialize CPU fallback engine: {e}")
            raise

    def _ensure_onnx_model(self, model_path: Path) -> Path:
        """Ensure ONNX model exists, export if needed."""
        onnx_path = model_path.with_suffix('.onnx')

        if not onnx_path.exists():
            logger.info("Exporting YOLO11 model to ONNX for CPU fallback...")

            model = YOLO(str(model_path))
            model.export(
                format='onnx',
                imgsz=self.config.input_size,
                dynamic=True,
                batch_size=1,
                opset=17,
                half=False,  # Use FP32 for CPU
                simplify=True,
            )

        return onnx_path

    def predict(self, frame: np.ndarray, frame_id: str, camera_id: str = "unknown") -> DetectionResult:
        """Run CPU inference using ONNX Runtime."""
        if not self.onnx_session:
            raise RuntimeError("CPU fallback engine not initialized")

        start_time = time.time()

        # Preprocess frame
        preprocess_start = time.time()
        input_tensor = self._preprocess_frame_cpu(frame)
        preprocess_time = (time.time() - preprocess_start) * 1000

        # Run inference
        inference_start = time.time()
        try:
            outputs = self.onnx_session.run(
                self.model_output_names,
                {self.model_input_name: input_tensor}
            )

            inference_time = (time.time() - inference_start) * 1000

            # Post-process results
            postprocess_start = time.time()
            result = self._postprocess_cpu_predictions(
                outputs[0], frame_id, camera_id, inference_time, preprocess_time, start_time
            )
            postprocess_time = (time.time() - postprocess_start) * 1000

            result.postprocessing_time_ms = postprocess_time
            result.total_time_ms = (time.time() - start_time) * 1000

            # Track performance
            self.inference_times.append(result.total_time_ms)

            return result

        except Exception as e:
            logger.error(f"CPU fallback inference failed: {e}")
            raise

    def _preprocess_frame_cpu(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for CPU inference."""
        h, w = frame.shape[:2]
        target_h, target_w = self.config.input_size

        # Resize with letterboxing
        scale = min(target_h / h, target_w / w)
        new_h, new_w = int(h * scale), int(w * scale)

        if scale != 1:
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Create padded canvas
        pad_h = (target_h - new_h) // 2
        pad_w = (target_w - new_w) // 2

        padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = frame

        # Convert to NCHW format and normalize
        input_tensor = padded.transpose(2, 0, 1).astype(np.float32)
        input_tensor = input_tensor / 255.0
        input_tensor = np.expand_dims(input_tensor, axis=0)

        return input_tensor

    def _postprocess_cpu_predictions(
        self,
        predictions: np.ndarray,
        frame_id: str,
        camera_id: str,
        inference_time: float,
        preprocess_time: float,
        start_time: float
    ) -> DetectionResult:
        """Post-process CPU inference results."""
        # Apply NMS and confidence filtering
        filtered_predictions = self._apply_cpu_nms(predictions)

        # Extract detections
        boxes, scores, classes = self._extract_cpu_detections(filtered_predictions)

        # Vehicle class mapping
        coco_to_vehicle = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
        class_names = [coco_to_vehicle.get(int(cls), f"class_{cls}") for cls in classes]

        # Calculate metrics
        detection_count = len(boxes)
        avg_confidence = float(np.mean(scores)) if len(scores) > 0 else 0.0

        return DetectionResult(
            boxes=boxes,
            scores=scores,
            classes=classes,
            class_names=class_names,
            frame_id=frame_id,
            camera_id=camera_id,
            timestamp=time.time(),
            inference_time_ms=inference_time,
            preprocessing_time_ms=preprocess_time,
            postprocessing_time_ms=0.0,  # Updated later
            total_time_ms=0.0,  # Updated later
            detection_count=detection_count,
            avg_confidence=avg_confidence,
            gpu_memory_used_mb=0.0,  # CPU fallback
        )

    def _apply_cpu_nms(self, predictions: np.ndarray) -> np.ndarray:
        """Apply NMS using CPU operations."""
        if len(predictions.shape) == 3:
            predictions = predictions[0]  # Remove batch dimension

        # Filter by confidence
        conf_mask = predictions[:, 4] >= self.config.conf_threshold
        filtered_preds = predictions[conf_mask]

        if len(filtered_preds) == 0:
            return np.array([]).reshape(0, predictions.shape[1])

        # Extract components
        boxes = filtered_preds[:, :4]
        scores = filtered_preds[:, 4]
        class_probs = filtered_preds[:, 5:]

        # Get class predictions
        class_scores = np.max(class_probs, axis=1)
        np.argmax(class_probs, axis=1)
        final_scores = scores * class_scores

        # Convert to corner format for NMS
        x_center, y_center, width, height = boxes.T
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

        # Simple CPU NMS implementation
        keep_indices = self._cpu_nms(boxes_xyxy, final_scores, self.config.iou_threshold)

        if len(keep_indices) > self.config.max_detections:
            keep_indices = keep_indices[:self.config.max_detections]

        return filtered_preds[keep_indices]

    def _cpu_nms(self, boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
        """CPU-based Non-Maximum Suppression."""
        if len(boxes) == 0:
            return np.array([], dtype=np.int32)

        # Sort by scores (highest first)
        indices = np.argsort(scores)[::-1]

        keep = []
        while len(indices) > 0:
            current = indices[0]
            keep.append(current)

            if len(indices) == 1:
                break

            # Calculate IoU with remaining boxes
            current_box = boxes[current]
            remaining_boxes = boxes[indices[1:]]

            # Calculate intersection
            x1 = np.maximum(current_box[0], remaining_boxes[:, 0])
            y1 = np.maximum(current_box[1], remaining_boxes[:, 1])
            x2 = np.minimum(current_box[2], remaining_boxes[:, 2])
            y2 = np.minimum(current_box[3], remaining_boxes[:, 3])

            intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

            # Calculate areas
            current_area = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
            remaining_areas = ((remaining_boxes[:, 2] - remaining_boxes[:, 0]) *
                             (remaining_boxes[:, 3] - remaining_boxes[:, 1]))

            # Calculate IoU
            union = current_area + remaining_areas - intersection
            iou = intersection / (union + 1e-8)

            # Keep boxes with IoU below threshold
            indices = indices[1:][iou <= iou_threshold]

        return np.array(keep, dtype=np.int32)

    def _extract_cpu_detections(self, predictions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract detections from CPU predictions."""
        if len(predictions) == 0:
            return np.array([]).reshape(0, 4), np.array([]), np.array([])

        # Extract components
        boxes_xywh = predictions[:, :4]
        confidences = predictions[:, 4]
        class_probs = predictions[:, 5:]

        # Get class predictions
        class_scores = np.max(class_probs, axis=1)
        class_indices = np.argmax(class_probs, axis=1)
        final_scores = confidences * class_scores

        # Convert to corner format
        x_center, y_center, width, height = boxes_xywh.T
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

        # Filter for vehicle classes
        vehicle_classes = [1, 2, 3, 5, 7]
        vehicle_mask = np.isin(class_indices, vehicle_classes)

        if not np.any(vehicle_mask):
            return np.array([]).reshape(0, 4), np.array([]), np.array([])

        # Apply vehicle filter and sort by confidence
        filtered_boxes = boxes_xyxy[vehicle_mask]
        filtered_scores = final_scores[vehicle_mask]
        filtered_classes = class_indices[vehicle_mask]

        sort_indices = np.argsort(filtered_scores)[::-1]

        return (
            filtered_boxes[sort_indices],
            filtered_scores[sort_indices],
            filtered_classes[sort_indices]
        )

    def get_performance_stats(self) -> dict[str, float]:
        """Get CPU fallback performance statistics."""
        if not self.inference_times:
            return {}

        times = list(self.inference_times)
        return {
            "avg_latency_ms": float(np.mean(times)),
            "p95_latency_ms": float(np.percentile(times, 95)),
            "throughput_fps": 1000.0 / float(np.mean(times)) if times else 0.0,
        }


class PerformanceBenchmarkSuite:
    """Comprehensive performance benchmarking and monitoring system."""

    def __init__(self, inference_engine: "OptimizedInferenceEngine"):
        self.inference_engine = inference_engine
        self.benchmark_results: dict[str, Any] = {}
        self.benchmark_history: list[dict[str, Any]] = []

    async def run_comprehensive_benchmark(
        self,
        test_images: list[np.ndarray] | None = None,
        duration_seconds: int = 60
    ) -> dict[str, Any]:
        """Run comprehensive performance benchmark."""
        logger.info(f"Starting comprehensive benchmark for {duration_seconds}s...")

        if test_images is None:
            test_images = self._generate_test_images()

        results = {
            "timestamp": time.time(),
            "duration_seconds": duration_seconds,
            "test_config": {
                "batch_sizes": [1, 4, 8, 16],
                "input_size": self.inference_engine.config.input_size,
                "precision": self.inference_engine.config.precision,
                "backend": self.inference_engine.config.backend.value,
            },
            "benchmarks": {}
        }

        # Single frame latency test
        results["benchmarks"]["single_frame"] = await self._benchmark_single_frame(test_images[0])

        # Batch throughput test
        results["benchmarks"]["batch_throughput"] = await self._benchmark_batch_throughput(test_images)

        # Sustained load test
        results["benchmarks"]["sustained_load"] = await self._benchmark_sustained_load(
            test_images, duration_seconds
        )

        # Memory usage test
        results["benchmarks"]["memory_usage"] = self._benchmark_memory_usage()

        # GPU utilization test
        results["benchmarks"]["gpu_utilization"] = self._benchmark_gpu_utilization()

        # Store results
        self.benchmark_results = results
        self.benchmark_history.append(results)

        # Keep only last 10 benchmark runs
        if len(self.benchmark_history) > 10:
            self.benchmark_history = self.benchmark_history[-10:]

        logger.info("Comprehensive benchmark completed")
        return results

    def _generate_test_images(self, count: int = 32) -> list[np.ndarray]:
        """Generate synthetic test images for benchmarking."""
        images = []
        h, w = self.inference_engine.config.input_size

        for _i in range(count):
            # Create realistic traffic scene patterns
            base_color = np.random.randint(80, 200, 3)
            img = np.full((h, w, 3), base_color, dtype=np.uint8)

            # Add some noise and patterns
            noise = np.random.randint(-30, 30, (h, w, 3), dtype=np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            # Add some geometric shapes (simulating vehicles)
            for _ in range(np.random.randint(1, 6)):
                x1, y1 = np.random.randint(0, w-100), np.random.randint(0, h-60)
                x2, y2 = x1 + np.random.randint(60, 120), y1 + np.random.randint(40, 80)
                color = np.random.randint(50, 255, 3)
                cv2.rectangle(img, (x1, y1), (x2, y2), color.tolist(), -1)

            images.append(img)

        return images

    async def _benchmark_single_frame(self, test_image: np.ndarray) -> dict[str, float]:
        """Benchmark single frame inference latency."""
        latencies = []

        # Warmup
        for _ in range(10):
            await self.inference_engine.predict_single(test_image, "warmup", "benchmark")

        # Measure latency
        for i in range(100):
            start_time = time.time()
            await self.inference_engine.predict_single(test_image, f"bench_{i}", "benchmark")
            latency = (time.time() - start_time) * 1000
            latencies.append(latency)

        return {
            "avg_latency_ms": float(np.mean(latencies)),
            "p50_latency_ms": float(np.percentile(latencies, 50)),
            "p95_latency_ms": float(np.percentile(latencies, 95)),
            "p99_latency_ms": float(np.percentile(latencies, 99)),
            "min_latency_ms": float(np.min(latencies)),
            "max_latency_ms": float(np.max(latencies)),
        }

    async def _benchmark_batch_throughput(self, test_images: list[np.ndarray]) -> dict[str, Any]:
        """Benchmark batch inference throughput."""
        results = {}

        for batch_size in [1, 4, 8, 16]:
            if batch_size > len(test_images):
                continue

            batch_images = test_images[:batch_size]
            batch_ids = [f"batch_{i}" for i in range(batch_size)]

            throughputs = []
            latencies = []

            # Warmup
            for _ in range(5):
                await self.inference_engine.predict_batch(batch_images, batch_ids)

            # Measure throughput
            for _i in range(20):
                start_time = time.time()
                batch_results = await self.inference_engine.predict_batch(batch_images, batch_ids)
                total_time = time.time() - start_time

                throughput = len(batch_results) / total_time
                avg_latency = (total_time / len(batch_results)) * 1000

                throughputs.append(throughput)
                latencies.append(avg_latency)

            results[f"batch_size_{batch_size}"] = {
                "avg_throughput_fps": float(np.mean(throughputs)),
                "p95_throughput_fps": float(np.percentile(throughputs, 95)),
                "avg_latency_per_frame_ms": float(np.mean(latencies)),
                "efficiency_ratio": float(np.mean(throughputs)) / batch_size,
            }

        return results

    async def _benchmark_sustained_load(self, test_images: list[np.ndarray], duration_seconds: int) -> dict[str, Any]:
        """Benchmark sustained load performance."""
        start_time = time.time()
        end_time = start_time + duration_seconds

        total_frames = 0
        total_batches = 0
        latencies = []
        errors = 0

        image_idx = 0
        batch_size = min(8, len(test_images))

        while time.time() < end_time:
            try:
                # Select batch
                batch_images = []
                batch_ids = []

                for i in range(batch_size):
                    batch_images.append(test_images[image_idx])
                    batch_ids.append(f"sustained_{total_frames + i}")
                    image_idx = (image_idx + 1) % len(test_images)

                # Process batch
                batch_start = time.time()
                results = await self.inference_engine.predict_batch(batch_images, batch_ids)
                batch_time = (time.time() - batch_start) * 1000

                total_frames += len(results)
                total_batches += 1
                latencies.append(batch_time / len(results))

                # Small delay to prevent overwhelming
                await asyncio.sleep(0.001)

            except Exception as e:
                errors += 1
                logger.warning(f"Sustained load error: {e}")
                await asyncio.sleep(0.01)

        actual_duration = time.time() - start_time

        return {
            "total_frames_processed": total_frames,
            "total_batches_processed": total_batches,
            "actual_duration_seconds": actual_duration,
            "avg_throughput_fps": total_frames / actual_duration,
            "avg_latency_per_frame_ms": float(np.mean(latencies)) if latencies else 0,
            "p95_latency_per_frame_ms": float(np.percentile(latencies, 95)) if latencies else 0,
            "error_count": errors,
            "error_rate": errors / total_batches if total_batches > 0 else 0,
        }

    def _benchmark_memory_usage(self) -> dict[str, float]:
        """Benchmark memory usage patterns."""
        if not self.inference_engine.config.device_ids:
            return {"cpu_only": True}

        memory_stats = {}

        for device_id in self.inference_engine.config.device_ids:
            try:
                allocated = torch.cuda.memory_allocated(device_id) / 1024 / 1024
                cached = torch.cuda.memory_reserved(device_id) / 1024 / 1024
                total = torch.cuda.get_device_properties(device_id).total_memory / 1024 / 1024

                memory_stats[f"gpu_{device_id}"] = {
                    "allocated_mb": float(allocated),
                    "cached_mb": float(cached),
                    "total_mb": float(total),
                    "utilization_percent": float(allocated / total * 100),
                }
            except Exception as e:
                memory_stats[f"gpu_{device_id}_error"] = str(e)

        return memory_stats

    def _benchmark_gpu_utilization(self) -> dict[str, float]:
        """Benchmark GPU utilization."""
        return self.inference_engine._get_gpu_memory_usage()

    def get_performance_summary(self) -> dict[str, Any]:
        """Get summary of recent performance benchmarks."""
        if not self.benchmark_history:
            return {"no_benchmarks": True}

        recent = self.benchmark_history[-1]

        return {
            "last_benchmark_time": recent["timestamp"],
            "single_frame_p95_ms": recent["benchmarks"]["single_frame"].get("p95_latency_ms", 0),
            "sustained_throughput_fps": recent["benchmarks"]["sustained_load"].get("avg_throughput_fps", 0),
            "memory_utilization_percent": self._calculate_avg_memory_utilization(recent),
            "meets_requirements": self._check_performance_requirements(recent),
        }

    def _calculate_avg_memory_utilization(self, benchmark_result: dict) -> float:
        """Calculate average memory utilization across GPUs."""
        memory_data = benchmark_result["benchmarks"].get("memory_usage", {})

        utilizations = []
        for _key, value in memory_data.items():
            if isinstance(value, dict) and "utilization_percent" in value:
                utilizations.append(value["utilization_percent"])

        return float(np.mean(utilizations)) if utilizations else 0.0

    def _check_performance_requirements(self, benchmark_result: dict) -> dict[str, bool]:
        """Check if performance meets requirements."""
        single_frame = benchmark_result["benchmarks"].get("single_frame", {})
        sustained = benchmark_result["benchmarks"].get("sustained_load", {})

        return {
            "sub_100ms_latency": single_frame.get("p95_latency_ms", 999) < 100,
            "30fps_sustained": sustained.get("avg_throughput_fps", 0) >= 30,
            "low_error_rate": sustained.get("error_rate", 1.0) < 0.01,
            "memory_efficient": self._calculate_avg_memory_utilization(benchmark_result) < 80,
        }


# Legacy alias for backward compatibility
TensorRTOptimizer = AdvancedTensorRTOptimizer


class AdvancedDynamicBatcher:
    """Advanced dynamic batching system with adaptive timeout and priority queues."""

    def __init__(self, config: InferenceConfig):
        self.config = config

        # Multi-priority queues for different latency requirements
        self.urgent_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=config.max_batch_size)
        self.normal_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=config.max_batch_size * 4)
        self.batch_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=config.max_batch_size * 8)

        self.result_futures: dict[str, asyncio.Future[Any]] = {}
        self.batch_processor: asyncio.Task[None] | None = None
        self.running = False

        # Advanced batching metrics
        self.total_requests: int = 0
        self.batched_requests: int = 0
        self.urgent_requests: int = 0
        self.avg_batch_size: float = 0
        self.adaptive_timeout: float = config.batch_timeout_ms / 1000.0
        self.timeout_history: deque[float] = deque(maxlen=100)
        self.latency_history: deque[float] = deque(maxlen=1000)

        # Performance tracking
        self.last_batch_time: float = time.time()
        self.batch_intervals: deque[float] = deque(maxlen=50)

    async def start(self) -> None:
        """Start the advanced dynamic batching processor."""
        self.running = True
        self.batch_processor = asyncio.create_task(self._process_batches_advanced())

        # Start adaptive timeout adjustment task
        asyncio.create_task(self._adjust_adaptive_timeout())

    async def stop(self) -> None:
        """Stop the advanced dynamic batching processor."""
        self.running = False
        if self.batch_processor:
            self.batch_processor.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.batch_processor

    async def predict(self, frame: np.ndarray, frame_id: str, camera_id: str = "unknown", priority: str = "normal") -> DetectionResult:
        """Add frame to appropriate priority queue and return prediction."""
        future = asyncio.Future()
        timestamp = time.time()

        request = {
            "frame": frame,
            "frame_id": frame_id,
            "camera_id": camera_id,
            "priority": priority,
            "future": future,
            "timestamp": timestamp,
            "queue_time": timestamp,
        }

        # Route to appropriate queue based on priority
        try:
            if priority == "urgent":
                await self.urgent_queue.put(request)
                self.urgent_requests += 1
            elif priority == "normal":
                await self.normal_queue.put(request)
            else:
                await self.batch_queue.put(request)

            self.total_requests += 1
        except asyncio.QueueFull:
            # Graceful degradation - use normal queue if preferred queue is full
            await self.normal_queue.put(request)
            self.total_requests += 1

        # Wait for result with timeout
        try:
            result = await asyncio.wait_for(future, timeout=5.0)  # 5 second timeout

            # Track latency
            latency = time.time() - timestamp
            self.latency_history.append(latency)

            return result
        except TimeoutError:
            logger.error(f"Prediction timeout for frame {frame_id}")
            raise RuntimeError(f"Prediction timeout for frame {frame_id}")

    async def _process_batches_advanced(self) -> None:
        """Process batches with advanced scheduling and priority handling."""
        while self.running:
            try:
                batch = await self._collect_priority_batch()
                if batch:
                    batch_start = time.time()
                    await self._process_batch(batch)

                    # Track batch processing intervals
                    batch_time = time.time() - batch_start
                    interval = batch_start - self.last_batch_time
                    self.batch_intervals.append(interval)
                    self.last_batch_time = batch_start

                    # Update adaptive timeout based on performance
                    self.timeout_history.append(batch_time)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Advanced batch processing error: {e}")
                # Small delay to prevent tight error loops
                await asyncio.sleep(0.001)

    async def _collect_priority_batch(self) -> list[dict[str, Any]]:
        """Collect batch with priority-aware scheduling and adaptive timeout."""
        batch = []
        deadline = time.time() + self.adaptive_timeout

        # First, try to get urgent requests (non-blocking)
        while len(batch) < self.config.max_batch_size // 4:  # Reserve 25% for urgent
            try:
                urgent_request = self.urgent_queue.get_nowait()
                batch.append(urgent_request)
            except asyncio.QueueEmpty:
                break

        # Then fill with normal and batch requests
        for queue in [self.normal_queue, self.batch_queue]:
            while len(batch) < self.config.max_batch_size and time.time() < deadline:
                try:
                    remaining_time = max(0.001, deadline - time.time())
                    request = await asyncio.wait_for(queue.get(), timeout=remaining_time)
                    batch.append(request)

                    # If we got a request, try to get more immediately
                    if len(batch) == 1:
                        deadline = time.time() + self.adaptive_timeout

                except TimeoutError:
                    break

            if batch:  # If we have requests, don't wait for other queues
                break

        # Sort batch by priority and timestamp
        if batch:
            batch.sort(key=lambda x: (x.get('priority', 'normal') != 'urgent', x['timestamp']))

        return batch

    async def _process_batch(self, batch: list[dict[str, Any]]) -> None:
        """Process a batch of requests with performance tracking."""
        if not batch:
            return

        batch_size = len(batch)
        self.batched_requests += batch_size

        # Update running average batch size
        alpha = 0.1  # Exponential moving average factor
        self.avg_batch_size = (1 - alpha) * self.avg_batch_size + alpha * batch_size

        # Extract frames and metadata
        [req['frame'] for req in batch]
        [req['frame_id'] for req in batch]
        [req.get('camera_id', 'unknown') for req in batch]

        try:
            # This will be called by the InferenceEngine
            logger.debug(f"Processing batch of {batch_size} frames")

            # For now, create mock results - this will be replaced by actual inference
            for _i, req in enumerate(batch):
                if not req['future'].done():
                    # Mock result - replace with actual inference results
                    mock_result = DetectionResult(
                        boxes=np.array([]).reshape(0, 4),
                        scores=np.array([]),
                        classes=np.array([], dtype=np.int32),
                        class_names=[],
                        frame_id=req['frame_id'],
                        camera_id=req.get('camera_id', 'unknown'),
                        timestamp=time.time(),
                        inference_time_ms=10.0,
                        preprocessing_time_ms=2.0,
                        postprocessing_time_ms=1.0,
                        total_time_ms=13.0,
                        detection_count=0,
                        avg_confidence=0.0,
                        gpu_memory_used_mb=100.0,
                    )
                    req['future'].set_result(mock_result)

        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            # Set exceptions for all futures in the batch
            for req in batch:
                if not req['future'].done():
                    req['future'].set_exception(e)

    async def _adjust_adaptive_timeout(self) -> None:
        """Continuously adjust timeout based on system performance."""
        while self.running:
            await asyncio.sleep(1.0)  # Adjust every second

            if len(self.timeout_history) < 10:
                continue

            # Calculate recent performance metrics
            recent_times = list(self.timeout_history)[-10:]
            sum(recent_times) / len(recent_times)

            # Adjust timeout based on system load and performance
            if len(self.latency_history) > 50:
                recent_latencies = list(self.latency_history)[-50:]
                p95_latency = np.percentile(recent_latencies, 95)

                # If P95 latency is high, reduce timeout to create smaller batches
                if p95_latency > 0.08:  # 80ms
                    self.adaptive_timeout = max(0.002, self.adaptive_timeout * 0.9)
                elif p95_latency < 0.03:  # 30ms
                    self.adaptive_timeout = min(0.02, self.adaptive_timeout * 1.1)

            # Ensure timeout stays within reasonable bounds
            min_timeout = 0.001  # 1ms
            max_timeout = self.config.batch_timeout_ms / 1000.0
            self.adaptive_timeout = max(min_timeout, min(max_timeout, self.adaptive_timeout))

    def get_performance_stats(self) -> dict[str, Any]:
        """Get detailed batching performance statistics."""
        return {
            "total_requests": self.total_requests,
            "batched_requests": self.batched_requests,
            "urgent_requests": self.urgent_requests,
            "avg_batch_size": self.avg_batch_size,
            "adaptive_timeout_ms": self.adaptive_timeout * 1000,
            "queue_sizes": {
                "urgent": self.urgent_queue.qsize(),
                "normal": self.normal_queue.qsize(),
                "batch": self.batch_queue.qsize(),
            },
            "recent_latency_p95_ms": np.percentile(list(self.latency_history)[-100:], 95) * 1000 if self.latency_history else 0,
            "batch_intervals_avg_ms": np.mean(list(self.batch_intervals)) * 1000 if self.batch_intervals else 0,
        }


# Legacy alias
DynamicBatcher = AdvancedDynamicBatcher


class OptimizedInferenceEngine:
    """High-performance YOLO11 inference engine optimized for traffic monitoring."""

    def __init__(self, config: InferenceConfig):
        self.config = config
        self.models: dict[int, Any] = {}
        self.tensorrt_engines: dict[int, Any] = {}  # ProductionTensorRTEngine when available
        self.current_device: int = 0

        # Enhanced performance components using new optimization modules
        if config.device_ids:
            # Use enhanced memory manager with tensor pools if available
            if ENHANCED_MEMORY_AVAILABLE:
                tensor_pool_config = TensorPoolConfig(
                    max_tensors_per_shape=16,
                    max_total_tensors=500,
                    max_memory_fraction=config.memory_fraction,
                    enable_pinned_memory=True,
                    enable_memory_profiling=True
                )
                self.memory_manager = MultiGPUMemoryManager(config.device_ids, tensor_pool_config)
            else:
                # Fallback to legacy memory manager
                self.memory_manager = GPUMemoryManager(
                    config.device_ids, config.memory_fraction
                )

            # Keep legacy stream manager for compatibility
            self.stream_manager = CUDAStreamManager(config.device_ids)
        else:
            self.memory_manager = None
            self.stream_manager = None

        # Use enhanced dynamic batcher if available
        if ENHANCED_BATCHING_AVAILABLE:
            batch_config = BatchConfiguration(
                min_batch_size=1,  # Default min batch size
                max_batch_size=config.max_batch_size,
                optimal_batch_size=config.batch_size,
                max_wait_time_ms=config.batch_timeout_ms,
                enable_adaptive_batching=True,
                latency_target_ms=75.0,  # Target sub-75ms latency
                throughput_target_fps=100.0
            )
            self.enhanced_batcher = NewAdvancedDynamicBatcher(batch_config)
        else:
            self.enhanced_batcher = None

        # Legacy batcher for compatibility
        self.batcher = AdvancedDynamicBatcher(config)

        # Enhanced TensorRT optimizer
        if config.backend == OptimizationBackend.TENSORRT:
            if ENHANCED_TRT_AVAILABLE:
                self.tensorrt_config = TensorRTConfig(
                    input_height=640,
                    input_width=640,
                    max_batch_size=config.max_batch_size,
                    use_fp16=True,
                    use_int8=False,  # Will be enabled with calibration data
                    workspace_size_gb=8,
                    enable_tactic_sources=True,
                    enable_sparse_weights=True,
                    enable_timing_cache=True
                )
                self.tensorrt_optimizer = ProductionTensorRTOptimizer(self.tensorrt_config)
            else:
                self.tensorrt_config = None
                self.tensorrt_optimizer = None

            # Legacy optimizer for compatibility
            self.optimizer = AdvancedTensorRTOptimizer(config)

        # Performance tracking
        self.inference_times: list[float] = []
        self.throughput_counter: int = 0
        self.last_throughput_time: float = time.time()

        self._setup_pytorch_optimizations()

    def _setup_pytorch_optimizations(self) -> None:
        """Configure PyTorch for optimal performance."""
        # Enable cuDNN benchmark for consistent input sizes
        if self.config.enable_cudnn_benchmark:
            torch.backends.cudnn.benchmark = True

        # Optimize for inference
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = True

        # Set thread count to avoid CPU oversubscription
        torch.set_num_threads(4)

    async def initialize(self, model_path: Path) -> None:
        """Initialize the inference engine with optimized models."""
        logger.info("Initializing optimized inference engine with enhanced components...")

        # Load models on each GPU
        for device_id in self.config.device_ids:
            await self._load_model_on_device(model_path, device_id)

        # Start enhanced dynamic batcher if available
        if self.enhanced_batcher:
            await self.enhanced_batcher.start()

        # Start legacy batcher for compatibility
        await self.batcher.start()

        logger.info(f"Inference engine initialized with {len(self.models)} models and {len(self.tensorrt_engines)} TensorRT engines")

    async def _load_model_on_device(self, model_path: Path, device_id: int) -> None:
        """Load and optimize model on specific GPU device."""
        device = f"cuda:{device_id}"
        torch.cuda.set_device(device_id)

        if self.config.backend == OptimizationBackend.TENSORRT and TRT_AVAILABLE:
            # Use enhanced TensorRT optimization
            trt_path = model_path.with_suffix(".trt")

            if not trt_path.exists():
                # First convert to ONNX if needed
                onnx_path = model_path.with_suffix(".onnx")
                if not onnx_path.exists():
                    logger.info("Converting YOLO11 model to ONNX...")
                    model = YOLO(str(model_path))
                    model.export(format="onnx", imgsz=640, dynamic=True, opset=16, half=True)

                # Optimize with production TensorRT optimizer
                if self.tensorrt_optimizer:
                    logger.info(f"Optimizing model with enhanced TensorRT for device {device_id}...")
                    await self.tensorrt_optimizer.optimize_model(str(onnx_path), str(trt_path))

            # Load enhanced TensorRT engine if available
            if ENHANCED_TRT_AVAILABLE and self.tensorrt_config:
                engine = ProductionTensorRTEngine(self.tensorrt_config, str(trt_path))
                self.tensorrt_engines[device_id] = engine

            # Also keep legacy TensorRT model for compatibility
            model = self._load_tensorrt_model(trt_path, device)
        else:
            # Load PyTorch model with optimization
            model = YOLO(str(model_path))
            model.to(device)

            # JIT compile for better performance
            try:
                model.model = torch.jit.script(model.model)
            except Exception as e:
                logger.warning(f"JIT compilation failed: {e}, using regular model")

            # Set to evaluation mode
            model.model.eval()

        self.models[device_id] = model
        logger.info(f"Enhanced model loaded on {device} with TensorRT engine: {device_id in self.tensorrt_engines}")

    def _load_tensorrt_model(self, engine_path: Path, device: str) -> Any:
        """Load TensorRT engine."""
        if not TRT_AVAILABLE:
            raise RuntimeError("TensorRT not available")

        import tensorrt as trt

        # Load serialized engine
        with open(engine_path, "rb") as f:
            engine_data = f.read()

        # Create runtime and deserialize engine
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        engine = runtime.deserialize_cuda_engine(engine_data)

        if engine is None:
            raise RuntimeError(f"Failed to load TensorRT engine from {engine_path}")

        # Create execution context
        context = engine.create_execution_context()

        return TensorRTModel(engine, context, device)

    async def predict_single_enhanced(
        self, frame: np.ndarray, frame_id: str, camera_id: str = "unknown"
    ) -> DetectionResult:
        """Enhanced single frame prediction using optimized TensorRT engines."""
        start_time = time.time()

        # Select optimal device based on load
        device_id = self._select_optimal_device()
        if device_id is None:
            raise RuntimeError("No GPU devices available for inference")

        # Use enhanced TensorRT engine if available
        if device_id in self.tensorrt_engines:
            engine = self.tensorrt_engines[device_id]

            # Preprocessing with enhanced memory management
            preprocess_start = time.time()
            input_data = self._preprocess_frame_for_tensorrt(frame, device_id)
            preprocess_time = (time.time() - preprocess_start) * 1000

            # Enhanced TensorRT inference
            inference_start = time.time()
            predictions = await engine.infer_async(input_data)
            inference_time = (time.time() - inference_start) * 1000

            # Post-processing
            postprocess_start = time.time()
            result = self._postprocess_tensorrt_predictions(
                predictions,
                frame_id,
                camera_id,
                inference_time,
                preprocess_time,
                start_time,
            )
            postprocess_time = (time.time() - postprocess_start) * 1000

            result.postprocessing_time_ms = postprocess_time
            result.total_time_ms = (time.time() - start_time) * 1000

            # Record performance for the enhanced TensorRT engine
            self.inference_times.append(result.total_time_ms)
            if len(self.inference_times) > 1000:
                self.inference_times = self.inference_times[-1000:]

            return result
        else:
            # Fallback to legacy prediction
            return await self.predict_single(frame, frame_id, camera_id)

    async def predict_single(
        self, frame: np.ndarray, frame_id: str, camera_id: str = "unknown"
    ) -> DetectionResult:
        """Single frame prediction with optimizations."""
        start_time = time.time()

        # Select optimal device based on load
        device_id = self._select_optimal_device()
        if device_id is None:
            raise RuntimeError("No GPU devices available for inference")
        model = self.models[device_id]

        # Preprocessing
        preprocess_start = time.time()
        input_tensor = self._preprocess_frame(frame, device_id)
        preprocess_time = (time.time() - preprocess_start) * 1000

        # Inference
        inference_start = time.time()
        with (
            torch.cuda.device(device_id),
            torch.inference_mode(),
            autocast(
                enabled=self.config.precision == "fp16",
                device_type="cuda" if torch.cuda.is_available() else "cpu",
            ),
        ):
            predictions = model(input_tensor)

        # Synchronize to get accurate timing
        torch.cuda.synchronize(device_id)
        inference_time = (time.time() - inference_start) * 1000

        # Post-processing
        postprocess_start = time.time()
        result = self._postprocess_predictions(
            predictions,
            frame_id,
            camera_id,
            inference_time,
            preprocess_time,
            start_time,
        )
        postprocess_time = (time.time() - postprocess_start) * 1000

        # Update performance metrics
        result.postprocessing_time_ms = postprocess_time
        result.total_time_ms = (time.time() - start_time) * 1000

        self._update_performance_metrics(result.total_time_ms)

        return result

    async def predict_batch(
        self,
        frames: list[np.ndarray[Any, np.dtype[Any]]],
        frame_ids: list[str],
        camera_ids: list[str] | None = None,
    ) -> list[DetectionResult]:
        """Batch prediction for maximum throughput."""
        if not frames:
            return []

        if camera_ids is None:
            camera_ids = ["unknown"] * len(frames)

        start_time = time.time()
        batch_size = len(frames)

        # Select optimal device
        device_id = self._select_optimal_device()
        if device_id is None:
            raise RuntimeError("No GPU devices available for inference")
        model = self.models[device_id]

        # Batch preprocessing
        preprocess_start = time.time()
        batch_tensor = self._preprocess_batch(frames, device_id)
        preprocess_time = (time.time() - preprocess_start) * 1000

        # Batch inference
        inference_start = time.time()
        with (
            torch.cuda.device(device_id),
            torch.inference_mode(),
            autocast(
                enabled=self.config.precision == "fp16",
                device_type="cuda" if torch.cuda.is_available() else "cpu",
            ),
        ):
            batch_predictions = model(batch_tensor)

        torch.cuda.synchronize(device_id)
        inference_time = (time.time() - inference_start) * 1000

        # Post-process each result in batch
        results = []
        for i, (frame_id, camera_id) in enumerate(
            zip(frame_ids, camera_ids, strict=False)
        ):
            # Extract single prediction from batch
            single_pred = self._extract_single_prediction(batch_predictions, i)

            result = self._postprocess_predictions(
                single_pred,
                frame_id,
                camera_id,
                inference_time / batch_size,
                preprocess_time / batch_size,
                start_time,
            )

            results.append(result)

        # Update throughput metrics
        self.throughput_counter += batch_size

        return results

    def _select_optimal_device(self) -> int | None:
        """Select the optimal GPU device based on current load."""
        if not self.config.device_ids:
            return None

        if len(self.config.device_ids) == 1:
            return self.config.device_ids[0]

        # Intelligent load balancing based on GPU utilization
        if len(self.config.device_ids) > 1:
            try:
                device_loads = []
                for device_id in self.config.device_ids:
                    memory_used = torch.cuda.memory_allocated(device_id)
                    memory_total = torch.cuda.get_device_properties(device_id).total_memory
                    load = memory_used / memory_total
                    device_loads.append((load, device_id))

                # Select device with lowest load
                device_loads.sort(key=lambda x: x[0])
                return device_loads[0][1]

            except Exception:
                # Fallback to round-robin
                device_id = self.config.device_ids[self.current_device]
                self.current_device = (self.current_device + 1) % len(self.config.device_ids)
                return device_id

        return self.config.device_ids[0]

    def _preprocess_frame(
        self, frame: np.ndarray[Any, np.dtype[Any]], device_id: int
    ) -> torch.Tensor:
        """Optimized frame preprocessing."""
        # Resize with letterboxing to maintain aspect ratio
        h, w = frame.shape[:2]
        target_h, target_w = self.config.input_size

        scale = min(target_h / h, target_w / w)
        new_h, new_w = int(h * scale), int(w * scale)

        # Resize frame
        if scale != 1:
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Create padded canvas
        pad_h = (target_h - new_h) // 2
        pad_w = (target_w - new_w) // 2

        padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        padded[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = frame

        # Convert to tensor format (CHW, normalized)
        tensor = torch.from_numpy(padded).to(f"cuda:{device_id}")
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).contiguous()
        tensor = tensor.half() if self.config.precision == "fp16" else tensor.float()
        tensor /= 255.0

        return tensor

    def _preprocess_batch(
        self, frames: list[np.ndarray[Any, np.dtype[Any]]], device_id: int
    ) -> torch.Tensor:
        """Optimized batch preprocessing."""
        # Process each frame and stack into batch tensor
        batch_tensors = []

        for frame in frames:
            tensor = self._preprocess_frame(frame, device_id)
            batch_tensors.append(tensor)

        # Stack into batch
        batch_tensor = torch.cat(batch_tensors, dim=0)

        return batch_tensor

    def _preprocess_frame_for_tensorrt(
        self, frame: np.ndarray, device_id: int
    ) -> np.ndarray:
        """Enhanced preprocessing for TensorRT engines with memory optimization."""
        # Resize with letterboxing to maintain aspect ratio
        h, w = frame.shape[:2]
        target_h, target_w = 640, 640  # YOLO11 standard input size

        scale = min(target_h / h, target_w / w)
        new_h, new_w = int(h * scale), int(w * scale)

        # Resize frame
        if scale != 1:
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Create padded canvas
        pad_h = (target_h - new_h) // 2
        pad_w = (target_w - new_w) // 2

        padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        padded[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = frame

        # Convert to CHW format and normalize for TensorRT
        processed = padded.transpose(2, 0, 1).astype(np.float32) / 255.0

        # Add batch dimension for TensorRT engine
        processed = np.expand_dims(processed, axis=0)

        return processed

    def _postprocess_tensorrt_predictions(
        self,
        predictions: np.ndarray,
        frame_id: str,
        camera_id: str,
        inference_time: float,
        preprocess_time: float,
        start_time: float,
    ) -> DetectionResult:
        """Enhanced post-processing for TensorRT predictions."""
        # TensorRT output format: [batch_size, num_detections, 85]
        # Where 85 = [x, y, w, h, confidence, class_probabilities...]

        detections = []

        if predictions.size > 0:
            # Remove batch dimension
            batch_predictions = predictions[0] if len(predictions.shape) == 3 else predictions

            # Apply confidence threshold
            confidence_mask = batch_predictions[:, 4] > self.config.conf_threshold
            filtered_predictions = batch_predictions[confidence_mask]

            if len(filtered_predictions) > 0:
                # Extract bounding boxes and scores
                boxes = filtered_predictions[:, :4]  # x, y, w, h
                scores = filtered_predictions[:, 4]  # confidence
                class_scores = filtered_predictions[:, 5:]  # class probabilities

                # Get class IDs
                class_ids = np.argmax(class_scores, axis=1)

                # Convert to Detection objects
                for i, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids, strict=False)):
                    x, y, w, h = box

                    # Convert from center format to corner format
                    x1 = float(x - w / 2)
                    y1 = float(y - h / 2)
                    x2 = float(x + w / 2)
                    y2 = float(y + h / 2)

                    detection = Detection(
                        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                        confidence=float(score),
                        class_id=int(class_id),
                        class_name=self._get_class_name(int(class_id)),
                        track_id=None,  # Will be assigned by tracker
                    )
                    detections.append(detection)

        return DetectionResult(
            frame_id=frame_id,
            camera_id=camera_id,
            detections=detections,
            inference_time_ms=inference_time,
            preprocessing_time_ms=preprocess_time,
            postprocessing_time_ms=0.0,  # Will be set by caller
            total_time_ms=0.0,  # Will be set by caller
            timestamp=start_time,
            model_info={
                "backend": "enhanced_tensorrt",
                "precision": "fp16" if self.tensorrt_config.use_fp16 else "fp32",
                "device": f"cuda:{self._select_optimal_device()}",
            },
        )

    def _get_class_name(self, class_id: int) -> str:
        """Get class name for given class ID."""
        # YOLO11 COCO class names for traffic monitoring
        class_names = [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
            "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
            "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
            "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
            "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
            "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
            "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
            "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
            "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
            "toothbrush"
        ]

        if 0 <= class_id < len(class_names):
            return class_names[class_id]
        return f"unknown_{class_id}"

    def _extract_single_prediction(
        self, batch_predictions: Any, index: int
    ) -> torch.Tensor | None:
        """Extract single prediction from batch results."""
        if batch_predictions is None or len(batch_predictions) == 0:
            return None

        # Handle different output formats from YOLO11
        if isinstance(batch_predictions, list | tuple):
            # Multiple outputs, take the first (main detection output)
            batch_predictions = batch_predictions[0]

        if isinstance(batch_predictions, torch.Tensor):
            if len(batch_predictions.shape) == 3:
                # Shape: [batch_size, num_detections, features]
                if index < batch_predictions.shape[0]:
                    return batch_predictions[index]
                else:
                    return torch.zeros(
                        (0, batch_predictions.shape[-1]),
                        device=batch_predictions.device,
                    )
            elif len(batch_predictions.shape) == 2:
                # Already single prediction
                return batch_predictions

        # If we can't extract, return empty tensor
        return torch.zeros(
            (0, 85), device="cuda" if torch.cuda.is_available() else "cpu"
        )

    def _postprocess_predictions(
        self,
        predictions: Any,
        frame_id: str,
        camera_id: str,
        inference_time: float,
        preprocess_time: float,
        _start_time: float,
    ) -> DetectionResult:
        """Convert model predictions to structured result."""
        # Apply NMS and confidence filtering
        filtered_predictions = self._apply_nms(predictions)

        # Extract detection components
        boxes, scores, classes = self._extract_detections(filtered_predictions)

        # Vehicle class names mapping (COCO classes)
        coco_to_vehicle = {
            1: "bicycle",
            2: "car",
            3: "motorcycle",
            5: "bus",
            7: "truck",
        }

        class_names = []
        if len(classes) > 0:
            class_names = [coco_to_vehicle.get(int(cls), f"class_{cls}") for cls in classes]

        # Calculate metrics
        detection_count = len(boxes)
        avg_confidence = float(np.mean(scores)) if len(scores) > 0 else 0.0
        gpu_memory = (
            torch.cuda.memory_allocated(self.current_device) / 1024 / 1024
            if torch.cuda.is_available() and self.current_device >= 0
            else 0.0
        )

        return DetectionResult(
            boxes=boxes,
            scores=scores,
            classes=classes,
            class_names=class_names,
            frame_id=frame_id,
            camera_id=camera_id,
            timestamp=time.time(),
            inference_time_ms=inference_time,
            preprocessing_time_ms=preprocess_time,
            postprocessing_time_ms=0.0,  # Updated later
            total_time_ms=0.0,  # Updated later
            detection_count=detection_count,
            avg_confidence=avg_confidence,
            gpu_memory_used_mb=gpu_memory,
        )

    def _apply_nms(self, predictions: Any) -> torch.Tensor:
        """Apply Non-Maximum Suppression to filter overlapping detections."""
        if predictions is None or len(predictions) == 0:
            return predictions

        # Extract detection components from YOLO11 predictions
        # YOLO11 output format: [batch_size, num_detections, 85]
        # where 85 = 4 (bbox) + 1 (conf) + 80 (classes)

        if isinstance(predictions, list | tuple):
            predictions = predictions[0]  # Take first output

        if len(predictions.shape) == 3:
            predictions = predictions[0]  # Remove batch dimension for single inference

        # Filter by confidence threshold
        conf_mask = predictions[:, 4] >= self.config.conf_threshold
        filtered_preds = predictions[conf_mask]

        if len(filtered_preds) == 0:
            return torch.zeros((0, 85), device=predictions.device)

        # Extract boxes and scores
        boxes = filtered_preds[:, :4]  # [x_center, y_center, width, height]
        scores = filtered_preds[:, 4]
        class_probs = filtered_preds[:, 5:]

        # Get class predictions
        class_scores, class_indices = torch.max(class_probs, dim=1)
        final_scores = scores * class_scores

        # Convert from center format to corner format for NMS
        x_center, y_center, width, height = boxes.unbind(1)
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)

        # Apply NMS per class
        keep_indices = []
        unique_classes = torch.unique(class_indices)

        for cls in unique_classes:
            cls_mask = class_indices == cls
            cls_boxes = boxes_xyxy[cls_mask]
            cls_scores = final_scores[cls_mask]
            cls_indices = torch.where(cls_mask)[0]

            # Apply NMS for this class
            from torchvision.ops import nms

            keep = nms(cls_boxes, cls_scores, self.config.iou_threshold)
            keep_indices.append(cls_indices[keep])

        if keep_indices:
            final_keep = torch.cat(keep_indices)
            final_keep = final_keep[: self.config.max_detections]
            return filtered_preds[final_keep]
        else:
            return torch.zeros((0, 85), device=predictions.device)

    def _extract_detections(
        self, predictions: Any
    ) -> tuple[
        np.ndarray[Any, np.dtype[np.float32]],
        np.ndarray[Any, np.dtype[np.float32]],
        np.ndarray[Any, np.dtype[np.int32]],
    ]:
        """Extract boxes, scores, and classes from predictions."""
        if predictions is None or len(predictions) == 0:
            return np.array([]).reshape(0, 4), np.array([]), np.array([])

        # Convert to CPU numpy if on GPU
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()

        if len(predictions) == 0:
            return np.array([]).reshape(0, 4), np.array([]), np.array([])

        # Extract components
        boxes_xywh = predictions[:, :4]  # [x_center, y_center, width, height]
        confidences = predictions[:, 4]
        class_probs = predictions[:, 5:]

        # Get final class predictions
        class_scores = np.max(class_probs, axis=1)
        class_indices = np.argmax(class_probs, axis=1)
        final_scores = confidences * class_scores

        # Convert boxes from center format to corner format
        x_center, y_center, width, height = boxes_xywh.T
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

        # Filter traffic-relevant classes (vehicles)
        # COCO class mapping for vehicles: 2=car, 3=motorcycle, 5=bus, 7=truck
        vehicle_classes = [2, 3, 5, 7, 1]  # Include bicycle (1)
        vehicle_mask = np.isin(class_indices, vehicle_classes)

        if not np.any(vehicle_mask):
            return np.array([]).reshape(0, 4), np.array([]), np.array([])

        # Apply vehicle filter
        filtered_boxes = boxes_xyxy[vehicle_mask]
        filtered_scores = final_scores[vehicle_mask]
        filtered_classes = class_indices[vehicle_mask]

        # Sort by confidence (highest first)
        sort_indices = np.argsort(filtered_scores)[::-1]

        return (
            filtered_boxes[sort_indices],
            filtered_scores[sort_indices],
            filtered_classes[sort_indices],
        )

    def _update_performance_metrics(self, total_time_ms: float) -> None:
        """Update running performance metrics."""
        self.inference_times.append(total_time_ms)

        # Keep only last 1000 measurements
        if len(self.inference_times) > 1000:
            self.inference_times = self.inference_times[-1000:]

    def get_performance_stats(self) -> dict[str, float | dict[str, float]]:
        """Get current performance statistics."""
        if not self.inference_times:
            return {}

        current_time = time.time()
        throughput = self.throughput_counter / max(
            1, current_time - self.last_throughput_time
        )

        return {
            "avg_latency_ms": float(np.mean(self.inference_times)),
            "p50_latency_ms": float(np.percentile(self.inference_times, 50)),
            "p95_latency_ms": float(np.percentile(self.inference_times, 95)),
            "p99_latency_ms": float(np.percentile(self.inference_times, 99)),
            "throughput_fps": throughput,
            "gpu_utilization": self._get_gpu_utilization(),
            "gpu_memory_used": self._get_gpu_memory_usage(),
        }

    def _get_gpu_utilization(self) -> float:
        """Get average GPU utilization across all devices."""
        if not self.config.device_ids:
            return 0.0

        try:
            import pynvml

            pynvml.nvmlInit()

            total_util = 0
            for device_id in self.config.device_ids:
                handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                total_util += util

            return total_util / len(self.config.device_ids)
        except Exception:
            return 0.0

    def _get_gpu_memory_usage(self) -> dict[str, float]:
        """Get GPU memory usage for all devices."""
        memory_stats = {}

        if not self.config.device_ids:
            return memory_stats

        for device_id in self.config.device_ids:
            try:
                allocated = torch.cuda.memory_allocated(device_id) / 1024 / 1024
                cached = torch.cuda.memory_reserved(device_id) / 1024 / 1024

                memory_stats[f"gpu_{device_id}_allocated_mb"] = float(allocated)
                memory_stats[f"gpu_{device_id}_cached_mb"] = float(cached)
            except Exception:
                memory_stats[f"gpu_{device_id}_allocated_mb"] = 0.0
                memory_stats[f"gpu_{device_id}_cached_mb"] = 0.0

        return memory_stats

    async def cleanup(self) -> None:
        """Clean up resources."""
        # Stop both enhanced and legacy batchers
        if self.enhanced_batcher:
            await self.enhanced_batcher.stop()
        await self.batcher.stop()

        # Clean up enhanced TensorRT engines
        for engine in self.tensorrt_engines.values():
            engine.cleanup()

        # Clean up memory and stream managers
        if self.memory_manager:
            self.memory_manager.cleanup()
        if self.stream_manager:
            self.stream_manager.cleanup()

    async def get_enhanced_performance_stats(self) -> dict[str, Any]:
        """Get comprehensive performance statistics for enhanced components."""
        stats = {
            'enhanced_tensorrt_engines': len(self.tensorrt_engines),
            'total_models': len(self.models),
            'enhanced_batcher_stats': self.enhanced_batcher.get_stats() if self.enhanced_batcher else None,
        }

        # Get TensorRT performance stats
        tensorrt_stats = {}
        for device_id, engine in self.tensorrt_engines.items():
            device_stats = engine.get_performance_stats()
            tensorrt_stats[f'device_{device_id}'] = device_stats

        stats['tensorrt_performance'] = tensorrt_stats

        # Get memory manager stats
        if ENHANCED_MEMORY_AVAILABLE and isinstance(self.memory_manager, MultiGPUMemoryManager):
            stats['memory_manager_stats'] = self.memory_manager.get_overall_stats()

        # Overall performance summary
        if self.inference_times:
            stats['overall_performance'] = {
                'avg_latency_ms': float(np.mean(self.inference_times)),
                'min_latency_ms': float(np.min(self.inference_times)),
                'max_latency_ms': float(np.max(self.inference_times)),
                'p95_latency_ms': float(np.percentile(self.inference_times, 95)),
                'p99_latency_ms': float(np.percentile(self.inference_times, 99)),
                'total_inferences': len(self.inference_times),
                'target_achieved': float(np.mean(self.inference_times)) < 75.0
            }

        return stats

    async def benchmark_enhanced_vs_legacy(self, test_frames: list[np.ndarray], iterations: int = 100) -> dict[str, Any]:
        """Benchmark enhanced vs legacy inference performance."""
        logger.info(f"Benchmarking enhanced vs legacy inference over {iterations} iterations...")

        enhanced_times = []
        legacy_times = []

        # Warmup
        for i in range(10):
            frame = test_frames[i % len(test_frames)]
            if self.tensorrt_engines:
                await self.predict_single_enhanced(frame, f"warmup_{i}")
            await self.predict_single(frame, f"warmup_{i}")

        # Benchmark enhanced inference
        for i in range(iterations):
            frame = test_frames[i % len(test_frames)]
            start_time = time.time()

            if self.tensorrt_engines:
                await self.predict_single_enhanced(frame, f"enhanced_{i}")
                enhanced_times.append((time.time() - start_time) * 1000)

        # Benchmark legacy inference
        for i in range(iterations):
            frame = test_frames[i % len(test_frames)]
            start_time = time.time()

            await self.predict_single(frame, f"legacy_{i}")
            legacy_times.append((time.time() - start_time) * 1000)

        # Calculate improvements
        enhanced_avg = np.mean(enhanced_times) if enhanced_times else float('inf')
        legacy_avg = np.mean(legacy_times)
        improvement_percent = ((legacy_avg - enhanced_avg) / legacy_avg * 100) if enhanced_times else 0

        results = {
            'enhanced_performance': {
                'avg_latency_ms': enhanced_avg,
                'min_latency_ms': float(np.min(enhanced_times)) if enhanced_times else 0,
                'max_latency_ms': float(np.max(enhanced_times)) if enhanced_times else 0,
                'p95_latency_ms': float(np.percentile(enhanced_times, 95)) if enhanced_times else 0,
                'iterations': len(enhanced_times)
            },
            'legacy_performance': {
                'avg_latency_ms': legacy_avg,
                'min_latency_ms': float(np.min(legacy_times)),
                'max_latency_ms': float(np.max(legacy_times)),
                'p95_latency_ms': float(np.percentile(legacy_times, 95)),
                'iterations': len(legacy_times)
            },
            'improvement_metrics': {
                'latency_improvement_percent': improvement_percent,
                'target_75ms_achieved': enhanced_avg < 75.0 if enhanced_times else False,
                'speedup_factor': legacy_avg / enhanced_avg if enhanced_times and enhanced_avg > 0 else 1.0
            }
        }

        logger.info(f"Benchmark complete: {improvement_percent:.1f}% improvement, "
                   f"enhanced avg: {enhanced_avg:.2f}ms, legacy avg: {legacy_avg:.2f}ms")

        return results


# Model Selection Utility Functions


def select_optimal_model_for_deployment(
    target_latency_ms: int,
    target_accuracy: float,
    available_memory_gb: float,
    device_type: str = "gpu",
) -> tuple[ModelType, InferenceConfig]:
    """
    Select optimal YOLO11 model configuration based on deployment requirements.

    Performance Guidelines:
    - NANO: 2-5ms inference, 85-88% mAP, 2.6M params, 5MB memory
    - SMALL: 5-12ms inference, 89-92% mAP, 9.4M params, 18MB memory
    - MEDIUM: 12-25ms inference, 92-94% mAP, 20.1M params, 40MB memory
    - LARGE: 25-40ms inference, 94-95% mAP, 25.3M params, 50MB memory
    - XLARGE: 40-60ms inference, 95-96% mAP, 56.9M params, 114MB memory
    """

    if target_latency_ms <= 10:
        if target_accuracy <= 0.88:
            model_type = ModelType.NANO
            batch_size = 16 if available_memory_gb >= 4 else 8
        else:
            model_type = ModelType.SMALL
            batch_size = 12 if available_memory_gb >= 6 else 6
    elif target_latency_ms <= 30:
        if target_accuracy <= 0.92:
            model_type = ModelType.SMALL
            batch_size = 8 if available_memory_gb >= 4 else 4
        else:
            model_type = ModelType.MEDIUM
            batch_size = 6 if available_memory_gb >= 8 else 3
    else:
        if target_accuracy <= 0.94:
            model_type = ModelType.MEDIUM
            batch_size = 4 if available_memory_gb >= 6 else 2
        else:
            model_type = ModelType.LARGE
            batch_size = 3 if available_memory_gb >= 10 else 1

    # Determine optimization backend
    backend = (
        OptimizationBackend.TENSORRT
        if device_type == "gpu"
        else OptimizationBackend.ONNX
    )
    precision = "fp16" if device_type == "gpu" else "fp32"

    config = InferenceConfig(
        model_type=model_type,
        backend=backend,
        precision=precision,
        batch_size=batch_size,
        max_batch_size=batch_size * 2,
        batch_timeout_ms=max(5, target_latency_ms // 4),
        conf_threshold=0.25,
        iou_threshold=0.45,
    )

    return model_type, config


def estimate_throughput(
    model_type: ModelType, batch_size: int, gpu_type: str = "T4"
) -> dict[str, float]:
    """
    Estimate throughput for different model and hardware combinations.

    Benchmarks based on NVIDIA T4, V100, and A10G GPUs.
    """

    # Base inference times (ms) for batch size 1
    base_times: dict[ModelType, dict[str, float]] = {
        ModelType.NANO: {"T4": 2.5, "V100": 1.8, "A10G": 1.5},
        ModelType.SMALL: {"T4": 6.0, "V100": 4.2, "A10G": 3.5},
        ModelType.MEDIUM: {"T4": 14.0, "V100": 9.8, "A10G": 8.2},
        ModelType.LARGE: {"T4": 22.0, "V100": 15.4, "A10G": 12.8},
        ModelType.XLARGE: {"T4": 42.0, "V100": 29.4, "A10G": 24.5},
    }

    base_time = base_times[model_type].get(gpu_type, base_times[model_type]["T4"])

    # Batch scaling factor (non-linear due to GPU parallelization)
    batch_scaling: dict[int, float] = {
        1: 1.0,
        2: 1.6,
        4: 2.8,
        8: 4.5,
        16: 7.2,
        32: 12.0,
    }

    scaling_factor = batch_scaling.get(batch_size, batch_size * 0.75)
    batch_time = base_time * scaling_factor

    return {
        "inference_time_ms": batch_time,
        "throughput_fps": (1000 * batch_size) / batch_time,
        "latency_ms": batch_time / batch_size,
        "gpu_utilization_pct": min(95, batch_size * 8),
        "memory_usage_mb": batch_size * base_times[model_type]["T4"] * 2,
    }
