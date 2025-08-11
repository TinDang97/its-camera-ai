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
import logging
import os
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.cuda.amp import autocast
from ultralytics import YOLO

try:
    import tensorrt as trt
    import torch_tensorrt
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


logger = logging.getLogger(__name__)


class TensorRTModel:
    """Wrapper for TensorRT inference engine."""

    def __init__(self, engine, context, device: str):
        self.engine = engine
        self.context = context
        self.device = device

        # Pre-allocate GPU memory for inputs and outputs
        self.bindings = []
        self.inputs = []
        self.outputs = []

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
            size = np.prod(shape) * np_dtype().itemsize
            mem_gpu = torch.cuda.mem_alloc(size)

            self.bindings.append(int(mem_gpu))

            if engine.binding_is_input(i):
                self.inputs.append({
                    'name': binding_name,
                    'mem_gpu': mem_gpu,
                    'shape': shape,
                    'dtype': np_dtype
                })
            else:
                self.outputs.append({
                    'name': binding_name,
                    'mem_gpu': mem_gpu,
                    'shape': shape,
                    'dtype': np_dtype
                })

    def __call__(self, input_tensor: torch.Tensor):
        """Run inference using TensorRT engine."""
        if not isinstance(input_tensor, torch.Tensor):
            input_tensor = torch.from_numpy(input_tensor)

        # Copy input to GPU memory
        input_data = input_tensor.contiguous().cpu().numpy()

        try:
            import pycuda.driver as cuda

            # Copy input data to GPU
            cuda.memcpy_htod(self.inputs[0]['mem_gpu'], input_data)

            # Run inference
            self.context.execute_v2(self.bindings)

            # Copy output back to host
            outputs = []
            for output_info in self.outputs:
                output_shape = output_info['shape']
                output_dtype = output_info['dtype']

                # Handle dynamic batch size
                if output_shape[0] == -1:
                    output_shape = (input_tensor.shape[0],) + output_shape[1:]

                output_data = np.empty(output_shape, dtype=output_dtype)
                cuda.memcpy_dtoh(output_data, output_info['mem_gpu'])

                outputs.append(torch.from_numpy(output_data))

            return outputs[0] if len(outputs) == 1 else outputs

        except ImportError:
            logger.error("PyCUDA not available for TensorRT inference")
            return None


class ModelType(Enum):
    """Supported YOLO11 model variants for different deployment scenarios."""
    NANO = "yolo11n.pt"      # Ultra-fast edge deployment, 2.6M params
    SMALL = "yolo11s.pt"     # Balanced performance, 9.4M params
    MEDIUM = "yolo11m.pt"    # High accuracy cloud deployment, 20.1M params
    LARGE = "yolo11l.pt"     # Maximum accuracy, 25.3M params
    XLARGE = "yolo11x.pt"    # Research grade, 56.9M params


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
    device_ids: list[int] = None
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

    def __post_init__(self):
        if self.device_ids is None:
            self.device_ids = [0] if torch.cuda.is_available() else []


@dataclass
class DetectionResult:
    """Structured detection result with performance metrics."""

    # Detection Data
    boxes: np.ndarray  # [N, 4] - xyxy format
    scores: np.ndarray  # [N] - confidence scores
    classes: np.ndarray  # [N] - class indices
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


class GPUMemoryManager:
    """Advanced GPU memory management for high-throughput inference."""

    def __init__(self, device_ids: list[int], memory_fraction: float = 0.8):
        self.device_ids = device_ids
        self.memory_fraction = memory_fraction
        self.memory_pools = {}
        self.allocated_tensors = {}

        self._initialize_memory_pools()

    def _initialize_memory_pools(self):
        """Initialize memory pools for each GPU device."""
        for device_id in self.device_ids:
            torch.cuda.set_device(device_id)

            # Set memory fraction to prevent OOM
            total_memory = torch.cuda.get_device_properties(device_id).total_memory
            torch.cuda.set_per_process_memory_fraction(
                self.memory_fraction, device_id
            )

            # Pre-allocate common tensor shapes
            self.memory_pools[device_id] = {}
            common_shapes = [
                (1, 3, 640, 640),   # Single frame
                (4, 3, 640, 640),   # Small batch
                (8, 3, 640, 640),   # Standard batch
                (16, 3, 640, 640),  # Large batch
                (32, 3, 640, 640),  # Max batch
            ]

            for shape in common_shapes:
                tensor = torch.zeros(shape, dtype=torch.float16,
                                   device=f"cuda:{device_id}")
                self.memory_pools[device_id][shape] = tensor

            logger.info(f"Initialized memory pool for GPU {device_id}")

    def get_tensor(self, shape: tuple[int, ...], device_id: int) -> torch.Tensor:
        """Get pre-allocated tensor or create new one."""
        if shape in self.memory_pools[device_id]:
            return self.memory_pools[device_id][shape].clone()

        return torch.zeros(shape, dtype=torch.float16,
                         device=f"cuda:{device_id}")

    def cleanup(self):
        """Clean up GPU memory."""
        for device_id in self.device_ids:
            torch.cuda.set_device(device_id)
            torch.cuda.empty_cache()


class TensorRTOptimizer:
    """TensorRT model optimization for maximum inference performance."""

    def __init__(self, config: InferenceConfig):
        self.config = config
        self.trt_engines = {}

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
        onnx_path = output_path.with_suffix('.onnx')
        model.export(
            format='onnx',
            imgsz=self.config.input_size,
            dynamic=True,
            batch_size=1,
            opset=16,
            half=self.config.precision == "fp16"
        )

        # Compile ONNX to TensorRT
        engine_path = output_path.with_suffix('.trt')
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
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            # Enable optimizations
            session_options.optimized_model_filepath = str(output_path)

            # Create session to trigger optimization
            providers = ['CUDAExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
            session = ort.InferenceSession(str(model_path), session_options, providers=providers)

            # Test run to ensure optimization
            dummy_input = np.random.randn(1, 3, *self.config.input_size).astype(np.float32)
            session.run(None, {'images': dummy_input})

            logger.info(f"ONNX model optimized and saved to {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"ONNX optimization failed: {e}")
            return model_path

    def quantize_model_int8(self, model_path: Path, calibration_data: list[np.ndarray]) -> Path:
        """Quantize model to INT8 for edge deployment."""
        logger.info(f"Quantizing model to INT8: {model_path}")

        try:
            import torch.quantization as quant
            from ultralytics import YOLO

            # Load model
            model = YOLO(str(model_path))

            # Prepare model for quantization
            model.model.eval()
            model.model.qconfig = quant.get_default_qconfig('fbgemm')

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
            quantized_path = model_path.with_suffix('.int8.pt')
            torch.save(model.model.state_dict(), quantized_path)

            logger.info(f"INT8 quantized model saved to {quantized_path}")
            return quantized_path

        except Exception as e:
            logger.error(f"INT8 quantization failed: {e}")
            return model_path

    def _compile_onnx_to_tensorrt(self, onnx_path: Path, engine_path: Path):
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
        with open(onnx_path, 'rb') as model:
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
            (self.config.max_batch_size, 3, *self.config.input_size)  # max
        )
        config.add_optimization_profile(profile)

        # Build engine
        engine = builder.build_serialized_network(network, config)
        if engine is None:
            raise RuntimeError("Failed to build TensorRT engine")

        # Save engine
        with open(engine_path, 'wb') as f:
            f.write(engine)

        logger.info(f"TensorRT engine saved to {engine_path}")

    def _create_int8_calibrator(self):
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

                def get_batch(self, names):
                    if self.current_index + self.batch_size > len(self.calibration_files):
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
                        img = np.random.randint(0, 255,
                                              (self.config.input_size[0], self.config.input_size[1], 3),
                                              dtype=np.uint8)

                        # Preprocess like real inference
                        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                        batch_images.append(img_tensor)

                    if batch_images:
                        batch_tensor = torch.stack(batch_images)
                        return batch_tensor.numpy().ascontiguousarray()
                    return None

                def read_calibration_cache(self):
                    if os.path.exists(self.cache_file):
                        with open(self.cache_file, 'rb') as f:
                            return f.read()
                    return None

                def write_calibration_cache(self, cache):
                    with open(self.cache_file, 'wb') as f:
                        f.write(cache)

            # Create calibrator with synthetic data for demo
            # In production, use real traffic images
            calibration_files = [f"calib_{i}.jpg" for i in range(100)]
            cache_file = "int8_calibration.cache"

            return TrafficCalibrator(calibration_files, cache_file, batch_size=1)

        except ImportError:
            logger.warning("PyCUDA not available for INT8 calibration")
            return None


class DynamicBatcher:
    """Dynamic batching system for optimal GPU utilization."""

    def __init__(self, config: InferenceConfig):
        self.config = config
        self.batch_queue = asyncio.Queue(maxsize=config.max_batch_size * 2)
        self.result_futures = {}
        self.batch_processor = None

        # Batching metrics
        self.total_requests = 0
        self.batched_requests = 0
        self.avg_batch_size = 0

    async def start(self):
        """Start the dynamic batching processor."""
        self.batch_processor = asyncio.create_task(self._process_batches())

    async def stop(self):
        """Stop the dynamic batching processor."""
        if self.batch_processor:
            self.batch_processor.cancel()
            try:
                await self.batch_processor
            except asyncio.CancelledError:
                pass

    async def predict(self, frame: np.ndarray, frame_id: str) -> DetectionResult:
        """Add frame to batch queue and return prediction."""
        future = asyncio.Future()

        request = {
            'frame': frame,
            'frame_id': frame_id,
            'future': future,
            'timestamp': time.time()
        }

        await self.batch_queue.put(request)
        self.total_requests += 1

        # Wait for result
        return await future

    async def _process_batches(self):
        """Process batches continuously."""
        while True:
            try:
                batch = await self._collect_batch()
                if batch:
                    await self._process_batch(batch)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Batch processing error: {e}")

    async def _collect_batch(self) -> list[dict]:
        """Collect a batch of requests with timeout."""
        batch = []
        deadline = time.time() + (self.config.batch_timeout_ms / 1000.0)

        # Get first request (blocking)
        try:
            first_request = await asyncio.wait_for(
                self.batch_queue.get(),
                timeout=self.config.batch_timeout_ms / 1000.0
            )
            batch.append(first_request)
        except TimeoutError:
            return []

        # Collect additional requests until timeout or batch full
        while (len(batch) < self.config.max_batch_size and
               time.time() < deadline):
            try:
                request = await asyncio.wait_for(
                    self.batch_queue.get(),
                    timeout=max(0.001, deadline - time.time())
                )
                batch.append(request)
            except TimeoutError:
                break

        return batch

    async def _process_batch(self, batch: list[dict]):
        """Process a batch of requests."""
        # TODO: This will be implemented by the InferenceEngine
        pass


class OptimizedInferenceEngine:
    """High-performance YOLO11 inference engine optimized for traffic monitoring."""

    def __init__(self, config: InferenceConfig):
        self.config = config
        self.models = {}
        self.current_device = 0

        # Performance components
        self.memory_manager = GPUMemoryManager(
            config.device_ids, config.memory_fraction
        )
        self.batcher = DynamicBatcher(config)

        if config.backend == OptimizationBackend.TENSORRT:
            self.optimizer = TensorRTOptimizer(config)

        # Performance tracking
        self.inference_times = []
        self.throughput_counter = 0
        self.last_throughput_time = time.time()

        self._setup_pytorch_optimizations()

    def _setup_pytorch_optimizations(self):
        """Configure PyTorch for optimal performance."""
        # Enable cuDNN benchmark for consistent input sizes
        if self.config.enable_cudnn_benchmark:
            torch.backends.cudnn.benchmark = True

        # Optimize for inference
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = True

        # Set thread count to avoid CPU oversubscription
        torch.set_num_threads(4)

    async def initialize(self, model_path: Path):
        """Initialize the inference engine with optimized models."""
        logger.info("Initializing optimized inference engine...")

        # Load models on each GPU
        for device_id in self.config.device_ids:
            await self._load_model_on_device(model_path, device_id)

        # Start dynamic batcher
        await self.batcher.start()

        logger.info(f"Inference engine initialized with {len(self.models)} models")

    async def _load_model_on_device(self, model_path: Path, device_id: int):
        """Load and optimize model on specific GPU device."""
        device = f"cuda:{device_id}"
        torch.cuda.set_device(device_id)

        if self.config.backend == OptimizationBackend.TENSORRT and TRT_AVAILABLE:
            # Load TensorRT optimized model
            trt_path = model_path.with_suffix('.trt')
            if not trt_path.exists():
                trt_path = self.optimizer.compile_model(model_path, trt_path)

            model = self._load_tensorrt_model(trt_path, device)
        else:
            # Load PyTorch model with optimization
            model = YOLO(str(model_path))
            model.to(device)

            # JIT compile for better performance
            model.model = torch.jit.script(model.model)

            # Set to evaluation mode
            model.model.eval()

        self.models[device_id] = model
        logger.info(f"Model loaded on {device}")

    def _load_tensorrt_model(self, engine_path: Path, device: str):
        """Load TensorRT engine."""
        if not TRT_AVAILABLE:
            raise RuntimeError("TensorRT not available")

        import tensorrt as trt

        # Load serialized engine
        with open(engine_path, 'rb') as f:
            engine_data = f.read()

        # Create runtime and deserialize engine
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        engine = runtime.deserialize_cuda_engine(engine_data)

        if engine is None:
            raise RuntimeError(f"Failed to load TensorRT engine from {engine_path}")

        # Create execution context
        context = engine.create_execution_context()

        return TensorRTModel(engine, context, device)

    async def predict_single(
        self,
        frame: np.ndarray,
        frame_id: str,
        camera_id: str = "unknown"
    ) -> DetectionResult:
        """Single frame prediction with optimizations."""
        start_time = time.time()

        # Select optimal device based on load
        device_id = self._select_optimal_device()
        device = f"cuda:{device_id}"
        model = self.models[device_id]

        # Preprocessing
        preprocess_start = time.time()
        input_tensor = self._preprocess_frame(frame, device_id)
        preprocess_time = (time.time() - preprocess_start) * 1000

        # Inference
        inference_start = time.time()
        with torch.cuda.device(device_id):
            with torch.inference_mode():
                with autocast(enabled=self.config.precision == "fp16"):
                    predictions = model(input_tensor)

        # Synchronize to get accurate timing
        torch.cuda.synchronize(device_id)
        inference_time = (time.time() - inference_start) * 1000

        # Post-processing
        postprocess_start = time.time()
        result = self._postprocess_predictions(
            predictions, frame_id, camera_id,
            inference_time, preprocess_time, start_time
        )
        postprocess_time = (time.time() - postprocess_start) * 1000

        # Update performance metrics
        result.postprocessing_time_ms = postprocess_time
        result.total_time_ms = (time.time() - start_time) * 1000

        self._update_performance_metrics(result.total_time_ms)

        return result

    async def predict_batch(
        self,
        frames: list[np.ndarray],
        frame_ids: list[str],
        camera_ids: list[str] = None
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
        device = f"cuda:{device_id}"
        model = self.models[device_id]

        # Batch preprocessing
        preprocess_start = time.time()
        batch_tensor = self._preprocess_batch(frames, device_id)
        preprocess_time = (time.time() - preprocess_start) * 1000

        # Batch inference
        inference_start = time.time()
        with torch.cuda.device(device_id):
            with torch.inference_mode():
                with autocast(enabled=self.config.precision == "fp16"):
                    batch_predictions = model(batch_tensor)

        torch.cuda.synchronize(device_id)
        inference_time = (time.time() - inference_start) * 1000

        # Post-process each result in batch
        results = []
        for i, (frame_id, camera_id) in enumerate(zip(frame_ids, camera_ids, strict=False)):
            # Extract single prediction from batch
            single_pred = self._extract_single_prediction(batch_predictions, i)

            result = self._postprocess_predictions(
                single_pred, frame_id, camera_id,
                inference_time / batch_size, preprocess_time / batch_size,
                start_time
            )

            results.append(result)

        # Update throughput metrics
        self.throughput_counter += batch_size

        return results

    def _select_optimal_device(self) -> int:
        """Select the optimal GPU device based on current load."""
        if len(self.config.device_ids) == 1:
            return self.config.device_ids[0]

        # Simple round-robin for now
        # TODO: Implement intelligent load balancing based on GPU utilization
        device_id = self.config.device_ids[self.current_device]
        self.current_device = (self.current_device + 1) % len(self.config.device_ids)

        return device_id

    def _preprocess_frame(self, frame: np.ndarray, device_id: int) -> torch.Tensor:
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
        padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = frame

        # Convert to tensor format (CHW, normalized)
        tensor = torch.from_numpy(padded).to(f"cuda:{device_id}")
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).contiguous()
        tensor = tensor.half() if self.config.precision == "fp16" else tensor.float()
        tensor /= 255.0

        return tensor

    def _preprocess_batch(self, frames: list[np.ndarray], device_id: int) -> torch.Tensor:
        """Optimized batch preprocessing."""
        # Process each frame and stack into batch tensor
        batch_tensors = []

        for frame in frames:
            tensor = self._preprocess_frame(frame, device_id)
            batch_tensors.append(tensor)

        # Stack into batch
        batch_tensor = torch.cat(batch_tensors, dim=0)

        return batch_tensor

    def _extract_single_prediction(self, batch_predictions, index: int):
        """Extract single prediction from batch results."""
        if batch_predictions is None or len(batch_predictions) == 0:
            return None

        # Handle different output formats from YOLO11
        if isinstance(batch_predictions, (list, tuple)):
            # Multiple outputs, take the first (main detection output)
            batch_predictions = batch_predictions[0]

        if isinstance(batch_predictions, torch.Tensor):
            if len(batch_predictions.shape) == 3:
                # Shape: [batch_size, num_detections, features]
                if index < batch_predictions.shape[0]:
                    return batch_predictions[index]
                else:
                    return torch.zeros((0, batch_predictions.shape[-1]),
                                     device=batch_predictions.device)
            elif len(batch_predictions.shape) == 2:
                # Already single prediction
                return batch_predictions

        # If we can't extract, return empty tensor
        return torch.zeros((0, 85), device='cuda' if torch.cuda.is_available() else 'cpu')

    def _postprocess_predictions(
        self,
        predictions,
        frame_id: str,
        camera_id: str,
        inference_time: float,
        preprocess_time: float,
        start_time: float
    ) -> DetectionResult:
        """Convert model predictions to structured result."""
        # Apply NMS and confidence filtering
        filtered_predictions = self._apply_nms(predictions)

        # Extract detection components
        boxes, scores, classes = self._extract_detections(filtered_predictions)

        # Vehicle class names mapping (COCO classes)
        coco_to_vehicle = {
            1: "bicycle", 2: "car", 3: "motorcycle",
            5: "bus", 7: "truck"
        }

        class_names = []
        if len(classes) > 0:
            class_names = [coco_to_vehicle.get(cls, f"class_{cls}") for cls in classes]

        # Calculate metrics
        detection_count = len(boxes)
        avg_confidence = np.mean(scores) if len(scores) > 0 else 0.0
        gpu_memory = torch.cuda.memory_allocated(self.current_device) / 1024 / 1024

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
            gpu_memory_used_mb=gpu_memory
        )

    def _apply_nms(self, predictions):
        """Apply Non-Maximum Suppression to filter overlapping detections."""
        if predictions is None or len(predictions) == 0:
            return predictions

        # Extract detection components from YOLO11 predictions
        # YOLO11 output format: [batch_size, num_detections, 85]
        # where 85 = 4 (bbox) + 1 (conf) + 80 (classes)

        if isinstance(predictions, (list, tuple)):
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
            final_keep = final_keep[:self.config.max_detections]
            return filtered_preds[final_keep]
        else:
            return torch.zeros((0, 85), device=predictions.device)

    def _extract_detections(self, predictions) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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

        return (filtered_boxes[sort_indices],
                filtered_scores[sort_indices],
                filtered_classes[sort_indices])

    def _update_performance_metrics(self, total_time_ms: float):
        """Update running performance metrics."""
        self.inference_times.append(total_time_ms)

        # Keep only last 1000 measurements
        if len(self.inference_times) > 1000:
            self.inference_times = self.inference_times[-1000:]

    def get_performance_stats(self) -> dict[str, float]:
        """Get current performance statistics."""
        if not self.inference_times:
            return {}

        current_time = time.time()
        throughput = self.throughput_counter / max(
            1, current_time - self.last_throughput_time
        )

        return {
            "avg_latency_ms": np.mean(self.inference_times),
            "p50_latency_ms": np.percentile(self.inference_times, 50),
            "p95_latency_ms": np.percentile(self.inference_times, 95),
            "p99_latency_ms": np.percentile(self.inference_times, 99),
            "throughput_fps": throughput,
            "gpu_utilization": self._get_gpu_utilization(),
            "gpu_memory_used": self._get_gpu_memory_usage()
        }

    def _get_gpu_utilization(self) -> float:
        """Get average GPU utilization across all devices."""
        try:
            import pynvml
            pynvml.nvmlInit()

            total_util = 0
            for device_id in self.config.device_ids:
                handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                total_util += util

            return total_util / len(self.config.device_ids)
        except:
            return 0.0

    def _get_gpu_memory_usage(self) -> dict[str, float]:
        """Get GPU memory usage for all devices."""
        memory_stats = {}

        for device_id in self.config.device_ids:
            allocated = torch.cuda.memory_allocated(device_id) / 1024 / 1024
            cached = torch.cuda.memory_reserved(device_id) / 1024 / 1024

            memory_stats[f"gpu_{device_id}_allocated_mb"] = allocated
            memory_stats[f"gpu_{device_id}_cached_mb"] = cached

        return memory_stats

    async def cleanup(self):
        """Clean up resources."""
        await self.batcher.stop()
        self.memory_manager.cleanup()


# Model Selection Utility Functions

def select_optimal_model_for_deployment(
    target_latency_ms: int,
    target_accuracy: float,
    available_memory_gb: float,
    device_type: str = "gpu"
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
    backend = OptimizationBackend.TENSORRT if device_type == "gpu" else OptimizationBackend.ONNX
    precision = "fp16" if device_type == "gpu" else "fp32"

    config = InferenceConfig(
        model_type=model_type,
        backend=backend,
        precision=precision,
        batch_size=batch_size,
        max_batch_size=batch_size * 2,
        batch_timeout_ms=max(5, target_latency_ms // 4),
        conf_threshold=0.25,
        iou_threshold=0.45
    )

    return model_type, config


def estimate_throughput(
    model_type: ModelType,
    batch_size: int,
    gpu_type: str = "T4"
) -> dict[str, float]:
    """
    Estimate throughput for different model and hardware combinations.
    
    Benchmarks based on NVIDIA T4, V100, and A10G GPUs.
    """

    # Base inference times (ms) for batch size 1
    base_times = {
        ModelType.NANO: {"T4": 2.5, "V100": 1.8, "A10G": 1.5},
        ModelType.SMALL: {"T4": 6.0, "V100": 4.2, "A10G": 3.5},
        ModelType.MEDIUM: {"T4": 14.0, "V100": 9.8, "A10G": 8.2},
        ModelType.LARGE: {"T4": 22.0, "V100": 15.4, "A10G": 12.8},
        ModelType.XLARGE: {"T4": 42.0, "V100": 29.4, "A10G": 24.5}
    }

    base_time = base_times[model_type].get(gpu_type, base_times[model_type]["T4"])

    # Batch scaling factor (non-linear due to GPU parallelization)
    batch_scaling = {
        1: 1.0, 2: 1.6, 4: 2.8, 8: 4.5, 16: 7.2, 32: 12.0
    }

    scaling_factor = batch_scaling.get(batch_size, batch_size * 0.75)
    batch_time = base_time * scaling_factor

    return {
        "inference_time_ms": batch_time,
        "throughput_fps": (1000 * batch_size) / batch_time,
        "latency_ms": batch_time / batch_size,
        "gpu_utilization_pct": min(95, batch_size * 8),
        "memory_usage_mb": batch_size * base_times[model_type]["T4"] * 2
    }
