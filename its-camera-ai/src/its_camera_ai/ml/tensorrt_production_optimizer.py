"""
Production-grade TensorRT optimization for ITS Camera AI.

Enhanced TensorRT implementation focusing on achieving sub-75ms latency
with 25-35% performance improvement over baseline YOLO11 inference.
"""

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from pydantic import BaseModel, Field

# Import TensorRT if available
try:
    import pycuda.autoinit  # noqa: F401
    import pycuda.driver as cuda
    import tensorrt as trt
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False
    trt = None
    cuda = None

logger = logging.getLogger(__name__)


class TensorRTConfig(BaseModel):
    """Configuration for TensorRT optimization."""

    # Model configuration
    input_height: int = Field(default=640)
    input_width: int = Field(default=640)
    max_batch_size: int = Field(default=32)

    # Precision settings
    use_fp16: bool = Field(default=True)
    use_int8: bool = Field(default=False)

    # Performance tuning
    workspace_size_gb: int = Field(default=8)
    max_aux_streams: int = Field(default=2)
    dla_core: int | None = Field(default=None)  # Deep Learning Accelerator

    # Optimization profiles
    min_batch_size: int = Field(default=1)
    opt_batch_size: int = Field(default=8)

    # Calibration settings
    calibration_dataset_size: int = Field(default=1000)
    calibration_batch_size: int = Field(default=8)

    # Engine settings
    enable_tactic_sources: bool = Field(default=True)
    enable_sparse_weights: bool = Field(default=True)
    enable_timing_cache: bool = Field(default=True)


class CUDAMemoryPool:
    """Optimized CUDA memory pool for TensorRT inference."""

    def __init__(self, config: TensorRTConfig):
        if not TRT_AVAILABLE:
            raise RuntimeError("TensorRT and PyCUDA are required for CUDAMemoryPool")

        self.config = config
        self.pools: dict[tuple[int, ...], list[Any]] = {}  # DeviceAllocation when available
        self.in_use: dict[int, Any] = {}  # DeviceAllocation when available
        self._lock = asyncio.Lock()

        # Pre-allocate common tensor sizes
        self._pre_allocate_tensors()

    def _pre_allocate_tensors(self) -> None:
        """Pre-allocate common tensor sizes for zero-copy operations."""
        common_shapes = [
            (1, 3, self.config.input_height, self.config.input_width),
            (4, 3, self.config.input_height, self.config.input_width),
            (8, 3, self.config.input_height, self.config.input_width),
            (16, 3, self.config.input_height, self.config.input_width),
            (32, 3, self.config.input_height, self.config.input_width),
        ]

        for shape in common_shapes:
            size_bytes = np.prod(shape) * np.dtype(np.float32).itemsize

            # Allocate multiple buffers per shape for concurrent use
            self.pools[shape] = []
            for _ in range(4):  # 4 buffers per shape
                try:
                    allocation = cuda.mem_alloc(size_bytes)
                    self.pools[shape].append(allocation)
                    logger.debug(f"Pre-allocated CUDA memory for shape {shape}: {size_bytes / 1024 / 1024:.1f} MB")
                except cuda.MemoryError as e:
                    logger.warning(f"Failed to pre-allocate memory for shape {shape}: {e}")
                    break

    async def get_buffer(self, shape: tuple[int, ...]) -> Any | None:
        """Get a CUDA buffer for the given shape."""
        async with self._lock:
            if shape in self.pools and self.pools[shape]:
                return self.pools[shape].pop()

        # Allocate new buffer if pool is empty
        size_bytes = np.prod(shape) * np.dtype(np.float32).itemsize
        try:
            return cuda.mem_alloc(size_bytes)
        except cuda.MemoryError:
            logger.error(f"Failed to allocate CUDA memory for shape {shape}")
            return None

    async def return_buffer(self, shape: tuple[int, ...], buffer: Any) -> None:
        """Return a buffer to the pool."""
        async with self._lock:
            if shape not in self.pools:
                self.pools[shape] = []

            if len(self.pools[shape]) < 8:  # Limit pool size
                self.pools[shape].append(buffer)
            # Otherwise let it be garbage collected

    def cleanup(self) -> None:
        """Clean up all allocated memory."""
        for pool in self.pools.values():
            for buffer in pool:
                try:
                    buffer.free()
                except:
                    pass
        self.pools.clear()


# Create INT8Calibrator class conditionally based on TensorRT availability
if TRT_AVAILABLE:
    class INT8Calibrator(trt.IInt8EntropyCalibrator2):
        """Custom INT8 calibrator for traffic camera data."""

        def __init__(self, config: TensorRTConfig, calibration_data: list[np.ndarray], cache_file: str):
            trt.IInt8EntropyCalibrator2.__init__(self)
else:
    class INT8Calibrator:
        """Mock INT8 calibrator when TensorRT is not available."""

        def __init__(self, config: TensorRTConfig, calibration_data: list[np.ndarray], cache_file: str):
            raise RuntimeError("TensorRT is required for INT8Calibrator")

class INT8CalibratorImpl(INT8Calibrator):
    """Implementation of INT8 calibrator functionality."""

    def __init__(self, config: TensorRTConfig, calibration_data: list[np.ndarray], cache_file: str):
        super().__init__(config, calibration_data, cache_file)
        self.config = config
        self.calibration_data = calibration_data
        self.cache_file = cache_file
        self.batch_size = config.calibration_batch_size
        self.current_index = 0

        # Allocate device memory for calibration
        self.device_input = None
        self.input_size = (
            self.batch_size,
            3,
            config.input_height,
            config.input_width
        )

        logger.info(f"Initialized INT8 calibrator with {len(calibration_data)} samples")

    def get_batch_size(self) -> int:
        return self.batch_size

    def get_batch(self, names: list[str]) -> list[int] | None:
        """Get next calibration batch."""
        if self.current_index + self.batch_size > len(self.calibration_data):
            return None

        # Prepare batch data
        batch_data = []
        for i in range(self.batch_size):
            if self.current_index + i < len(self.calibration_data):
                img = self.calibration_data[self.current_index + i]
                # Preprocess image
                processed = self._preprocess_image(img)
                batch_data.append(processed)

        if not batch_data:
            return None

        # Stack into batch array
        batch_array = np.stack(batch_data, axis=0).astype(np.float32)

        # Allocate device memory if needed
        if self.device_input is None:
            self.device_input = cuda.mem_alloc(batch_array.nbytes)

        # Copy to device
        cuda.memcpy_htod(self.device_input, batch_array)

        self.current_index += self.batch_size

        logger.debug(f"Calibration batch {self.current_index // self.batch_size}: processed {len(batch_data)} images")

        return [int(self.device_input)]

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess calibration image to match inference format."""
        # Resize with letterboxing
        h, w = image.shape[:2]
        scale = min(self.config.input_height / h, self.config.input_width / w)

        new_h, new_w = int(h * scale), int(w * scale)

        if scale != 1.0:
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Create padded canvas
        pad_h = (self.config.input_height - new_h) // 2
        pad_w = (self.config.input_width - new_w) // 2

        padded = np.full((self.config.input_height, self.config.input_width, 3), 114, dtype=np.uint8)
        padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = image

        # Convert to CHW format and normalize
        processed = padded.transpose(2, 0, 1).astype(np.float32) / 255.0

        return processed

    def read_calibration_cache(self) -> bytes | None:
        """Read calibration cache if exists."""
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache: bytes) -> None:
        """Write calibration cache."""
        with open(self.cache_file, 'wb') as f:
            f.write(cache)
        logger.info(f"Wrote INT8 calibration cache to {self.cache_file}")


class ProductionTensorRTEngine:
    """Production-optimized TensorRT inference engine."""

    def __init__(self, config: TensorRTConfig, engine_path: str):
        self.config = config
        self.engine_path = engine_path

        # TensorRT components
        self.engine: trt.ICudaEngine | None = None
        self.context: trt.IExecutionContext | None = None
        self.stream: cuda.Stream | None = None

        # Memory management
        self.memory_pool = CUDAMemoryPool(config)
        self.input_buffers: dict[int, cuda.DeviceAllocation] = {}
        self.output_buffers: dict[int, cuda.DeviceAllocation] = {}

        # Host memory for async transfers
        self.host_inputs: dict[int, np.ndarray] = {}
        self.host_outputs: dict[int, np.ndarray] = {}

        # Performance tracking
        self.inference_times: list[float] = []

        self._initialize()

    def _initialize(self) -> None:
        """Initialize TensorRT engine and allocate buffers."""
        if not TRT_AVAILABLE:
            raise RuntimeError("TensorRT is not available")

        # Load engine
        logger.info(f"Loading TensorRT engine from {self.engine_path}")

        trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(trt_logger)

        with open(self.engine_path, 'rb') as f:
            engine_data = f.read()

        self.engine = runtime.deserialize_cuda_engine(engine_data)
        if self.engine is None:
            raise RuntimeError("Failed to load TensorRT engine")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create execution context")

        # Create CUDA stream for async operations
        self.stream = cuda.Stream()

        # Allocate buffers
        self._allocate_buffers()

        logger.info("TensorRT engine initialized successfully")

    def _allocate_buffers(self) -> None:
        """Allocate input and output buffers."""
        for i in range(self.engine.num_bindings):
            binding_name = self.engine.get_binding_name(i)
            binding_shape = self.engine.get_binding_shape(i)
            binding_dtype = trt.nptype(self.engine.get_binding_dtype(i))

            # Calculate buffer size
            size = np.prod(binding_shape) * np.dtype(binding_dtype).itemsize

            # Allocate device memory
            device_buffer = cuda.mem_alloc(size)

            # Allocate pinned host memory for async transfers
            host_buffer = cuda.pagelocked_empty(binding_shape, binding_dtype)

            if self.engine.binding_is_input(i):
                self.input_buffers[i] = device_buffer
                self.host_inputs[i] = host_buffer
                logger.debug(f"Allocated input buffer {i} ({binding_name}): {binding_shape} ({size / 1024 / 1024:.1f} MB)")
            else:
                self.output_buffers[i] = device_buffer
                self.host_outputs[i] = host_buffer
                logger.debug(f"Allocated output buffer {i} ({binding_name}): {binding_shape} ({size / 1024 / 1024:.1f} MB)")

    async def infer_async(self, input_data: np.ndarray) -> np.ndarray:
        """Asynchronous inference with optimal memory management."""
        start_time = time.perf_counter()

        # Ensure input data is in correct format
        if input_data.dtype != np.float32:
            input_data = input_data.astype(np.float32)

        # Set dynamic batch size if needed
        batch_size = input_data.shape[0]
        if self.engine.has_implicit_batch_dimension:
            # Legacy TensorRT with implicit batch
            if batch_size != self.config.opt_batch_size:
                logger.warning(f"Batch size mismatch: expected {self.config.opt_batch_size}, got {batch_size}")
        else:
            # Explicit batch dimension - set dynamic shape
            input_shape = input_data.shape
            self.context.set_binding_shape(0, input_shape)

        # Copy input data to pinned host memory
        np.copyto(self.host_inputs[0], input_data.ravel())

        # Async copy to device
        cuda.memcpy_htod_async(self.input_buffers[0], self.host_inputs[0], self.stream)

        # Prepare binding pointers
        bindings = []
        for i in range(self.engine.num_bindings):
            if self.engine.binding_is_input(i):
                bindings.append(int(self.input_buffers[i]))
            else:
                bindings.append(int(self.output_buffers[i]))

        # Execute inference asynchronously
        success = self.context.execute_async_v2(bindings, self.stream.handle)
        if not success:
            raise RuntimeError("TensorRT inference failed")

        # Async copy output from device
        output_binding = len(self.input_buffers)
        cuda.memcpy_dtoh_async(self.host_outputs[output_binding], self.output_buffers[output_binding], self.stream)

        # Synchronize stream
        self.stream.synchronize()

        # Get output shape for reshaping
        output_shape = self.context.get_binding_shape(output_binding)
        if output_shape[0] == -1:  # Dynamic batch
            output_shape = (batch_size,) + output_shape[1:]

        # Reshape output
        output_data = self.host_outputs[output_binding].reshape(output_shape)

        # Record performance
        inference_time = (time.perf_counter() - start_time) * 1000
        self.inference_times.append(inference_time)

        # Keep only recent measurements
        if len(self.inference_times) > 1000:
            self.inference_times = self.inference_times[-1000:]

        return output_data.copy()

    def get_performance_stats(self) -> dict[str, float]:
        """Get performance statistics."""
        if not self.inference_times:
            return {}

        return {
            'avg_latency_ms': float(np.mean(self.inference_times)),
            'min_latency_ms': float(np.min(self.inference_times)),
            'max_latency_ms': float(np.max(self.inference_times)),
            'p50_latency_ms': float(np.percentile(self.inference_times, 50)),
            'p95_latency_ms': float(np.percentile(self.inference_times, 95)),
            'p99_latency_ms': float(np.percentile(self.inference_times, 99)),
        }

    def cleanup(self) -> None:
        """Clean up resources."""
        # Free device buffers
        for buffer in self.input_buffers.values():
            buffer.free()

        for buffer in self.output_buffers.values():
            buffer.free()

        # Clean up memory pool
        self.memory_pool.cleanup()

        # Destroy TensorRT objects
        if self.context:
            del self.context
        if self.engine:
            del self.engine

        logger.info("TensorRT engine cleanup completed")


class ProductionTensorRTOptimizer:
    """Production-grade TensorRT model optimizer."""

    def __init__(self, config: TensorRTConfig):
        self.config = config
        self.logger = trt.Logger(trt.Logger.INFO)

        if not TRT_AVAILABLE:
            raise RuntimeError("TensorRT is not available")

    async def optimize_model(
        self,
        onnx_path: str,
        output_path: str,
        calibration_data: list[np.ndarray] | None = None
    ) -> str:
        """Optimize ONNX model to TensorRT engine with production settings."""

        logger.info(f"Optimizing ONNX model {onnx_path} to TensorRT engine {output_path}")

        # Create builder and network
        builder = trt.Builder(self.logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, self.logger)

        # Parse ONNX model
        with open(onnx_path, 'rb') as model_file:
            if not parser.parse(model_file.read()):
                for error in range(parser.num_errors):
                    logger.error(parser.get_error(error))
                raise RuntimeError("Failed to parse ONNX model")

        logger.info("ONNX model parsed successfully")

        # Create builder config
        config = builder.create_builder_config()

        # Set workspace size
        config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE,
            self.config.workspace_size_gb * 1024 * 1024 * 1024
        )

        # Enable optimization flags
        if self.config.enable_tactic_sources:
            config.set_tactic_sources(
                1 << int(trt.TacticSource.CUBLAS) |
                1 << int(trt.TacticSource.CUBLAS_LT) |
                1 << int(trt.TacticSource.CUDNN) |
                1 << int(trt.TacticSource.EDGE_MASK_CONVOLUTIONS)
            )

        # Enable sparse weights if supported
        if self.config.enable_sparse_weights and hasattr(trt.BuilderFlag, 'SPARSE_WEIGHTS'):
            config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)

        # Set precision
        if self.config.use_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            logger.info("Enabled FP16 precision")

        if self.config.use_int8:
            if calibration_data is None:
                raise ValueError("Calibration data required for INT8 optimization")

            config.set_flag(trt.BuilderFlag.INT8)

            # Create calibrator
            cache_file = f"{output_path}.int8_cache"
            calibrator = INT8CalibratorImpl(self.config, calibration_data, cache_file)
            config.int8_calibrator = calibrator

            logger.info("Enabled INT8 precision with calibration")

        # Set optimization profiles for dynamic batching
        profile = builder.create_optimization_profile()

        # Assume first input is the main input
        input_name = network.get_input(0).name

        # Set dynamic batch size ranges
        min_shape = (self.config.min_batch_size, 3, self.config.input_height, self.config.input_width)
        opt_shape = (self.config.opt_batch_size, 3, self.config.input_height, self.config.input_width)
        max_shape = (self.config.max_batch_size, 3, self.config.input_height, self.config.input_width)

        profile.set_shape(input_name, min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)

        logger.info(f"Set dynamic batch profiles: min={min_shape}, opt={opt_shape}, max={max_shape}")

        # Enable timing cache for faster subsequent builds
        if self.config.enable_timing_cache:
            timing_cache_path = f"{output_path}.timing_cache"
            if os.path.exists(timing_cache_path):
                with open(timing_cache_path, 'rb') as f:
                    timing_cache = f.read()
                    config.set_timing_cache(timing_cache, verify=True)
                logger.info(f"Loaded timing cache from {timing_cache_path}")

        # Build engine
        logger.info("Building TensorRT engine (this may take several minutes)...")
        start_time = time.time()

        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            raise RuntimeError("Failed to build TensorRT engine")

        build_time = time.time() - start_time
        logger.info(f"TensorRT engine built in {build_time:.1f} seconds")

        # Save engine
        with open(output_path, 'wb') as f:
            f.write(serialized_engine)

        # Save timing cache for future builds
        if self.config.enable_timing_cache:
            timing_cache = config.get_timing_cache()
            if timing_cache:
                timing_cache_path = f"{output_path}.timing_cache"
                with open(timing_cache_path, 'wb') as f:
                    f.write(timing_cache.serialize())
                logger.info(f"Saved timing cache to {timing_cache_path}")

        logger.info(f"TensorRT engine saved to {output_path}")

        return output_path

    async def benchmark_engine(self, engine_path: str, num_iterations: int = 1000) -> dict[str, float]:
        """Benchmark TensorRT engine performance."""
        logger.info(f"Benchmarking TensorRT engine {engine_path}")

        # Create engine instance
        engine = ProductionTensorRTEngine(self.config, engine_path)

        # Warm up
        dummy_input = np.random.randn(
            self.config.opt_batch_size,
            3,
            self.config.input_height,
            self.config.input_width
        ).astype(np.float32)

        for _ in range(10):
            await engine.infer_async(dummy_input)

        # Benchmark
        start_time = time.time()

        for _ in range(num_iterations):
            await engine.infer_async(dummy_input)

        total_time = time.time() - start_time

        # Get performance stats
        stats = engine.get_performance_stats()
        stats.update({
            'total_benchmark_time_s': total_time,
            'throughput_fps': (num_iterations * self.config.opt_batch_size) / total_time,
            'iterations': num_iterations,
        })

        engine.cleanup()

        logger.info(f"Benchmark completed: {stats['avg_latency_ms']:.2f}ms avg latency, {stats['throughput_fps']:.1f} FPS throughput")

        return stats


async def optimize_yolo11_for_production(
    model_path: str,
    output_dir: str,
    calibration_images: list[str] | None = None,
    target_latency_ms: float = 75.0
) -> dict[str, Any]:
    """
    Optimize YOLO11 model for production deployment with TensorRT.
    
    Args:
        model_path: Path to YOLO11 PyTorch model (.pt file)
        output_dir: Directory to save optimized engines
        calibration_images: List of image paths for INT8 calibration
        target_latency_ms: Target inference latency in milliseconds
    
    Returns:
        Dictionary with optimization results and performance metrics
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'models': {},
        'best_model': None,
        'target_achieved': False
    }

    # Export to ONNX first
    from ultralytics import YOLO

    logger.info(f"Loading YOLO11 model from {model_path}")
    model = YOLO(model_path)

    onnx_path = output_dir / "model.onnx"
    model.export(
        format="onnx",
        imgsz=640,
        dynamic=True,
        batch_size=1,
        opset=16,
        half=True
    )

    logger.info(f"Exported ONNX model to {onnx_path}")

    # Load calibration data if provided
    calibration_data = None
    if calibration_images and len(calibration_images) > 0:
        logger.info(f"Loading {len(calibration_images)} calibration images")
        calibration_data = []

        for img_path in calibration_images[:1000]:  # Limit to 1000 images
            try:
                img = cv2.imread(img_path)
                if img is not None:
                    calibration_data.append(img)
            except Exception as e:
                logger.warning(f"Failed to load calibration image {img_path}: {e}")

        logger.info(f"Loaded {len(calibration_data)} calibration images")

    # Test different optimization configurations
    test_configs = [
        # FP32 baseline
        TensorRTConfig(
            use_fp16=False,
            use_int8=False,
            max_batch_size=16,
            opt_batch_size=4
        ),

        # FP16 optimized
        TensorRTConfig(
            use_fp16=True,
            use_int8=False,
            max_batch_size=32,
            opt_batch_size=8,
            workspace_size_gb=8
        ),

        # INT8 with calibration (if data available)
        TensorRTConfig(
            use_fp16=True,
            use_int8=True if calibration_data else False,
            max_batch_size=32,
            opt_batch_size=8,
            workspace_size_gb=8,
            calibration_batch_size=8
        ) if calibration_data else None,
    ]

    # Remove None configs
    test_configs = [cfg for cfg in test_configs if cfg is not None]

    best_latency = float('inf')
    best_config_name = None

    for i, config in enumerate(test_configs):
        config_name = f"config_{i}_"
        if config.use_int8:
            config_name += "int8"
        elif config.use_fp16:
            config_name += "fp16"
        else:
            config_name += "fp32"

        try:
            logger.info(f"Testing configuration: {config_name}")

            # Build engine
            optimizer = ProductionTensorRTOptimizer(config)
            engine_path = output_dir / f"{config_name}.trt"

            await optimizer.optimize_model(
                str(onnx_path),
                str(engine_path),
                calibration_data if config.use_int8 else None
            )

            # Benchmark
            stats = await optimizer.benchmark_engine(str(engine_path), 500)

            results['models'][config_name] = {
                'engine_path': str(engine_path),
                'config': config.dict(),
                'performance': stats
            }

            avg_latency = stats['avg_latency_ms']
            if avg_latency < best_latency:
                best_latency = avg_latency
                best_config_name = config_name

            logger.info(f"{config_name}: {avg_latency:.2f}ms avg latency")

        except Exception as e:
            logger.error(f"Failed to optimize with {config_name}: {e}")
            continue

    if best_config_name:
        results['best_model'] = best_config_name
        results['target_achieved'] = best_latency <= target_latency_ms

        logger.info(f"Best model: {best_config_name} with {best_latency:.2f}ms latency")

        if results['target_achieved']:
            logger.info(f"✅ Target latency of {target_latency_ms}ms achieved!")
        else:
            logger.info(f"⚠️ Target latency of {target_latency_ms}ms not achieved. Best: {best_latency:.2f}ms")

    return results


# CLI interface for optimization
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Optimize YOLO11 model with TensorRT")
    parser.add_argument("--model", required=True, help="Path to YOLO11 .pt model file")
    parser.add_argument("--output", required=True, help="Output directory for optimized models")
    parser.add_argument("--calibration-dir", help="Directory containing calibration images for INT8")
    parser.add_argument("--target-latency", type=float, default=75.0, help="Target latency in ms")

    args = parser.parse_args()

    # Gather calibration images if directory provided
    calibration_images = []
    if args.calibration_dir:
        calibration_dir = Path(args.calibration_dir)
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            calibration_images.extend(list(calibration_dir.glob(ext)))

        calibration_images = [str(p) for p in calibration_images]

    # Run optimization
    async def main():
        results = await optimize_yolo11_for_production(
            args.model,
            args.output,
            calibration_images,
            args.target_latency
        )

        print("\n" + "="*50)
        print("OPTIMIZATION RESULTS")
        print("="*50)

        for name, model_info in results['models'].items():
            perf = model_info['performance']
            print(f"\n{name}:")
            print(f"  Avg Latency: {perf['avg_latency_ms']:.2f}ms")
            print(f"  P95 Latency: {perf['p95_latency_ms']:.2f}ms")
            print(f"  Throughput: {perf['throughput_fps']:.1f} FPS")

        if results['best_model']:
            print(f"\nBest Model: {results['best_model']}")
            print(f"Target Achieved: {'✅ Yes' if results['target_achieved'] else '❌ No'}")

    asyncio.run(main())
