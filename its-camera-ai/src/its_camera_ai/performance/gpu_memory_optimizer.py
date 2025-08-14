"""GPU Memory Optimization for YOLO11 inference with production-ready performance.

This module implements comprehensive GPU memory optimization strategies including
TensorRT optimization, dynamic batching, memory pooling, and model quantization
to achieve >85% GPU utilization with <100ms latency.
"""

import gc
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager, contextmanager
from typing import Any

import structlog

from ..core.exceptions import GPUOptimizationError
from .optimization_config import GPUOptimizationConfig

logger = structlog.get_logger(__name__)

# GPU and ML imports with availability checks
try:
    import torch
    import torch.cuda
    import torch.nn as nn
    from torch.cuda.amp import GradScaler, autocast
    from torch.utils.data import DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    logger.warning("PyTorch not available - GPU optimization disabled")
    TORCH_AVAILABLE = False
    torch = None

try:
    import tensorrt as trt
    import torch_tensorrt
    TRT_AVAILABLE = True
except ImportError:
    logger.info("TensorRT not available - using PyTorch optimization only")
    TRT_AVAILABLE = False

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    logger.info("CuPy not available - using PyTorch memory management")
    CUPY_AVAILABLE = False

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    logger.warning("Ultralytics not available - YOLO11 optimization disabled")
    ULTRALYTICS_AVAILABLE = False


class GPUMemoryPool:
    """Advanced GPU memory pool manager for efficient memory allocation.
    
    Implements memory pooling with pre-allocation and reuse to minimize
    GPU memory fragmentation and allocation overhead.
    """

    def __init__(self, pool_size_gb: float = 8.0, enable_pinned_memory: bool = True):
        """Initialize GPU memory pool.
        
        Args:
            pool_size_gb: Total pool size in gigabytes
            enable_pinned_memory: Enable pinned memory for faster transfers
        """
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            raise GPUOptimizationError("CUDA not available for memory pool")

        self.pool_size_bytes = int(pool_size_gb * 1024**3)
        self.enable_pinned_memory = enable_pinned_memory

        # Memory pool storage
        self.allocated_tensors: dict[str, torch.Tensor] = {}
        self.free_tensors: dict[tuple[int, ...], list[torch.Tensor]] = {}
        self.pinned_buffers: dict[str, torch.Tensor] = {}

        # Statistics
        self.allocation_stats = {
            "total_allocations": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "memory_used_bytes": 0,
            "peak_memory_bytes": 0,
        }

        # Initialize memory pool
        self._initialize_pool()

        logger.info(f"GPU memory pool initialized: {pool_size_gb:.1f}GB")

    def _initialize_pool(self) -> None:
        """Initialize memory pool with common tensor sizes."""
        if not torch.cuda.is_available():
            return

        # Common tensor shapes for YOLO11 inference
        common_shapes = [
            (1, 3, 640, 640),    # Single image
            (4, 3, 640, 640),    # Small batch
            (8, 3, 640, 640),    # Medium batch
            (16, 3, 640, 640),   # Large batch
            (32, 3, 640, 640),   # Max batch
        ]

        for shape in common_shapes:
            try:
                tensor = torch.empty(shape, dtype=torch.float16, device="cuda")
                if shape not in self.free_tensors:
                    self.free_tensors[shape] = []
                self.free_tensors[shape].append(tensor)

                if self.enable_pinned_memory:
                    # Create pinned CPU buffer for fast transfers
                    cpu_tensor = torch.empty(shape, dtype=torch.float16, pin_memory=True)
                    self.pinned_buffers[f"cpu_{shape}"] = cpu_tensor

            except Exception as e:
                logger.warning(f"Failed to pre-allocate tensor {shape}: {e}")

    @contextmanager
    def get_tensor(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype = torch.float16,
        device: str = "cuda"
    ):
        """Get tensor from pool or allocate new one.
        
        Args:
            shape: Tensor shape
            dtype: Data type
            device: Device placement
            
        Yields:
            torch.Tensor: Allocated tensor
        """
        tensor = None

        try:
            # Try to get from pool
            if shape in self.free_tensors and self.free_tensors[shape]:
                tensor = self.free_tensors[shape].pop()
                self.allocation_stats["cache_hits"] += 1
            else:
                # Allocate new tensor
                tensor = torch.empty(shape, dtype=dtype, device=device)
                self.allocation_stats["cache_misses"] += 1

            self.allocation_stats["total_allocations"] += 1
            self.allocation_stats["memory_used_bytes"] += tensor.numel() * tensor.element_size()

            yield tensor

        finally:
            if tensor is not None:
                # Return tensor to pool
                tensor.zero_()  # Clear data
                if shape not in self.free_tensors:
                    self.free_tensors[shape] = []
                self.free_tensors[shape].append(tensor)

    def get_memory_stats(self) -> dict[str, Any]:
        """Get memory pool statistics.
        
        Returns:
            Dict[str, Any]: Memory pool statistics
        """
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_stats()
            return {
                **self.allocation_stats,
                "gpu_memory_allocated": gpu_memory.get("allocated_bytes.all.current", 0),
                "gpu_memory_reserved": gpu_memory.get("reserved_bytes.all.current", 0),
                "gpu_memory_free": torch.cuda.get_device_properties(0).total_memory -
                                  gpu_memory.get("reserved_bytes.all.current", 0),
                "cache_hit_rate": (
                    self.allocation_stats["cache_hits"] /
                    max(1, self.allocation_stats["total_allocations"])
                ),
            }
        return self.allocation_stats

    def clear_pool(self) -> None:
        """Clear memory pool and free GPU memory."""
        self.free_tensors.clear()
        self.pinned_buffers.clear()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        logger.info("GPU memory pool cleared")


class ModelQuantizer:
    """Model quantization for inference optimization.
    
    Implements INT8 quantization with calibration dataset to maintain
    >90% accuracy while reducing memory footprint and inference time.
    """

    def __init__(self, calibration_dataset: DataLoader | None = None):
        """Initialize model quantizer.
        
        Args:
            calibration_dataset: Dataset for INT8 calibration
        """
        self.calibration_dataset = calibration_dataset
        self.quantization_stats = {
            "models_quantized": 0,
            "memory_reduction_percent": 0.0,
            "inference_speedup": 0.0,
        }

    @asynccontextmanager
    async def quantize_model(
        self,
        model: torch.nn.Module,
        quantization_mode: str = "dynamic",
        target_precision: str = "int8"
    ) -> AsyncGenerator[torch.nn.Module, None]:
        """Quantize model for optimized inference.
        
        Args:
            model: PyTorch model to quantize
            quantization_mode: Quantization mode (dynamic, static)
            target_precision: Target precision (int8, fp16)
            
        Yields:
            torch.nn.Module: Quantized model
        """
        if not TORCH_AVAILABLE:
            yield model
            return

        original_size = self._get_model_size(model)

        try:
            if target_precision == "int8" and quantization_mode == "dynamic":
                # Dynamic INT8 quantization
                quantized_model = torch.quantization.quantize_dynamic(
                    model,
                    {torch.nn.Conv2d, torch.nn.Linear},
                    dtype=torch.qint8
                )
            elif target_precision == "int8" and quantization_mode == "static":
                # Static INT8 quantization with calibration
                quantized_model = await self._static_quantization(model)
            elif target_precision == "fp16":
                # FP16 quantization
                quantized_model = model.half()
            else:
                logger.warning(f"Unsupported quantization: {target_precision}/{quantization_mode}")
                quantized_model = model

            quantized_size = self._get_model_size(quantized_model)
            memory_reduction = ((original_size - quantized_size) / original_size) * 100

            self.quantization_stats["models_quantized"] += 1
            self.quantization_stats["memory_reduction_percent"] = memory_reduction

            logger.info(f"Model quantized: {memory_reduction:.1f}% memory reduction")

            yield quantized_model

        except Exception as e:
            logger.error(f"Model quantization failed: {e}")
            yield model  # Return original model on failure

    async def _static_quantization(self, model: torch.nn.Module) -> torch.nn.Module:
        """Perform static INT8 quantization with calibration.
        
        Args:
            model: Model to quantize
            
        Returns:
            torch.nn.Module: Statically quantized model
        """
        if self.calibration_dataset is None:
            logger.warning("No calibration dataset - falling back to dynamic quantization")
            return torch.quantization.quantize_dynamic(
                model,
                {torch.nn.Conv2d, torch.nn.Linear},
                dtype=torch.qint8
            )

        # Prepare model for static quantization
        model.eval()
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')

        # Prepare model
        prepared_model = torch.quantization.prepare(model, inplace=False)

        # Calibration with sample data
        with torch.no_grad():
            for batch_data, _ in self.calibration_dataset:
                if isinstance(batch_data, (list, tuple)):
                    batch_data = batch_data[0]
                prepared_model(batch_data)

        # Convert to quantized model
        quantized_model = torch.quantization.convert(prepared_model, inplace=False)

        return quantized_model

    def _get_model_size(self, model: torch.nn.Module) -> int:
        """Get model size in bytes.
        
        Args:
            model: Model to measure
            
        Returns:
            int: Model size in bytes
        """
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        return param_size + buffer_size


class TensorRTOptimizer:
    """TensorRT optimization for YOLO11 models.
    
    Implements TensorRT optimization with dynamic batching support
    to achieve maximum inference performance on NVIDIA GPUs.
    """

    def __init__(self, config: GPUOptimizationConfig):
        """Initialize TensorRT optimizer.
        
        Args:
            config: GPU optimization configuration
        """
        if not TRT_AVAILABLE:
            logger.warning("TensorRT not available - optimization disabled")

        self.config = config
        self.optimized_models: dict[str, Any] = {}
        self.optimization_stats = {
            "models_optimized": 0,
            "average_speedup": 0.0,
            "tensorrt_available": TRT_AVAILABLE,
        }

    async def optimize_yolo11_model(
        self,
        model: torch.nn.Module,
        input_shape: tuple[int, int, int, int],
        model_id: str = "yolo11_model"
    ) -> torch.nn.Module:
        """Optimize YOLO11 model with TensorRT.
        
        Args:
            model: YOLO11 model to optimize
            input_shape: Input tensor shape (batch, channels, height, width)
            model_id: Unique model identifier
            
        Returns:
            torch.nn.Module: TensorRT optimized model
        """
        if not TRT_AVAILABLE or not torch.cuda.is_available():
            logger.warning("TensorRT optimization unavailable - returning original model")
            return model

        try:
            # Check cache first
            if model_id in self.optimized_models:
                logger.info(f"Using cached TensorRT model: {model_id}")
                return self.optimized_models[model_id]

            logger.info(f"Optimizing YOLO11 model with TensorRT: {model_id}")

            # Prepare model for optimization
            model.eval()
            model = model.cuda()

            # Create example input
            example_input = torch.randn(
                self.config.tensorrt_min_batch_size,
                input_shape[1], input_shape[2], input_shape[3],
                dtype=torch.float16 if self.config.tensorrt_precision == "fp16" else torch.float32,
                device="cuda"
            )

            # TensorRT optimization settings
            optimization_settings = {
                "enabled_precisions": self._get_precision_set(),
                "workspace_size": int(self.config.tensorrt_workspace_size_gb * 1024**3),
                "max_batch_size": self.config.tensorrt_max_batch_size,
                "min_block_size": 1,
                "use_cuda_graph": self.config.enable_cuda_graphs,
            }

            # Dynamic shape configuration
            if self.config.enable_dynamic_batching:
                optimization_settings["inputs"] = [
                    torch_tensorrt.Input(
                        min_shape=(self.config.tensorrt_min_batch_size, *input_shape[1:]),
                        opt_shape=(self.config.tensorrt_opt_batch_size, *input_shape[1:]),
                        max_shape=(self.config.tensorrt_max_batch_size, *input_shape[1:]),
                        dtype=torch.float16 if self.config.tensorrt_precision == "fp16" else torch.float32,
                    )
                ]

            # Compile model with TensorRT
            start_time = time.perf_counter()

            optimized_model = torch_tensorrt.compile(
                model,
                inputs=[example_input] if not self.config.enable_dynamic_batching else None,
                **optimization_settings
            )

            compilation_time = time.perf_counter() - start_time

            # Benchmark optimization
            speedup = await self._benchmark_model_speedup(model, optimized_model, example_input)

            # Cache optimized model
            self.optimized_models[model_id] = optimized_model
            self.optimization_stats["models_optimized"] += 1
            self.optimization_stats["average_speedup"] = speedup

            logger.info(
                f"TensorRT optimization completed: {model_id}, "
                f"compilation_time={compilation_time:.2f}s, speedup={speedup:.2f}x"
            )

            return optimized_model

        except Exception as e:
            logger.error(f"TensorRT optimization failed: {e}")
            return model  # Return original model on failure

    def _get_precision_set(self) -> set:
        """Get TensorRT precision set based on configuration.
        
        Returns:
            set: Set of enabled precisions
        """
        precision_map = {
            "fp32": {torch.float32},
            "fp16": {torch.float16, torch.float32},
            "int8": {torch.int8, torch.float16, torch.float32},
        }
        return precision_map.get(self.config.tensorrt_precision, {torch.float32})

    async def _benchmark_model_speedup(
        self,
        original_model: torch.nn.Module,
        optimized_model: torch.nn.Module,
        example_input: torch.Tensor,
        num_iterations: int = 100
    ) -> float:
        """Benchmark model speedup with TensorRT optimization.
        
        Args:
            original_model: Original model
            optimized_model: TensorRT optimized model
            example_input: Example input tensor
            num_iterations: Number of benchmark iterations
            
        Returns:
            float: Speedup ratio (optimized/original)
        """
        def benchmark_model(model: torch.nn.Module, input_tensor: torch.Tensor) -> float:
            """Benchmark single model inference time."""
            model.eval()

            # Warmup
            with torch.no_grad():
                for _ in range(10):
                    _ = model(input_tensor)

            torch.cuda.synchronize()

            # Benchmark
            start_time = time.perf_counter()
            with torch.no_grad():
                for _ in range(num_iterations):
                    _ = model(input_tensor)

            torch.cuda.synchronize()
            end_time = time.perf_counter()

            return (end_time - start_time) / num_iterations

        try:
            original_time = benchmark_model(original_model, example_input)
            optimized_time = benchmark_model(optimized_model, example_input)

            speedup = original_time / optimized_time if optimized_time > 0 else 1.0
            return speedup

        except Exception as e:
            logger.warning(f"Benchmark failed: {e}")
            return 1.0


class GPUMemoryOptimizer:
    """Comprehensive GPU memory optimizer for production YOLO11 inference.
    
    Coordinates GPU memory optimization strategies including memory pooling,
    model quantization, TensorRT optimization, and dynamic batching to achieve
    >85% GPU utilization with <100ms latency.
    """

    def __init__(self, config: GPUOptimizationConfig):
        """Initialize GPU memory optimizer.
        
        Args:
            config: GPU optimization configuration
        """
        self.config = config
        self.memory_pool: GPUMemoryPool | None = None
        self.model_quantizer: ModelQuantizer | None = None
        self.tensorrt_optimizer: TensorRTOptimizer | None = None

        # Optimization metrics
        self.metrics = {
            "gpu_utilization_percent": 0.0,
            "memory_efficiency_percent": 0.0,
            "average_batch_size": 0.0,
            "inference_latency_ms": 0.0,
            "throughput_fps": 0.0,
        }

        self.is_initialized = False

    async def initialize(self) -> None:
        """Initialize GPU optimization components."""
        if self.is_initialized:
            return

        try:
            if not TORCH_AVAILABLE or not torch.cuda.is_available():
                logger.warning("CUDA not available - GPU optimization disabled")
                return

            # Initialize memory pool
            if self.config.gpu_memory_pool_size_gb > 0:
                self.memory_pool = GPUMemoryPool(
                    pool_size_gb=self.config.gpu_memory_pool_size_gb,
                    enable_pinned_memory=self.config.enable_memory_pinning
                )

            # Initialize model quantizer
            self.model_quantizer = ModelQuantizer()

            # Initialize TensorRT optimizer
            if self.config.enable_tensorrt and TRT_AVAILABLE:
                self.tensorrt_optimizer = TensorRTOptimizer(self.config)

            # Set PyTorch optimizations
            if self.config.compile_model and hasattr(torch, 'compile'):
                logger.info("PyTorch 2.0+ compilation enabled")

            if self.config.use_channels_last:
                logger.info("Channels-last memory format enabled")

            self.is_initialized = True
            logger.info("GPU memory optimizer initialized successfully")

        except Exception as e:
            logger.error(f"GPU optimizer initialization failed: {e}")
            raise GPUOptimizationError(f"Initialization failed: {e}") from e

    @asynccontextmanager
    async def optimize_inference_pipeline(
        self,
        model: torch.nn.Module,
        input_shape: tuple[int, int, int, int],
        model_id: str = "inference_model"
    ) -> AsyncGenerator[torch.nn.Module, None]:
        """Optimize complete inference pipeline for production use.
        
        Args:
            model: Model to optimize
            input_shape: Input tensor shape (batch, channels, height, width)
            model_id: Unique model identifier
            
        Yields:
            torch.nn.Module: Fully optimized model
        """
        if not self.is_initialized:
            await self.initialize()

        optimized_model = model

        try:
            # Step 1: Model quantization
            if self.config.enable_quantization and self.model_quantizer:
                async with self.model_quantizer.quantize_model(
                    model,
                    quantization_mode=self.config.quantization_mode,
                    target_precision="fp16" if self.config.enable_mixed_precision else "fp32"
                ) as quantized_model:
                    optimized_model = quantized_model

                    # Step 2: TensorRT optimization
                    if self.config.enable_tensorrt and self.tensorrt_optimizer:
                        optimized_model = await self.tensorrt_optimizer.optimize_yolo11_model(
                            optimized_model, input_shape, model_id
                        )

                    # Step 3: Apply additional PyTorch optimizations
                    optimized_model = self._apply_pytorch_optimizations(optimized_model)

                    yield optimized_model
            else:
                # No quantization - apply other optimizations
                if self.config.enable_tensorrt and self.tensorrt_optimizer:
                    optimized_model = await self.tensorrt_optimizer.optimize_yolo11_model(
                        model, input_shape, model_id
                    )

                optimized_model = self._apply_pytorch_optimizations(optimized_model)
                yield optimized_model

        except Exception as e:
            logger.error(f"Inference pipeline optimization failed: {e}")
            yield model  # Return original model on failure

    def _apply_pytorch_optimizations(self, model: torch.nn.Module) -> torch.nn.Module:
        """Apply PyTorch-specific optimizations.
        
        Args:
            model: Model to optimize
            
        Returns:
            torch.nn.Module: Optimized model
        """
        if not TORCH_AVAILABLE:
            return model

        try:
            # Enable eval mode
            model.eval()

            # Channels-last memory format for CNN optimization
            if self.config.use_channels_last:
                model = model.to(memory_format=torch.channels_last)

            # PyTorch 2.0+ compilation
            if self.config.compile_model and hasattr(torch, 'compile'):
                model = torch.compile(model, dynamic=self.config.enable_dynamic_batching)

            # JIT optimization for static models
            if self.config.enable_jit_fusion and not self.config.enable_dynamic_batching:
                model = torch.jit.optimize_for_inference(
                    torch.jit.script(model)
                )

            return model

        except Exception as e:
            logger.warning(f"PyTorch optimization failed: {e}")
            return model

    async def update_metrics(self, batch_size: int, inference_time_ms: float) -> None:
        """Update optimization metrics.
        
        Args:
            batch_size: Current batch size
            inference_time_ms: Inference time in milliseconds
        """
        if not torch.cuda.is_available():
            return

        try:
            # GPU utilization
            gpu_util = torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0.0

            # Memory efficiency
            gpu_memory = torch.cuda.memory_stats()
            allocated_bytes = gpu_memory.get("allocated_bytes.all.current", 0)
            reserved_bytes = gpu_memory.get("reserved_bytes.all.current", 0)
            memory_efficiency = (allocated_bytes / max(1, reserved_bytes)) * 100

            # Update metrics
            self.metrics.update({
                "gpu_utilization_percent": gpu_util,
                "memory_efficiency_percent": memory_efficiency,
                "average_batch_size": batch_size,
                "inference_latency_ms": inference_time_ms,
                "throughput_fps": (batch_size / max(0.001, inference_time_ms / 1000)),
            })

        except Exception as e:
            logger.warning(f"Metrics update failed: {e}")

    def get_optimization_metrics(self) -> dict[str, Any]:
        """Get comprehensive optimization metrics.
        
        Returns:
            Dict[str, Any]: Optimization metrics
        """
        base_metrics = self.metrics.copy()

        # Add memory pool metrics if available
        if self.memory_pool:
            base_metrics["memory_pool"] = self.memory_pool.get_memory_stats()

        # Add quantization metrics if available
        if self.model_quantizer:
            base_metrics["quantization"] = self.model_quantizer.quantization_stats

        # Add TensorRT metrics if available
        if self.tensorrt_optimizer:
            base_metrics["tensorrt"] = self.tensorrt_optimizer.optimization_stats

        # System metrics
        if torch.cuda.is_available():
            base_metrics["system"] = {
                "cuda_available": True,
                "gpu_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
                "gpu_name": torch.cuda.get_device_name(),
            }

        return base_metrics

    async def cleanup(self) -> None:
        """Cleanup GPU optimization resources."""
        if self.memory_pool:
            self.memory_pool.clear_pool()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.is_initialized = False
        logger.info("GPU memory optimizer cleanup completed")


async def create_gpu_optimizer(
    config: GPUOptimizationConfig
) -> GPUMemoryOptimizer:
    """Create and initialize GPU memory optimizer.
    
    Args:
        config: GPU optimization configuration
        
    Returns:
        GPUMemoryOptimizer: Initialized GPU optimizer
    """
    optimizer = GPUMemoryOptimizer(config)
    await optimizer.initialize()
    return optimizer
