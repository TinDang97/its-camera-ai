"""
TensorRT Optimization for License Plate Recognition OCR Models.

This module provides specialized TensorRT optimization for OCR models used in 
license plate recognition, targeting sub-15ms inference latency with >95% accuracy.
"""

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

# TensorRT imports with fallback
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


class LPRTensorRTConfig:
    """Configuration for LPR TensorRT optimization."""

    def __init__(
        self,
        # OCR Model settings
        input_height: int = 32,
        input_width: int = 128,
        max_batch_size: int = 16,

        # Precision settings
        use_fp16: bool = True,
        use_int8: bool = True,

        # Performance settings
        workspace_size_gb: int = 4,
        max_aux_streams: int = 2,

        # Optimization settings
        enable_tactic_sources: bool = True,
        enable_timing_cache: bool = True,

        # Calibration settings
        calibration_batch_size: int = 8,
        calibration_dataset_size: int = 500,
    ):
        self.input_height = input_height
        self.input_width = input_width
        self.max_batch_size = max_batch_size
        self.use_fp16 = use_fp16
        self.use_int8 = use_int8
        self.workspace_size_gb = workspace_size_gb
        self.max_aux_streams = max_aux_streams
        self.enable_tactic_sources = enable_tactic_sources
        self.enable_timing_cache = enable_timing_cache
        self.calibration_batch_size = calibration_batch_size
        self.calibration_dataset_size = calibration_dataset_size


class LPROCRCalibrator:
    """INT8 calibrator for LPR OCR models."""

    def __init__(self, config: LPRTensorRTConfig, calibration_data: list[np.ndarray], cache_file: str):
        if not TRT_AVAILABLE:
            raise RuntimeError("TensorRT is required for LPROCRCalibrator")

        # Initialize parent class
        trt.IInt8EntropyCalibrator2.__init__(self)

        self.config = config
        self.calibration_data = calibration_data
        self.cache_file = cache_file
        self.batch_size = config.calibration_batch_size
        self.current_index = 0

        # Allocate device memory for calibration
        self.device_input = None
        self.input_size = (
            self.batch_size,
            1,  # Grayscale for OCR
            config.input_height,
            config.input_width
        )

        logger.info(f"Initialized LPR OCR calibrator with {len(calibration_data)} samples")

    def get_batch_size(self) -> int:
        return self.batch_size

    def get_batch(self, names: list[str]) -> list[int] | None:
        """Get next calibration batch for OCR model."""
        if self.current_index + self.batch_size > len(self.calibration_data):
            return None

        # Prepare batch data
        batch_data = []
        for i in range(self.batch_size):
            if self.current_index + i < len(self.calibration_data):
                plate_img = self.calibration_data[self.current_index + i]
                # Preprocess for OCR CRNN model
                processed = self._preprocess_plate_for_ocr(plate_img)
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

        logger.debug(f"OCR calibration batch {self.current_index // self.batch_size}: processed {len(batch_data)} plates")

        return [int(self.device_input)]

    def _preprocess_plate_for_ocr(self, plate_image: np.ndarray) -> np.ndarray:
        """Preprocess license plate image for OCR CRNN model."""
        # Convert to grayscale
        if len(plate_image.shape) == 3:
            gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_image.copy()

        # Resize to target dimensions with aspect ratio preservation
        h, w = gray.shape
        scale = min(self.config.input_height / h, self.config.input_width / w)

        new_h, new_w = int(h * scale), int(w * scale)
        if scale != 1.0:
            gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Pad to target size
        pad_h = (self.config.input_height - new_h) // 2
        pad_w = (self.config.input_width - new_w) // 2

        padded = np.full((self.config.input_height, self.config.input_width), 128, dtype=np.uint8)
        padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = gray

        # Normalize and add channel dimension
        normalized = padded.astype(np.float32) / 255.0

        # Return as (1, H, W) for CRNN input
        return normalized.reshape(1, self.config.input_height, self.config.input_width)

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
        logger.info(f"Wrote LPR OCR calibration cache to {self.cache_file}")


# Make calibrator class inherit from TensorRT base if available
if TRT_AVAILABLE:
    class LPROCRCalibratorImpl(trt.IInt8EntropyCalibrator2, LPROCRCalibrator):
        pass
else:
    class LPROCRCalibratorImpl(LPROCRCalibrator):
        pass


class LPRTensorRTOptimizer:
    """TensorRT optimizer specialized for LPR OCR models."""

    def __init__(self, config: LPRTensorRTConfig):
        if not TRT_AVAILABLE:
            raise RuntimeError("TensorRT is not available")

        self.config = config
        self.logger = trt.Logger(trt.Logger.INFO)

    async def optimize_ocr_model(
        self,
        onnx_path: str,
        output_path: str,
        calibration_data: list[np.ndarray] | None = None
    ) -> str:
        """Optimize ONNX OCR model to TensorRT engine for LPR."""

        logger.info(f"Optimizing LPR OCR model {onnx_path} to TensorRT engine {output_path}")

        # Create builder and network
        builder = trt.Builder(self.logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, self.logger)

        # Parse ONNX model
        with open(onnx_path, 'rb') as model_file:
            if not parser.parse(model_file.read()):
                for error in range(parser.num_errors):
                    logger.error(parser.get_error(error))
                raise RuntimeError("Failed to parse OCR ONNX model")

        logger.info("OCR ONNX model parsed successfully")

        # Create builder config
        config = builder.create_builder_config()

        # Set workspace size (smaller for OCR models)
        config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE,
            self.config.workspace_size_gb * 1024 * 1024 * 1024
        )

        # Enable optimization tactics
        if self.config.enable_tactic_sources:
            config.set_tactic_sources(
                1 << int(trt.TacticSource.CUBLAS) |
                1 << int(trt.TacticSource.CUBLAS_LT) |
                1 << int(trt.TacticSource.CUDNN)
            )

        # Set precision for maximum performance
        if self.config.use_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            logger.info("Enabled FP16 precision for OCR model")

        if self.config.use_int8 and calibration_data:
            config.set_flag(trt.BuilderFlag.INT8)

            # Create calibrator for OCR model
            cache_file = f"{output_path}.ocr_int8_cache"
            calibrator = LPROCRCalibratorImpl(self.config, calibration_data, cache_file)
            config.int8_calibrator = calibrator

            logger.info("Enabled INT8 precision with OCR-specific calibration")

        # Set optimization profiles for dynamic batching
        profile = builder.create_optimization_profile()

        # OCR input is typically (batch, 1, height, width)
        input_name = network.get_input(0).name

        min_shape = (1, 1, self.config.input_height, self.config.input_width)
        opt_shape = (self.config.calibration_batch_size, 1, self.config.input_height, self.config.input_width)
        max_shape = (self.config.max_batch_size, 1, self.config.input_height, self.config.input_width)

        profile.set_shape(input_name, min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)

        logger.info(f"Set OCR dynamic batch profiles: min={min_shape}, opt={opt_shape}, max={max_shape}")

        # Enable timing cache for faster subsequent builds
        if self.config.enable_timing_cache:
            timing_cache_path = f"{output_path}.ocr_timing_cache"
            if os.path.exists(timing_cache_path):
                with open(timing_cache_path, 'rb') as f:
                    timing_cache = f.read()
                    config.set_timing_cache(timing_cache, verify=True)
                logger.info(f"Loaded OCR timing cache from {timing_cache_path}")

        # Build engine with OCR-specific optimizations
        logger.info("Building TensorRT OCR engine (optimized for sub-15ms latency)...")
        start_time = time.time()

        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            raise RuntimeError("Failed to build TensorRT OCR engine")

        build_time = time.time() - start_time
        logger.info(f"TensorRT OCR engine built in {build_time:.1f} seconds")

        # Save engine
        with open(output_path, 'wb') as f:
            f.write(serialized_engine)

        # Save timing cache for future builds
        if self.config.enable_timing_cache:
            timing_cache = config.get_timing_cache()
            if timing_cache:
                timing_cache_path = f"{output_path}.ocr_timing_cache"
                with open(timing_cache_path, 'wb') as f:
                    f.write(timing_cache.serialize())
                logger.info(f"Saved OCR timing cache to {timing_cache_path}")

        logger.info(f"TensorRT OCR engine saved to {output_path}")

        return output_path

    async def benchmark_ocr_engine(self, engine_path: str, num_iterations: int = 1000) -> dict[str, float]:
        """Benchmark TensorRT OCR engine performance."""
        logger.info(f"Benchmarking TensorRT OCR engine {engine_path}")

        # Load and initialize engine for benchmarking
        trt_logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(trt_logger)
            engine = runtime.deserialize_cuda_engine(f.read())

        context = engine.create_execution_context()
        stream = cuda.Stream()

        # Allocate buffers
        input_shape = (self.config.calibration_batch_size, 1, self.config.input_height, self.config.input_width)
        input_size = np.prod(input_shape) * np.dtype(np.float32).itemsize

        # Get output shape
        output_shape = context.get_binding_shape(1)
        if output_shape[0] == -1:  # Dynamic batch
            output_shape = (self.config.calibration_batch_size,) + output_shape[1:]
        output_size = np.prod(output_shape) * np.dtype(np.float32).itemsize

        # Allocate device memory
        input_buffer = cuda.mem_alloc(input_size)
        output_buffer = cuda.mem_alloc(output_size)

        # Allocate host memory
        host_input = cuda.pagelocked_empty(input_shape, np.float32)
        host_output = cuda.pagelocked_empty(output_shape, np.float32)

        # Create dummy input data
        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        np.copyto(host_input, dummy_input.ravel())

        # Warm up
        for _ in range(10):
            cuda.memcpy_htod_async(input_buffer, host_input, stream)
            context.execute_async_v2([int(input_buffer), int(output_buffer)], stream.handle)
            cuda.memcpy_dtoh_async(host_output, output_buffer, stream)
            stream.synchronize()

        # Benchmark
        start_time = time.time()
        latencies = []

        for _ in range(num_iterations):
            iter_start = time.perf_counter()

            cuda.memcpy_htod_async(input_buffer, host_input, stream)
            context.execute_async_v2([int(input_buffer), int(output_buffer)], stream.handle)
            cuda.memcpy_dtoh_async(host_output, output_buffer, stream)
            stream.synchronize()

            iter_time = (time.perf_counter() - iter_start) * 1000
            latencies.append(iter_time)

        total_time = time.time() - start_time

        # Calculate statistics
        latencies = np.array(latencies)
        stats = {
            'avg_latency_ms': float(np.mean(latencies)),
            'min_latency_ms': float(np.min(latencies)),
            'max_latency_ms': float(np.max(latencies)),
            'p50_latency_ms': float(np.percentile(latencies, 50)),
            'p95_latency_ms': float(np.percentile(latencies, 95)),
            'p99_latency_ms': float(np.percentile(latencies, 99)),
            'throughput_fps': (num_iterations * self.config.calibration_batch_size) / total_time,
            'sub_15ms_rate': float(np.mean(latencies <= 15.0) * 100),
            'sub_10ms_rate': float(np.mean(latencies <= 10.0) * 100),
            'iterations': num_iterations,
        }

        # Cleanup
        input_buffer.free()
        output_buffer.free()
        del context
        del engine

        logger.info(f"OCR benchmark completed: {stats['avg_latency_ms']:.2f}ms avg, "
                   f"{stats['sub_15ms_rate']:.1f}% under 15ms")

        return stats


async def optimize_lpr_ocr_for_production(
    pytorch_model_path: str,
    output_dir: str,
    calibration_plates: list[str] | None = None,
    target_latency_ms: float = 15.0
) -> dict[str, Any]:
    """
    Optimize LPR OCR model for production deployment with TensorRT.
    
    Args:
        pytorch_model_path: Path to PyTorch OCR model (.pt file)
        output_dir: Directory to save optimized engines
        calibration_plates: List of license plate image paths for INT8 calibration
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

    # Load and export PyTorch model to ONNX
    logger.info(f"Loading PyTorch OCR model from {pytorch_model_path}")

    # Create dummy input for ONNX export
    dummy_input = torch.randn(1, 1, 32, 128)  # OCR CRNN input format

    # Load PyTorch model
    model = torch.load(pytorch_model_path, map_location='cpu')
    model.eval()

    # Export to ONNX
    onnx_path = output_dir / "ocr_model.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=16,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    logger.info(f"Exported ONNX model to {onnx_path}")

    # Load calibration data if provided
    calibration_data = None
    if calibration_plates and len(calibration_plates) > 0:
        logger.info(f"Loading {len(calibration_plates)} calibration plate images")
        calibration_data = []

        for img_path in calibration_plates[:500]:  # Limit for faster calibration
            try:
                img = cv2.imread(img_path)
                if img is not None:
                    calibration_data.append(img)
            except Exception as e:
                logger.warning(f"Failed to load calibration image {img_path}: {e}")

        logger.info(f"Loaded {len(calibration_data)} calibration images")

    # Test different optimization configurations
    test_configs = [
        # FP16 optimized (fastest)
        LPRTensorRTConfig(
            use_fp16=True,
            use_int8=False,
            max_batch_size=16,
            calibration_batch_size=4
        ),

        # INT8 with calibration (smallest and potentially fastest)
        LPRTensorRTConfig(
            use_fp16=True,
            use_int8=True,
            max_batch_size=16,
            calibration_batch_size=4,
            calibration_dataset_size=len(calibration_data) if calibration_data else 100
        ) if calibration_data else None,

        # FP32 baseline
        LPRTensorRTConfig(
            use_fp16=False,
            use_int8=False,
            max_batch_size=8,
            calibration_batch_size=4
        ),
    ]

    # Remove None configs
    test_configs = [cfg for cfg in test_configs if cfg is not None]

    best_latency = float('inf')
    best_config_name = None

    for i, config in enumerate(test_configs):
        config_name = f"ocr_config_{i}_"
        if config.use_int8:
            config_name += "int8"
        elif config.use_fp16:
            config_name += "fp16"
        else:
            config_name += "fp32"

        try:
            logger.info(f"Testing OCR configuration: {config_name}")

            # Build engine
            optimizer = LPRTensorRTOptimizer(config)
            engine_path = output_dir / f"{config_name}.trt"

            await optimizer.optimize_ocr_model(
                str(onnx_path),
                str(engine_path),
                calibration_data if config.use_int8 else None
            )

            # Benchmark
            stats = await optimizer.benchmark_ocr_engine(str(engine_path), 500)

            results['models'][config_name] = {
                'engine_path': str(engine_path),
                'config': config.__dict__,
                'performance': stats
            }

            avg_latency = stats['avg_latency_ms']
            if avg_latency < best_latency:
                best_latency = avg_latency
                best_config_name = config_name

            logger.info(f"{config_name}: {avg_latency:.2f}ms avg latency, "
                       f"{stats['sub_15ms_rate']:.1f}% under 15ms")

        except Exception as e:
            logger.error(f"Failed to optimize OCR with {config_name}: {e}")
            continue

    if best_config_name:
        results['best_model'] = best_config_name
        results['target_achieved'] = best_latency <= target_latency_ms

        logger.info(f"Best OCR model: {best_config_name} with {best_latency:.2f}ms latency")

        if results['target_achieved']:
            logger.info(f"✅ Target OCR latency of {target_latency_ms}ms achieved!")
        else:
            logger.info(f"⚠️ Target OCR latency of {target_latency_ms}ms not achieved. Best: {best_latency:.2f}ms")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Optimize LPR OCR model with TensorRT")
    parser.add_argument("--model", required=True, help="Path to PyTorch OCR model (.pt file)")
    parser.add_argument("--output", required=True, help="Output directory for optimized models")
    parser.add_argument("--calibration-dir", help="Directory containing license plate images for INT8")
    parser.add_argument("--target-latency", type=float, default=15.0, help="Target latency in ms")

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
        results = await optimize_lpr_ocr_for_production(
            args.model,
            args.output,
            calibration_images,
            args.target_latency
        )

        print("\n" + "="*60)
        print("LPR OCR OPTIMIZATION RESULTS")
        print("="*60)

        for name, model_info in results['models'].items():
            perf = model_info['performance']
            print(f"\n{name}:")
            print(f"  Avg Latency: {perf['avg_latency_ms']:.2f}ms")
            print(f"  P95 Latency: {perf['p95_latency_ms']:.2f}ms")
            print(f"  Sub-15ms Rate: {perf['sub_15ms_rate']:.1f}%")
            print(f"  Sub-10ms Rate: {perf['sub_10ms_rate']:.1f}%")
            print(f"  Throughput: {perf['throughput_fps']:.1f} FPS")

        if results['best_model']:
            print(f"\nBest Model: {results['best_model']}")
            print(f"Target Achieved: {'✅ Yes' if results['target_achieved'] else '❌ No'}")

    asyncio.run(main())
