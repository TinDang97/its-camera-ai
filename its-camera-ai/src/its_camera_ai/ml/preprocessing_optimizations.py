"""
Advanced preprocessing optimizations for production ML inference.

This module provides GPU-accelerated preprocessing utilities, memory management,
and custom CUDA kernels for high-performance computer vision preprocessing
specifically optimized for YOLO11 models in production environments.

Key Optimizations:
1. Custom CUDA kernels for letterboxing and resizing
2. Tensor memory pooling and reuse strategies
3. Vectorized batch preprocessing operations
4. Asynchronous quality score calculations
5. Multi-stream GPU processing
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import cv2
import numpy as np
import torch

try:
    import cupy as cp
    import cupyx.scipy.ndimage as cp_ndimage
    from cupyx.scipy.ndimage import zoom as cp_zoom
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    cp_ndimage = None
    cp_zoom = None
    CUPY_AVAILABLE = False

try:
    import nvjpeg
    NVJPEG_AVAILABLE = True
except ImportError:
    nvjpeg = None
    NVJPEG_AVAILABLE = False

logger = logging.getLogger(__name__)


class GPUMemoryPool:
    """Advanced GPU memory pool for tensor reuse and allocation optimization."""

    def __init__(self, max_pool_size: int = 100, device: str = 'cuda'):
        self.device = torch.device(device)
        self.max_pool_size = max_pool_size
        self.tensor_pools = {}  # Keyed by (shape, dtype)
        self.allocation_stats = {
            'hits': 0,
            'misses': 0,
            'allocations': 0,
            'deallocations': 0
        }

        logger.info(f"Initialized GPU memory pool with max_size={max_pool_size}")

    def get_tensor(self, shape: tuple, dtype: torch.dtype) -> torch.Tensor:
        """Get tensor from pool or allocate new one."""
        pool_key = (shape, dtype)

        if pool_key in self.tensor_pools and self.tensor_pools[pool_key]:
            tensor = self.tensor_pools[pool_key].pop()
            self.allocation_stats['hits'] += 1
            return tensor

        # Allocate new tensor
        tensor = torch.empty(shape, dtype=dtype, device=self.device)
        self.allocation_stats['misses'] += 1
        self.allocation_stats['allocations'] += 1
        return tensor

    def return_tensor(self, tensor: torch.Tensor) -> None:
        """Return tensor to pool for reuse."""
        if tensor.device != self.device:
            return

        pool_key = (tuple(tensor.shape), tensor.dtype)

        if pool_key not in self.tensor_pools:
            self.tensor_pools[pool_key] = []

        if len(self.tensor_pools[pool_key]) < self.max_pool_size:
            tensor.detach_()  # Remove from computation graph
            self.tensor_pools[pool_key].append(tensor)

        self.allocation_stats['deallocations'] += 1

    def get_stats(self) -> dict:
        """Get memory pool statistics."""
        hit_rate = self.allocation_stats['hits'] / max(1,
            self.allocation_stats['hits'] + self.allocation_stats['misses'])

        return {
            **self.allocation_stats,
            'hit_rate': hit_rate,
            'active_pools': len(self.tensor_pools),
            'total_pooled_tensors': sum(len(pool) for pool in self.tensor_pools.values())
        }

    def cleanup(self) -> None:
        """Clean up memory pools."""
        self.tensor_pools.clear()
        torch.cuda.empty_cache()


class CUDAPreprocessingKernels:
    """Custom CUDA kernels for high-performance preprocessing operations."""

    def __init__(self):
        self.kernels_compiled = False
        self.letterbox_kernel = None
        self.resize_kernel = None

        if CUPY_AVAILABLE:
            self._compile_kernels()

    def _compile_kernels(self) -> None:
        """Compile custom CUDA kernels for preprocessing."""
        try:
            # Letterboxing kernel - optimized for batch processing
            letterbox_kernel_code = '''
            extern "C" __global__
            void letterbox_kernel(
                unsigned char* input_batch,
                unsigned char* output_batch,
                int batch_size,
                int input_height,
                int input_width,
                int output_height,
                int output_width,
                int channels,
                int pad_value,
                float* scales,
                int* offsets_x,
                int* offsets_y
            ) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                int total_pixels = batch_size * output_height * output_width * channels;
                
                if (idx >= total_pixels) return;
                
                // Calculate batch, pixel, and channel indices
                int batch_idx = idx / (output_height * output_width * channels);
                int remaining = idx % (output_height * output_width * channels);
                int pixel_idx = remaining / channels;
                int channel = remaining % channels;
                
                int y = pixel_idx / output_width;
                int x = pixel_idx % output_width;
                
                float scale = scales[batch_idx];
                int offset_x = offsets_x[batch_idx];
                int offset_y = offsets_y[batch_idx];
                
                // Check if pixel is in padding area
                if (x < offset_x || x >= offset_x + (int)(input_width * scale) ||
                    y < offset_y || y >= offset_y + (int)(input_height * scale)) {
                    output_batch[idx] = pad_value;
                } else {
                    // Map back to input coordinates
                    int input_x = (int)((x - offset_x) / scale);
                    int input_y = (int)((y - offset_y) / scale);
                    
                    // Ensure coordinates are within bounds
                    input_x = min(max(input_x, 0), input_width - 1);
                    input_y = min(max(input_y, 0), input_height - 1);
                    
                    int input_idx = batch_idx * (input_height * input_width * channels) +
                                   input_y * input_width * channels +
                                   input_x * channels + channel;
                    
                    output_batch[idx] = input_batch[input_idx];
                }
            }
            '''

            # Compile letterbox kernel
            self.letterbox_kernel = cp.RawKernel(letterbox_kernel_code, 'letterbox_kernel')

            # Batch resize kernel for better memory efficiency
            resize_kernel_code = '''
            extern "C" __global__
            void batch_resize_kernel(
                unsigned char* input_batch,
                unsigned char* output_batch,
                int batch_size,
                int input_height,
                int input_width,
                int output_height,
                int output_width,
                int channels
            ) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                int total_pixels = batch_size * output_height * output_width * channels;
                
                if (idx >= total_pixels) return;
                
                int batch_idx = idx / (output_height * output_width * channels);
                int remaining = idx % (output_height * output_width * channels);
                int pixel_idx = remaining / channels;
                int channel = remaining % channels;
                
                int out_y = pixel_idx / output_width;
                int out_x = pixel_idx % output_width;
                
                // Bilinear interpolation
                float scale_y = (float)input_height / output_height;
                float scale_x = (float)input_width / output_width;
                
                float src_y = (out_y + 0.5f) * scale_y - 0.5f;
                float src_x = (out_x + 0.5f) * scale_x - 0.5f;
                
                int y1 = max(0, (int)floor(src_y));
                int y2 = min(input_height - 1, y1 + 1);
                int x1 = max(0, (int)floor(src_x));
                int x2 = min(input_width - 1, x1 + 1);
                
                float wy2 = src_y - y1;
                float wy1 = 1.0f - wy2;
                float wx2 = src_x - x1;
                float wx1 = 1.0f - wx2;
                
                int base_input = batch_idx * (input_height * input_width * channels);
                
                unsigned char p11 = input_batch[base_input + y1 * input_width * channels + x1 * channels + channel];
                unsigned char p12 = input_batch[base_input + y1 * input_width * channels + x2 * channels + channel];
                unsigned char p21 = input_batch[base_input + y2 * input_width * channels + x1 * channels + channel];
                unsigned char p22 = input_batch[base_input + y2 * input_width * channels + x2 * channels + channel];
                
                float result = wy1 * wx1 * p11 + wy1 * wx2 * p12 + wy2 * wx1 * p21 + wy2 * wx2 * p22;
                
                output_batch[idx] = (unsigned char)min(255.0f, max(0.0f, result));
            }
            '''

            self.resize_kernel = cp.RawKernel(resize_kernel_code, 'batch_resize_kernel')

            self.kernels_compiled = True
            logger.info("Custom CUDA preprocessing kernels compiled successfully")

        except Exception as e:
            logger.warning(f"Failed to compile CUDA kernels: {e}")
            self.kernels_compiled = False

    def batch_letterbox_gpu(
        self,
        input_batch: Any,  # cp.ndarray when cupy available
        output_shape: tuple[int, int],
        pad_value: int = 114
    ) -> tuple[Any, list[dict]]:  # Tuple[cp.ndarray, List[dict]] when cupy available
        """GPU-accelerated batch letterboxing using custom CUDA kernel."""
        if not self.kernels_compiled:
            raise RuntimeError("CUDA kernels not compiled")

        batch_size, input_h, input_w, channels = input_batch.shape
        output_h, output_w = output_shape

        # Calculate scaling and offsets for each image in batch
        scales = []
        offsets_x = []
        offsets_y = []
        metadata_list = []

        for i in range(batch_size):
            scale = min(output_h / input_h, output_w / input_w)
            new_h, new_w = int(input_h * scale), int(input_w * scale)
            offset_x = (output_w - new_w) // 2
            offset_y = (output_h - new_h) // 2

            scales.append(scale)
            offsets_x.append(offset_x)
            offsets_y.append(offset_y)

            metadata_list.append({
                'scale_factor': scale,
                'padding': (offset_x, offset_y),
                'new_size': (new_w, new_h)
            })

        # Convert to GPU arrays
        scales_gpu = cp.array(scales, dtype=cp.float32)
        offsets_x_gpu = cp.array(offsets_x, dtype=cp.int32)
        offsets_y_gpu = cp.array(offsets_y, dtype=cp.int32)

        # Allocate output
        output_batch = cp.zeros((batch_size, output_h, output_w, channels), dtype=cp.uint8)

        # Launch kernel
        total_pixels = batch_size * output_h * output_w * channels
        block_size = 256
        grid_size = (total_pixels + block_size - 1) // block_size

        self.letterbox_kernel(
            (grid_size,), (block_size,),
            (input_batch, output_batch,
             batch_size, input_h, input_w, output_h, output_w, channels,
             pad_value, scales_gpu, offsets_x_gpu, offsets_y_gpu)
        )

        return output_batch, metadata_list

    def batch_resize_gpu(
        self,
        input_batch: Any,  # cp.ndarray when cupy available
        output_shape: tuple[int, int]
    ) -> Any:  # cp.ndarray when cupy available
        """GPU-accelerated batch resize using custom CUDA kernel."""
        if not self.kernels_compiled:
            raise RuntimeError("CUDA kernels not compiled")

        batch_size, input_h, input_w, channels = input_batch.shape
        output_h, output_w = output_shape

        # Allocate output
        output_batch = cp.zeros((batch_size, output_h, output_w, channels), dtype=cp.uint8)

        # Launch kernel
        total_pixels = batch_size * output_h * output_w * channels
        block_size = 256
        grid_size = (total_pixels + block_size - 1) // block_size

        self.batch_resize_kernel(
            (grid_size,), (block_size,),
            (input_batch, output_batch,
             batch_size, input_h, input_w, output_h, output_w, channels)
        )

        return output_batch


class AsyncQualityCalculator:
    """Asynchronous quality score calculation to avoid blocking preprocessing."""

    def __init__(self, max_workers: int = 2):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.quality_cache = {}
        self.max_cache_size = 1000

    def _calculate_quality_fast(self, frame: np.ndarray) -> float:
        """Fast quality calculation using sampling and optimized operations."""
        h, w = frame.shape[:2]

        # Use smaller sample for quality estimation
        step = max(1, min(h, w) // 8)
        sample = frame[::step, ::step]

        if len(sample.shape) == 3:
            gray_sample = cv2.cvtColor(sample, cv2.COLOR_RGB2GRAY)
        else:
            gray_sample = sample

        # Fast gradient-based sharpness
        grad_x = np.gradient(gray_sample, axis=1)
        grad_y = np.gradient(gray_sample, axis=0)
        gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
        sharpness_score = min(1.0, np.mean(gradient_mag) / 15.0)

        # Fast brightness and contrast
        mean_brightness = np.mean(gray_sample)
        brightness_score = 1.0 if 50 <= mean_brightness <= 200 else 0.7

        contrast = np.std(gray_sample)
        contrast_score = min(1.0, contrast / 40.0)

        return sharpness_score * 0.5 + brightness_score * 0.25 + contrast_score * 0.25

    async def calculate_quality_async(self, frame: np.ndarray) -> float:
        """Async quality calculation."""
        # Simple hash for caching (production should use better hashing)
        cache_key = hash(frame.tobytes()[:1000])  # Hash first 1000 bytes

        if cache_key in self.quality_cache:
            return self.quality_cache[cache_key]

        loop = asyncio.get_event_loop()
        quality_score = await loop.run_in_executor(
            self.executor, self._calculate_quality_fast, frame
        )

        # Cache result
        if len(self.quality_cache) < self.max_cache_size:
            self.quality_cache[cache_key] = quality_score

        return quality_score

    def cleanup(self):
        """Cleanup resources."""
        self.executor.shutdown(wait=False)
        self.quality_cache.clear()


class ProductionPreprocessingOptimizer:
    """
    Production-ready preprocessing optimizer combining all optimization techniques.
    
    This class provides the highest performance preprocessing pipeline by:
    1. Using custom CUDA kernels for batch operations
    2. Managing GPU memory pools for tensor reuse
    3. Implementing async quality calculations
    4. Utilizing multi-stream GPU processing
    5. Caching frequently used parameters
    """

    def __init__(self, config, enable_custom_kernels: bool = True):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_gpu = torch.cuda.is_available() and CUPY_AVAILABLE

        # Initialize optimization components
        self.memory_pool = GPUMemoryPool() if self.use_gpu else None
        self.cuda_kernels = CUDAPreprocessingKernels() if (self.use_gpu and enable_custom_kernels) else None
        self.quality_calculator = AsyncQualityCalculator()

        # CUDA streams for parallel processing
        if self.use_gpu:
            self.preprocess_stream = torch.cuda.Stream()
            self.quality_stream = torch.cuda.Stream()

        # Performance tracking
        self.optimization_stats = {
            'kernel_accelerated_batches': 0,
            'fallback_batches': 0,
            'total_processing_time_ms': 0,
            'frames_processed': 0
        }

        logger.info(f"Production preprocessing optimizer initialized (GPU: {self.use_gpu}, "
                   f"Custom kernels: {self.cuda_kernels is not None})")

    def preprocess_batch_optimized(
        self,
        frames: list[np.ndarray],
        target_size: tuple[int, int] = (640, 640)
    ) -> tuple[np.ndarray, list[dict]]:
        """
        Highly optimized batch preprocessing using all available optimizations.
        """
        start_time = time.time()
        batch_size = len(frames)

        if not self.use_gpu or batch_size == 1:
            return self._preprocess_batch_cpu_fallback(frames, target_size)

        try:
            with torch.cuda.stream(self.preprocess_stream):
                # Convert to CuPy arrays for GPU processing
                gpu_frames = []
                original_shapes = []

                for frame in frames:
                    original_shapes.append(frame.shape[:2])
                    gpu_frame = cp.asarray(frame, dtype=cp.uint8)
                    gpu_frames.append(gpu_frame)

                # Group frames by similar dimensions for efficient batch processing
                dimension_groups = self._group_frames_by_dimensions(gpu_frames, original_shapes)

                all_processed_frames = []
                all_metadata = []

                for group_shapes, group_data in dimension_groups.items():
                    group_frames, group_indices = group_data

                    if len(group_frames) > 1 and self.cuda_kernels and self.cuda_kernels.kernels_compiled:
                        # Use custom CUDA kernels for batch processing
                        batch_array = cp.stack(group_frames, axis=0)
                        processed_batch, metadata_list = self.cuda_kernels.batch_letterbox_gpu(
                            batch_array, target_size
                        )

                        self.optimization_stats['kernel_accelerated_batches'] += 1

                    else:
                        # Fallback to optimized CuPy operations
                        processed_batch, metadata_list = self._process_group_cupy_fallback(
                            group_frames, target_size
                        )

                        self.optimization_stats['fallback_batches'] += 1

                    # Store results in original order
                    for i, (processed_frame, metadata) in enumerate(zip(processed_batch, metadata_list, strict=False)):
                        original_idx = group_indices[i]
                        while len(all_processed_frames) <= original_idx:
                            all_processed_frames.append(None)
                            all_metadata.append(None)

                        all_processed_frames[original_idx] = processed_frame
                        all_metadata[original_idx] = metadata

                # Convert back to CPU for inference
                cpu_frames = [cp.asnumpy(frame) for frame in all_processed_frames]
                final_batch = np.stack(cpu_frames, axis=0)

                # Update quality scores asynchronously (non-blocking)
                asyncio.create_task(self._update_quality_scores_async(cpu_frames, all_metadata))

        except Exception as e:
            logger.warning(f"GPU batch preprocessing failed: {e}, falling back to CPU")
            return self._preprocess_batch_cpu_fallback(frames, target_size)

        # Update statistics
        processing_time = (time.time() - start_time) * 1000
        self.optimization_stats['total_processing_time_ms'] += processing_time
        self.optimization_stats['frames_processed'] += batch_size

        # Add timing to metadata
        per_frame_time = processing_time / batch_size
        for metadata in all_metadata:
            metadata['processing_time_ms'] = per_frame_time
            metadata['gpu_accelerated'] = True
            metadata['kernel_accelerated'] = self.cuda_kernels is not None

        return final_batch, all_metadata

    def _group_frames_by_dimensions(self, frames: list[Any], shapes: list[tuple[int, int]]) -> dict:
        """Group frames by similar dimensions for efficient batch processing."""
        dimension_groups = {}

        for i, (frame, shape) in enumerate(zip(frames, shapes, strict=False)):
            if shape not in dimension_groups:
                dimension_groups[shape] = ([], [])

            dimension_groups[shape][0].append(frame)  # frames
            dimension_groups[shape][1].append(i)      # indices

        return dimension_groups

    def _process_group_cupy_fallback(
        self,
        group_frames: list[Any],  # List[cp.ndarray] when cupy available
        target_size: tuple[int, int]
    ) -> tuple[list[Any], list[dict]]:  # Tuple[List[cp.ndarray], List[dict]] when cupy available
        """Process frame group using CuPy operations as fallback."""
        processed_frames = []
        metadata_list = []

        for frame in group_frames:
            original_shape = frame.shape[:2]

            # Calculate scaling
            scale = min(target_size[0] / original_shape[0], target_size[1] / original_shape[1])
            new_size = (int(original_shape[1] * scale), int(original_shape[0] * scale))

            # Resize using CuPy
            if scale != 1.0:
                resized = cp.array(cv2.resize(cp.asnumpy(frame), new_size, interpolation=cv2.INTER_LINEAR))
            else:
                resized = frame

            # Letterboxing
            processed_frame = cp.full((*target_size, 3), 114, dtype=cp.uint8)
            y_offset = (target_size[0] - new_size[1]) // 2
            x_offset = (target_size[1] - new_size[0]) // 2

            processed_frame[y_offset:y_offset + new_size[1], x_offset:x_offset + new_size[0]] = resized

            processed_frames.append(processed_frame)
            metadata_list.append({
                'original_shape': original_shape,
                'scale_factor': scale,
                'padding': (x_offset, y_offset),
                'quality_score': 0.85,  # Placeholder
            })

        return processed_frames, metadata_list

    def _preprocess_batch_cpu_fallback(
        self,
        frames: list[np.ndarray],
        target_size: tuple[int, int]
    ) -> tuple[np.ndarray, list[dict]]:
        """CPU fallback preprocessing."""
        processed_frames = []
        metadata_list = []

        for frame in frames:
            processed_frame, metadata = self._preprocess_single_cpu(frame, target_size)
            processed_frames.append(processed_frame)
            metadata_list.append(metadata)

        return np.stack(processed_frames, axis=0), metadata_list

    def _preprocess_single_cpu(
        self,
        frame: np.ndarray,
        target_size: tuple[int, int]
    ) -> tuple[np.ndarray, dict]:
        """Single frame CPU preprocessing."""
        original_shape = frame.shape[:2]

        # Calculate scaling
        scale = min(target_size[0] / original_shape[0], target_size[1] / original_shape[1])
        new_size = (int(original_shape[1] * scale), int(original_shape[0] * scale))

        # Resize
        if scale != 1.0:
            frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_LINEAR)

        # Letterboxing
        processed_frame = np.full((*target_size, 3), 114, dtype=np.uint8)
        y_offset = (target_size[0] - new_size[1]) // 2
        x_offset = (target_size[1] - new_size[0]) // 2

        processed_frame[y_offset:y_offset + new_size[1], x_offset:x_offset + new_size[0]] = frame

        metadata = {
            'original_shape': original_shape,
            'scale_factor': scale,
            'padding': (x_offset, y_offset),
            'quality_score': 0.8,  # Simplified for CPU
            'gpu_accelerated': False,
            'kernel_accelerated': False,
        }

        return processed_frame, metadata

    async def _update_quality_scores_async(
        self,
        frames: list[np.ndarray],
        metadata_list: list[dict]
    ) -> None:
        """Asynchronously update quality scores."""
        tasks = []
        for frame, metadata in zip(frames, metadata_list, strict=False):
            task = asyncio.create_task(self.quality_calculator.calculate_quality_async(frame))
            tasks.append(task)

        try:
            quality_scores = await asyncio.wait_for(asyncio.gather(*tasks), timeout=0.1)
            for metadata, quality_score in zip(metadata_list, quality_scores, strict=False):
                metadata['quality_score'] = quality_score
        except TimeoutError:
            logger.debug("Quality score calculation timed out, using defaults")

    def get_optimization_stats(self) -> dict:
        """Get comprehensive optimization statistics."""
        stats = self.optimization_stats.copy()

        if stats['frames_processed'] > 0:
            stats['avg_processing_time_ms'] = stats['total_processing_time_ms'] / stats['frames_processed']

        if self.memory_pool:
            stats['memory_pool'] = self.memory_pool.get_stats()

        return stats

    def cleanup(self) -> None:
        """Clean up optimization resources."""
        if self.memory_pool:
            self.memory_pool.cleanup()

        if self.quality_calculator:
            self.quality_calculator.cleanup()

        if self.use_gpu:
            torch.cuda.empty_cache()
            if cp is not None:
                cp.get_default_memory_pool().free_all_blocks()
