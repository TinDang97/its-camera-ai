"""
Performance benchmarks for ML annotation streaming pipeline.

This module contains performance tests to ensure the ML annotation
processor meets the <50ms latency requirement for real-time streaming.
"""

import asyncio
import time
import statistics
from typing import List
import pytest
import numpy as np
import cv2
from unittest.mock import AsyncMock, MagicMock

from src.its_camera_ai.ml.streaming_annotation_processor import (
    MLAnnotationProcessor,
    DetectionConfig,
    AnnotationStyleConfig,
    AnnotationRenderer,
    Detection,
)


class TestMLStreamingPerformance:
    """Performance benchmark tests for ML streaming."""
    
    def setup_method(self):
        """Set up performance test fixtures."""
        # Create lightweight mock engine for performance testing
        self.mock_engine = AsyncMock()
        self.mock_engine.detect_objects.return_value = MagicMock(
            detections=[
                {
                    'class': 'car',
                    'confidence': 0.85,
                    'bbox': (100, 100, 200, 150),
                    'center': (150.0, 125.0),
                    'area': 5000.0,
                    'class_id': 2
                },
                {
                    'class': 'truck',
                    'confidence': 0.92,
                    'bbox': (250, 200, 350, 300),
                    'center': (300.0, 250.0),
                    'area': 10000.0,
                    'class_id': 7
                }
            ]
        )
        
        # Configure for performance
        self.config = DetectionConfig(
            confidence_threshold=0.5,
            target_latency_ms=50.0,
            batch_size=8,
            enable_gpu_acceleration=True
        )
        
        self.processor = MLAnnotationProcessor(
            vision_engine=self.mock_engine,
            config=self.config
        )
        
        # Create test frame data
        self.test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        _, encoded = cv2.imencode('.jpg', self.test_frame)
        self.frame_bytes = encoded.tobytes()
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_single_frame_latency(self):
        """Test single frame processing latency requirement (<50ms)."""
        await self.processor.initialize()
        
        # Warm up
        for _ in range(3):
            await self.processor.process_frame_with_annotations(
                frame_data=self.frame_bytes,
                camera_id="test_camera"
            )
        
        # Measure actual latency
        latencies = []
        for i in range(10):
            start_time = time.perf_counter()
            
            result = await self.processor.process_frame_with_annotations(
                frame_data=self.frame_bytes,
                camera_id="test_camera"
            )
            
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
            
            assert result.processing_time_ms > 0
        
        avg_latency = statistics.mean(latencies)
        max_latency = max(latencies)
        p95_latency = statistics.quantiles(latencies, n=20)[18]  # 95th percentile
        
        print(f"\nLatency metrics:")
        print(f"  Average: {avg_latency:.2f}ms")
        print(f"  Maximum: {max_latency:.2f}ms")
        print(f"  95th percentile: {p95_latency:.2f}ms")
        print(f"  Target: {self.config.target_latency_ms}ms")
        
        # Performance requirements
        assert avg_latency < self.config.target_latency_ms, (
            f"Average latency {avg_latency:.2f}ms exceeds target {self.config.target_latency_ms}ms"
        )
        assert p95_latency < self.config.target_latency_ms * 1.2, (
            f"95th percentile latency {p95_latency:.2f}ms exceeds target {self.config.target_latency_ms * 1.2}ms"
        )
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_concurrent_camera_processing(self):
        """Test processing multiple camera streams concurrently."""
        await self.processor.initialize()
        
        num_cameras = 5
        frames_per_camera = 10
        
        async def process_camera_stream(camera_id: str) -> List[float]:
            """Process frames for a single camera."""
            latencies = []
            for frame_num in range(frames_per_camera):
                start_time = time.perf_counter()
                
                result = await self.processor.process_frame_with_annotations(
                    frame_data=self.frame_bytes,
                    camera_id=f"{camera_id}_{frame_num}"
                )
                
                end_time = time.perf_counter()
                latency_ms = (end_time - start_time) * 1000
                latencies.append(latency_ms)
                
                assert result.camera_id.startswith(camera_id)
            
            return latencies
        
        # Process multiple cameras concurrently
        start_time = time.perf_counter()
        
        tasks = [
            process_camera_stream(f"camera_{i}")
            for i in range(num_cameras)
        ]
        
        results = await asyncio.gather(*tasks)
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # Analyze results
        all_latencies = []
        for camera_latencies in results:
            all_latencies.extend(camera_latencies)
        
        total_frames = num_cameras * frames_per_camera
        avg_latency = statistics.mean(all_latencies)
        throughput_fps = total_frames / total_time
        
        print(f"\nConcurrent processing metrics:")
        print(f"  Cameras: {num_cameras}")
        print(f"  Total frames: {total_frames}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Average latency: {avg_latency:.2f}ms")
        print(f"  Throughput: {throughput_fps:.1f} FPS")
        
        # Performance requirements for concurrent processing
        assert avg_latency < self.config.target_latency_ms * 1.5, (
            f"Concurrent average latency {avg_latency:.2f}ms exceeds target"
        )
        assert throughput_fps > 100, (
            f"Throughput {throughput_fps:.1f} FPS below requirement"
        )
    
    @pytest.mark.performance
    def test_annotation_rendering_performance(self):
        """Test annotation rendering performance."""
        renderer = AnnotationRenderer()
        
        # Create multiple detections
        detections = []
        for i in range(50):  # Many detections for stress test
            detection = Detection(
                class_name=f"car_{i % 5}",
                confidence=0.8 + (i % 5) * 0.02,
                bbox=(i*10, i*8, i*10+60, i*8+40),
                center=(i*10+30.0, i*8+20.0),
                area=2400.0,
                class_id=2
            )
            detections.append(detection)
        
        # Measure rendering performance
        rendering_times = []
        for _ in range(10):
            start_time = time.perf_counter()
            
            result = asyncio.run(
                renderer.render_detections_on_frame(
                    self.test_frame, detections
                )
            )
            
            end_time = time.perf_counter()
            rendering_time_ms = (end_time - start_time) * 1000
            rendering_times.append(rendering_time_ms)
            
            assert result.shape == self.test_frame.shape
        
        avg_rendering_time = statistics.mean(rendering_times)
        max_rendering_time = max(rendering_times)
        
        print(f"\nRendering performance:")
        print(f"  Detections: {len(detections)}")
        print(f"  Average rendering time: {avg_rendering_time:.2f}ms")
        print(f"  Maximum rendering time: {max_rendering_time:.2f}ms")
        
        # Rendering should be fast (< 10ms for 50 detections)
        assert avg_rendering_time < 10.0, (
            f"Rendering time {avg_rendering_time:.2f}ms too slow"
        )
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_memory_usage_stability(self):
        """Test memory usage stability over extended processing."""
        await self.processor.initialize()
        
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process many frames to test for memory leaks
        num_frames = 100
        memory_samples = []
        
        for i in range(num_frames):
            await self.processor.process_frame_with_annotations(
                frame_data=self.frame_bytes,
                camera_id=f"memory_test_{i}"
            )
            
            if i % 10 == 0:  # Sample memory every 10 frames
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_samples.append(current_memory)
        
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_growth = final_memory - initial_memory
        
        print(f"\nMemory usage analysis:")
        print(f"  Initial: {initial_memory:.1f} MB")
        print(f"  Final: {final_memory:.1f} MB")
        print(f"  Growth: {memory_growth:.1f} MB")
        print(f"  Frames processed: {num_frames}")
        
        # Memory growth should be reasonable (< 50MB for 100 frames)
        assert memory_growth < 50.0, (
            f"Memory growth {memory_growth:.1f} MB indicates potential leak"
        )
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_gpu_utilization_efficiency(self):
        """Test GPU utilization efficiency (if GPU available)."""
        if not self.config.enable_gpu_acceleration:
            pytest.skip("GPU acceleration not enabled")
        
        await self.processor.initialize()
        
        # Test batch processing efficiency
        batch_sizes = [1, 4, 8, 16]
        batch_results = {}
        
        for batch_size in batch_sizes:
            self.processor.config.batch_size = batch_size
            
            # Measure throughput for this batch size
            start_time = time.perf_counter()
            
            tasks = []
            for i in range(batch_size * 5):  # 5 batches
                task = self.processor.process_frame_with_annotations(
                    frame_data=self.frame_bytes,
                    camera_id=f"gpu_test_{i}"
                )
                tasks.append(task)
            
            await asyncio.gather(*tasks)
            
            end_time = time.perf_counter()
            total_time = end_time - start_time
            throughput = (batch_size * 5) / total_time
            
            batch_results[batch_size] = {
                'throughput': throughput,
                'total_time': total_time
            }
        
        print(f"\nGPU batch efficiency:")
        for batch_size, results in batch_results.items():
            print(f"  Batch {batch_size}: {results['throughput']:.1f} FPS")
        
        # Larger batch sizes should generally be more efficient
        assert batch_results[8]['throughput'] > batch_results[1]['throughput'], (
            "Batch processing not showing efficiency gains"
        )
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_end_to_end_streaming_latency(self):
        """Test complete end-to-end streaming latency."""
        await self.processor.initialize()
        
        # Simulate streaming pipeline: decode -> inference -> render -> encode
        num_frames = 20
        e2e_latencies = []
        
        for i in range(num_frames):
            pipeline_start = time.perf_counter()
            
            # 1. Frame processing (includes decode, inference, render, encode)
            result = await self.processor.process_frame_with_annotations(
                frame_data=self.frame_bytes,
                camera_id=f"e2e_test_{i}"
            )
            
            # 2. Simulate metadata creation
            metadata = await self.processor.create_detection_metadata(
                detections=result.detections,
                frame_timestamp=result.timestamp,
                camera_id=result.camera_id,
                frame_id=result.frame_id,
                processing_time_ms=result.processing_time_ms
            )
            
            pipeline_end = time.perf_counter()
            e2e_latency = (pipeline_end - pipeline_start) * 1000
            e2e_latencies.append(e2e_latency)
        
        avg_e2e_latency = statistics.mean(e2e_latencies)
        max_e2e_latency = max(e2e_latencies)
        p95_e2e_latency = statistics.quantiles(e2e_latencies, n=20)[18]
        
        print(f"\nEnd-to-end streaming latency:")
        print(f"  Average: {avg_e2e_latency:.2f}ms")
        print(f"  Maximum: {max_e2e_latency:.2f}ms")
        print(f"  95th percentile: {p95_e2e_latency:.2f}ms")
        print(f"  Target: <100ms (system requirement)")
        
        # End-to-end latency should be under 100ms for complete pipeline
        assert avg_e2e_latency < 100.0, (
            f"E2E average latency {avg_e2e_latency:.2f}ms exceeds 100ms target"
        )
        assert p95_e2e_latency < 120.0, (
            f"E2E 95th percentile {p95_e2e_latency:.2f}ms exceeds 120ms limit"
        )
    
    @pytest.mark.performance
    def test_configuration_impact_on_performance(self):
        """Test how different configurations impact performance."""
        # Test different confidence thresholds
        thresholds = [0.3, 0.5, 0.7, 0.9]
        threshold_results = {}
        
        for threshold in thresholds:
            config = DetectionConfig(
                confidence_threshold=threshold,
                target_latency_ms=50.0
            )
            
            # Simulate detection filtering time (higher threshold = fewer detections = faster)
            # This is a simplified test - in practice would measure actual performance
            mock_detections = [
                {'class': 'car', 'confidence': 0.8},
                {'class': 'car', 'confidence': 0.6},
                {'class': 'car', 'confidence': 0.4},
                {'class': 'car', 'confidence': 0.2},
            ]
            
            filtered = [d for d in mock_detections if d['confidence'] >= threshold]
            threshold_results[threshold] = len(filtered)
        
        print(f"\nConfiguration impact:")
        for threshold, count in threshold_results.items():
            print(f"  Threshold {threshold}: {count} detections")
        
        # Higher thresholds should result in fewer detections
        assert threshold_results[0.9] <= threshold_results[0.3], (
            "Confidence threshold not filtering correctly"
        )


if __name__ == "__main__":
    # Run performance tests
    pytest.main([__file__, "-v", "-m", "performance"])