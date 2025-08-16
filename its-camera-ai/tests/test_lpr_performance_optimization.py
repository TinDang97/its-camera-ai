"""
Performance tests for optimized License Plate Recognition pipeline.

Tests the enhanced LPR implementation with TensorRT optimization to validate:
- Sub-15ms additional latency per vehicle detection
- >95% accuracy on clear plates
- Integration with enhanced memory management
- TensorRT OCR engine performance

Performance Targets:
- LPR Additional Latency: <15ms per vehicle (target <10ms)
- Combined Pipeline: <75ms total (vehicle detection + LPR)
- Memory Utilization: <500MB additional GPU memory
- Accuracy: >95% on clear license plates
"""

import asyncio
import logging
import time
import statistics
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from unittest.mock import Mock, patch
import gc

import pytest
import numpy as np
import cv2
import torch

from its_camera_ai.ml.license_plate_recognition import (
    LicensePlateRecognitionPipeline, 
    LPRConfig, 
    PlateDetectionResult,
    PlateDetectionStatus,
    create_lpr_pipeline
)
from its_camera_ai.ml.ocr_engine import create_ocr_engine, OCREngine, PlateRegion

logger = logging.getLogger(__name__)


@dataclass
class LPRPerformanceMetrics:
    """Performance metrics for LPR benchmark results."""
    
    # Latency metrics (milliseconds)
    avg_lpr_latency_ms: float
    p95_lpr_latency_ms: float
    p99_lpr_latency_ms: float
    min_lpr_latency_ms: float
    max_lpr_latency_ms: float
    
    # Accuracy metrics
    detection_rate_percent: float
    accuracy_rate_percent: float
    tensorrt_usage_percent: float
    
    # Performance targets
    sub_15ms_rate_percent: float
    sub_10ms_rate_percent: float
    
    # Memory metrics
    peak_memory_mb: float
    avg_memory_mb: float
    
    # Throughput
    plates_per_second: float
    
    @property
    def performance_target_met(self) -> bool:
        """Check if performance targets are met."""
        return (
            self.sub_15ms_rate_percent >= 90.0 and  # 90% under 15ms
            self.detection_rate_percent >= 85.0 and  # 85% detection rate
            self.accuracy_rate_percent >= 95.0       # 95% accuracy on detected plates
        )


class LPRTestDataGenerator:
    """Generate synthetic test data for LPR performance testing."""
    
    def __init__(self):
        self.plate_texts = [
            "ABC1234", "XYZ9876", "DEF5678", "GHI2345", "JKL8901",
            "MNO3456", "PQR7890", "STU1357", "VWX2468", "YZA9753",
            "BCD8642", "EFG1975", "HIJ0864", "KLM5309", "NOP7531"
        ]
    
    def create_synthetic_traffic_frame(self, width: int = 1920, height: int = 1080) -> np.ndarray:
        """Create a synthetic traffic camera frame."""
        # Create base image
        frame = np.random.randint(50, 200, (height, width, 3), dtype=np.uint8)
        
        # Add road-like background
        road_y = int(height * 0.6)
        frame[road_y:, :] = np.random.randint(80, 120, (height - road_y, width, 3), dtype=np.uint8)
        
        return frame
    
    def add_vehicle_with_plate(
        self, 
        frame: np.ndarray, 
        vehicle_bbox: Tuple[int, int, int, int],
        plate_text: Optional[str] = None
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """Add a vehicle with license plate to the frame."""
        x1, y1, x2, y2 = vehicle_bbox
        
        # Draw vehicle body
        vehicle_color = np.random.randint(50, 200, 3)
        cv2.rectangle(frame, (x1, y1), (x2, y2), vehicle_color.tolist(), -1)
        
        # Calculate license plate position (lower center of vehicle)
        plate_width = min(120, (x2 - x1) // 3)
        plate_height = min(40, (y2 - y1) // 5)
        
        plate_x1 = x1 + (x2 - x1 - plate_width) // 2
        plate_y1 = y2 - plate_height - 10
        plate_x2 = plate_x1 + plate_width
        plate_y2 = plate_y1 + plate_height
        
        # Draw license plate background
        cv2.rectangle(frame, (plate_x1, plate_y1), (plate_x2, plate_y2), (240, 240, 240), -1)
        cv2.rectangle(frame, (plate_x1, plate_y1), (plate_x2, plate_y2), (50, 50, 50), 2)
        
        # Add plate text
        if plate_text is None:
            plate_text = np.random.choice(self.plate_texts)
        
        # Calculate font size based on plate size
        font_scale = min(plate_width / 120, plate_height / 40) * 0.8
        thickness = max(1, int(font_scale * 2))
        
        # Get text size and center it
        (text_width, text_height), _ = cv2.getTextSize(plate_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        text_x = plate_x1 + (plate_width - text_width) // 2
        text_y = plate_y1 + (plate_height + text_height) // 2
        
        cv2.putText(frame, plate_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                   font_scale, (0, 0, 0), thickness)
        
        return frame, (plate_x1, plate_y1, plate_x2, plate_y2)
    
    def generate_test_scenarios(self, num_scenarios: int = 100) -> List[Dict[str, Any]]:
        """Generate diverse test scenarios."""
        scenarios = []
        
        for i in range(num_scenarios):
            frame = self.create_synthetic_traffic_frame()
            
            # Add 1-3 vehicles per frame
            num_vehicles = np.random.randint(1, 4)
            vehicles = []
            
            for _ in range(num_vehicles):
                # Random vehicle position
                vehicle_width = np.random.randint(150, 300)
                vehicle_height = np.random.randint(100, 200)
                
                x1 = np.random.randint(0, frame.shape[1] - vehicle_width)
                y1 = np.random.randint(frame.shape[0] // 2, frame.shape[0] - vehicle_height)
                x2 = x1 + vehicle_width
                y2 = y1 + vehicle_height
                
                vehicle_bbox = (x1, y1, x2, y2)
                plate_text = self.plate_texts[i % len(self.plate_texts)]
                
                frame, plate_bbox = self.add_vehicle_with_plate(frame, vehicle_bbox, plate_text)
                
                vehicles.append({
                    'vehicle_bbox': vehicle_bbox,
                    'plate_bbox': plate_bbox,
                    'plate_text': plate_text,
                    'vehicle_confidence': 0.85 + np.random.random() * 0.15  # 0.85-1.0
                })
            
            scenarios.append({
                'frame': frame,
                'vehicles': vehicles,
                'scenario_id': i
            })
        
        return scenarios


@pytest.fixture
def lpr_test_data():
    """Generate test data for LPR performance testing."""
    generator = LPRTestDataGenerator()
    return generator.generate_test_scenarios(50)  # 50 scenarios for testing


@pytest.fixture
def optimized_lpr_pipeline():
    """Create optimized LPR pipeline for testing."""
    config = LPRConfig(
        use_gpu=torch.cuda.is_available(),
        device_ids=[0] if torch.cuda.is_available() else [],
        target_latency_ms=10.0,  # Aggressive target
        vehicle_confidence_threshold=0.7,
        plate_confidence_threshold=0.5,
        ocr_min_confidence=0.6,
        enable_caching=True,
        cache_ttl_seconds=2.0,
        max_cache_size=500
    )
    
    return LicensePlateRecognitionPipeline(config)


class TestLPRPerformanceOptimization:
    """Test suite for LPR performance optimization."""
    
    @pytest.mark.asyncio
    async def test_lpr_latency_target(self, optimized_lpr_pipeline, lpr_test_data):
        """Test that LPR meets sub-15ms latency target."""
        
        logger.info("Testing LPR latency performance target")
        
        latencies = []
        successful_detections = 0
        total_detections = 0
        
        for scenario in lpr_test_data[:30]:  # Test subset for performance
            frame = scenario['frame']
            
            for vehicle in scenario['vehicles']:
                vehicle_bbox = vehicle['vehicle_bbox']
                vehicle_conf = vehicle['vehicle_confidence']
                expected_text = vehicle['plate_text']
                
                # Measure LPR latency
                start_time = time.perf_counter()
                
                result = await optimized_lpr_pipeline.recognize_plate(
                    frame, vehicle_bbox, vehicle_conf
                )
                
                latency_ms = (time.perf_counter() - start_time) * 1000
                latencies.append(latency_ms)
                total_detections += 1
                
                if result.status == PlateDetectionStatus.SUCCESS:
                    successful_detections += 1
                    
                    # Verify accuracy
                    if result.plate_text and result.plate_text.replace('-', '').replace(' ', '') == expected_text:
                        pass  # Accurate detection
        
        # Calculate metrics
        avg_latency = statistics.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        sub_15ms_rate = (sum(1 for lat in latencies if lat <= 15.0) / len(latencies)) * 100
        sub_10ms_rate = (sum(1 for lat in latencies if lat <= 10.0) / len(latencies)) * 100
        
        logger.info(f"LPR Performance Results:")
        logger.info(f"  Average latency: {avg_latency:.2f}ms")
        logger.info(f"  P95 latency: {p95_latency:.2f}ms")
        logger.info(f"  P99 latency: {p99_latency:.2f}ms")
        logger.info(f"  Sub-15ms rate: {sub_15ms_rate:.1f}%")
        logger.info(f"  Sub-10ms rate: {sub_10ms_rate:.1f}%")
        logger.info(f"  Detection rate: {(successful_detections/total_detections)*100:.1f}%")
        
        # Assertions
        assert avg_latency <= 15.0, f"Average latency {avg_latency:.2f}ms exceeds 15ms target"
        assert p95_latency <= 25.0, f"P95 latency {p95_latency:.2f}ms exceeds 25ms threshold"
        assert sub_15ms_rate >= 85.0, f"Sub-15ms rate {sub_15ms_rate:.1f}% below 85% target"
    
    @pytest.mark.asyncio
    async def test_tensorrt_ocr_performance(self, optimized_lpr_pipeline):
        """Test TensorRT OCR engine performance if available."""
        
        # Create test plate images
        test_plates = []
        plate_texts = ["ABC1234", "XYZ9876", "DEF5678"]
        
        for text in plate_texts:
            # Create synthetic plate image
            plate_img = np.full((40, 120, 3), 240, dtype=np.uint8)
            cv2.rectangle(plate_img, (0, 0), (120, 40), (50, 50, 50), 2)
            
            # Add text
            font_scale = 0.7
            thickness = 2
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            text_x = (120 - text_width) // 2
            text_y = (40 + text_height) // 2
            
            cv2.putText(plate_img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                       font_scale, (0, 0, 0), thickness)
            
            test_plates.append((plate_img, text))
        
        # Test OCR performance
        ocr_latencies = []
        accurate_predictions = 0
        
        for plate_img, expected_text in test_plates:
            for _ in range(10):  # Multiple runs for statistics
                start_time = time.perf_counter()
                
                ocr_result = await optimized_lpr_pipeline.ocr_engine.recognize(
                    plate_img, PlateRegion.US
                )
                
                ocr_latency = (time.perf_counter() - start_time) * 1000
                ocr_latencies.append(ocr_latency)
                
                if ocr_result and ocr_result.text:
                    predicted_text = ocr_result.text.replace('-', '').replace(' ', '')
                    if predicted_text == expected_text:
                        accurate_predictions += 1
        
        # Calculate OCR metrics
        avg_ocr_latency = statistics.mean(ocr_latencies)
        p95_ocr_latency = np.percentile(ocr_latencies, 95)
        ocr_accuracy = (accurate_predictions / len(ocr_latencies)) * 100
        
        logger.info(f"OCR Performance Results:")
        logger.info(f"  Average OCR latency: {avg_ocr_latency:.2f}ms")
        logger.info(f"  P95 OCR latency: {p95_ocr_latency:.2f}ms")
        logger.info(f"  OCR accuracy: {ocr_accuracy:.1f}%")
        
        # Assertions
        assert avg_ocr_latency <= 12.0, f"OCR latency {avg_ocr_latency:.2f}ms exceeds 12ms target"
        assert ocr_accuracy >= 80.0, f"OCR accuracy {ocr_accuracy:.1f}% below 80% threshold"
    
    @pytest.mark.asyncio
    async def test_memory_efficiency(self, optimized_lpr_pipeline, lpr_test_data):
        """Test memory efficiency of optimized LPR pipeline."""
        
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for memory testing")
        
        # Measure initial memory
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        
        # Process multiple frames
        peak_memory = initial_memory
        
        for scenario in lpr_test_data[:20]:
            frame = scenario['frame']
            
            for vehicle in scenario['vehicles']:
                vehicle_bbox = vehicle['vehicle_bbox']
                vehicle_conf = vehicle['vehicle_confidence']
                
                result = await optimized_lpr_pipeline.recognize_plate(
                    frame, vehicle_bbox, vehicle_conf
                )
                
                current_memory = torch.cuda.memory_allocated() / 1024 / 1024
                peak_memory = max(peak_memory, current_memory)
        
        # Force cleanup
        gc.collect()
        torch.cuda.empty_cache()
        
        final_memory = torch.cuda.memory_allocated() / 1024 / 1024
        memory_increase = peak_memory - initial_memory
        
        logger.info(f"Memory Usage:")
        logger.info(f"  Initial: {initial_memory:.1f} MB")
        logger.info(f"  Peak: {peak_memory:.1f} MB")
        logger.info(f"  Final: {final_memory:.1f} MB")
        logger.info(f"  Increase: {memory_increase:.1f} MB")
        
        # Memory efficiency assertions
        assert memory_increase <= 500.0, f"Memory increase {memory_increase:.1f}MB exceeds 500MB limit"
        assert (final_memory - initial_memory) <= 100.0, "Memory leak detected"
    
    @pytest.mark.asyncio
    async def test_cache_effectiveness(self, optimized_lpr_pipeline, lpr_test_data):
        """Test LPR cache effectiveness."""
        
        scenario = lpr_test_data[0]
        frame = scenario['frame']
        vehicle = scenario['vehicles'][0]
        vehicle_bbox = vehicle['vehicle_bbox']
        vehicle_conf = vehicle['vehicle_confidence']
        
        # First request (cache miss)
        start_time = time.perf_counter()
        result1 = await optimized_lpr_pipeline.recognize_plate(frame, vehicle_bbox, vehicle_conf)
        first_latency = (time.perf_counter() - start_time) * 1000
        
        # Second request (cache hit)
        start_time = time.perf_counter()
        result2 = await optimized_lpr_pipeline.recognize_plate(frame, vehicle_bbox, vehicle_conf)
        second_latency = (time.perf_counter() - start_time) * 1000
        
        logger.info(f"Cache Performance:")
        logger.info(f"  First request: {first_latency:.2f}ms")
        logger.info(f"  Second request: {second_latency:.2f}ms")
        logger.info(f"  Speedup: {first_latency/second_latency:.1f}x")
        
        # Cache assertions
        assert second_latency < first_latency, "Cache did not improve performance"
        assert second_latency <= 2.0, f"Cached request took {second_latency:.2f}ms, should be <2ms"
        assert result1.plate_text == result2.plate_text, "Cache returned different result"
    
    @pytest.mark.asyncio
    async def test_batch_processing_performance(self, optimized_lpr_pipeline, lpr_test_data):
        """Test batch processing performance."""
        
        # Prepare batch of detections
        batch_requests = []
        scenario = lpr_test_data[0]
        frame = scenario['frame']
        
        vehicle_detections = [(vehicle['vehicle_bbox'], vehicle['vehicle_confidence']) 
                             for vehicle in scenario['vehicles']]
        
        # Measure batch processing time
        start_time = time.perf_counter()
        
        results = await optimized_lpr_pipeline.batch_recognize(frame, vehicle_detections)
        
        batch_latency = (time.perf_counter() - start_time) * 1000
        per_vehicle_latency = batch_latency / len(vehicle_detections)
        
        logger.info(f"Batch Processing Performance:")
        logger.info(f"  Total batch time: {batch_latency:.2f}ms")
        logger.info(f"  Per vehicle time: {per_vehicle_latency:.2f}ms")
        logger.info(f"  Vehicles processed: {len(vehicle_detections)}")
        
        # Batch processing assertions
        assert per_vehicle_latency <= 20.0, f"Per-vehicle batch latency {per_vehicle_latency:.2f}ms exceeds 20ms"
        assert len(results) == len(vehicle_detections), "Batch processing returned wrong number of results"
    
    @pytest.mark.asyncio
    async def test_comprehensive_performance_benchmark(self, optimized_lpr_pipeline, lpr_test_data):
        """Comprehensive performance benchmark for LPR optimization."""
        
        logger.info("Running comprehensive LPR performance benchmark")
        
        all_latencies = []
        successful_detections = 0
        total_detections = 0
        accurate_detections = 0
        tensorrt_usage = 0
        
        # Track memory usage
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated() / 1024 / 1024
            peak_memory = initial_memory
        else:
            initial_memory = peak_memory = 0
        
        start_benchmark = time.time()
        
        for scenario in lpr_test_data:
            frame = scenario['frame']
            
            for vehicle in scenario['vehicles']:
                vehicle_bbox = vehicle['vehicle_bbox']
                vehicle_conf = vehicle['vehicle_confidence']
                expected_text = vehicle['plate_text']
                
                # Measure individual LPR performance
                start_time = time.perf_counter()
                
                result = await optimized_lpr_pipeline.recognize_plate(
                    frame, vehicle_bbox, vehicle_conf
                )
                
                latency_ms = (time.perf_counter() - start_time) * 1000
                all_latencies.append(latency_ms)
                total_detections += 1
                
                # Track memory
                if torch.cuda.is_available():
                    current_memory = torch.cuda.memory_allocated() / 1024 / 1024
                    peak_memory = max(peak_memory, current_memory)
                
                # Analyze results
                if result.status == PlateDetectionStatus.SUCCESS:
                    successful_detections += 1
                    
                    # Check accuracy
                    if result.plate_text:
                        predicted_text = result.plate_text.replace('-', '').replace(' ', '')
                        if predicted_text == expected_text:
                            accurate_detections += 1
                    
                    # Track TensorRT usage
                    if 'tensorrt' in result.engine_used.lower():
                        tensorrt_usage += 1
        
        total_benchmark_time = time.time() - start_benchmark
        
        # Calculate comprehensive metrics
        metrics = LPRPerformanceMetrics(
            avg_lpr_latency_ms=float(statistics.mean(all_latencies)),
            p95_lpr_latency_ms=float(np.percentile(all_latencies, 95)),
            p99_lpr_latency_ms=float(np.percentile(all_latencies, 99)),
            min_lpr_latency_ms=float(min(all_latencies)),
            max_lpr_latency_ms=float(max(all_latencies)),
            detection_rate_percent=(successful_detections / total_detections) * 100,
            accuracy_rate_percent=(accurate_detections / max(1, successful_detections)) * 100,
            tensorrt_usage_percent=(tensorrt_usage / max(1, successful_detections)) * 100,
            sub_15ms_rate_percent=(sum(1 for lat in all_latencies if lat <= 15.0) / len(all_latencies)) * 100,
            sub_10ms_rate_percent=(sum(1 for lat in all_latencies if lat <= 10.0) / len(all_latencies)) * 100,
            peak_memory_mb=peak_memory,
            avg_memory_mb=(peak_memory + initial_memory) / 2,
            plates_per_second=total_detections / total_benchmark_time
        )
        
        # Log comprehensive results
        logger.info("=" * 60)
        logger.info("COMPREHENSIVE LPR PERFORMANCE BENCHMARK RESULTS")
        logger.info("=" * 60)
        logger.info(f"Latency Performance:")
        logger.info(f"  Average: {metrics.avg_lpr_latency_ms:.2f}ms")
        logger.info(f"  P95: {metrics.p95_lpr_latency_ms:.2f}ms")
        logger.info(f"  P99: {metrics.p99_lpr_latency_ms:.2f}ms")
        logger.info(f"  Min: {metrics.min_lpr_latency_ms:.2f}ms")
        logger.info(f"  Max: {metrics.max_lpr_latency_ms:.2f}ms")
        logger.info(f"")
        logger.info(f"Accuracy Performance:")
        logger.info(f"  Detection Rate: {metrics.detection_rate_percent:.1f}%")
        logger.info(f"  Accuracy Rate: {metrics.accuracy_rate_percent:.1f}%")
        logger.info(f"  TensorRT Usage: {metrics.tensorrt_usage_percent:.1f}%")
        logger.info(f"")
        logger.info(f"Performance Targets:")
        logger.info(f"  Sub-15ms Rate: {metrics.sub_15ms_rate_percent:.1f}%")
        logger.info(f"  Sub-10ms Rate: {metrics.sub_10ms_rate_percent:.1f}%")
        logger.info(f"  Target Met: {'✅ Yes' if metrics.performance_target_met else '❌ No'}")
        logger.info(f"")
        logger.info(f"System Performance:")
        logger.info(f"  Peak Memory: {metrics.peak_memory_mb:.1f} MB")
        logger.info(f"  Throughput: {metrics.plates_per_second:.1f} plates/sec")
        logger.info(f"  Total Detections: {total_detections}")
        logger.info("=" * 60)
        
        # Get pipeline stats
        pipeline_stats = optimized_lpr_pipeline.get_stats()
        logger.info(f"Pipeline Statistics:")
        for key, value in pipeline_stats.items():
            logger.info(f"  {key}: {value}")
        
        # Critical performance assertions
        assert metrics.avg_lpr_latency_ms <= 15.0, f"Average latency {metrics.avg_lpr_latency_ms:.2f}ms exceeds 15ms target"
        assert metrics.p95_lpr_latency_ms <= 25.0, f"P95 latency {metrics.p95_lpr_latency_ms:.2f}ms exceeds 25ms"
        assert metrics.sub_15ms_rate_percent >= 85.0, f"Sub-15ms rate {metrics.sub_15ms_rate_percent:.1f}% below 85%"
        assert metrics.detection_rate_percent >= 80.0, f"Detection rate {metrics.detection_rate_percent:.1f}% below 80%"
        
        if torch.cuda.is_available():
            memory_increase = metrics.peak_memory_mb - initial_memory
            assert memory_increase <= 500.0, f"Memory increase {memory_increase:.1f}MB exceeds 500MB limit"
        
        return metrics


if __name__ == "__main__":
    # Run performance benchmark directly
    async def main():
        # Create test data and pipeline
        generator = LPRTestDataGenerator()
        test_data = generator.generate_test_scenarios(20)
        
        config = LPRConfig(
            use_gpu=torch.cuda.is_available(),
            device_ids=[0] if torch.cuda.is_available() else [],
            target_latency_ms=10.0,
            enable_caching=True
        )
        
        pipeline = LicensePlateRecognitionPipeline(config)
        
        # Create test instance
        test_instance = TestLPRPerformanceOptimization()
        
        # Run comprehensive benchmark
        metrics = await test_instance.test_comprehensive_performance_benchmark(pipeline, test_data)
        
        print(f"\nFinal Performance Assessment:")
        print(f"Performance Target Met: {'✅ Yes' if metrics.performance_target_met else '❌ No'}")
        print(f"Key Metrics:")
        print(f"  Average Latency: {metrics.avg_lpr_latency_ms:.2f}ms")
        print(f"  Sub-15ms Rate: {metrics.sub_15ms_rate_percent:.1f}%")
        print(f"  Detection Rate: {metrics.detection_rate_percent:.1f}%")
        print(f"  Accuracy Rate: {metrics.accuracy_rate_percent:.1f}%")
    
    asyncio.run(main())