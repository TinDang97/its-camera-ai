"""
Ultra-fast YOLO11 inference pipeline integration validator and benchmark suite.

This module validates that all optimized components work together to achieve
the <50ms end-to-end latency target for real-time camera processing.

Key Components Validated:
- YOLO11 engine with INT8 quantization and custom NMS
- CUDA preprocessor with graphs and DALI
- Adaptive batcher with 3ms timeout and priority lanes
- Streaming pipeline with async validation
- Performance monitor with P99 tracking
- Edge deployment optimizations

Performance Targets:
- End-to-end latency: <50ms (P99)
- Inference time: <10ms
- Preprocessing: <2ms
- Batch collection: <3ms
- Quality validation: <5ms (async)
- 100+ concurrent streams support
"""

import asyncio
import logging
import statistics
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .adaptive_batcher import AdaptiveBatchProcessor, BatchPriority
from .gpu_preprocessor import CUDAPreprocessor
from .ultra_fast_performance_monitor import (
    UltraFastPerformanceMonitor,
    create_performance_monitor,
)
from .ultra_fast_streaming_pipeline import (
    UltraFastStreamingPipeline,
    create_ultra_fast_pipeline,
)

# Import our optimized components
from .ultra_fast_yolo11_engine import (
    UltraFastYOLO11Engine,
    create_ultra_fast_yolo11_engine,
)

logger = logging.getLogger(__name__)


@dataclass
class ValidationResults:
    """Comprehensive validation results."""

    # Latency metrics (milliseconds)
    end_to_end_latencies: list[float] = field(default_factory=list)
    inference_latencies: list[float] = field(default_factory=list)
    preprocessing_latencies: list[float] = field(default_factory=list)
    batch_collection_latencies: list[float] = field(default_factory=list)
    quality_validation_latencies: list[float] = field(default_factory=list)

    # Throughput metrics
    total_frames_processed: int = 0
    total_processing_time: float = 0.0
    concurrent_streams_tested: int = 0

    # Performance targets
    target_p99_latency_ms: float = 50.0
    target_avg_latency_ms: float = 30.0
    target_throughput_fps: float = 30.0

    # Success metrics
    frames_meeting_targets: int = 0
    sla_violations: int = 0
    errors: int = 0

    # Component-specific metrics
    yolo_engine_performance: dict[str, Any] = field(default_factory=dict)
    preprocessor_performance: dict[str, Any] = field(default_factory=dict)
    batcher_performance: dict[str, Any] = field(default_factory=dict)
    pipeline_performance: dict[str, Any] = field(default_factory=dict)

    def add_measurement(
        self,
        end_to_end_ms: float,
        inference_ms: float = 0.0,
        preprocessing_ms: float = 0.0,
        batch_collection_ms: float = 0.0,
        quality_validation_ms: float = 0.0
    ):
        """Add a complete measurement."""
        self.end_to_end_latencies.append(end_to_end_ms)
        self.inference_latencies.append(inference_ms)
        self.preprocessing_latencies.append(preprocessing_ms)
        self.batch_collection_latencies.append(batch_collection_ms)
        self.quality_validation_latencies.append(quality_validation_ms)

        self.total_frames_processed += 1

        # Check targets
        if end_to_end_ms <= self.target_p99_latency_ms:
            self.frames_meeting_targets += 1
        else:
            self.sla_violations += 1

    def get_percentiles(self, data: list[float]) -> dict[str, float]:
        """Calculate percentiles for a dataset."""
        if not data:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "avg": 0.0, "min": 0.0, "max": 0.0}

        return {
            "p50": np.percentile(data, 50),
            "p95": np.percentile(data, 95),
            "p99": np.percentile(data, 99),
            "avg": statistics.mean(data),
            "min": min(data),
            "max": max(data)
        }

    def generate_report(self) -> dict[str, Any]:
        """Generate comprehensive performance report."""
        return {
            "validation_summary": {
                "total_frames": self.total_frames_processed,
                "frames_meeting_targets": self.frames_meeting_targets,
                "sla_violations": self.sla_violations,
                "success_rate": self.frames_meeting_targets / max(1, self.total_frames_processed),
                "error_rate": self.errors / max(1, self.total_frames_processed),
                "concurrent_streams": self.concurrent_streams_tested,
                "avg_throughput_fps": self.total_frames_processed / max(1, self.total_processing_time)
            },
            "latency_breakdown": {
                "end_to_end": self.get_percentiles(self.end_to_end_latencies),
                "inference": self.get_percentiles(self.inference_latencies),
                "preprocessing": self.get_percentiles(self.preprocessing_latencies),
                "batch_collection": self.get_percentiles(self.batch_collection_latencies),
                "quality_validation": self.get_percentiles(self.quality_validation_latencies)
            },
            "target_compliance": {
                "p99_latency_target_met": (
                    self.get_percentiles(self.end_to_end_latencies)["p99"] <= self.target_p99_latency_ms
                ),
                "avg_latency_target_met": (
                    self.get_percentiles(self.end_to_end_latencies)["avg"] <= self.target_avg_latency_ms
                ),
                "throughput_target_met": (
                    (self.total_frames_processed / max(1, self.total_processing_time)) >= self.target_throughput_fps
                ),
                "overall_targets_met": (
                    self.get_percentiles(self.end_to_end_latencies)["p99"] <= self.target_p99_latency_ms and
                    self.frames_meeting_targets / max(1, self.total_frames_processed) >= 0.95
                )
            },
            "component_performance": {
                "yolo_engine": self.yolo_engine_performance,
                "preprocessor": self.preprocessor_performance,
                "batcher": self.batcher_performance,
                "pipeline": self.pipeline_performance
            }
        }


class SyntheticTrafficGenerator:
    """Generate synthetic traffic camera frames for testing."""

    def __init__(self, width: int = 1920, height: int = 1080, fps: int = 30):
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_count = 0

        # Create base traffic scene
        self.base_scene = self._create_base_scene()

        # Vehicle templates for synthetic traffic
        self.vehicle_templates = self._create_vehicle_templates()

    def _create_base_scene(self) -> np.ndarray:
        """Create a base traffic scene background."""
        # Create road scene with lanes
        scene = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Add road surface (gray)
        scene[self.height//3:2*self.height//3, :] = [80, 80, 80]

        # Add lane markings (white)
        for y in range(self.height//3, 2*self.height//3, 60):
            for x in range(0, self.width, 100):
                cv2.rectangle(scene, (x, y), (x+40, y+5), (255, 255, 255), -1)

        # Add some static background elements
        # Traffic lights
        cv2.rectangle(scene, (self.width//4, 50), (self.width//4+30, 120), (100, 100, 100), -1)
        cv2.circle(scene, (self.width//4+15, 70), 8, (0, 255, 0), -1)  # Green light

        return scene

    def _create_vehicle_templates(self) -> list[tuple[np.ndarray, str]]:
        """Create simple vehicle templates for different vehicle types."""
        templates = []

        # Car template (blue rectangle)
        car = np.zeros((40, 80, 3), dtype=np.uint8)
        car[:, :] = [200, 100, 50]  # Blue color
        templates.append((car, "car"))

        # Truck template (larger, different color)
        truck = np.zeros((50, 120, 3), dtype=np.uint8)
        truck[:, :] = [50, 50, 200]  # Red color
        templates.append((truck, "truck"))

        # Emergency vehicle (with distinct markings)
        emergency = np.zeros((45, 90, 3), dtype=np.uint8)
        emergency[:, :] = [0, 0, 255]  # Bright red
        # Add flashing pattern
        emergency[10:15, 10:80] = [255, 255, 255]  # White stripe
        templates.append((emergency, "emergency"))

        return templates

    async def generate_frame(
        self,
        camera_id: str,
        vehicle_count: int = 5,
        include_emergency: bool = False
    ) -> tuple[np.ndarray, str, dict[str, Any]]:
        """Generate a synthetic traffic frame."""
        frame = self.base_scene.copy()
        frame_id = f"{camera_id}_frame_{self.frame_count:06d}"
        self.frame_count += 1

        # Add vehicles randomly
        vehicles_added = []
        for i in range(vehicle_count):
            template, vehicle_type = self.vehicle_templates[
                np.random.randint(0, len(self.vehicle_templates) - (0 if include_emergency else 1))
            ]

            # Random position on road
            x = np.random.randint(0, self.width - template.shape[1])
            y = np.random.randint(
                self.height//3,
                2*self.height//3 - template.shape[0]
            )

            # Add vehicle to frame
            try:
                frame[y:y+template.shape[0], x:x+template.shape[1]] = template
                vehicles_added.append({
                    "type": vehicle_type,
                    "x": x, "y": y,
                    "width": template.shape[1],
                    "height": template.shape[0]
                })
            except Exception:
                pass  # Skip if position is invalid

        # Add emergency vehicle if requested
        if include_emergency and np.random.random() < 0.3:  # 30% chance
            emergency_template, _ = self.vehicle_templates[-1]  # Emergency is last template
            x = np.random.randint(0, self.width - emergency_template.shape[1])
            y = np.random.randint(
                self.height//3,
                2*self.height//3 - emergency_template.shape[0]
            )

            try:
                frame[y:y+emergency_template.shape[0], x:x+emergency_template.shape[1]] = emergency_template
                vehicles_added.append({
                    "type": "emergency",
                    "x": x, "y": y,
                    "width": emergency_template.shape[1],
                    "height": emergency_template.shape[0]
                })
            except Exception:
                pass

        # Add some noise for realism
        noise = np.random.normal(0, 10, frame.shape).astype(np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        metadata = {
            "vehicles": vehicles_added,
            "timestamp": time.time(),
            "camera_id": camera_id,
            "frame_id": frame_id,
            "resolution": (self.width, self.height),
            "has_emergency": any(v["type"] == "emergency" for v in vehicles_added)
        }

        return frame, frame_id, metadata


class UltraFastIntegrationValidator:
    """
    Comprehensive integration validator for ultra-fast YOLO11 pipeline.
    
    Validates that all optimized components work together to meet performance targets.
    """

    def __init__(
        self,
        model_path: Path | None = None,
        device_id: int = 0,
        target_p99_latency_ms: float = 50.0,
        concurrent_streams: int = 100
    ):
        self.model_path = model_path
        self.device_id = device_id
        self.target_p99_latency_ms = target_p99_latency_ms
        self.concurrent_streams = concurrent_streams

        # Components (will be initialized)
        self.yolo_engine: UltraFastYOLO11Engine | None = None
        self.preprocessor: CUDAPreprocessor | None = None
        self.batcher: AdaptiveBatchProcessor | None = None
        self.streaming_pipeline: UltraFastStreamingPipeline | None = None
        self.performance_monitor: UltraFastPerformanceMonitor | None = None

        # Test infrastructure
        self.traffic_generator = SyntheticTrafficGenerator()
        self.validation_results = ValidationResults()

        logger.info(f"Ultra-fast integration validator initialized: "
                   f"target_p99={target_p99_latency_ms}ms, streams={concurrent_streams}")

    async def initialize_components(self) -> None:
        """Initialize all optimized components."""
        logger.info("Initializing ultra-fast pipeline components...")

        try:
            # Initialize YOLO11 engine with INT8 optimization
            self.yolo_engine = await create_ultra_fast_yolo11_engine(
                model_path=self.model_path,
                device_id=self.device_id,
                enable_int8=True,
                enable_cuda_graphs=True,
                max_batch_size=32,
                target_latency_ms=10.0
            )
            logger.info("✓ YOLO11 engine initialized with INT8 quantization")

            # Initialize CUDA preprocessor with DALI
            self.preprocessor = CUDAPreprocessor(
                input_size=(640, 640),
                device_id=self.device_id,
                max_batch_size=32,
                target_latency_ms=2.0,
                use_cuda_graphs=True,
                use_dali=True
            )
            logger.info("✓ CUDA preprocessor initialized with DALI support")

            # Initialize adaptive batcher with 3ms timeout
            async def inference_wrapper(frames, frame_ids, camera_ids):
                """Wrapper to integrate preprocessor and YOLO engine."""
                start_time = time.time()

                # Preprocess frames
                preprocessed = await self.preprocessor.preprocess_batch_gpu(frames)
                preprocess_time = (time.time() - start_time) * 1000

                # Run inference
                inference_start = time.time()
                results = await self.yolo_engine.batch_inference(preprocessed, frame_ids, camera_ids)
                inference_time = (time.time() - inference_start) * 1000

                # Update validation results with component timings
                for _ in frame_ids:
                    self.validation_results.preprocessing_latencies.append(preprocess_time / len(frame_ids))
                    self.validation_results.inference_latencies.append(inference_time / len(frame_ids))

                return results

            self.batcher = AdaptiveBatchProcessor(
                inference_func=inference_wrapper,
                max_batch_size=32,
                base_timeout_ms=3,  # Ultra-low latency
                enable_micro_batching=True,
                target_latency_ms=self.target_p99_latency_ms,
                priority_lane_timeout_ms=1
            )
            await self.batcher.start()
            logger.info("✓ Adaptive batcher started with 3ms timeout and priority lanes")

            # Initialize streaming pipeline
            self.streaming_pipeline = await create_ultra_fast_pipeline(
                inference_engine=self.yolo_engine,
                max_concurrent_streams=self.concurrent_streams,
                target_latency_ms=self.target_p99_latency_ms
            )
            logger.info("✓ Ultra-fast streaming pipeline initialized")

            # Initialize performance monitor
            self.performance_monitor = await create_performance_monitor(
                target_latency_p99_ms=self.target_p99_latency_ms,
                enable_alerting=True
            )
            logger.info("✓ Performance monitor started with P99 tracking")

            logger.info("🚀 All components initialized successfully!")

        except Exception as e:
            logger.error(f"Component initialization failed: {e}")
            raise

    async def validate_single_frame_performance(self, iterations: int = 100) -> dict[str, Any]:
        """Validate single frame processing performance."""
        logger.info(f"Validating single frame performance ({iterations} iterations)...")

        results = []

        for i in range(iterations):
            # Generate synthetic frame
            frame, frame_id, metadata = await self.traffic_generator.generate_frame(
                camera_id="test_cam_single",
                vehicle_count=5
            )

            # Measure end-to-end processing
            start_time = time.time()

            try:
                # Submit to batcher for processing
                result = await self.batcher.submit_request(
                    frame=frame,
                    frame_id=frame_id,
                    camera_id="test_cam_single",
                    priority=BatchPriority.NORMAL,
                    deadline_ms=int(self.target_p99_latency_ms)
                )

                end_to_end_time = (time.time() - start_time) * 1000
                results.append(end_to_end_time)

                # Record in performance monitor
                if self.performance_monitor:
                    self.performance_monitor.record_latency(
                        latency_ms=end_to_end_time,
                        camera_id="test_cam_single",
                        frame_id=frame_id
                    )

                # Update validation results
                self.validation_results.add_measurement(end_to_end_ms=end_to_end_time)

                if i % 20 == 0:
                    logger.debug(f"Processed frame {i+1}/{iterations}, latency: {end_to_end_time:.1f}ms")

            except Exception as e:
                logger.error(f"Single frame processing failed at iteration {i}: {e}")
                self.validation_results.errors += 1
                if self.performance_monitor:
                    self.performance_monitor.record_error(camera_id="test_cam_single")

        # Calculate statistics
        percentiles = self.validation_results.get_percentiles(results)

        single_frame_results = {
            "iterations": iterations,
            "latency_statistics": percentiles,
            "target_compliance": {
                "p99_under_target": percentiles["p99"] <= self.target_p99_latency_ms,
                "avg_under_target": percentiles["avg"] <= self.target_p99_latency_ms * 0.6,
                "success_rate": (iterations - self.validation_results.errors) / iterations
            },
            "errors": self.validation_results.errors
        }

        logger.info(f"Single frame validation completed: "
                   f"P99={percentiles['p99']:.1f}ms, "
                   f"Avg={percentiles['avg']:.1f}ms, "
                   f"Target={self.target_p99_latency_ms}ms")

        return single_frame_results

    async def validate_concurrent_streams(self, num_streams: int = 50, duration_seconds: int = 30) -> dict[str, Any]:
        """Validate concurrent stream processing performance."""
        logger.info(f"Validating concurrent streams ({num_streams} streams for {duration_seconds}s)...")

        self.validation_results.concurrent_streams_tested = num_streams

        # Create concurrent stream tasks
        stream_tasks = []
        start_time = time.time()

        for stream_id in range(num_streams):
            task = asyncio.create_task(
                self._simulate_camera_stream(
                    camera_id=f"cam_{stream_id:03d}",
                    duration_seconds=duration_seconds,
                    fps=10  # 10 FPS per stream
                )
            )
            stream_tasks.append(task)

        # Wait for all streams to complete
        try:
            await asyncio.gather(*stream_tasks, return_exceptions=True)
        except Exception as e:
            logger.error(f"Concurrent stream validation error: {e}")

        total_time = time.time() - start_time
        self.validation_results.total_processing_time = total_time

        # Collect performance metrics from all components
        concurrent_results = {
            "num_streams": num_streams,
            "duration_seconds": duration_seconds,
            "total_frames": self.validation_results.total_frames_processed,
            "overall_throughput_fps": self.validation_results.total_frames_processed / total_time,
            "latency_breakdown": {
                "end_to_end": self.validation_results.get_percentiles(self.validation_results.end_to_end_latencies)
            }
        }

        # Get component performance metrics
        if self.batcher:
            concurrent_results["batcher_metrics"] = self.batcher.get_performance_metrics()

        if self.performance_monitor:
            concurrent_results["performance_summary"] = self.performance_monitor.get_performance_summary()

        logger.info(f"Concurrent streams validation completed: "
                   f"{self.validation_results.total_frames_processed} frames, "
                   f"{concurrent_results['overall_throughput_fps']:.1f} FPS overall")

        return concurrent_results

    async def _simulate_camera_stream(self, camera_id: str, duration_seconds: int, fps: int) -> None:
        """Simulate a single camera stream."""
        frame_interval = 1.0 / fps
        end_time = time.time() + duration_seconds

        while time.time() < end_time:
            try:
                # Generate frame with occasional emergency vehicles
                include_emergency = np.random.random() < 0.1  # 10% chance
                frame, frame_id, metadata = await self.traffic_generator.generate_frame(
                    camera_id=camera_id,
                    vehicle_count=np.random.randint(3, 8),
                    include_emergency=include_emergency
                )

                # Determine priority based on content
                priority = BatchPriority.EMERGENCY if include_emergency else BatchPriority.NORMAL

                # Measure processing
                start_time = time.time()

                try:
                    if include_emergency and self.batcher.enable_micro_batching:
                        # Use priority lane for emergency vehicles
                        result = await self.batcher.submit_priority_request(
                            frame=frame,
                            frame_id=frame_id,
                            camera_id=camera_id,
                            is_emergency=True
                        )
                    else:
                        # Regular processing
                        result = await self.batcher.submit_request(
                            frame=frame,
                            frame_id=frame_id,
                            camera_id=camera_id,
                            priority=priority,
                            deadline_ms=int(self.target_p99_latency_ms * 2)  # Give some buffer for concurrent load
                        )

                    processing_time = (time.time() - start_time) * 1000

                    # Record metrics
                    if self.performance_monitor:
                        self.performance_monitor.record_latency(
                            latency_ms=processing_time,
                            camera_id=camera_id,
                            frame_id=frame_id
                        )

                    self.validation_results.add_measurement(end_to_end_ms=processing_time)

                except TimeoutError:
                    logger.warning(f"Frame {frame_id} timed out")
                    self.validation_results.errors += 1
                    if self.performance_monitor:
                        self.performance_monitor.record_error(camera_id=camera_id)

                except Exception as e:
                    logger.error(f"Stream {camera_id} processing error: {e}")
                    self.validation_results.errors += 1
                    if self.performance_monitor:
                        self.performance_monitor.record_error(camera_id=camera_id)

                # Maintain frame rate
                await asyncio.sleep(frame_interval)

            except Exception as e:
                logger.error(f"Camera stream {camera_id} error: {e}")
                break

    async def validate_priority_lane_performance(self, iterations: int = 50) -> dict[str, Any]:
        """Validate priority lane (emergency vehicle) processing performance."""
        logger.info(f"Validating priority lane performance ({iterations} emergency frames)...")

        if not self.batcher or not self.batcher.enable_micro_batching:
            return {"error": "Priority lane not enabled"}

        priority_results = []

        for i in range(iterations):
            # Generate emergency vehicle frame
            frame, frame_id, metadata = await self.traffic_generator.generate_frame(
                camera_id="emergency_test",
                vehicle_count=2,
                include_emergency=True
            )

            start_time = time.time()

            try:
                result = await self.batcher.submit_priority_request(
                    frame=frame,
                    frame_id=frame_id,
                    camera_id="emergency_test",
                    is_emergency=True
                )

                processing_time = (time.time() - start_time) * 1000
                priority_results.append(processing_time)

                if i % 10 == 0:
                    logger.debug(f"Priority frame {i+1}/{iterations}, latency: {processing_time:.1f}ms")

            except Exception as e:
                logger.error(f"Priority processing failed at iteration {i}: {e}")

        percentiles = self.validation_results.get_percentiles(priority_results)

        priority_validation = {
            "iterations": iterations,
            "latency_statistics": percentiles,
            "ultra_fast_compliance": {
                "p99_under_10ms": percentiles["p99"] <= 10.0,
                "avg_under_5ms": percentiles["avg"] <= 5.0,
                "all_under_15ms": max(priority_results) <= 15.0 if priority_results else False
            }
        }

        logger.info(f"Priority lane validation completed: "
                   f"P99={percentiles['p99']:.1f}ms, "
                   f"Avg={percentiles['avg']:.1f}ms")

        return priority_validation

    async def run_comprehensive_validation(self) -> dict[str, Any]:
        """Run complete validation suite."""
        logger.info("🚀 Starting comprehensive ultra-fast pipeline validation...")

        validation_report = {
            "validation_timestamp": time.time(),
            "target_p99_latency_ms": self.target_p99_latency_ms,
            "concurrent_streams_target": self.concurrent_streams,
            "component_versions": {
                "yolo_engine": "ultra_fast_v1.0",
                "cuda_preprocessor": "dali_enabled_v1.0",
                "adaptive_batcher": "3ms_timeout_v1.0",
                "streaming_pipeline": "async_validation_v1.0",
                "performance_monitor": "p99_tracking_v1.0"
            }
        }

        try:
            # Initialize all components
            await self.initialize_components()

            # 1. Single frame performance validation
            logger.info("\n📊 Phase 1: Single Frame Performance")
            single_frame_results = await self.validate_single_frame_performance(iterations=200)
            validation_report["single_frame_performance"] = single_frame_results

            # 2. Priority lane validation
            logger.info("\n🚨 Phase 2: Priority Lane (Emergency) Performance")
            priority_lane_results = await self.validate_priority_lane_performance(iterations=100)
            validation_report["priority_lane_performance"] = priority_lane_results

            # 3. Concurrent streams validation
            logger.info(f"\n🎥 Phase 3: Concurrent Streams ({self.concurrent_streams//2} streams)")
            concurrent_results = await self.validate_concurrent_streams(
                num_streams=self.concurrent_streams//2,  # Start with half load
                duration_seconds=30
            )
            validation_report["concurrent_streams_performance"] = concurrent_results

            # 4. Generate final validation report
            validation_report["final_results"] = self.validation_results.generate_report()

            # 5. Overall compliance assessment
            overall_compliance = self._assess_overall_compliance(validation_report)
            validation_report["overall_compliance"] = overall_compliance

            logger.info("\n✅ Comprehensive validation completed!")

        except Exception as e:
            logger.error(f"Validation failed: {e}")
            validation_report["validation_error"] = str(e)

        finally:
            # Cleanup
            await self._cleanup_components()

        return validation_report

    def _assess_overall_compliance(self, validation_report: dict[str, Any]) -> dict[str, Any]:
        """Assess overall compliance with performance targets."""

        # Extract key metrics
        single_frame = validation_report.get("single_frame_performance", {})
        priority_lane = validation_report.get("priority_lane_performance", {})
        concurrent = validation_report.get("concurrent_streams_performance", {})

        # Single frame compliance
        single_p99 = single_frame.get("latency_statistics", {}).get("p99", float('inf'))
        single_avg = single_frame.get("latency_statistics", {}).get("avg", float('inf'))

        # Priority lane compliance
        priority_p99 = priority_lane.get("latency_statistics", {}).get("p99", float('inf'))
        priority_avg = priority_lane.get("latency_statistics", {}).get("avg", float('inf'))

        # Concurrent streams compliance
        concurrent_throughput = concurrent.get("overall_throughput_fps", 0)
        concurrent_latency = concurrent.get("latency_breakdown", {}).get("end_to_end", {}).get("p99", float('inf'))

        compliance = {
            "targets_met": {
                "single_frame_p99_target": single_p99 <= self.target_p99_latency_ms,
                "single_frame_avg_target": single_avg <= self.target_p99_latency_ms * 0.6,
                "priority_lane_ultra_fast": priority_p99 <= 10.0,  # Ultra-fast target
                "concurrent_throughput": concurrent_throughput >= 200,  # Target FPS for concurrent
                "concurrent_latency": concurrent_latency <= self.target_p99_latency_ms * 1.5  # Allow some buffer
            },
            "performance_summary": {
                "single_frame_p99_ms": single_p99,
                "single_frame_avg_ms": single_avg,
                "priority_lane_p99_ms": priority_p99,
                "priority_lane_avg_ms": priority_avg,
                "concurrent_throughput_fps": concurrent_throughput,
                "concurrent_p99_ms": concurrent_latency
            }
        }

        # Overall assessment
        targets_met = list(compliance["targets_met"].values())
        compliance["overall_success"] = all(targets_met)
        compliance["targets_met_percentage"] = sum(targets_met) / len(targets_met) * 100

        # Performance grade
        if compliance["targets_met_percentage"] >= 90:
            compliance["performance_grade"] = "EXCELLENT"
        elif compliance["targets_met_percentage"] >= 80:
            compliance["performance_grade"] = "GOOD"
        elif compliance["targets_met_percentage"] >= 70:
            compliance["performance_grade"] = "ACCEPTABLE"
        else:
            compliance["performance_grade"] = "NEEDS_IMPROVEMENT"

        return compliance

    async def _cleanup_components(self) -> None:
        """Clean up all initialized components."""
        logger.info("Cleaning up components...")

        try:
            if self.batcher:
                await self.batcher.stop()

            if self.performance_monitor:
                await self.performance_monitor.stop_monitoring()

            # Other components will be garbage collected

        except Exception as e:
            logger.error(f"Cleanup error: {e}")

    @asynccontextmanager
    async def validation_context(self):
        """Async context manager for validation."""
        try:
            await self.initialize_components()
            yield self
        finally:
            await self._cleanup_components()


# Factory function
async def create_integration_validator(
    model_path: Path | None = None,
    target_p99_latency_ms: float = 50.0,
    concurrent_streams: int = 100,
    device_id: int = 0
) -> UltraFastIntegrationValidator:
    """Create integration validator with specified targets."""

    validator = UltraFastIntegrationValidator(
        model_path=model_path,
        device_id=device_id,
        target_p99_latency_ms=target_p99_latency_ms,
        concurrent_streams=concurrent_streams
    )

    return validator


# CLI interface for standalone validation
async def main():
    """Run comprehensive validation from command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Ultra-fast YOLO11 pipeline validation")
    parser.add_argument("--model-path", type=Path, help="Path to YOLO11 model")
    parser.add_argument("--target-latency", type=float, default=50.0,
                       help="Target P99 latency in ms (default: 50.0)")
    parser.add_argument("--concurrent-streams", type=int, default=100,
                       help="Number of concurrent streams to test (default: 100)")
    parser.add_argument("--device-id", type=int, default=0,
                       help="CUDA device ID (default: 0)")
    parser.add_argument("--output-report", type=Path,
                       help="Path to save validation report JSON")

    args = parser.parse_args()

    # Create validator
    validator = await create_integration_validator(
        model_path=args.model_path,
        target_p99_latency_ms=args.target_latency,
        concurrent_streams=args.concurrent_streams,
        device_id=args.device_id
    )

    # Run validation
    async with validator.validation_context():
        report = await validator.run_comprehensive_validation()

    # Print summary
    print("\n" + "="*80)
    print("🚀 ULTRA-FAST YOLO11 PIPELINE VALIDATION SUMMARY")
    print("="*80)

    compliance = report.get("overall_compliance", {})
    print(f"Performance Grade: {compliance.get('performance_grade', 'UNKNOWN')}")
    print(f"Targets Met: {compliance.get('targets_met_percentage', 0):.1f}%")
    print(f"Overall Success: {'✅ YES' if compliance.get('overall_success') else '❌ NO'}")

    performance = compliance.get("performance_summary", {})
    print("\nKey Metrics:")
    print(f"  Single Frame P99: {performance.get('single_frame_p99_ms', 0):.1f}ms (target: {args.target_latency}ms)")
    print(f"  Priority Lane P99: {performance.get('priority_lane_p99_ms', 0):.1f}ms (target: 10ms)")
    print(f"  Concurrent Throughput: {performance.get('concurrent_throughput_fps', 0):.1f} FPS")

    # Save report if requested
    if args.output_report:
        import json
        with open(args.output_report, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nDetailed report saved to: {args.output_report}")

    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())
