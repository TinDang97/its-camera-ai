"""
Core Computer Vision Engine Demonstration.

This example demonstrates the complete functionality of the Core Computer Vision Engine
for real-time traffic monitoring, including model optimization, batch processing,
performance monitoring, and ML pipeline integration.

Run this script to see the engine in action with synthetic traffic data.
"""

import asyncio
import logging
import time

import cv2
import numpy as np

# Import our vision components
from its_camera_ai.ml.core_vision_engine import (
    CoreVisionEngine,
    ModelType,
    VisionConfig,
    benchmark_engine,
    create_optimal_config,
)
from its_camera_ai.ml.inference_optimizer import OptimizationBackend

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def generate_synthetic_traffic_scene(
    width: int = 640,
    height: int = 640,
    num_vehicles: int = None,
) -> np.ndarray:
    """
    Generate synthetic traffic scene for demonstration.

    Creates realistic-looking traffic scenes with vehicle-like objects
    for testing the computer vision pipeline.
    """
    if num_vehicles is None:
        num_vehicles = np.random.randint(2, 8)

    # Create base road scene
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    # Add road surface (dark gray)
    road_color = (50, 50, 50)
    frame[:, :] = road_color

    # Add road markings (white lines)
    # Center divider
    cv2.line(frame, (width // 2, 0), (width // 2, height), (255, 255, 255), 3)

    # Lane markings
    for y in range(0, height, 60):
        cv2.line(frame, (width // 4, y), (width // 4, y + 30), (255, 255, 255), 2)
        cv2.line(
            frame, (3 * width // 4, y), (3 * width // 4, y + 30), (255, 255, 255), 2
        )

    # Add vehicles with realistic colors and sizes
    vehicle_colors = [
        (220, 220, 220),  # White
        (100, 100, 100),  # Gray
        (20, 20, 150),  # Red
        (150, 20, 20),  # Blue
        (20, 150, 20),  # Green
        (0, 0, 0),  # Black
    ]

    vehicles_added = 0
    attempts = 0
    max_attempts = 50

    placed_vehicles = []

    while vehicles_added < num_vehicles and attempts < max_attempts:
        attempts += 1

        # Random vehicle properties
        vehicle_width = np.random.randint(80, 140)
        vehicle_height = np.random.randint(40, 70)

        # Position in driving lanes
        lane_centers = [width // 4, 3 * width // 4]
        lane_center = np.random.choice(lane_centers)
        vehicle_x = lane_center - vehicle_width // 2 + np.random.randint(-20, 20)
        vehicle_y = np.random.randint(50, height - vehicle_height - 50)

        # Check for overlap with existing vehicles
        overlap = False
        for existing in placed_vehicles:
            if (
                abs(vehicle_x - existing["x"]) < vehicle_width + 20
                and abs(vehicle_y - existing["y"]) < vehicle_height + 20
            ):
                overlap = True
                break

        if overlap:
            continue

        # Choose vehicle color
        color = np.random.choice(vehicle_colors)

        # Draw main vehicle body (rectangle)
        cv2.rectangle(
            frame,
            (vehicle_x, vehicle_y),
            (vehicle_x + vehicle_width, vehicle_y + vehicle_height),
            color,
            -1,
        )

        # Add some vehicle details
        # Front/rear lights
        if np.random.random() > 0.5:  # Front lights
            cv2.circle(frame, (vehicle_x + 10, vehicle_y + 10), 3, (255, 255, 200), -1)
            cv2.circle(
                frame,
                (vehicle_x + vehicle_width - 10, vehicle_y + 10),
                3,
                (255, 255, 200),
                -1,
            )
        else:  # Rear lights
            cv2.circle(
                frame,
                (vehicle_x + 10, vehicle_y + vehicle_height - 10),
                3,
                (0, 0, 200),
                -1,
            )
            cv2.circle(
                frame,
                (vehicle_x + vehicle_width - 10, vehicle_y + vehicle_height - 10),
                3,
                (0, 0, 200),
                -1,
            )

        # Add slight border for realism
        cv2.rectangle(
            frame,
            (vehicle_x, vehicle_y),
            (vehicle_x + vehicle_width, vehicle_y + vehicle_height),
            (max(0, color[0] - 30), max(0, color[1] - 30), max(0, color[2] - 30)),
            2,
        )

        # Record vehicle placement
        placed_vehicles.append(
            {
                "x": vehicle_x,
                "y": vehicle_y,
                "width": vehicle_width,
                "height": vehicle_height,
            }
        )

        vehicles_added += 1

    # Add some visual noise and details
    # Random spots and imperfections
    for _ in range(np.random.randint(10, 30)):
        x = np.random.randint(0, width)
        y = np.random.randint(0, height)
        noise_color = np.random.randint(0, 100, 3)
        cv2.circle(
            frame, (x, y), np.random.randint(1, 3), tuple(noise_color.tolist()), -1
        )

    # Add slight Gaussian blur for realism
    frame = cv2.GaussianBlur(frame, (3, 3), 0.5)

    return frame


async def demo_basic_usage():
    """Demonstrate basic Core Vision Engine usage."""
    logger.info("=== Core Vision Engine Basic Demo ===")

    # 1. Create configuration for production scenario
    config = create_optimal_config(
        deployment_scenario="production", available_memory_gb=8.0, target_cameras=4
    )

    logger.info(
        f"Configuration: {config.model_type.value} model, batch_size={config.batch_size}"
    )

    # 2. Initialize engine
    engine = CoreVisionEngine(config)
    await engine.initialize()

    logger.info("Engine initialized successfully")

    # 3. Process single frame
    logger.info("Processing single frame...")
    frame = generate_synthetic_traffic_scene()

    start_time = time.time()
    result = await engine.process_frame(frame, "demo_frame_001", "demo_camera")
    processing_time = (time.time() - start_time) * 1000

    logger.info("Single frame results:")
    logger.info(f"  - Detected vehicles: {result.detection_count}")
    logger.info(f"  - Average confidence: {result.avg_confidence:.3f}")
    logger.info(f"  - Processing time: {processing_time:.1f}ms")
    logger.info(f"  - Engine latency: {result.total_processing_time_ms:.1f}ms")

    # 4. Process batch of frames
    logger.info("Processing batch of frames...")
    batch_size = 4
    batch_frames = [generate_synthetic_traffic_scene() for _ in range(batch_size)]
    frame_ids = [f"demo_batch_{i}" for i in range(batch_size)]
    camera_ids = [
        f"camera_{i%2}" for i in range(batch_size)
    ]  # Alternate between 2 cameras

    batch_start = time.time()
    batch_results = await engine.process_batch(batch_frames, frame_ids, camera_ids)
    batch_time = (time.time() - batch_start) * 1000

    logger.info("Batch processing results:")
    logger.info(f"  - Frames processed: {len(batch_results)}")
    logger.info(f"  - Total batch time: {batch_time:.1f}ms")
    logger.info(f"  - Average per frame: {batch_time/len(batch_results):.1f}ms")
    logger.info(f"  - Batch throughput: {len(batch_results)/(batch_time/1000):.1f} FPS")

    # Show individual results
    for i, result in enumerate(batch_results):
        logger.info(
            f"    Frame {i}: {result.detection_count} vehicles, "
            f"{result.avg_confidence:.3f} confidence, "
            f"{result.total_processing_time_ms:.1f}ms"
        )

    # 5. Get performance metrics
    logger.info("Performance metrics:")
    metrics = engine.get_performance_metrics()

    if metrics.get("performance"):
        perf = metrics["performance"]
        if perf.get("latency"):
            logger.info(
                f"  - Average latency: {perf['latency'].get('avg_ms', 0):.1f}ms"
            )
            logger.info(f"  - P95 latency: {perf['latency'].get('p95_ms', 0):.1f}ms")
        if perf.get("throughput"):
            logger.info(
                f"  - Current throughput: {perf['throughput'].get('current_fps', 0):.1f} FPS"
            )

    # 6. Health status
    health = engine.get_health_status()
    logger.info(f"  - Health score: {health['health_score']:.2f}")
    logger.info(f"  - Status: {health['status']}")

    # Cleanup
    await engine.cleanup()
    logger.info("Basic demo completed successfully")


async def demo_performance_monitoring():
    """Demonstrate performance monitoring and alerting."""
    logger.info("=== Performance Monitoring Demo ===")

    # Create configuration with monitoring enabled
    config = VisionConfig(
        model_type=ModelType.NANO,  # Use nano for faster processing
        target_latency_ms=50,  # Strict target for demo
        target_accuracy=0.90,
        batch_size=4,
        max_concurrent_cameras=4,
        enable_performance_monitoring=True,
        drift_detection_enabled=True,
    )

    engine = CoreVisionEngine(config)
    await engine.initialize()

    # Process multiple frames to build up metrics
    logger.info("Processing frames to build performance history...")

    total_detections = 0
    total_frames = 50

    for i in range(total_frames):
        # Vary scene complexity to test performance monitoring
        num_vehicles = np.random.randint(1, 8)
        frame = generate_synthetic_traffic_scene(num_vehicles=num_vehicles)

        result = await engine.process_frame(frame, f"perf_frame_{i}", "perf_camera")
        total_detections += result.detection_count

        # Log progress every 10 frames
        if (i + 1) % 10 == 0:
            logger.info(f"Processed {i+1}/{total_frames} frames")

    # Get comprehensive metrics
    metrics = engine.get_performance_metrics()
    health = engine.get_health_status()

    logger.info(f"Performance monitoring results after {total_frames} frames:")
    logger.info(f"  - Total detections: {total_detections}")
    logger.info(
        f"  - Average detections per frame: {total_detections/total_frames:.1f}"
    )

    if metrics.get("performance"):
        perf = metrics["performance"]

        if perf.get("latency"):
            lat = perf["latency"]
            logger.info("  - Latency metrics:")
            logger.info(f"    * Average: {lat.get('avg_ms', 0):.1f}ms")
            logger.info(f"    * P95: {lat.get('p95_ms', 0):.1f}ms")
            logger.info(f"    * Target: {config.target_latency_ms}ms")
            logger.info(f"    * Meets target: {lat.get('meets_target', False)}")

        if perf.get("quality"):
            qual = perf["quality"]
            logger.info("  - Quality metrics:")
            logger.info(f"    * Average score: {qual.get('avg_score', 0):.3f}")
            logger.info(f"    * Target: {config.target_accuracy:.3f}")
            logger.info(f"    * Meets target: {qual.get('meets_target', False)}")

    # Health status
    logger.info(f"  - Health score: {health['health_score']:.2f}")
    logger.info(f"  - System status: {health['status']}")

    # Check for alerts
    if health.get("alerts"):
        alert_info = health["alerts"]
        logger.info(f"  - Total alerts: {alert_info.get('total', 0)}")
        logger.info(f"  - Critical alerts: {alert_info.get('critical', 0)}")
        logger.info(f"  - Warning alerts: {alert_info.get('warning', 0)}")

        if alert_info.get("recent"):
            logger.info("  - Recent alerts:")
            for alert in alert_info["recent"]:
                logger.info(f"    * {alert['type']}: {alert['message']}")

    await engine.cleanup()
    logger.info("Performance monitoring demo completed")


async def demo_benchmark():
    """Demonstrate comprehensive benchmarking."""
    logger.info("=== Benchmark Demo ===")

    # Test different configurations
    configs = [
        ("Edge (Nano)", create_optimal_config("edge", 4.0, 2)),
        ("Production (Small)", create_optimal_config("production", 8.0, 6)),
        ("Cloud (Medium)", create_optimal_config("cloud", 16.0, 10)),
    ]

    benchmark_results = {}

    for config_name, config in configs:
        logger.info(f"Benchmarking {config_name} configuration...")

        try:
            # Run benchmark
            results = await benchmark_engine(
                config=config, num_frames=50, frame_size=(640, 640)  # Reduced for demo
            )

            benchmark_results[config_name] = results

            # Log key results
            single_perf = results.get("single_frame_performance", {})
            batch_perf = results.get("batch_performance", {})

            logger.info(f"  {config_name} Results:")
            logger.info(
                f"    - Average latency: {single_perf.get('avg_latency_ms', 0):.1f}ms"
            )
            logger.info(
                f"    - P95 latency: {single_perf.get('p95_latency_ms', 0):.1f}ms"
            )
            logger.info(
                f"    - Throughput: {single_perf.get('throughput_fps', 0):.1f} FPS"
            )
            logger.info(
                f"    - Batch efficiency: {batch_perf.get('batch_efficiency', 0):.2f}x"
            )
            logger.info(
                f"    - Meets targets: {results.get('benchmark_summary', {}).get('meets_performance_targets', False)}"
            )

        except Exception as e:
            logger.error(f"Benchmark failed for {config_name}: {e}")
            benchmark_results[config_name] = {"error": str(e)}

    # Summary comparison
    logger.info("\n=== Benchmark Summary Comparison ===")
    logger.info(
        f"{'Configuration':<20} {'Avg Latency':<12} {'P95 Latency':<12} {'Throughput':<12} {'Targets':<8}"
    )
    logger.info("-" * 70)

    for config_name, results in benchmark_results.items():
        if "error" in results:
            logger.info(f"{config_name:<20} ERROR: {results['error']}")
            continue

        single_perf = results.get("single_frame_performance", {})
        summary = results.get("benchmark_summary", {})

        avg_lat = single_perf.get("avg_latency_ms", 0)
        p95_lat = single_perf.get("p95_latency_ms", 0)
        throughput = single_perf.get("throughput_fps", 0)
        meets_targets = "✓" if summary.get("meets_performance_targets", False) else "✗"

        logger.info(
            f"{config_name:<20} {avg_lat:<12.1f} {p95_lat:<12.1f} {throughput:<12.1f} {meets_targets:<8}"
        )

    logger.info("Benchmark demo completed")


async def demo_advanced_features():
    """Demonstrate advanced features like different model types and optimization backends."""
    logger.info("=== Advanced Features Demo ===")

    # Test different model types
    model_configs = [
        (ModelType.NANO, "Ultra-fast edge processing"),
        (ModelType.SMALL, "Balanced performance"),
    ]

    for model_type, description in model_configs:
        logger.info(f"\nTesting {model_type.value.upper()} model ({description})...")

        config = VisionConfig(
            model_type=model_type,
            target_latency_ms=100,
            target_accuracy=0.90,
            batch_size=4,
            optimization_backend=OptimizationBackend.TENSORRT,  # Try TensorRT first
        )

        try:
            engine = CoreVisionEngine(config)
            await engine.initialize()

            # Process test frames
            test_frames = [generate_synthetic_traffic_scene() for _ in range(5)]
            frame_ids = [f"advanced_frame_{i}" for i in range(5)]
            camera_ids = ["advanced_camera"] * 5

            start_time = time.time()
            results = await engine.process_batch(test_frames, frame_ids, camera_ids)
            total_time = (time.time() - start_time) * 1000

            # Calculate statistics
            total_detections = sum(r.detection_count for r in results)
            avg_confidence = np.mean([r.avg_confidence for r in results])
            avg_latency = np.mean([r.total_processing_time_ms for r in results])

            logger.info(f"  Results for {model_type.value}:")
            logger.info(f"    - Total detections: {total_detections}")
            logger.info(f"    - Average confidence: {avg_confidence:.3f}")
            logger.info(f"    - Average latency: {avg_latency:.1f}ms")
            logger.info(f"    - Batch processing time: {total_time:.1f}ms")
            logger.info(f"    - Throughput: {len(results)/(total_time/1000):.1f} FPS")

            # Get model information
            model_info = engine.model_manager.get_model_info()
            if model_info.get("performance"):
                perf = model_info["performance"]
                logger.info(
                    f"    - GPU utilization: {perf.get('gpu_utilization', 0):.1f}%"
                )
                if perf.get("gpu_memory_used"):
                    memory_info = perf["gpu_memory_used"]
                    if isinstance(memory_info, dict):
                        for gpu_id, memory_mb in memory_info.items():
                            logger.info(f"    - {gpu_id}: {memory_mb:.1f}MB")

            await engine.cleanup()

        except Exception as e:
            logger.error(f"  Failed to test {model_type.value}: {e}")


async def demo_integration_features():
    """Demonstrate ML pipeline integration features."""
    logger.info("=== Integration Features Demo ===")

    try:
        # Create integrated pipeline (without full ML pipeline for demo)
        from its_camera_ai.ml.vision_integration import (
            IntegrationConfig,
            VisionPipelineIntegration,
        )

        vision_config = create_optimal_config("production", 8.0, 4)
        integration_config = IntegrationConfig(
            vision_config=vision_config,
            enable_model_registry=False,  # Disable for demo
            enable_ab_testing=False,  # Disable for demo
            enable_drift_detection=True,  # Keep drift detection
            metrics_update_interval=10,  # Fast updates for demo
        )

        integration = VisionPipelineIntegration(integration_config, ml_pipeline=None)
        await integration.initialize()

        logger.info("Integrated pipeline initialized")

        # Process frames through integration layer
        logger.info("Processing frames through integration layer...")

        for i in range(10):
            frame = generate_synthetic_traffic_scene()
            result = await integration.process_frame(
                frame, f"integration_frame_{i}", "integration_camera"
            )

            if i % 3 == 0:  # Log every 3rd frame
                logger.info(
                    f"  Frame {i}: {result.detection_count} detections, "
                    f"{result.avg_confidence:.3f} confidence, "
                    f"{result.total_processing_time_ms:.1f}ms"
                )

        # Get integration metrics
        metrics = integration.get_integration_metrics()
        logger.info("Integration metrics:")
        logger.info(f"  - Total processed: {metrics['integration']['total_processed']}")
        logger.info(
            f"  - Drift detections: {metrics['integration']['drift_detections']}"
        )
        logger.info(
            f"  - Baseline established: {metrics['integration']['baseline_established']}"
        )
        logger.info(f"  - Health score: {metrics['health_status']['health_score']:.2f}")

        if metrics.get("baseline_metrics"):
            baseline = metrics["baseline_metrics"]
            logger.info(
                f"  - Baseline latency: {baseline.get('avg_latency_ms', 0):.1f}ms"
            )
            logger.info(
                f"  - Baseline confidence: {baseline.get('avg_confidence', 0):.3f}"
            )

        await integration.cleanup()
        logger.info("Integration features demo completed")

    except Exception as e:
        logger.error(f"Integration demo failed: {e}")


async def main():
    """Run all demonstration scenarios."""
    logger.info("Starting Core Computer Vision Engine Demonstration")
    logger.info("=" * 60)

    try:
        # Basic usage demo
        await demo_basic_usage()
        await asyncio.sleep(2)

        # Performance monitoring demo
        await demo_performance_monitoring()
        await asyncio.sleep(2)

        # Benchmark demo
        await demo_benchmark()
        await asyncio.sleep(2)

        # Advanced features demo
        await demo_advanced_features()
        await asyncio.sleep(2)

        # Integration features demo
        await demo_integration_features()

        logger.info("=" * 60)
        logger.info("All demonstrations completed successfully!")

    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        raise


if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(main())
