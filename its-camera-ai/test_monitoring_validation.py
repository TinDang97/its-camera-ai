#!/usr/bin/env python3
"""
Thread-Safety and Performance Validation for ML/CV Monitoring Implementations

This script validates:
1. Thread-safety of CPU monitoring in PostProcessor
2. Performance impact of monitoring on inference latency
3. Memory leak detection accuracy
4. Concurrent access patterns
5. High-throughput stress testing
"""

import asyncio
import concurrent.futures
import time
from typing import Any

import numpy as np

from src.its_camera_ai.api.routers.health import _check_ml_models_health

# Import the monitoring components we implemented
from src.its_camera_ai.ml.core_vision_engine import (
    CoreVisionEngine,
    create_optimal_config,
)
from src.its_camera_ai.services.streaming_service import StreamingDataProcessor


class MonitoringValidationSuite:
    """Comprehensive validation suite for monitoring implementations."""

    def __init__(self):
        self.results = {
            "thread_safety": {},
            "performance_impact": {},
            "memory_tracking": {},
            "stress_testing": {},
        }

    async def run_all_tests(self) -> dict[str, Any]:
        """Run all validation tests."""
        print("🔍 Starting Monitoring Validation Suite...")

        # Test 1: Thread Safety Validation
        print("\n1️⃣ Testing thread safety...")
        await self.test_thread_safety()

        # Test 2: Performance Impact Assessment
        print("\n2️⃣ Testing performance impact...")
        await self.test_performance_impact()

        # Test 3: Memory Tracking Accuracy
        print("\n3️⃣ Testing memory tracking...")
        await self.test_memory_tracking()

        # Test 4: Stress Testing
        print("\n4️⃣ Running stress tests...")
        await self.test_stress_conditions()

        # Generate final report
        return self.generate_report()

    async def test_thread_safety(self) -> None:
        """Test thread safety of monitoring implementations."""
        print("  Testing CPU monitoring thread safety...")

        # Create a vision engine for testing
        config = create_optimal_config("edge", available_memory_gb=2.0, target_cameras=1)
        engine = CoreVisionEngine(config)
        await engine.initialize()

        try:
            # Test concurrent CPU monitoring calls
            cpu_values = []
            errors = []

            def cpu_monitoring_worker(worker_id: int, iterations: int):
                """Worker thread for concurrent CPU monitoring."""
                try:
                    for i in range(iterations):
                        # Simulate frame processing with CPU monitoring
                        dummy_frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
                        result = asyncio.run(engine.process_frame(
                            dummy_frame, f"thread_{worker_id}_frame_{i}", f"test_camera_{worker_id}"
                        ))
                        cpu_values.append(result.cpu_utilization)
                except Exception as e:
                    errors.append(f"Worker {worker_id}: {str(e)}")

            # Launch multiple threads
            num_threads = 10
            iterations_per_thread = 20

            with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [
                    executor.submit(cpu_monitoring_worker, i, iterations_per_thread)
                    for i in range(num_threads)
                ]
                concurrent.futures.wait(futures)

            # Analyze results
            self.results["thread_safety"] = {
                "cpu_monitoring": {
                    "total_operations": len(cpu_values),
                    "expected_operations": num_threads * iterations_per_thread,
                    "errors": len(errors),
                    "error_rate": len(errors) / (num_threads * iterations_per_thread),
                    "cpu_values_collected": len(cpu_values),
                    "cpu_value_range": [min(cpu_values), max(cpu_values)] if cpu_values else [0, 0],
                    "thread_safe": len(errors) == 0,
                    "error_details": errors[:5],  # First 5 errors
                }
            }

            print(f"    ✅ CPU monitoring: {len(errors)} errors in {len(cpu_values)} operations")

        finally:
            await engine.cleanup()

    async def test_performance_impact(self) -> None:
        """Test performance impact of monitoring on inference latency."""
        print("  Measuring performance impact...")

        config = create_optimal_config("edge", available_memory_gb=2.0, target_cameras=1)

        # Test without monitoring
        engine_no_monitoring = CoreVisionEngine(config)
        engine_no_monitoring.performance_monitor.enabled = False
        await engine_no_monitoring.initialize()

        # Test with monitoring
        engine_with_monitoring = CoreVisionEngine(config)
        engine_with_monitoring.performance_monitor.enabled = True
        await engine_with_monitoring.initialize()

        try:
            num_test_frames = 50
            dummy_frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

            # Benchmark without monitoring
            latencies_no_monitoring = []
            for i in range(num_test_frames):
                start_time = time.time()
                await engine_no_monitoring.process_frame(dummy_frame, f"no_mon_frame_{i}", "test_camera")
                latencies_no_monitoring.append((time.time() - start_time) * 1000)

            # Benchmark with monitoring
            latencies_with_monitoring = []
            for i in range(num_test_frames):
                start_time = time.time()
                await engine_with_monitoring.process_frame(dummy_frame, f"mon_frame_{i}", "test_camera")
                latencies_with_monitoring.append((time.time() - start_time) * 1000)

            # Calculate impact
            avg_no_monitoring = np.mean(latencies_no_monitoring)
            avg_with_monitoring = np.mean(latencies_with_monitoring)
            performance_impact = ((avg_with_monitoring - avg_no_monitoring) / avg_no_monitoring) * 100

            self.results["performance_impact"] = {
                "avg_latency_no_monitoring_ms": round(avg_no_monitoring, 2),
                "avg_latency_with_monitoring_ms": round(avg_with_monitoring, 2),
                "performance_impact_percent": round(performance_impact, 2),
                "acceptable_impact": performance_impact < 5.0,  # <5% impact is acceptable
                "p95_no_monitoring_ms": round(np.percentile(latencies_no_monitoring, 95), 2),
                "p95_with_monitoring_ms": round(np.percentile(latencies_with_monitoring, 95), 2),
                "meets_latency_target": avg_with_monitoring < config.target_latency_ms,
            }

            print(f"    📊 Performance impact: {performance_impact:.1f}% ({avg_no_monitoring:.1f}ms → {avg_with_monitoring:.1f}ms)")

        finally:
            await engine_no_monitoring.cleanup()
            await engine_with_monitoring.cleanup()

    async def test_memory_tracking(self) -> None:
        """Test accuracy of memory tracking implementations."""
        print("  Testing memory tracking accuracy...")

        # Test streaming service memory tracking
        streaming_processor = StreamingDataProcessor()

        # Get baseline memory
        baseline_memory = streaming_processor._get_current_memory_usage()

        # Simulate memory allocation
        large_arrays = []
        for i in range(10):
            # Allocate ~10MB arrays
            array = np.random.bytes(10 * 1024 * 1024)
            large_arrays.append(array)

        # Check if memory tracking detected the increase
        current_memory = streaming_processor._get_current_memory_usage()
        memory_increase = current_memory - baseline_memory
        expected_increase = 100  # ~100MB allocated

        # Test GPU memory tracking if available
        gpu_memory_status = streaming_processor._get_gpu_memory_usage()
        gpu_available = gpu_memory_status["total_mb"] > 0

        # Test comprehensive memory status
        comprehensive_status = streaming_processor.get_comprehensive_memory_status()

        self.results["memory_tracking"] = {
            "baseline_memory_mb": round(baseline_memory, 2),
            "current_memory_mb": round(current_memory, 2),
            "detected_increase_mb": round(memory_increase, 2),
            "expected_increase_mb": expected_increase,
            "tracking_accuracy": abs(memory_increase - expected_increase) < 50,  # Within 50MB
            "gpu_tracking_available": gpu_available,
            "gpu_memory_status": gpu_memory_status,
            "comprehensive_tracking_works": comprehensive_status.get("tracking_enabled", False),
            "leak_detection_works": "leak_detection" in comprehensive_status,
        }

        # Cleanup
        del large_arrays

        print(f"    💾 Memory increase detected: {memory_increase:.1f}MB (expected ~{expected_increase}MB)")

    async def test_stress_conditions(self) -> None:
        """Test monitoring under stress conditions."""
        print("  Running stress tests...")

        # Test ML model health checks under load
        start_time = time.time()
        health_check_tasks = []

        for i in range(20):  # 20 concurrent health checks
            task = asyncio.create_task(_check_ml_models_health())
            health_check_tasks.append(task)

        health_results = await asyncio.gather(*health_check_tasks, return_exceptions=True)
        health_check_duration = time.time() - start_time

        # Count successful vs failed health checks
        successful_checks = sum(1 for result in health_results if isinstance(result, dict) and result.get("status") == "healthy")
        failed_checks = len(health_results) - successful_checks

        # Test high-frequency monitoring calls
        config = create_optimal_config("edge", available_memory_gb=1.0, target_cameras=1)
        engine = CoreVisionEngine(config)
        await engine.initialize()

        try:
            # Rapid-fire processing
            rapid_fire_start = time.time()
            rapid_fire_results = []

            for i in range(100):  # 100 rapid frames
                dummy_frame = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)  # Smaller for speed
                result = await engine.process_frame(dummy_frame, f"stress_frame_{i}", "stress_camera")
                rapid_fire_results.append(result)

            rapid_fire_duration = time.time() - rapid_fire_start
            avg_processing_time = (rapid_fire_duration / 100) * 1000  # ms per frame

            self.results["stress_testing"] = {
                "health_checks": {
                    "concurrent_checks": len(health_check_tasks),
                    "successful_checks": successful_checks,
                    "failed_checks": failed_checks,
                    "total_duration_s": round(health_check_duration, 2),
                    "avg_check_time_ms": round((health_check_duration / len(health_check_tasks)) * 1000, 2),
                    "success_rate": successful_checks / len(health_check_tasks),
                },
                "rapid_processing": {
                    "frames_processed": len(rapid_fire_results),
                    "total_duration_s": round(rapid_fire_duration, 2),
                    "avg_processing_time_ms": round(avg_processing_time, 2),
                    "throughput_fps": round(100 / rapid_fire_duration, 1),
                    "all_frames_successful": all(r.detection_count >= 0 for r in rapid_fire_results),
                },
            }

            print(f"    🚀 Stress test: {successful_checks}/{len(health_check_tasks)} health checks, {avg_processing_time:.1f}ms avg processing")

        finally:
            await engine.cleanup()

    def generate_report(self) -> dict[str, Any]:
        """Generate comprehensive validation report."""
        print("\n📋 Generating Validation Report...")

        # Overall assessment
        thread_safety_pass = self.results["thread_safety"]["cpu_monitoring"]["thread_safe"]
        performance_pass = self.results["performance_impact"]["acceptable_impact"]
        memory_tracking_pass = self.results["memory_tracking"]["tracking_accuracy"]
        stress_test_pass = (
            self.results["stress_testing"]["health_checks"]["success_rate"] > 0.8 and
            self.results["stress_testing"]["rapid_processing"]["all_frames_successful"]
        )

        all_tests_pass = all([thread_safety_pass, performance_pass, memory_tracking_pass, stress_test_pass])

        report = {
            "validation_timestamp": time.time(),
            "overall_status": "PASS" if all_tests_pass else "FAIL",
            "test_results": self.results,
            "summary": {
                "thread_safety": "PASS" if thread_safety_pass else "FAIL",
                "performance_impact": "PASS" if performance_pass else "FAIL",
                "memory_tracking": "PASS" if memory_tracking_pass else "FAIL",
                "stress_testing": "PASS" if stress_test_pass else "FAIL",
            },
            "recommendations": [],
        }

        # Add recommendations based on results
        if not thread_safety_pass:
            report["recommendations"].append("Review thread safety in CPU monitoring implementation")

        if not performance_pass:
            impact = self.results["performance_impact"]["performance_impact_percent"]
            report["recommendations"].append(f"Optimize monitoring performance (current impact: {impact:.1f}%)")

        if not memory_tracking_pass:
            report["recommendations"].append("Improve memory tracking accuracy")

        if not stress_test_pass:
            report["recommendations"].append("Enhance monitoring reliability under stress conditions")

        if all_tests_pass:
            report["recommendations"].append("All monitoring implementations are production-ready!")

        return report


async def main():
    """Run the monitoring validation suite."""
    print("🔧 ITS Camera AI - Monitoring Implementation Validation")
    print("=" * 60)

    validator = MonitoringValidationSuite()
    report = await validator.run_all_tests()

    print("\n" + "=" * 60)
    print("📊 VALIDATION REPORT")
    print("=" * 60)
    print(f"Overall Status: {report['overall_status']}")
    print(f"Timestamp: {time.ctime(report['validation_timestamp'])}")
    print()

    print("Test Results:")
    for test_name, status in report["summary"].items():
        status_icon = "✅" if status == "PASS" else "❌"
        print(f"  {status_icon} {test_name.replace('_', ' ').title()}: {status}")

    print("\nRecommendations:")
    for i, rec in enumerate(report["recommendations"], 1):
        print(f"  {i}. {rec}")

    print("\nDetailed Results:")
    print(f"  • Thread Safety: {report['test_results']['thread_safety']['cpu_monitoring']['errors']} errors")
    print(f"  • Performance Impact: {report['test_results']['performance_impact']['performance_impact_percent']:.1f}%")
    print(f"  • Memory Tracking: {report['test_results']['memory_tracking']['detected_increase_mb']:.1f}MB detected")
    print(f"  • Stress Testing: {report['test_results']['stress_testing']['health_checks']['success_rate']:.1%} success rate")

    if report['overall_status'] == "PASS":
        print("\n🎉 All monitoring implementations are production-ready!")
    else:
        print("\n⚠️  Some monitoring implementations need attention before production deployment.")

    return report


if __name__ == "__main__":
    # Run the validation
    asyncio.run(main())
