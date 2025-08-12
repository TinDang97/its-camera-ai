#!/usr/bin/env python3
"""Streaming Service Demo.

This demo shows how to use the ITS Camera AI Streaming Service for
high-performance real-time camera stream processing.

Features demonstrated:
- Camera registration and stream setup
- Real-time frame processing and quality validation
- Redis queue integration for batch processing
- gRPC server for distributed processing
- Performance monitoring and health checks
- RTSP/WebRTC stream handling

Usage:
    python examples/streaming_service_demo.py [--cameras N] [--duration SECONDS]
"""

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

import grpc
import numpy as np

# Ensure the project root is in the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from its_camera_ai.proto import processed_frame_pb2 as frame_pb
from its_camera_ai.proto import streaming_service_pb2 as pb
from its_camera_ai.proto import streaming_service_pb2_grpc as pb_grpc
from its_camera_ai.services.streaming_container import (
    get_streaming_server,
    initialize_streaming_container,
)


class StreamingDemo:
    """Comprehensive demo of the streaming service."""

    def __init__(self, host: str = "localhost", port: int = 50051):
        self.host = host
        self.port = port
        self.server = None
        self.client_channel = None
        self.client_stub = None

    async def setup_server(self, max_streams: int = 10) -> None:
        """Set up and start the streaming server."""
        print(f"🚀 Starting streaming server on {self.host}:{self.port}")

        # Initialize container with demo configuration
        config = {
            "redis": {
                "url": "redis://localhost:6379",
                "pool_size": 10,
                "timeout": 30,
                "retry_on_failure": True,
            },
            "streaming": {
                "grpc_host": self.host,
                "grpc_port": self.port,
                "max_concurrent_streams": max_streams,
                "frame_processing_timeout": 0.01,
                "min_quality_score": 0.5,
            },
        }

        initialize_streaming_container(config)
        self.server = get_streaming_server()

        # Start the server in the background
        asyncio.create_task(self.server.start())

        # Wait a moment for server to start
        await asyncio.sleep(2)

        print("✅ Streaming server started successfully")

    async def setup_client(self) -> None:
        """Set up the gRPC client."""
        self.client_channel = grpc.aio.insecure_channel(f"{self.host}:{self.port}")
        self.client_stub = pb_grpc.StreamingServiceStub(self.client_channel)
        print("📱 gRPC client connected")

    async def demo_health_check(self) -> None:
        """Demonstrate health check functionality."""
        print("\n🏥 Testing Health Check...")

        request = pb.HealthCheckRequest(service_name="streaming_service")

        try:
            response = await self.client_stub.HealthCheck(request)

            status_text = {
                pb.HealthCheckResponse.Status.SERVING: "SERVING",
                pb.HealthCheckResponse.Status.NOT_SERVING: "NOT_SERVING",
                pb.HealthCheckResponse.Status.UNKNOWN: "UNKNOWN",
            }.get(response.status, "UNKNOWN")

            print(f"   Status: {status_text}")
            print(f"   Message: {response.message}")
            print(f"   Response Time: {response.response_time_ms:.2f}ms")

            if response.status == pb.HealthCheckResponse.Status.SERVING:
                print("   ✅ Service is healthy!")
            else:
                print("   ⚠️ Service has issues")

        except Exception as e:
            print(f"   ❌ Health check failed: {e}")

    async def demo_camera_registration(self, num_cameras: int = 5) -> list[str]:
        """Demonstrate camera registration."""
        print(f"\n📹 Registering {num_cameras} test cameras...")

        camera_ids = []

        for i in range(num_cameras):
            camera_id = f"demo_camera_{i:03d}"

            config = frame_pb.CameraStreamConfig(
                camera_id=camera_id,
                location=f"Demo Location {i}",
                latitude=40.7128 + (i * 0.001),  # NYC area
                longitude=-74.0060 + (i * 0.001),
                width=1280,
                height=720,
                fps=30,
                encoding="h264",
                quality_threshold=0.7,
                processing_enabled=True,
            )

            # Add some ROI boxes
            roi1 = frame_pb.ROIBox(
                x=100, y=100, width=300, height=200, label="intersection"
            )
            roi2 = frame_pb.ROIBox(
                x=500, y=300, width=200, height=150, label="crosswalk"
            )
            config.roi_boxes.extend([roi1, roi2])

            try:
                response = await self.client_stub.RegisterStream(config)

                if response.success:
                    camera_ids.append(camera_id)
                    print(f"   ✅ {camera_id}: {response.message}")
                else:
                    print(f"   ❌ {camera_id}: {response.message}")

            except Exception as e:
                print(f"   ❌ {camera_id}: Registration failed - {e}")

        print(
            f"\n   📊 Successfully registered {len(camera_ids)}/{num_cameras} cameras"
        )
        return camera_ids

    async def demo_frame_processing(
        self, camera_ids: list[str], num_frames: int = 20
    ) -> None:
        """Demonstrate frame processing with quality validation."""
        print(f"\n🎬 Processing {num_frames} frames per camera...")

        total_frames = 0
        successful_frames = 0
        failed_frames = 0
        latencies = []

        for camera_id in camera_ids[:3]:  # Process frames for first 3 cameras
            print(f"\n   📹 Processing frames for {camera_id}")

            for frame_idx in range(num_frames):
                try:
                    # Create synthetic frame data
                    self._create_synthetic_frame(
                        1280, 720, quality="high" if frame_idx % 4 != 0 else "low"
                    )

                    # Convert to protobuf format
                    image_data = frame_pb.ImageData(
                        compressed_data=b"synthetic_frame_data",  # In real use, this would be JPEG/PNG bytes
                        width=1280,
                        height=720,
                        channels=3,
                        compression_format="jpeg",
                        quality=85,
                    )

                    processed_frame = frame_pb.ProcessedFrame(
                        frame_id=f"{camera_id}_{frame_idx:04d}",
                        camera_id=camera_id,
                        timestamp=time.time(),
                        original_image=image_data,
                    )

                    # Create batch with single frame
                    batch = frame_pb.ProcessedFrameBatch(
                        frames=[processed_frame],
                        batch_id=f"batch_{camera_id}_{frame_idx}",
                        batch_timestamp=time.time(),
                        batch_size=1,
                    )

                    # Process with timing
                    start_time = time.perf_counter()
                    response = await self.client_stub.ProcessFrameBatch(batch)
                    end_time = time.perf_counter()

                    latency_ms = (end_time - start_time) * 1000
                    latencies.append(latency_ms)

                    total_frames += 1

                    if response.success:
                        successful_frames += response.processed_count
                        failed_frames += response.failed_count

                        if frame_idx < 5:  # Show details for first few frames
                            print(
                                f"      Frame {frame_idx:04d}: ✅ Processed ({latency_ms:.2f}ms)"
                            )
                    else:
                        failed_frames += 1
                        if frame_idx < 5:
                            print(
                                f"      Frame {frame_idx:04d}: ❌ Failed - {response.errors[0].error_message if response.errors else 'Unknown error'}"
                            )

                    # Small delay to simulate realistic frame rate
                    await asyncio.sleep(0.033)  # ~30 FPS

                except Exception as e:
                    total_frames += 1
                    failed_frames += 1
                    if frame_idx < 5:
                        print(f"      Frame {frame_idx:04d}: ❌ Exception - {e}")

        # Calculate and display statistics
        print("\n   📊 Processing Statistics:")
        print(f"      Total frames: {total_frames}")
        print(f"      Successful: {successful_frames}")
        print(f"      Failed: {failed_frames}")

        if total_frames > 0:
            success_rate = (successful_frames / total_frames) * 100
            print(f"      Success rate: {success_rate:.2f}%")

        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            min_latency = min(latencies)
            max_latency = max(latencies)
            p95_latency = (
                sorted(latencies)[int(0.95 * len(latencies))]
                if len(latencies) > 20
                else max_latency
            )

            print(f"      Avg latency: {avg_latency:.2f}ms")
            print(f"      Min latency: {min_latency:.2f}ms")
            print(f"      Max latency: {max_latency:.2f}ms")
            print(f"      P95 latency: {p95_latency:.2f}ms")

            # Check performance requirements
            if avg_latency <= 10.0:
                print("      ✅ Meets <10ms latency requirement")
            else:
                print("      ⚠️ Exceeds 10ms latency requirement")

    async def demo_system_metrics(self) -> None:
        """Demonstrate system metrics collection."""
        print("\n📊 Collecting System Metrics...")

        request = pb.SystemMetricsRequest(
            include_queue_metrics=True, include_performance_metrics=True
        )

        try:
            response = await self.client_stub.GetSystemMetrics(request)

            print("   Performance Metrics:")
            perf = response.performance_metrics
            print(f"      Frames processed: {perf.frames_processed}")
            print(f"      Frames rejected: {perf.frames_rejected}")
            print(f"      Avg processing time: {perf.avg_processing_time_ms:.2f}ms")
            print(f"      Throughput: {perf.throughput_fps:.1f} fps")
            print(f"      Error count: {perf.error_count}")
            print(f"      Active connections: {perf.active_connections}")

            if response.queue_metrics:
                print("\n   Queue Metrics:")
                for queue in response.queue_metrics:
                    print(f"      {queue.queue_name}:")
                    print(f"         Pending: {queue.pending_count}")
                    print(f"         Processing: {queue.processing_count}")
                    print(f"         Completed: {queue.completed_count}")
                    print(f"         Failed: {queue.failed_count}")
                    print(f"         Throughput: {queue.throughput_fps:.1f} fps")

        except Exception as e:
            print(f"   ❌ Failed to get metrics: {e}")

    async def demo_load_test(self, num_cameras: int = 10, duration: int = 30) -> None:
        """Demonstrate load testing with multiple cameras."""
        print(f"\n🚀 Load Test: {num_cameras} cameras for {duration} seconds")

        # Register cameras for load test
        camera_ids = []
        for i in range(num_cameras):
            camera_id = f"load_test_camera_{i:03d}"

            config = frame_pb.CameraStreamConfig(
                camera_id=camera_id,
                location=f"Load Test Location {i}",
                latitude=40.7128 + (i * 0.001),
                longitude=-74.0060 + (i * 0.001),
                width=640,  # Smaller resolution for load test
                height=480,
                fps=15,  # Lower FPS for load test
                encoding="h264",
                quality_threshold=0.5,  # Lower quality threshold
                processing_enabled=True,
            )

            try:
                response = await self.client_stub.RegisterStream(config)
                if response.success:
                    camera_ids.append(camera_id)
            except Exception as e:
                print(f"   ⚠️ Failed to register {camera_id}: {e}")

        print(f"   📹 Registered {len(camera_ids)} cameras for load test")

        if not camera_ids:
            print("   ❌ No cameras registered, aborting load test")
            return

        # Performance tracking
        start_time = time.time()
        frames_sent = 0
        frames_processed = 0
        frames_failed = 0
        latencies = []

        async def camera_load_task(camera_id: str):
            """Send frames for one camera during the load test."""
            nonlocal frames_sent, frames_processed, frames_failed

            fps = 15  # 15 FPS per camera
            camera_start = time.time()

            while time.time() - start_time < duration:
                try:
                    # Create frame
                    frame_data = frame_pb.ImageData(
                        compressed_data=b"load_test_frame_data",
                        width=640,
                        height=480,
                        channels=3,
                        compression_format="jpeg",
                        quality=85,
                    )

                    processed_frame = frame_pb.ProcessedFrame(
                        frame_id=f"{camera_id}_{int(time.time() * 1000)}",
                        camera_id=camera_id,
                        timestamp=time.time(),
                        original_image=frame_data,
                    )

                    batch = frame_pb.ProcessedFrameBatch(
                        frames=[processed_frame],
                        batch_id=f"load_{camera_id}_{frames_sent}",
                        batch_timestamp=time.time(),
                        batch_size=1,
                    )

                    # Send frame with timing
                    frame_start = time.perf_counter()
                    response = await self.client_stub.ProcessFrameBatch(batch)
                    frame_end = time.perf_counter()

                    frames_sent += 1
                    latency_ms = (frame_end - frame_start) * 1000
                    latencies.append(latency_ms)

                    if response.success:
                        frames_processed += response.processed_count
                        frames_failed += response.failed_count
                    else:
                        frames_failed += 1

                    # Wait for next frame time
                    elapsed = time.time() - camera_start
                    expected_frames = int(elapsed * fps) + 1
                    next_frame_time = camera_start + (expected_frames / fps)
                    sleep_time = max(0, next_frame_time - time.time())

                    if sleep_time > 0:
                        await asyncio.sleep(sleep_time)

                except Exception:
                    frames_sent += 1
                    frames_failed += 1
                    # Continue on errors during load test

        # Start all camera tasks
        print("   🎬 Starting load test...")
        camera_tasks = [
            asyncio.create_task(camera_load_task(camera_id)) for camera_id in camera_ids
        ]

        # Wait for completion
        await asyncio.gather(*camera_tasks, return_exceptions=True)

        # Calculate results
        end_time = time.time()
        actual_duration = end_time - start_time

        print("\n   📊 Load Test Results:")
        print(f"      Duration: {actual_duration:.1f}s")
        print(f"      Cameras: {len(camera_ids)}")
        print(f"      Frames sent: {frames_sent}")
        print(f"      Frames processed: {frames_processed}")
        print(f"      Frames failed: {frames_failed}")

        if frames_sent > 0:
            success_rate = (frames_processed / frames_sent) * 100
            throughput = frames_sent / actual_duration

            print(f"      Success rate: {success_rate:.2f}%")
            print(f"      Total throughput: {throughput:.1f} fps")
            print(
                f"      Per-camera throughput: {throughput / len(camera_ids):.1f} fps"
            )

            if latencies:
                avg_latency = sum(latencies) / len(latencies)
                p95_latency = (
                    sorted(latencies)[int(0.95 * len(latencies))]
                    if len(latencies) > 20
                    else max(latencies)
                )
                p99_latency = (
                    sorted(latencies)[int(0.99 * len(latencies))]
                    if len(latencies) > 100
                    else max(latencies)
                )

                print(f"      Avg latency: {avg_latency:.2f}ms")
                print(f"      P95 latency: {p95_latency:.2f}ms")
                print(f"      P99 latency: {p99_latency:.2f}ms")

                # Performance assessment
                meets_latency = avg_latency <= 10.0
                meets_success_rate = success_rate >= 99.9
                meets_throughput = throughput >= 100.0

                print("\n   🎯 Performance Assessment:")
                print(
                    f"      Latency < 10ms: {'✅' if meets_latency else '❌'} ({avg_latency:.2f}ms)"
                )
                print(
                    f"      Success rate > 99.9%: {'✅' if meets_success_rate else '❌'} ({success_rate:.2f}%)"
                )
                print(
                    f"      Throughput > 100 fps: {'✅' if meets_throughput else '❌'} ({throughput:.1f} fps)"
                )

    def _create_synthetic_frame(
        self, width: int, height: int, quality: str = "high"
    ) -> np.ndarray:
        """Create a synthetic frame for testing."""
        if quality == "high":
            # Create a frame with clear patterns and good contrast
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[::20, :] = 255  # Horizontal lines
            frame[:, ::20] = 255  # Vertical lines
            frame += np.random.randint(0, 50, frame.shape, dtype=np.uint8)  # Some noise
        else:
            # Create a blurry, low-contrast frame
            frame = np.random.randint(100, 150, (height, width, 3), dtype=np.uint8)

        return frame

    async def cleanup(self) -> None:
        """Clean up resources."""
        if self.client_channel:
            await self.client_channel.close()
            print("📱 Client disconnected")

        if self.server:
            await self.server.stop()
            print("🛑 Server stopped")

    async def run_complete_demo(self) -> None:
        """Run the complete streaming service demonstration."""
        try:
            print("🎪 ITS Camera AI Streaming Service Demo")
            print("=" * 50)

            # Setup
            await self.setup_server(max_streams=50)
            await self.setup_client()

            # Demo sequence
            await self.demo_health_check()
            camera_ids = await self.demo_camera_registration(5)

            if camera_ids:
                await self.demo_frame_processing(camera_ids, 10)
                await self.demo_system_metrics()
                await self.demo_load_test(10, 15)

            print("\n🎉 Demo completed successfully!")

        except Exception as e:
            print(f"❌ Demo failed: {e}")
            import traceback

            traceback.print_exc()
        finally:
            await self.cleanup()


async def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description="ITS Camera AI Streaming Service Demo")
    parser.add_argument("--host", default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=50051, help="Server port")
    parser.add_argument(
        "--demo",
        choices=["health", "register", "process", "metrics", "load", "all"],
        default="all",
        help="Which demo to run",
    )
    parser.add_argument(
        "--cameras", type=int, default=5, help="Number of cameras for demos"
    )
    parser.add_argument(
        "--frames", type=int, default=10, help="Number of frames per camera"
    )
    parser.add_argument(
        "--duration", type=int, default=15, help="Load test duration in seconds"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    demo = StreamingDemo(host=args.host, port=args.port)

    try:
        if args.demo == "all":
            await demo.run_complete_demo()
        else:
            # Setup
            await demo.setup_server(max_streams=50)
            await demo.setup_client()

            # Run specific demo
            if args.demo == "health":
                await demo.demo_health_check()
            elif args.demo == "register":
                await demo.demo_camera_registration(args.cameras)
            elif args.demo == "process":
                camera_ids = await demo.demo_camera_registration(args.cameras)
                if camera_ids:
                    await demo.demo_frame_processing(camera_ids, args.frames)
            elif args.demo == "metrics":
                await demo.demo_system_metrics()
            elif args.demo == "load":
                await demo.demo_load_test(args.cameras, args.duration)

            await demo.cleanup()

    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted")
        await demo.cleanup()
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        await demo.cleanup()
        raise


if __name__ == "__main__":
    asyncio.run(main())
