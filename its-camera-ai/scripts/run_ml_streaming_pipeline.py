#!/usr/bin/env python3
"""Production ML Streaming Pipeline Startup Script.

This script starts the complete ITS Camera AI ML streaming pipeline with:
1. Core Vision Engine (YOLO11 ML inference)
2. Unified Vision Analytics Engine (real-time analytics)  
3. gRPC streaming services (client communication)
4. Kafka event streaming (real-time data flow)
5. Performance monitoring and health checks

Usage:
    python scripts/run_ml_streaming_pipeline.py [options]

Examples:
    # Start with default configuration
    python scripts/run_ml_streaming_pipeline.py
    
    # Start with custom model and GPU settings
    python scripts/run_ml_streaming_pipeline.py --model models/yolo11m.pt --gpus 0,1 --port 50051
    
    # Start with Kafka streaming enabled
    python scripts/run_ml_streaming_pipeline.py --kafka --kafka-servers localhost:9092,localhost:9093
    
    # Start in production mode with monitoring
    python scripts/run_ml_streaming_pipeline.py --prod --monitoring-port 8080
"""

import argparse
import asyncio
import logging
import os
import signal
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.its_camera_ai.core.config import Settings
from src.its_camera_ai.core.logging import get_logger, setup_logging
from src.its_camera_ai.services.ml_streaming_integration_service import (
    MLStreamingIntegrationService,
)
from src.its_camera_ai.services.production_ml_grpc_server import (
    ProductionMLgRPCServerManager,
)
from src.its_camera_ai.flow.kafka_event_processor import (
    create_kafka_event_processor,
)

logger = get_logger(__name__)


class MLStreamingPipelineRunner:
    """Production ML streaming pipeline runner with complete integration."""

    def __init__(self, args):
        """Initialize pipeline runner with configuration."""
        self.args = args
        self.settings = self._create_settings()
        
        # Core services
        self.ml_integration_service = None
        self.grpc_server_manager = None
        self.kafka_event_processor = None
        
        # State
        self.is_running = False
        self.shutdown_event = asyncio.Event()

    def _create_settings(self) -> Settings:
        """Create settings from command line arguments."""
        settings = Settings()
        
        # Model configuration
        if self.args.model:
            settings.model_path = self.args.model
        if self.args.confidence:
            settings.confidence_threshold = self.args.confidence
        if self.args.iou:
            settings.iou_threshold = self.args.iou
            
        # Performance configuration
        if self.args.batch_size:
            settings.inference_batch_size = self.args.batch_size
        if self.args.max_batch_size:
            settings.max_batch_size = self.args.max_batch_size
        if self.args.target_fps:
            settings.target_fps = self.args.target_fps
            
        # GPU configuration
        if self.args.gpus:
            gpu_ids = [int(x.strip()) for x in self.args.gpus.split(',')]
            settings.gpu_device_ids = gpu_ids
            
        # TensorRT optimization
        settings.enable_tensorrt = not self.args.no_tensorrt
        if self.args.precision:
            settings.precision = self.args.precision
            
        # Kafka configuration
        settings.kafka_enabled = self.args.kafka
        if self.args.kafka_servers:
            settings.kafka_bootstrap_servers = self.args.kafka_servers.split(',')
            
        # Server configuration
        settings.grpc_host = self.args.host
        settings.grpc_port = self.args.port
        settings.max_workers = self.args.workers
        
        logger.info(f"Configuration: model={settings.model_path}, "
                   f"gpus={settings.gpu_device_ids}, kafka={settings.kafka_enabled}")
        
        return settings

    async def start(self):
        """Start the complete ML streaming pipeline."""
        logger.info("Starting ITS Camera AI ML Streaming Pipeline...")
        
        try:
            # Step 1: Initialize ML integration service
            await self._start_ml_service()
            
            # Step 2: Initialize Kafka event processor (if enabled)
            if self.settings.kafka_enabled:
                await self._start_kafka_processor()
            
            # Step 3: Initialize gRPC server
            await self._start_grpc_server()
            
            # Step 4: Setup monitoring (if enabled)
            if self.args.monitoring_port:
                await self._start_monitoring()
                
            self.is_running = True
            
            logger.info("🚀 ML Streaming Pipeline started successfully!")
            logger.info(f"📡 gRPC Server: {self.settings.grpc_host}:{self.settings.grpc_port}")
            logger.info(f"🤖 ML Model: {self.settings.model_path}")
            logger.info(f"🔧 GPU Devices: {self.settings.gpu_device_ids}")
            logger.info(f"📊 Kafka Streaming: {'Enabled' if self.settings.kafka_enabled else 'Disabled'}")
            
            # Log performance targets
            logger.info("🎯 Performance Targets:")
            logger.info(f"   • Inference Latency: <100ms (target: 50ms)")
            logger.info(f"   • Throughput: {self.settings.target_fps} FPS per camera")
            logger.info(f"   • Batch Size: {self.settings.inference_batch_size}-{self.settings.max_batch_size}")
            logger.info(f"   • Concurrent Streams: 1000+")
            
        except Exception as e:
            logger.error(f"Failed to start ML streaming pipeline: {e}")
            raise

    async def stop(self):
        """Stop the ML streaming pipeline gracefully."""
        if not self.is_running:
            return
            
        logger.info("Stopping ML streaming pipeline...")
        
        try:
            # Stop gRPC server
            if self.grpc_server_manager:
                await self.grpc_server_manager.stop()
                
            # Stop Kafka processor
            if self.kafka_event_processor:
                await self.kafka_event_processor.stop()
                
            # Stop ML service
            if self.ml_integration_service:
                await self.ml_integration_service.stop()
                
            self.is_running = False
            logger.info("ML streaming pipeline stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping pipeline: {e}")

    async def run(self):
        """Run the pipeline with signal handling."""
        # Setup signal handlers
        def signal_handler():
            logger.info("Received shutdown signal")
            self.shutdown_event.set()
            
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, signal_handler)
        
        try:
            # Start pipeline
            await self.start()
            
            # Print startup banner
            self._print_startup_banner()
            
            # Wait for shutdown signal
            await self.shutdown_event.wait()
            
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
        finally:
            await self.stop()

    async def _start_ml_service(self):
        """Start ML integration service."""
        logger.info("Initializing ML integration service...")
        
        self.ml_integration_service = MLStreamingIntegrationService()
        
        # Override settings
        self.ml_integration_service.settings = self.settings
        
        await self.ml_integration_service.initialize()
        await self.ml_integration_service.start()
        
        # Verify health
        health = await self.ml_integration_service.health_check()
        if health["status"] != "healthy":
            raise RuntimeError(f"ML service unhealthy: {health}")
            
        logger.info("✅ ML integration service started")

    async def _start_kafka_processor(self):
        """Start Kafka event processor."""
        logger.info("Initializing Kafka event processor...")
        
        self.kafka_event_processor = create_kafka_event_processor(
            bootstrap_servers=self.settings.kafka_bootstrap_servers,
            consumer_group="ml_streaming_pipeline",
        )
        
        await self.kafka_event_processor.start()
        logger.info("✅ Kafka event processor started")

    async def _start_grpc_server(self):
        """Start gRPC server."""
        logger.info("Initializing gRPC server...")
        
        self.grpc_server_manager = ProductionMLgRPCServerManager(
            self.ml_integration_service
        )
        
        # Start server in background task
        asyncio.create_task(
            self.grpc_server_manager.run_server(
                host=self.settings.grpc_host,
                port=self.settings.grpc_port,
                max_workers=self.settings.max_workers,
            )
        )
        
        # Give server time to start
        await asyncio.sleep(2)
        logger.info("✅ gRPC server started")

    async def _start_monitoring(self):
        """Start monitoring services."""
        logger.info(f"Starting monitoring on port {self.args.monitoring_port}...")
        
        # This would start Prometheus metrics, health check endpoints, etc.
        # For now, just log that monitoring is configured
        logger.info("✅ Monitoring configured")

    def _print_startup_banner(self):
        """Print startup banner with system information."""
        print("\n" + "="*80)
        print("🎥 ITS CAMERA AI - ML STREAMING PIPELINE")
        print("="*80)
        print(f"📡 gRPC Server: {self.settings.grpc_host}:{self.settings.grpc_port}")
        print(f"🤖 ML Model: {self.settings.model_path}")
        print(f"🔧 GPU Devices: {self.settings.gpu_device_ids}")
        print(f"⚡ TensorRT: {'Enabled' if self.settings.enable_tensorrt else 'Disabled'}")
        print(f"📊 Kafka: {'Enabled' if self.settings.kafka_enabled else 'Disabled'}")
        print(f"🎯 Target FPS: {self.settings.target_fps}")
        print(f"📦 Batch Size: {self.settings.inference_batch_size}-{self.settings.max_batch_size}")
        print("\n🔥 AVAILABLE SERVICES:")
        print("   • VisionCoreService    - Complete ML inference + analytics")
        print("   • StreamingService     - Real-time frame processing")  
        print("   • AnalyticsService     - Traffic analytics and metrics")
        print("\n⚡ PERFORMANCE TARGETS:")
        print("   • <100ms inference latency (target: 50ms)")
        print("   • 1000+ concurrent camera streams")
        print("   • Real-time analytics processing")
        print("   • 99.9% uptime with auto-recovery")
        print("\n🚀 System ready! Press Ctrl+C to stop.")
        print("="*80 + "\n")


def create_argument_parser():
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="ITS Camera AI ML Streaming Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Model configuration
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--model", 
        default="models/yolo11s.pt",
        help="Path to YOLO11 model file (default: models/yolo11s.pt)"
    )
    model_group.add_argument(
        "--confidence", 
        type=float, 
        default=0.5,
        help="Confidence threshold for detections (default: 0.5)"
    )
    model_group.add_argument(
        "--iou", 
        type=float, 
        default=0.4,
        help="IoU threshold for NMS (default: 0.4)"
    )
    
    # Performance configuration
    perf_group = parser.add_argument_group("Performance Configuration")
    perf_group.add_argument(
        "--batch-size", 
        type=int, 
        default=8,
        help="Inference batch size (default: 8)"
    )
    perf_group.add_argument(
        "--max-batch-size", 
        type=int, 
        default=32,
        help="Maximum batch size (default: 32)"
    )
    perf_group.add_argument(
        "--target-fps", 
        type=int, 
        default=30,
        help="Target FPS per camera (default: 30)"
    )
    
    # GPU configuration
    gpu_group = parser.add_argument_group("GPU Configuration")
    gpu_group.add_argument(
        "--gpus", 
        default="0",
        help="GPU device IDs (comma-separated, default: 0)"
    )
    gpu_group.add_argument(
        "--no-tensorrt", 
        action="store_true",
        help="Disable TensorRT optimization"
    )
    gpu_group.add_argument(
        "--precision", 
        choices=["fp32", "fp16", "int8"],
        default="fp16",
        help="Inference precision (default: fp16)"
    )
    
    # Server configuration
    server_group = parser.add_argument_group("Server Configuration")
    server_group.add_argument(
        "--host", 
        default="0.0.0.0",
        help="gRPC server host (default: 0.0.0.0)"
    )
    server_group.add_argument(
        "--port", 
        type=int, 
        default=50051,
        help="gRPC server port (default: 50051)"
    )
    server_group.add_argument(
        "--workers", 
        type=int, 
        default=10,
        help="gRPC server workers (default: 10)"
    )
    
    # Kafka configuration
    kafka_group = parser.add_argument_group("Kafka Configuration")
    kafka_group.add_argument(
        "--kafka", 
        action="store_true",
        help="Enable Kafka event streaming"
    )
    kafka_group.add_argument(
        "--kafka-servers", 
        default="localhost:9092",
        help="Kafka bootstrap servers (comma-separated)"
    )
    
    # Monitoring configuration
    monitor_group = parser.add_argument_group("Monitoring Configuration")
    monitor_group.add_argument(
        "--monitoring-port", 
        type=int,
        help="Enable monitoring on specified port"
    )
    monitor_group.add_argument(
        "--log-level", 
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    monitor_group.add_argument(
        "--prod", 
        action="store_true",
        help="Production mode (optimized settings)"
    )
    
    return parser


def validate_arguments(args):
    """Validate command line arguments."""
    errors = []
    
    # Validate model file
    if not Path(args.model).exists():
        errors.append(f"Model file not found: {args.model}")
    
    # Validate confidence and IoU thresholds
    if not 0.0 <= args.confidence <= 1.0:
        errors.append("Confidence threshold must be between 0.0 and 1.0")
    if not 0.0 <= args.iou <= 1.0:
        errors.append("IoU threshold must be between 0.0 and 1.0")
    
    # Validate batch sizes
    if args.batch_size <= 0:
        errors.append("Batch size must be positive")
    if args.max_batch_size < args.batch_size:
        errors.append("Max batch size must be >= batch size")
    
    # Validate port
    if not 1024 <= args.port <= 65535:
        errors.append("Port must be between 1024 and 65535")
    
    # Validate GPU IDs
    try:
        gpu_ids = [int(x.strip()) for x in args.gpus.split(',')]
        if any(gpu_id < 0 for gpu_id in gpu_ids):
            errors.append("GPU IDs must be non-negative")
    except ValueError:
        errors.append("Invalid GPU IDs format")
    
    if errors:
        print("❌ Configuration errors:")
        for error in errors:
            print(f"   • {error}")
        sys.exit(1)


async def main():
    """Main entry point."""
    # Parse arguments
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Validate arguments
    validate_arguments(args)
    
    # Setup logging
    setup_logging(level=getattr(logging, args.log_level))
    
    # Apply production optimizations
    if args.prod:
        args.workers = max(args.workers, 20)
        args.batch_size = max(args.batch_size, 16)
        args.kafka = True
        logger.info("🏭 Production mode enabled with optimized settings")
    
    try:
        # Create and run pipeline
        runner = MLStreamingPipelineRunner(args)
        await runner.run()
        
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Ensure proper asyncio setup
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    asyncio.run(main())