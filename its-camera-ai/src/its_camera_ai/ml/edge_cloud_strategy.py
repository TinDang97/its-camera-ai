"""
Edge vs Cloud Inference Strategy for ITS Camera AI Traffic Monitoring System.

This module implements intelligent deployment strategies that optimize between edge
and cloud inference based on:
1. Network conditions and bandwidth availability
2. Edge device capabilities and load
3. Latency requirements and fallback mechanisms
4. Cost optimization and power consumption

Key Features:
- Adaptive model distribution (edge vs cloud)
- Network-aware inference routing
- Multi-tier fallback mechanisms
- Bandwidth optimization and caching
- Dynamic model switching based on conditions
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np

from .inference_optimizer import (
    DetectionResult,
    InferenceConfig,
    ModelType,
    OptimizationBackend,
    OptimizedInferenceEngine,
)

logger = logging.getLogger(__name__)


class InferenceLocation(Enum):
    """Inference execution location options."""

    EDGE_ONLY = "edge_only"
    CLOUD_ONLY = "cloud_only"
    EDGE_PRIMARY = "edge_primary"  # Edge first, cloud fallback
    CLOUD_PRIMARY = "cloud_primary"  # Cloud first, edge fallback
    ADAPTIVE = "adaptive"  # Dynamic based on conditions


class NetworkCondition(Enum):
    """Network condition assessment."""

    EXCELLENT = "excellent"  # <10ms latency, >50Mbps, 100% uptime
    GOOD = "good"  # <50ms latency, >10Mbps, >99% uptime
    FAIR = "fair"  # <100ms latency, >2Mbps, >95% uptime
    POOR = "poor"  # >100ms latency, <2Mbps, <95% uptime
    OFFLINE = "offline"  # No connectivity


@dataclass
class EdgeDevice:
    """Edge device configuration and capabilities."""

    device_id: str
    device_type: str  # "jetson_nano", "jetson_xavier", "intel_ncs2", etc.

    # Hardware capabilities
    gpu_memory_gb: float
    cpu_cores: int
    max_power_watts: float

    # Performance characteristics
    supported_models: list[ModelType] = field(default_factory=list)
    max_throughput_fps: int = 30
    typical_latency_ms: float = 50

    # Current status
    current_load: float = 0.0
    temperature_celsius: float = 25.0
    power_consumption_watts: float = 0.0
    is_available: bool = True

    # Network connection
    network_condition: NetworkCondition = NetworkCondition.GOOD
    bandwidth_mbps: float = 10.0
    latency_to_cloud_ms: float = 50.0


@dataclass
class InferenceRequest:
    """Inference request with routing metadata."""

    frame: np.ndarray
    frame_id: str
    camera_id: str
    timestamp: float

    # Request preferences
    max_latency_ms: int = 100
    min_accuracy: float = 0.9
    priority: str = "normal"

    # Routing metadata
    preferred_location: InferenceLocation | None = None
    fallback_enabled: bool = True
    retry_count: int = 0
    max_retries: int = 2


@dataclass
class NetworkMetrics:
    """Network performance metrics."""

    bandwidth_mbps: float = 0.0
    latency_ms: float = 0.0
    packet_loss_pct: float = 0.0
    jitter_ms: float = 0.0
    uptime_pct: float = 100.0

    # Recent measurements
    recent_latencies: list[float] = field(default_factory=list)
    last_update: float = field(default_factory=time.time)

    def update_metrics(self, latency_ms: float, success: bool):
        """Update network metrics with new measurement."""
        self.recent_latencies.append(latency_ms)
        if len(self.recent_latencies) > 100:
            self.recent_latencies.pop(0)

        # Calculate running averages
        if self.recent_latencies:
            self.latency_ms = np.mean(self.recent_latencies)
            self.jitter_ms = np.std(self.recent_latencies)

        # Update uptime
        if success:
            self.uptime_pct = min(100.0, self.uptime_pct + 0.1)
        else:
            self.uptime_pct = max(0.0, self.uptime_pct - 1.0)

        self.last_update = time.time()

    def get_condition(self) -> NetworkCondition:
        """Assess current network condition."""
        if self.uptime_pct < 90:
            return NetworkCondition.OFFLINE
        elif self.latency_ms < 10 and self.bandwidth_mbps > 50:
            return NetworkCondition.EXCELLENT
        elif self.latency_ms < 50 and self.bandwidth_mbps > 10:
            return NetworkCondition.GOOD
        elif self.latency_ms < 100 and self.bandwidth_mbps > 2:
            return NetworkCondition.FAIR
        else:
            return NetworkCondition.POOR


class BandwidthOptimizer:
    """Optimize bandwidth usage through compression and caching."""

    def __init__(self, max_cache_size_mb: int = 500):
        self.max_cache_size_mb = max_cache_size_mb
        self.frame_cache = {}
        self.compression_enabled = True

        # Compression settings by network condition
        self.compression_settings = {
            NetworkCondition.EXCELLENT: {"quality": 95, "resize_factor": 1.0},
            NetworkCondition.GOOD: {"quality": 85, "resize_factor": 0.9},
            NetworkCondition.FAIR: {"quality": 70, "resize_factor": 0.8},
            NetworkCondition.POOR: {"quality": 50, "resize_factor": 0.7},
            NetworkCondition.OFFLINE: {"quality": 30, "resize_factor": 0.6},
        }

    def compress_frame(
        self, frame: np.ndarray, network_condition: NetworkCondition
    ) -> bytes:
        """Compress frame based on network conditions."""
        if not self.compression_enabled:
            return frame.tobytes()

        settings = self.compression_settings[network_condition]

        # Resize frame if needed
        if settings["resize_factor"] < 1.0:
            import cv2

            h, w = frame.shape[:2]
            new_h = int(h * settings["resize_factor"])
            new_w = int(w * settings["resize_factor"])
            frame = cv2.resize(frame, (new_w, new_h))

        # JPEG compression
        try:
            import cv2

            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), settings["quality"]]
            result, compressed = cv2.imencode(".jpg", frame, encode_param)

            if result:
                return compressed.tobytes()
            else:
                return frame.tobytes()

        except ImportError:
            return frame.tobytes()

    def cache_result(self, _frame_id: str, result: DetectionResult):
        """Cache inference result for similar frames."""
        # Simple caching based on frame similarity
        # In production, use perceptual hashing
        cache_key = f"{result.camera_id}_{len(result.boxes)}"

        self.frame_cache[cache_key] = {"result": result, "timestamp": time.time()}

        # Cleanup old cache entries
        self._cleanup_cache()

    def get_cached_result(
        self, frame: np.ndarray, camera_id: str
    ) -> DetectionResult | None:
        """Try to get cached result for similar frame."""
        # Simplified similarity check
        # In production, use more sophisticated matching
        cache_key = f"{camera_id}_{frame.shape[0]}_{frame.shape[1]}"

        if cache_key in self.frame_cache:
            cached = self.frame_cache[cache_key]

            # Check if cache is still fresh (within 5 seconds)
            if time.time() - cached["timestamp"] < 5.0:
                return cached["result"]

        return None

    def _cleanup_cache(self):
        """Remove old cache entries to manage memory."""
        current_time = time.time()

        # Remove entries older than 30 seconds
        expired_keys = [
            key
            for key, value in self.frame_cache.items()
            if current_time - value["timestamp"] > 30.0
        ]

        for key in expired_keys:
            del self.frame_cache[key]


class EdgeCloudRouter:
    """Route inference requests between edge and cloud based on conditions."""

    def __init__(
        self,
        edge_devices: list[EdgeDevice],
        cloud_endpoint: str = "https://api.its-camera-ai.com/inference",
    ):
        self.edge_devices = {device.device_id: device for device in edge_devices}
        self.cloud_endpoint = cloud_endpoint

        # Initialize inference engines
        self.edge_engines = {}
        self.cloud_client = None

        # Network monitoring
        self.network_metrics = NetworkMetrics()
        self.bandwidth_optimizer = BandwidthOptimizer()

        # Routing statistics
        self.routing_stats = {
            "edge_requests": 0,
            "cloud_requests": 0,
            "fallback_requests": 0,
            "cache_hits": 0,
            "total_requests": 0,
        }

        # Performance tracking
        self.edge_latencies = []
        self.cloud_latencies = []

    async def initialize(self):
        """Initialize edge engines and cloud client."""
        logger.info("Initializing edge-cloud router...")

        # Initialize edge inference engines
        for _device_id, device in self.edge_devices.items():
            if device.is_available:
                await self._initialize_edge_engine(device)

        # Initialize cloud client
        await self._initialize_cloud_client()

        # Start monitoring tasks
        self._start_monitoring_tasks()

        logger.info(f"Router initialized with {len(self.edge_engines)} edge devices")

    async def _initialize_edge_engine(self, device: EdgeDevice):
        """Initialize inference engine for edge device."""
        try:
            # Select optimal model for device
            model_type = self._select_model_for_device(device)

            # Configure for edge deployment
            config = InferenceConfig(
                model_type=model_type,
                backend=(
                    OptimizationBackend.ONNX
                    if device.gpu_memory_gb < 4
                    else OptimizationBackend.TENSORRT
                ),
                precision="fp16" if device.gpu_memory_gb >= 4 else "int8",
                batch_size=min(4, int(device.gpu_memory_gb)),
                max_batch_size=min(8, int(device.gpu_memory_gb * 2)),
                batch_timeout_ms=20,
                device_ids=[0],  # Assume single GPU per edge device
                enable_edge_optimization=True,
            )

            engine = OptimizedInferenceEngine(config)

            # Load appropriate model
            model_path = self._get_model_path(model_type)
            await engine.initialize(model_path)

            self.edge_engines[device.device_id] = engine

            logger.info(
                f"Initialized edge engine for {device.device_id} with {model_type.value}"
            )

        except Exception as e:
            logger.error(
                f"Failed to initialize edge engine for {device.device_id}: {e}"
            )
            device.is_available = False

    def _select_model_for_device(self, device: EdgeDevice) -> ModelType:
        """Select optimal model type for edge device capabilities."""
        if device.gpu_memory_gb >= 8:
            return ModelType.MEDIUM
        elif device.gpu_memory_gb >= 4:
            return ModelType.SMALL
        else:
            return ModelType.NANO

    def _get_model_path(self, model_type: ModelType) -> Path:
        """Get path to model file."""
        # In production, this would be configurable
        models_dir = Path("models")
        return models_dir / model_type.value

    async def _initialize_cloud_client(self):
        """Initialize HTTP client for cloud inference."""
        try:
            import httpx

            self.cloud_client = httpx.AsyncClient(
                base_url=self.cloud_endpoint,
                timeout=httpx.Timeout(30.0),
                limits=httpx.Limits(max_connections=20),
            )

            logger.info("Cloud inference client initialized")

        except ImportError:
            logger.warning("httpx not available - cloud inference disabled")
            self.cloud_client = None

    def _start_monitoring_tasks(self):
        """Start background monitoring tasks."""
        # In production, these would be proper async tasks
        logger.info("Network monitoring tasks started")

    async def infer(self, request: InferenceRequest) -> DetectionResult:
        """Route inference request to optimal location."""
        self.routing_stats["total_requests"] += 1
        start_time = time.time()

        try:
            # Check cache first
            cached_result = self.bandwidth_optimizer.get_cached_result(
                request.frame, request.camera_id
            )
            if cached_result:
                self.routing_stats["cache_hits"] += 1
                logger.debug(f"Cache hit for {request.frame_id}")
                return cached_result

            # Determine optimal inference location
            location = self._select_inference_location(request)

            # Execute inference
            result = await self._execute_inference(request, location)

            # Cache result
            self.bandwidth_optimizer.cache_result(request.frame_id, result)

            # Update performance metrics
            total_time = (time.time() - start_time) * 1000
            self._update_performance_stats(location, total_time)

            return result

        except Exception as e:
            logger.error(f"Inference failed for {request.frame_id}: {e}")

            # Try fallback if enabled
            if request.fallback_enabled and request.retry_count < request.max_retries:
                request.retry_count += 1
                return await self._try_fallback(request)

            raise

    def _select_inference_location(
        self, request: InferenceRequest
    ) -> InferenceLocation:
        """Select optimal inference location based on current conditions."""

        # Use explicit preference if set
        if request.preferred_location:
            return request.preferred_location

        # Get current network condition
        network_condition = self.network_metrics.get_condition()

        # Check edge device availability
        available_edge_devices = [
            device
            for device in self.edge_devices.values()
            if device.is_available and device.current_load < 0.8
        ]

        # Decision logic
        if not available_edge_devices:
            return InferenceLocation.CLOUD_ONLY

        if network_condition == NetworkCondition.OFFLINE:
            return InferenceLocation.EDGE_ONLY

        # High priority or low latency requirements favor edge
        if request.priority == "high" or request.max_latency_ms < 50:
            return InferenceLocation.EDGE_PRIMARY

        # Good network conditions and high accuracy requirements favor cloud
        if (
            network_condition in [NetworkCondition.EXCELLENT, NetworkCondition.GOOD]
            and request.min_accuracy > 0.95
        ):
            return InferenceLocation.CLOUD_PRIMARY

        # Default adaptive behavior
        if len(available_edge_devices) > 0:
            return InferenceLocation.EDGE_PRIMARY
        else:
            return InferenceLocation.CLOUD_PRIMARY

    async def _execute_inference(
        self, request: InferenceRequest, location: InferenceLocation
    ) -> DetectionResult:
        """Execute inference at specified location."""

        if location == InferenceLocation.EDGE_ONLY:
            return await self._infer_edge(request)

        elif location == InferenceLocation.CLOUD_ONLY:
            return await self._infer_cloud(request)

        elif location == InferenceLocation.EDGE_PRIMARY:
            try:
                return await self._infer_edge(request)
            except Exception as e:
                logger.warning(f"Edge inference failed, trying cloud: {e}")
                return await self._infer_cloud(request)

        elif location == InferenceLocation.CLOUD_PRIMARY:
            try:
                return await self._infer_cloud(request)
            except Exception as e:
                logger.warning(f"Cloud inference failed, trying edge: {e}")
                return await self._infer_edge(request)

        else:  # ADAPTIVE
            return await self._infer_adaptive(request)

    async def _infer_edge(self, request: InferenceRequest) -> DetectionResult:
        """Execute inference on edge device."""
        # Select best available edge device
        device = self._select_best_edge_device()

        if not device or device.device_id not in self.edge_engines:
            raise RuntimeError("No available edge devices")

        engine = self.edge_engines[device.device_id]

        self.routing_stats["edge_requests"] += 1
        device.current_load += 0.1  # Simulate load increase

        try:
            result = await engine.predict_single(
                request.frame, request.frame_id, request.camera_id
            )

            logger.debug(f"Edge inference completed for {request.frame_id}")
            return result

        finally:
            device.current_load = max(0, device.current_load - 0.1)

    async def _infer_cloud(self, request: InferenceRequest) -> DetectionResult:
        """Execute inference on cloud."""
        if not self.cloud_client:
            raise RuntimeError("Cloud client not available")

        self.routing_stats["cloud_requests"] += 1

        # Compress frame for transmission
        network_condition = self.network_metrics.get_condition()
        compressed_frame = self.bandwidth_optimizer.compress_frame(
            request.frame, network_condition
        )

        # Prepare request payload
        payload = {
            "frame_id": request.frame_id,
            "camera_id": request.camera_id,
            "timestamp": request.timestamp,
            "frame_data": compressed_frame.hex(),  # Hex encode for JSON
            "max_latency_ms": request.max_latency_ms,
            "min_accuracy": request.min_accuracy,
        }

        # Send request to cloud
        try:
            response = await self.cloud_client.post(
                "/infer", json=payload, timeout=request.max_latency_ms / 1000.0
            )

            response.raise_for_status()
            result_data = response.json()

            # Convert response to DetectionResult
            result = self._parse_cloud_response(result_data, request)

            logger.debug(f"Cloud inference completed for {request.frame_id}")
            return result

        except Exception as e:
            logger.error(f"Cloud inference failed: {e}")
            raise

    async def _infer_adaptive(self, request: InferenceRequest) -> DetectionResult:
        """Adaptive inference based on real-time conditions."""
        # Simple adaptive logic - could be much more sophisticated
        edge_load = np.mean(
            [device.current_load for device in self.edge_devices.values()]
        )
        network_condition = self.network_metrics.get_condition()

        if edge_load < 0.5 and network_condition != NetworkCondition.EXCELLENT:
            return await self._infer_edge(request)
        else:
            return await self._infer_cloud(request)

    def _select_best_edge_device(self) -> EdgeDevice | None:
        """Select best available edge device based on load and capabilities."""
        available_devices = [
            device
            for device in self.edge_devices.values()
            if device.is_available and device.current_load < 0.9
        ]

        if not available_devices:
            return None

        # Select device with lowest load
        return min(available_devices, key=lambda d: d.current_load)

    def _parse_cloud_response(
        self, response_data: dict, request: InferenceRequest
    ) -> DetectionResult:
        """Parse cloud inference response into DetectionResult."""

        # Extract detection data
        boxes = np.array(response_data.get("boxes", []))
        scores = np.array(response_data.get("scores", []))
        classes = np.array(response_data.get("classes", []))
        class_names = response_data.get("class_names", [])

        # Extract performance metrics
        inference_time = response_data.get("inference_time_ms", 0)
        total_time = response_data.get("total_time_ms", 0)

        return DetectionResult(
            boxes=boxes,
            scores=scores,
            classes=classes,
            class_names=class_names,
            frame_id=request.frame_id,
            camera_id=request.camera_id,
            timestamp=request.timestamp,
            inference_time_ms=inference_time,
            preprocessing_time_ms=0,  # Not available from cloud
            postprocessing_time_ms=0,  # Not available from cloud
            total_time_ms=total_time,
            detection_count=len(boxes),
            avg_confidence=np.mean(scores) if len(scores) > 0 else 0.0,
            gpu_memory_used_mb=0,  # Not available from cloud
        )

    async def _try_fallback(self, request: InferenceRequest) -> DetectionResult:
        """Try fallback inference location."""
        self.routing_stats["fallback_requests"] += 1

        # Determine fallback location
        if request.preferred_location == InferenceLocation.EDGE_ONLY:
            fallback_location = InferenceLocation.CLOUD_ONLY
        else:
            fallback_location = InferenceLocation.EDGE_ONLY

        return await self._execute_inference(request, fallback_location)

    def _update_performance_stats(self, location: InferenceLocation, latency_ms: float):
        """Update performance statistics."""
        if "edge" in location.value.lower():
            self.edge_latencies.append(latency_ms)
            if len(self.edge_latencies) > 1000:
                self.edge_latencies.pop(0)
        else:
            self.cloud_latencies.append(latency_ms)
            if len(self.cloud_latencies) > 1000:
                self.cloud_latencies.pop(0)

    def get_performance_stats(self) -> dict:
        """Get comprehensive performance statistics."""
        stats = {
            "routing_stats": self.routing_stats.copy(),
            "network_metrics": {
                "condition": self.network_metrics.get_condition().value,
                "latency_ms": self.network_metrics.latency_ms,
                "bandwidth_mbps": self.network_metrics.bandwidth_mbps,
                "uptime_pct": self.network_metrics.uptime_pct,
            },
            "edge_performance": {
                "available_devices": len(
                    [d for d in self.edge_devices.values() if d.is_available]
                ),
                "avg_load": np.mean(
                    [d.current_load for d in self.edge_devices.values()]
                ),
                "avg_latency_ms": (
                    np.mean(self.edge_latencies) if self.edge_latencies else 0
                ),
                "p95_latency_ms": (
                    np.percentile(self.edge_latencies, 95) if self.edge_latencies else 0
                ),
            },
            "cloud_performance": {
                "client_available": self.cloud_client is not None,
                "avg_latency_ms": (
                    np.mean(self.cloud_latencies) if self.cloud_latencies else 0
                ),
                "p95_latency_ms": (
                    np.percentile(self.cloud_latencies, 95)
                    if self.cloud_latencies
                    else 0
                ),
            },
        }

        return stats

    async def cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up edge-cloud router...")

        # Cleanup edge engines
        for engine in self.edge_engines.values():
            await engine.cleanup()

        # Cleanup cloud client
        if self.cloud_client:
            await self.cloud_client.aclose()

        logger.info("Edge-cloud router cleanup completed")


# Utility functions for deployment configuration


def create_jetson_device_config(
    device_id: str,
    model: str = "nano",  # "nano", "xavier_nx", "agx_xavier"
) -> EdgeDevice:
    """Create EdgeDevice configuration for NVIDIA Jetson devices."""

    jetson_specs = {
        "nano": {
            "gpu_memory_gb": 4.0,
            "cpu_cores": 4,
            "max_power_watts": 10.0,
            "supported_models": [ModelType.NANO, ModelType.SMALL],
            "max_throughput_fps": 15,
            "typical_latency_ms": 80,
        },
        "xavier_nx": {
            "gpu_memory_gb": 8.0,
            "cpu_cores": 6,
            "max_power_watts": 15.0,
            "supported_models": [ModelType.NANO, ModelType.SMALL, ModelType.MEDIUM],
            "max_throughput_fps": 30,
            "typical_latency_ms": 40,
        },
        "agx_xavier": {
            "gpu_memory_gb": 16.0,
            "cpu_cores": 8,
            "max_power_watts": 30.0,
            "supported_models": [
                ModelType.NANO,
                ModelType.SMALL,
                ModelType.MEDIUM,
                ModelType.LARGE,
            ],
            "max_throughput_fps": 60,
            "typical_latency_ms": 25,
        },
    }

    specs = jetson_specs.get(model, jetson_specs["nano"])

    return EdgeDevice(device_id=device_id, device_type=f"jetson_{model}", **specs)


def create_intel_device_config(
    device_id: str,
    model: str = "ncs2",  # "ncs2", "cpu_inference"
) -> EdgeDevice:
    """Create EdgeDevice configuration for Intel devices."""

    intel_specs = {
        "ncs2": {
            "gpu_memory_gb": 1.0,  # Neural Compute Stick memory
            "cpu_cores": 1,  # Virtual CPU for NCS2
            "max_power_watts": 2.5,
            "supported_models": [ModelType.NANO],
            "max_throughput_fps": 8,
            "typical_latency_ms": 120,
        },
        "cpu_inference": {
            "gpu_memory_gb": 0.0,  # CPU-only inference
            "cpu_cores": 8,
            "max_power_watts": 65.0,
            "supported_models": [ModelType.NANO, ModelType.SMALL],
            "max_throughput_fps": 5,
            "typical_latency_ms": 200,
        },
    }

    specs = intel_specs.get(model, intel_specs["ncs2"])

    return EdgeDevice(device_id=device_id, device_type=f"intel_{model}", **specs)


async def benchmark_edge_cloud_performance(
    router: EdgeCloudRouter, num_requests: int = 100
) -> dict:
    """Benchmark edge vs cloud performance."""

    logger.info(
        f"Starting edge-cloud performance benchmark with {num_requests} requests"
    )

    # Generate test requests
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Test edge-only performance
    edge_requests = []
    for i in range(num_requests // 2):
        request = InferenceRequest(
            frame=test_frame,
            frame_id=f"edge_test_{i}",
            camera_id="benchmark_camera",
            timestamp=time.time(),
            preferred_location=InferenceLocation.EDGE_ONLY,
        )
        edge_requests.append(request)

    edge_start = time.time()
    edge_results = []
    for request in edge_requests:
        try:
            result = await router.infer(request)
            edge_results.append(result)
        except Exception as e:
            logger.error(f"Edge benchmark request failed: {e}")

    edge_time = time.time() - edge_start

    # Test cloud-only performance (if available)
    cloud_requests = []
    for i in range(num_requests // 2):
        request = InferenceRequest(
            frame=test_frame,
            frame_id=f"cloud_test_{i}",
            camera_id="benchmark_camera",
            timestamp=time.time(),
            preferred_location=InferenceLocation.CLOUD_ONLY,
        )
        cloud_requests.append(request)

    cloud_start = time.time()
    cloud_results = []
    for request in cloud_requests:
        try:
            result = await router.infer(request)
            cloud_results.append(result)
        except Exception as e:
            logger.error(f"Cloud benchmark request failed: {e}")

    cloud_time = time.time() - cloud_start

    # Calculate metrics
    benchmark_results = {
        "edge_performance": {
            "requests": len(edge_requests),
            "successful": len(edge_results),
            "total_time_s": edge_time,
            "avg_latency_ms": (
                np.mean([r.total_time_ms for r in edge_results]) if edge_results else 0
            ),
            "throughput_rps": len(edge_results) / edge_time if edge_time > 0 else 0,
        },
        "cloud_performance": {
            "requests": len(cloud_requests),
            "successful": len(cloud_results),
            "total_time_s": cloud_time,
            "avg_latency_ms": (
                np.mean([r.total_time_ms for r in cloud_results])
                if cloud_results
                else 0
            ),
            "throughput_rps": len(cloud_results) / cloud_time if cloud_time > 0 else 0,
        },
        "router_stats": router.get_performance_stats(),
    }

    logger.info(f"Benchmark completed: {benchmark_results}")
    return benchmark_results
