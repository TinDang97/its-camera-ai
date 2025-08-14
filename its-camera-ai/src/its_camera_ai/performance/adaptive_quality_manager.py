"""Adaptive quality management for streaming system under load conditions.

This module implements dynamic quality adjustment based on system performance
to maintain SLA compliance under high load with priority streaming and
gradual quality recovery mechanisms.
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import psutil
import structlog

from .optimization_config import AdaptiveQualityConfig

logger = structlog.get_logger(__name__)

# GPU monitoring imports with availability checks
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    import GPUtil
    GPU_UTIL_AVAILABLE = True
except ImportError:
    GPU_UTIL_AVAILABLE = False
    GPUtil = None


class QualityLevel(str, Enum):
    """Quality levels for adaptive adjustment."""

    ULTRA_LOW = "ultra_low"      # Emergency mode: 240p, 15fps, minimal ML
    LOW = "low"                  # Degraded: 360p, 20fps, simplified ML
    MEDIUM = "medium"            # Balanced: 720p, 25fps, standard ML
    HIGH = "high"                # Optimal: 1080p, 30fps, full ML
    ULTRA_HIGH = "ultra_high"    # Premium: 4K, 30fps, enhanced ML


class LoadCondition(str, Enum):
    """System load conditions."""

    OPTIMAL = "optimal"          # System running optimally
    ELEVATED = "elevated"        # Slightly elevated load
    HIGH = "high"                # High load, quality reduction needed
    CRITICAL = "critical"        # Critical load, emergency mode
    OVERLOAD = "overload"        # System overloaded, reject new streams


@dataclass
class SystemMetrics:
    """System performance metrics for quality decisions."""

    cpu_percent: float
    memory_percent: float
    gpu_percent: float = 0.0
    gpu_memory_percent: float = 0.0
    active_streams: int = 0
    average_latency_ms: float = 0.0
    queue_depth: int = 0
    error_rate: float = 0.0
    timestamp: float = field(default_factory=time.time)

    @property
    def load_condition(self) -> LoadCondition:
        """Determine overall load condition."""
        # Critical thresholds
        if (self.cpu_percent > 90 or
            self.memory_percent > 95 or
            self.gpu_percent > 95 or
            self.average_latency_ms > 200):
            return LoadCondition.CRITICAL

        # High load thresholds
        if (self.cpu_percent > 80 or
            self.memory_percent > 85 or
            self.gpu_percent > 85 or
            self.average_latency_ms > 150):
            return LoadCondition.HIGH

        # Elevated load thresholds
        if (self.cpu_percent > 60 or
            self.memory_percent > 70 or
            self.gpu_percent > 70 or
            self.average_latency_ms > 120):
            return LoadCondition.ELEVATED

        # Overload condition (reject new streams)
        if (self.cpu_percent > 95 or
            self.memory_percent > 98 or
            self.error_rate > 0.1):
            return LoadCondition.OVERLOAD

        return LoadCondition.OPTIMAL


@dataclass
class QualityProfile:
    """Quality profile defining stream parameters."""

    level: QualityLevel
    resolution: tuple[int, int]  # (width, height)
    framerate: int
    bitrate_kbps: int
    ml_complexity: str  # "minimal", "simple", "standard", "full", "enhanced"
    encoding_preset: str  # "ultrafast", "fast", "medium", "slow"

    @property
    def computational_cost(self) -> float:
        """Estimated computational cost (0.0 to 1.0)."""
        # Resolution factor
        pixel_count = self.resolution[0] * self.resolution[1]
        resolution_factor = pixel_count / (1920 * 1080)  # Normalize to 1080p

        # Framerate factor
        framerate_factor = self.framerate / 30.0  # Normalize to 30fps

        # ML complexity factor
        ml_factors = {
            "minimal": 0.1,
            "simple": 0.3,
            "standard": 0.6,
            "full": 1.0,
            "enhanced": 1.5,
        }
        ml_factor = ml_factors.get(self.ml_complexity, 0.6)

        # Encoding factor (inverse - faster presets cost more CPU)
        encoding_factors = {
            "ultrafast": 1.0,
            "fast": 0.8,
            "medium": 0.6,
            "slow": 0.4,
        }
        encoding_factor = encoding_factors.get(self.encoding_preset, 0.6)

        return (resolution_factor * framerate_factor * ml_factor * encoding_factor)


class QualityProfileManager:
    """Manages quality profiles for different quality levels."""

    def __init__(self):
        """Initialize quality profile manager."""
        self.profiles = self._create_default_profiles()

    def _create_default_profiles(self) -> dict[QualityLevel, QualityProfile]:
        """Create default quality profiles.
        
        Returns:
            Dict[QualityLevel, QualityProfile]: Quality profiles
        """
        return {
            QualityLevel.ULTRA_LOW: QualityProfile(
                level=QualityLevel.ULTRA_LOW,
                resolution=(426, 240),  # 240p
                framerate=15,
                bitrate_kbps=200,
                ml_complexity="minimal",
                encoding_preset="ultrafast"
            ),
            QualityLevel.LOW: QualityProfile(
                level=QualityLevel.LOW,
                resolution=(640, 360),  # 360p
                framerate=20,
                bitrate_kbps=500,
                ml_complexity="simple",
                encoding_preset="fast"
            ),
            QualityLevel.MEDIUM: QualityProfile(
                level=QualityLevel.MEDIUM,
                resolution=(1280, 720),  # 720p
                framerate=25,
                bitrate_kbps=1500,
                ml_complexity="standard",
                encoding_preset="medium"
            ),
            QualityLevel.HIGH: QualityProfile(
                level=QualityLevel.HIGH,
                resolution=(1920, 1080),  # 1080p
                framerate=30,
                bitrate_kbps=3000,
                ml_complexity="full",
                encoding_preset="medium"
            ),
            QualityLevel.ULTRA_HIGH: QualityProfile(
                level=QualityLevel.ULTRA_HIGH,
                resolution=(3840, 2160),  # 4K
                framerate=30,
                bitrate_kbps=8000,
                ml_complexity="enhanced",
                encoding_preset="slow"
            ),
        }

    def get_profile(self, level: QualityLevel) -> QualityProfile:
        """Get quality profile for level.
        
        Args:
            level: Quality level
            
        Returns:
            QualityProfile: Quality profile
        """
        return self.profiles[level]

    def get_optimal_quality_for_load(self, system_metrics: SystemMetrics) -> QualityLevel:
        """Get optimal quality level for current system load.
        
        Args:
            system_metrics: Current system metrics
            
        Returns:
            QualityLevel: Recommended quality level
        """
        load_condition = system_metrics.load_condition

        # Map load conditions to quality levels
        quality_mapping = {
            LoadCondition.OPTIMAL: QualityLevel.HIGH,
            LoadCondition.ELEVATED: QualityLevel.MEDIUM,
            LoadCondition.HIGH: QualityLevel.LOW,
            LoadCondition.CRITICAL: QualityLevel.ULTRA_LOW,
            LoadCondition.OVERLOAD: QualityLevel.ULTRA_LOW,
        }

        return quality_mapping.get(load_condition, QualityLevel.MEDIUM)


class SystemMonitor:
    """Monitors system resources for quality decisions."""

    def __init__(self, monitoring_interval: float = 5.0):
        """Initialize system monitor.
        
        Args:
            monitoring_interval: Monitoring interval in seconds
        """
        self.monitoring_interval = monitoring_interval
        self.is_monitoring = False
        self.monitoring_task: asyncio.Task | None = None
        self.current_metrics: SystemMetrics | None = None
        self.metrics_history: list[SystemMetrics] = []
        self.max_history = 100  # Keep last 100 measurements

    async def start_monitoring(self) -> None:
        """Start system monitoring."""
        if self.is_monitoring:
            return

        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("System monitoring started")

    async def stop_monitoring(self) -> None:
        """Stop system monitoring."""
        self.is_monitoring = False

        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass

        logger.info("System monitoring stopped")

    async def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while self.is_monitoring:
            try:
                # Collect system metrics
                metrics = await self._collect_metrics()

                # Update current metrics
                self.current_metrics = metrics

                # Add to history
                self.metrics_history.append(metrics)
                if len(self.metrics_history) > self.max_history:
                    self.metrics_history.pop(0)

                await asyncio.sleep(self.monitoring_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"System monitoring error: {e}")
                await asyncio.sleep(self.monitoring_interval)

    async def _collect_metrics(self) -> SystemMetrics:
        """Collect current system metrics.
        
        Returns:
            SystemMetrics: Current system metrics
        """
        try:
            # CPU and memory metrics
            cpu_percent = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()
            memory_percent = memory.percent

            # GPU metrics
            gpu_percent = 0.0
            gpu_memory_percent = 0.0

            if TORCH_AVAILABLE and torch.cuda.is_available():
                try:
                    # PyTorch GPU utilization (approximate)
                    gpu_memory_allocated = torch.cuda.memory_allocated()
                    gpu_memory_total = torch.cuda.get_device_properties(0).total_memory
                    gpu_memory_percent = (gpu_memory_allocated / gpu_memory_total) * 100

                    # Try to get GPU utilization if available
                    if GPU_UTIL_AVAILABLE:
                        gpus = GPUtil.getGPUs()
                        if gpus:
                            gpu_percent = gpus[0].load * 100

                except Exception as e:
                    logger.debug(f"GPU metrics collection failed: {e}")

            return SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                gpu_percent=gpu_percent,
                gpu_memory_percent=gpu_memory_percent,
                timestamp=time.time()
            )

        except Exception as e:
            logger.error(f"Metrics collection failed: {e}")
            return SystemMetrics(
                cpu_percent=0.0,
                memory_percent=0.0,
                gpu_percent=0.0,
                gpu_memory_percent=0.0,
                timestamp=time.time()
            )

    def get_current_metrics(self) -> SystemMetrics | None:
        """Get current system metrics.
        
        Returns:
            Optional[SystemMetrics]: Current metrics or None
        """
        return self.current_metrics

    def get_average_metrics(self, window_seconds: int = 60) -> SystemMetrics | None:
        """Get average metrics over a time window.
        
        Args:
            window_seconds: Time window in seconds
            
        Returns:
            Optional[SystemMetrics]: Average metrics or None
        """
        if not self.metrics_history:
            return None

        current_time = time.time()
        cutoff_time = current_time - window_seconds

        # Filter metrics within window
        recent_metrics = [
            m for m in self.metrics_history
            if m.timestamp > cutoff_time
        ]

        if not recent_metrics:
            return None

        # Calculate averages
        avg_cpu = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_percent for m in recent_metrics) / len(recent_metrics)
        avg_gpu = sum(m.gpu_percent for m in recent_metrics) / len(recent_metrics)
        avg_gpu_memory = sum(m.gpu_memory_percent for m in recent_metrics) / len(recent_metrics)

        return SystemMetrics(
            cpu_percent=avg_cpu,
            memory_percent=avg_memory,
            gpu_percent=avg_gpu,
            gpu_memory_percent=avg_gpu_memory,
            timestamp=current_time
        )


@dataclass
class CameraQualityState:
    """Tracks quality state for individual cameras."""

    camera_id: str
    current_quality: QualityLevel
    target_quality: QualityLevel
    is_priority: bool
    last_adjustment: float
    adjustment_count: int
    stable_since: float

    @property
    def is_stable(self) -> bool:
        """Check if quality has been stable."""
        return (time.time() - self.stable_since) > 60.0  # Stable for 1 minute

    def update_quality(self, new_quality: QualityLevel) -> None:
        """Update quality level.
        
        Args:
            new_quality: New quality level
        """
        if new_quality != self.current_quality:
            self.current_quality = new_quality
            self.last_adjustment = time.time()
            self.adjustment_count += 1
            self.stable_since = time.time()
        else:
            # Quality unchanged, reset stable timer if it was recent
            if (time.time() - self.stable_since) < 30.0:
                self.stable_since = time.time()


class AdaptiveQualityManager:
    """Adaptive quality management for streaming system performance.
    
    Implements dynamic quality adjustment based on system performance
    to maintain SLA compliance under high load conditions with priority
    streaming and gradual quality recovery.
    """

    def __init__(self, config: AdaptiveQualityConfig):
        """Initialize adaptive quality manager.
        
        Args:
            config: Adaptive quality configuration
        """
        self.config = config

        # Quality management components
        self.profile_manager = QualityProfileManager()
        self.system_monitor = SystemMonitor(config.quality_adjustment_interval_seconds)

        # Camera quality tracking
        self.camera_states: dict[str, CameraQualityState] = {}
        self.priority_cameras: set[str] = set(config.priority_camera_ids)

        # Quality adjustment state
        self.global_quality_level = QualityLevel(config.default_quality)
        self.last_global_adjustment = time.time()
        self.adjustment_history: list[tuple[float, QualityLevel, str]] = []

        # Recovery tracking
        self.recovery_in_progress = False
        self.recovery_start_time: float | None = None

        # Statistics
        self.total_adjustments = 0
        self.quality_distribution: dict[QualityLevel, int] = dict.fromkeys(QualityLevel, 0)

        logger.info("AdaptiveQualityManager initialized")

    async def start(self) -> None:
        """Start adaptive quality management."""
        await self.system_monitor.start_monitoring()

        # Start quality adjustment task
        asyncio.create_task(self._quality_adjustment_loop())

        logger.info("Adaptive quality management started")

    async def stop(self) -> None:
        """Stop adaptive quality management."""
        await self.system_monitor.stop_monitoring()
        logger.info("Adaptive quality management stopped")

    async def register_camera(
        self,
        camera_id: str,
        is_priority: bool = False
    ) -> CameraQualityState:
        """Register camera for quality management.
        
        Args:
            camera_id: Camera identifier
            is_priority: Whether camera is high priority
            
        Returns:
            CameraQualityState: Camera quality state
        """
        if camera_id in self.camera_states:
            return self.camera_states[camera_id]

        # Set priority status
        if is_priority or camera_id in self.priority_cameras:
            self.priority_cameras.add(camera_id)
            is_priority = True

        # Create camera state
        state = CameraQualityState(
            camera_id=camera_id,
            current_quality=self.global_quality_level,
            target_quality=self.global_quality_level,
            is_priority=is_priority,
            last_adjustment=time.time(),
            adjustment_count=0,
            stable_since=time.time()
        )

        self.camera_states[camera_id] = state
        self.quality_distribution[self.global_quality_level] += 1

        logger.info(f"Registered camera {camera_id} (priority: {is_priority})")
        return state

    async def unregister_camera(self, camera_id: str) -> None:
        """Unregister camera from quality management.
        
        Args:
            camera_id: Camera identifier
        """
        if camera_id in self.camera_states:
            state = self.camera_states.pop(camera_id)
            self.quality_distribution[state.current_quality] -= 1
            logger.info(f"Unregistered camera {camera_id}")

    async def adjust_quality_based_on_load(
        self,
        system_metrics: SystemMetrics
    ) -> dict[str, QualityLevel]:
        """Adjust quality based on current system load.
        
        Args:
            system_metrics: Current system metrics
            
        Returns:
            Dict[str, QualityLevel]: Camera ID -> new quality level
        """
        try:
            # Determine optimal quality for current load
            optimal_quality = self.profile_manager.get_optimal_quality_for_load(system_metrics)

            quality_adjustments: dict[str, QualityLevel] = {}

            # Check if global adjustment is needed
            if (optimal_quality != self.global_quality_level and
                time.time() - self.last_global_adjustment > self.config.quality_adjustment_interval_seconds):

                # Apply quality adjustment strategy
                adjustments = await self._apply_quality_strategy(
                    optimal_quality, system_metrics
                )
                quality_adjustments.update(adjustments)

                # Update global quality level
                self.global_quality_level = optimal_quality
                self.last_global_adjustment = time.time()

                # Record adjustment
                self.adjustment_history.append((
                    time.time(),
                    optimal_quality,
                    f"Load condition: {system_metrics.load_condition.value}"
                ))

                self.total_adjustments += 1

                logger.info(
                    f"Quality adjustment: {len(quality_adjustments)} cameras to "
                    f"{optimal_quality.value} (load: {system_metrics.load_condition.value})"
                )

            # Check for quality recovery opportunity
            elif (self.config.quality_recovery_enabled and
                  system_metrics.load_condition in [LoadCondition.OPTIMAL, LoadCondition.ELEVATED] and
                  not self.recovery_in_progress):

                recovery_adjustments = await self._attempt_quality_recovery(system_metrics)
                quality_adjustments.update(recovery_adjustments)

            return quality_adjustments

        except Exception as e:
            logger.error(f"Quality adjustment failed: {e}")
            return {}

    async def _apply_quality_strategy(
        self,
        target_quality: QualityLevel,
        system_metrics: SystemMetrics
    ) -> dict[str, QualityLevel]:
        """Apply quality adjustment strategy.
        
        Args:
            target_quality: Target quality level
            system_metrics: Current system metrics
            
        Returns:
            Dict[str, QualityLevel]: Camera adjustments
        """
        adjustments: dict[str, QualityLevel] = {}

        if self.config.gradual_quality_reduction:
            # Gradual reduction: adjust cameras in priority order
            adjustments = await self._gradual_quality_adjustment(target_quality, system_metrics)
        else:
            # Immediate adjustment: adjust all cameras to target quality
            for camera_id, state in self.camera_states.items():
                # Priority cameras get special treatment
                if (state.is_priority and
                    self.config.priority_quality_protection and
                    target_quality < state.current_quality):
                    # Protect priority cameras from quality reduction
                    continue

                state.update_quality(target_quality)
                adjustments[camera_id] = target_quality

                # Update statistics
                self.quality_distribution[state.current_quality] += 1
                if state.current_quality != target_quality:
                    self.quality_distribution[state.current_quality] -= 1

        return adjustments

    async def _gradual_quality_adjustment(
        self,
        target_quality: QualityLevel,
        system_metrics: SystemMetrics
    ) -> dict[str, QualityLevel]:
        """Apply gradual quality adjustment.
        
        Args:
            target_quality: Target quality level
            system_metrics: Current system metrics
            
        Returns:
            Dict[str, QualityLevel]: Camera adjustments
        """
        adjustments: dict[str, QualityLevel] = {}

        # Sort cameras by priority (non-priority cameras first for reduction)
        cameras_by_priority = sorted(
            self.camera_states.items(),
            key=lambda x: (x[1].is_priority, -x[1].adjustment_count),
            reverse=(target_quality > self.global_quality_level)  # Reverse for quality increase
        )

        # Calculate how many cameras to adjust
        total_cameras = len(self.camera_states)
        adjustment_batch_size = max(1, total_cameras // 4)  # Adjust 25% at a time

        adjusted_count = 0
        for camera_id, state in cameras_by_priority:
            if adjusted_count >= adjustment_batch_size:
                break

            # Skip priority cameras if protected
            if (state.is_priority and
                self.config.priority_quality_protection and
                target_quality < state.current_quality):
                continue

            # Apply adjustment
            old_quality = state.current_quality
            state.update_quality(target_quality)
            adjustments[camera_id] = target_quality

            # Update statistics
            self.quality_distribution[old_quality] -= 1
            self.quality_distribution[target_quality] += 1

            adjusted_count += 1

        return adjustments

    async def _attempt_quality_recovery(
        self,
        system_metrics: SystemMetrics
    ) -> dict[str, QualityLevel]:
        """Attempt gradual quality recovery when system load improves.
        
        Args:
            system_metrics: Current system metrics
            
        Returns:
            Dict[str, QualityLevel]: Recovery adjustments
        """
        if not self.config.quality_recovery_enabled:
            return {}

        # Check if recovery conditions are met
        if system_metrics.load_condition not in [LoadCondition.OPTIMAL, LoadCondition.ELEVATED]:
            return {}

        # Find cameras that can be upgraded
        upgrade_candidates = []
        for camera_id, state in self.camera_states.items():
            if (state.current_quality < QualityLevel.HIGH and
                state.is_stable and
                time.time() - state.last_adjustment > self.config.recovery_stability_period_minutes * 60):
                upgrade_candidates.append((camera_id, state))

        if not upgrade_candidates:
            return {}

        # Start recovery process
        self.recovery_in_progress = True
        self.recovery_start_time = time.time()

        adjustments: dict[str, QualityLevel] = {}

        # Upgrade a small batch of cameras (gradual recovery)
        upgrade_batch_size = max(1, len(upgrade_candidates) // 8)  # Upgrade 12.5% at a time

        # Prioritize cameras for upgrade
        upgrade_candidates.sort(key=lambda x: (x[1].is_priority, -x[1].last_adjustment), reverse=True)

        for camera_id, state in upgrade_candidates[:upgrade_batch_size]:
            # Determine next quality level
            quality_levels = list(QualityLevel)
            current_index = quality_levels.index(state.current_quality)

            if current_index < len(quality_levels) - 1:
                next_quality = quality_levels[current_index + 1]

                old_quality = state.current_quality
                state.update_quality(next_quality)
                adjustments[camera_id] = next_quality

                # Update statistics
                self.quality_distribution[old_quality] -= 1
                self.quality_distribution[next_quality] += 1

        logger.info(f"Quality recovery: upgraded {len(adjustments)} cameras")

        # Reset recovery flag after delay
        asyncio.create_task(self._reset_recovery_flag())

        return adjustments

    async def _reset_recovery_flag(self) -> None:
        """Reset recovery flag after stability period."""
        await asyncio.sleep(self.config.recovery_stability_period_minutes * 60)
        self.recovery_in_progress = False
        self.recovery_start_time = None

    async def _quality_adjustment_loop(self) -> None:
        """Background loop for quality adjustments."""
        while True:
            try:
                await asyncio.sleep(self.config.quality_adjustment_interval_seconds)

                if not self.config.enable_adaptive_quality:
                    continue

                # Get current system metrics
                current_metrics = self.system_monitor.get_current_metrics()
                if not current_metrics:
                    continue

                # Update metrics with streaming-specific data
                current_metrics.active_streams = len(self.camera_states)

                # Apply quality adjustments
                adjustments = await self.adjust_quality_based_on_load(current_metrics)

                # Log significant adjustments
                if adjustments:
                    logger.info(f"Applied {len(adjustments)} quality adjustments")

            except Exception as e:
                logger.error(f"Quality adjustment loop error: {e}")
                await asyncio.sleep(self.config.quality_adjustment_interval_seconds)

    def get_quality_profile(self, camera_id: str) -> QualityProfile | None:
        """Get current quality profile for camera.
        
        Args:
            camera_id: Camera identifier
            
        Returns:
            Optional[QualityProfile]: Quality profile or None
        """
        if camera_id not in self.camera_states:
            return None

        state = self.camera_states[camera_id]
        return self.profile_manager.get_profile(state.current_quality)

    def implement_priority_streaming(
        self,
        priority_cameras: list[str]
    ) -> dict[str, bool]:
        """Implement priority streaming for specified cameras.
        
        Args:
            priority_cameras: List of priority camera IDs
            
        Returns:
            Dict[str, bool]: Camera ID -> priority status set
        """
        results = {}

        for camera_id in priority_cameras:
            if camera_id in self.camera_states:
                self.camera_states[camera_id].is_priority = True
                self.priority_cameras.add(camera_id)
                results[camera_id] = True
            else:
                results[camera_id] = False

        logger.info(f"Set priority status for {sum(results.values())} cameras")
        return results

    def get_comprehensive_metrics(self) -> dict[str, Any]:
        """Get comprehensive quality management metrics.
        
        Returns:
            Dict[str, Any]: Quality management metrics
        """
        system_metrics = self.system_monitor.get_current_metrics()

        return {
            "enabled": self.config.enable_adaptive_quality,
            "global_quality_level": self.global_quality_level.value,
            "total_cameras": len(self.camera_states),
            "priority_cameras": len(self.priority_cameras),
            "total_adjustments": self.total_adjustments,
            "recovery_in_progress": self.recovery_in_progress,
            "quality_distribution": {
                level.value: count for level, count in self.quality_distribution.items()
            },
            "camera_states": {
                camera_id: {
                    "current_quality": state.current_quality.value,
                    "is_priority": state.is_priority,
                    "adjustment_count": state.adjustment_count,
                    "is_stable": state.is_stable,
                }
                for camera_id, state in self.camera_states.items()
            },
            "system_metrics": system_metrics.__dict__ if system_metrics else None,
            "recent_adjustments": [
                {
                    "timestamp": timestamp,
                    "quality": quality.value,
                    "reason": reason,
                }
                for timestamp, quality, reason in self.adjustment_history[-10:]
            ],
        }


async def create_adaptive_quality_manager(
    config: AdaptiveQualityConfig
) -> AdaptiveQualityManager:
    """Create and initialize adaptive quality manager.
    
    Args:
        config: Adaptive quality configuration
        
    Returns:
        AdaptiveQualityManager: Initialized quality manager
    """
    manager = AdaptiveQualityManager(config)
    await manager.start()
    return manager
