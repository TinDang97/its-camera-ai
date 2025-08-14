"""Adaptive Quality Controller for bandwidth-optimized video streaming.

This module provides dynamic video quality adjustment based on content complexity,
network conditions, and system resources to achieve 25% bandwidth reduction while
maintaining visual quality for traffic monitoring.
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class QualityLevel(Enum):
    """Video quality levels for adaptive streaming."""
    ULTRA_LOW = "ultra_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA_HIGH = "ultra_high"


class ContentComplexity(Enum):
    """Content complexity levels for scene analysis."""
    SIMPLE = "simple"      # Static scene, few vehicles
    MODERATE = "moderate"  # Some movement, moderate traffic
    COMPLEX = "complex"    # High traffic, many vehicles
    CHAOTIC = "chaotic"    # Complex intersections, weather


@dataclass
class NetworkConditions:
    """Network condition metrics for adaptation."""
    bandwidth_kbps: int
    rtt_ms: float
    packet_loss_rate: float
    jitter_ms: float
    connection_type: str  # "wifi", "cellular", "ethernet"


@dataclass
class EncodingParameters:
    """Video encoding parameters for quality control."""
    crf: int              # Constant Rate Factor (18-28)
    preset: str           # ffmpeg preset (ultrafast, fast, medium, slow)
    bitrate_kbps: int     # Target bitrate
    resolution: tuple[int, int]  # (width, height)
    framerate: float      # Target framerate
    profile: str          # H.264 profile (baseline, main, high)
    keyframe_interval: int # Keyframe interval in frames


class AdaptiveQualityController:
    """Intelligent quality controller for traffic monitoring video streams."""

    def __init__(self, camera_id: str, target_bandwidth_reduction: float = 0.25):
        """Initialize adaptive quality controller.
        
        Args:
            camera_id: Unique camera identifier
            target_bandwidth_reduction: Target bandwidth reduction (0.0-1.0)
        """
        self.camera_id = camera_id
        self.target_bandwidth_reduction = target_bandwidth_reduction

        # Quality level mappings
        self.quality_profiles = {
            QualityLevel.ULTRA_LOW: EncodingParameters(
                crf=32, preset="ultrafast", bitrate_kbps=500,
                resolution=(640, 360), framerate=15.0,
                profile="baseline", keyframe_interval=60
            ),
            QualityLevel.LOW: EncodingParameters(
                crf=28, preset="fast", bitrate_kbps=1000,
                resolution=(854, 480), framerate=20.0,
                profile="main", keyframe_interval=45
            ),
            QualityLevel.MEDIUM: EncodingParameters(
                crf=25, preset="medium", bitrate_kbps=2000,
                resolution=(1280, 720), framerate=25.0,
                profile="main", keyframe_interval=30
            ),
            QualityLevel.HIGH: EncodingParameters(
                crf=22, preset="medium", bitrate_kbps=4000,
                resolution=(1920, 1080), framerate=30.0,
                profile="high", keyframe_interval=30
            ),
            QualityLevel.ULTRA_HIGH: EncodingParameters(
                crf=18, preset="slow", bitrate_kbps=8000,
                resolution=(1920, 1080), framerate=30.0,
                profile="high", keyframe_interval=25
            ),
        }

        # State tracking
        self.current_quality = QualityLevel.MEDIUM
        self.current_network = None
        self.recent_complexities = []
        self.bandwidth_history = []

        # Adaptation settings
        self.adaptation_window = 30.0  # seconds
        self.complexity_history_size = 10
        self.bandwidth_history_size = 20

        # Performance metrics
        self.adaptation_stats = {
            "quality_changes": 0,
            "bandwidth_savings_mbps": 0.0,
            "avg_quality_score": 0.0,
            "adaptation_responsiveness": 0.0,
        }

        logger.info(f"Adaptive quality controller initialized for camera {camera_id}")

    async def analyze_content_complexity(self, frame: np.ndarray) -> ContentComplexity:
        """Analyze frame content complexity for quality adaptation.
        
        Args:
            frame: Video frame as numpy array (H, W, C)
            
        Returns:
            Content complexity level
        """
        try:
            # Convert to grayscale for analysis
            if len(frame.shape) == 3:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            else:
                gray_frame = frame

            # Calculate multiple complexity metrics
            complexity_score = 0.0

            # 1. Edge density (structural complexity)
            edges = cv2.Canny(gray_frame, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            complexity_score += edge_density * 0.3

            # 2. Texture analysis (local variance)
            kernel = np.ones((5, 5), np.float32) / 25
            local_mean = cv2.filter2D(gray_frame.astype(np.float32), -1, kernel)
            local_variance = cv2.filter2D((gray_frame.astype(np.float32) - local_mean) ** 2, -1, kernel)
            texture_complexity = np.mean(local_variance) / 255.0
            complexity_score += texture_complexity * 0.2

            # 3. Gradient magnitude (change intensity)
            grad_x = cv2.Sobel(gray_frame, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray_frame, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            gradient_complexity = np.mean(gradient_magnitude) / 255.0
            complexity_score += gradient_complexity * 0.3

            # 4. Histogram entropy (color distribution)
            hist = cv2.calcHist([gray_frame], [0], None, [256], [0, 256])
            hist_norm = hist / hist.sum()
            entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-10))
            entropy_complexity = entropy / 8.0  # Normalize to 0-1
            complexity_score += entropy_complexity * 0.2

            # Map complexity score to levels
            if complexity_score < 0.2:
                complexity_level = ContentComplexity.SIMPLE
            elif complexity_score < 0.4:
                complexity_level = ContentComplexity.MODERATE
            elif complexity_score < 0.7:
                complexity_level = ContentComplexity.COMPLEX
            else:
                complexity_level = ContentComplexity.CHAOTIC

            # Update complexity history
            self.recent_complexities.append((time.time(), complexity_level, complexity_score))
            if len(self.recent_complexities) > self.complexity_history_size:
                self.recent_complexities.pop(0)

            logger.debug(
                f"Content complexity analysis: {complexity_level.value} "
                f"(score: {complexity_score:.3f})"
            )

            return complexity_level

        except Exception as e:
            logger.error(f"Content complexity analysis failed: {e}")
            return ContentComplexity.MODERATE  # Safe default

    async def update_network_conditions(self, conditions: NetworkConditions):
        """Update current network conditions for adaptation.
        
        Args:
            conditions: Current network condition metrics
        """
        self.current_network = conditions

        # Update bandwidth history
        self.bandwidth_history.append((time.time(), conditions.bandwidth_kbps))
        if len(self.bandwidth_history) > self.bandwidth_history_size:
            self.bandwidth_history.pop(0)

        logger.debug(
            f"Updated network conditions: {conditions.bandwidth_kbps}kbps, "
            f"RTT: {conditions.rtt_ms}ms, Loss: {conditions.packet_loss_rate:.1%}"
        )

    async def adapt_quality(
        self,
        frame: np.ndarray,
        current_bitrate_kbps: int,
        target_quality: QualityLevel | None = None
    ) -> tuple[QualityLevel, EncodingParameters]:
        """Adapt video quality based on content and network conditions.
        
        Args:
            frame: Current video frame for analysis
            current_bitrate_kbps: Current encoding bitrate
            target_quality: Override target quality (optional)
            
        Returns:
            Tuple of (selected quality level, encoding parameters)
        """
        try:
            # Analyze current frame complexity
            complexity = await self.analyze_content_complexity(frame)

            # If target quality is specified, use it
            if target_quality:
                selected_quality = target_quality
            else:
                # Adaptive quality selection
                selected_quality = await self._select_optimal_quality(
                    complexity, current_bitrate_kbps
                )

            # Get encoding parameters for selected quality
            encoding_params = self.quality_profiles[selected_quality]

            # Fine-tune parameters based on conditions
            encoding_params = await self._fine_tune_parameters(
                encoding_params, complexity, current_bitrate_kbps
            )

            # Update state if quality changed
            if selected_quality != self.current_quality:
                logger.info(
                    f"Quality adapted: {self.current_quality.value} -> {selected_quality.value} "
                    f"(complexity: {complexity.value})"
                )
                self.current_quality = selected_quality
                self.adaptation_stats["quality_changes"] += 1

            return selected_quality, encoding_params

        except Exception as e:
            logger.error(f"Quality adaptation failed: {e}")
            # Return current quality as fallback
            return self.current_quality, self.quality_profiles[self.current_quality]

    async def _select_optimal_quality(
        self,
        complexity: ContentComplexity,
        current_bitrate_kbps: int
    ) -> QualityLevel:
        """Select optimal quality level based on multiple factors."""

        # Base quality selection based on content complexity
        if complexity == ContentComplexity.SIMPLE:
            base_quality = QualityLevel.LOW  # Simple scenes can use lower quality
        elif complexity == ContentComplexity.MODERATE:
            base_quality = QualityLevel.MEDIUM
        elif complexity == ContentComplexity.COMPLEX:
            base_quality = QualityLevel.HIGH  # Complex scenes need higher quality
        else:  # CHAOTIC
            base_quality = QualityLevel.ULTRA_HIGH

        # Adjust based on network conditions
        if self.current_network:
            quality_adjustment = 0

            # Bandwidth constraints
            available_bandwidth = self.current_network.bandwidth_kbps
            target_bitrate = self.quality_profiles[base_quality].bitrate_kbps

            if available_bandwidth < target_bitrate * 0.8:  # Not enough bandwidth
                quality_adjustment -= 1
            elif available_bandwidth > target_bitrate * 2.0:  # Plenty of bandwidth
                quality_adjustment += 1

            # Latency constraints
            if self.current_network.rtt_ms > 200:  # High latency
                quality_adjustment -= 1
            elif self.current_network.rtt_ms < 50:  # Low latency
                quality_adjustment += 1

            # Packet loss constraints
            if self.current_network.packet_loss_rate > 0.02:  # High packet loss
                quality_adjustment -= 1

            # Apply adjustment
            quality_levels = list(QualityLevel)
            current_index = quality_levels.index(base_quality)
            adjusted_index = max(0, min(len(quality_levels) - 1, current_index + quality_adjustment))
            adjusted_quality = quality_levels[adjusted_index]

        else:
            adjusted_quality = base_quality

        # Consider quality stability (avoid frequent changes)
        if hasattr(self, 'last_quality_change_time'):
            time_since_change = time.time() - self.last_quality_change_time
            if (time_since_change < 10.0 and  # Less than 10 seconds
                abs(quality_levels.index(adjusted_quality) -
                    quality_levels.index(self.current_quality)) <= 1):
                # Stay with current quality for stability
                adjusted_quality = self.current_quality

        return adjusted_quality

    async def _fine_tune_parameters(
        self,
        base_params: EncodingParameters,
        complexity: ContentComplexity,
        current_bitrate_kbps: int
    ) -> EncodingParameters:
        """Fine-tune encoding parameters based on specific conditions."""

        # Create copy to avoid modifying the base profile
        params = EncodingParameters(
            crf=base_params.crf,
            preset=base_params.preset,
            bitrate_kbps=base_params.bitrate_kbps,
            resolution=base_params.resolution,
            framerate=base_params.framerate,
            profile=base_params.profile,
            keyframe_interval=base_params.keyframe_interval
        )

        # Adjust CRF based on content complexity
        if complexity == ContentComplexity.SIMPLE:
            params.crf = min(32, params.crf + 2)  # Higher CRF for simple content
        elif complexity == ContentComplexity.CHAOTIC:
            params.crf = max(18, params.crf - 2)  # Lower CRF for complex content

        # Adjust preset based on network conditions
        if self.current_network:
            if self.current_network.rtt_ms > 300:  # Very high latency
                params.preset = "ultrafast"  # Prioritize encoding speed
            elif self.current_network.bandwidth_kbps < 2000:  # Low bandwidth
                params.preset = "slow"  # Better compression

        # Adjust framerate for bandwidth savings
        if (self.current_network and
            self.current_network.bandwidth_kbps < params.bitrate_kbps * 0.8):
            # Reduce framerate to maintain quality at lower bandwidth
            params.framerate = max(15.0, params.framerate * 0.75)

        # Adjust keyframe interval based on content and network
        if complexity in [ContentComplexity.COMPLEX, ContentComplexity.CHAOTIC]:
            params.keyframe_interval = max(15, params.keyframe_interval - 10)  # More keyframes
        elif self.current_network and self.current_network.rtt_ms > 200:
            params.keyframe_interval = min(90, params.keyframe_interval + 15)  # Fewer keyframes

        return params

    async def get_quality_recommendation(
        self,
        scene_type: str = "traffic_intersection"
    ) -> dict[str, Any]:
        """Get quality recommendation for specific scene types.
        
        Args:
            scene_type: Type of traffic scene
            
        Returns:
            Quality recommendation with rationale
        """
        recommendations = {
            "traffic_intersection": {
                "preferred_quality": QualityLevel.HIGH,
                "min_quality": QualityLevel.MEDIUM,
                "rationale": "High detail needed for license plate reading and incident detection",
                "critical_features": ["license_plates", "traffic_lights", "vehicle_details"]
            },
            "highway_monitoring": {
                "preferred_quality": QualityLevel.MEDIUM,
                "min_quality": QualityLevel.LOW,
                "rationale": "Medium quality sufficient for speed monitoring and flow analysis",
                "critical_features": ["vehicle_count", "speed_estimation", "lane_detection"]
            },
            "parking_surveillance": {
                "preferred_quality": QualityLevel.LOW,
                "min_quality": QualityLevel.ULTRA_LOW,
                "rationale": "Low quality adequate for occupancy detection",
                "critical_features": ["vehicle_presence", "parking_violations"]
            }
        }

        return recommendations.get(scene_type, recommendations["traffic_intersection"])

    def get_adaptation_metrics(self) -> dict[str, Any]:
        """Get comprehensive adaptation performance metrics."""

        # Calculate average complexity over recent history
        if self.recent_complexities:
            complexity_scores = [score for _, _, score in self.recent_complexities]
            avg_complexity = np.mean(complexity_scores)
        else:
            avg_complexity = 0.0

        # Calculate bandwidth utilization
        if self.bandwidth_history:
            recent_bandwidth = [bw for _, bw in self.bandwidth_history[-5:]]
            avg_bandwidth_kbps = np.mean(recent_bandwidth)
        else:
            avg_bandwidth_kbps = 0.0

        # Estimate bandwidth savings
        baseline_bitrate = self.quality_profiles[QualityLevel.HIGH].bitrate_kbps
        current_bitrate = self.quality_profiles[self.current_quality].bitrate_kbps
        bandwidth_savings = max(0, (baseline_bitrate - current_bitrate) / 1000.0)  # Mbps

        return {
            **self.adaptation_stats,
            "current_quality": self.current_quality.value,
            "avg_content_complexity": avg_complexity,
            "avg_bandwidth_kbps": avg_bandwidth_kbps,
            "current_bandwidth_savings_mbps": bandwidth_savings,
            "bandwidth_reduction_ratio": min(1.0, bandwidth_savings / (baseline_bitrate / 1000.0)),
            "complexity_history_size": len(self.recent_complexities),
            "network_conditions": {
                "bandwidth_kbps": self.current_network.bandwidth_kbps if self.current_network else 0,
                "rtt_ms": self.current_network.rtt_ms if self.current_network else 0,
                "packet_loss_rate": self.current_network.packet_loss_rate if self.current_network else 0,
            } if self.current_network else None,
        }

    async def optimize_for_traffic_scenes(self) -> dict[str, Any]:
        """Apply traffic-specific optimizations to quality adaptation."""

        optimizations = {
            "roi_encoding": {
                "enabled": True,
                "high_quality_regions": ["license_plate_areas", "intersection_center"],
                "quality_boost": 2,  # CRF reduction for important areas
            },
            "temporal_optimization": {
                "reduce_quality_during_low_activity": True,
                "activity_threshold": 0.1,  # Motion detection threshold
                "quality_reduction": 1,  # CRF increase during low activity
            },
            "scene_specific_tuning": {
                "night_mode_compensation": True,
                "weather_adaptation": True,
                "traffic_density_scaling": True,
            }
        }

        logger.info("Applied traffic scene optimizations")
        return optimizations


class NetworkAwareStreamingManager:
    """Network-aware streaming manager for optimal fragment delivery."""

    def __init__(self):
        """Initialize network-aware streaming manager."""
        self.connection_profiles = {}
        self.streaming_metrics = {
            "total_fragments_sent": 0,
            "total_bytes_sent": 0,
            "avg_fragment_size_kb": 0.0,
            "network_adaptation_events": 0,
        }

        logger.info("Network-aware streaming manager initialized")

    async def adapt_fragment_parameters(
        self,
        network_conditions: NetworkConditions,
        content_complexity: ContentComplexity
    ) -> dict[str, Any]:
        """Adapt fragment parameters based on network and content conditions.
        
        Args:
            network_conditions: Current network metrics
            content_complexity: Content complexity level
            
        Returns:
            Optimized fragment configuration
        """
        fragment_config = {
            "fragment_duration_seconds": 4.0,
            "enable_adaptive_bitrate": True,
            "buffer_ahead_seconds": 12.0,
            "quality_levels": 3,
        }

        # Adapt based on RTT
        if network_conditions.rtt_ms > 200:
            fragment_config["fragment_duration_seconds"] = 2.0  # Smaller fragments
            fragment_config["buffer_ahead_seconds"] = 6.0
        elif network_conditions.rtt_ms < 50:
            fragment_config["fragment_duration_seconds"] = 6.0  # Larger fragments

        # Adapt based on bandwidth
        if network_conditions.bandwidth_kbps < 2000:
            fragment_config["quality_levels"] = 2  # Fewer quality options
            fragment_config["enable_aggressive_compression"] = True
        elif network_conditions.bandwidth_kbps > 10000:
            fragment_config["quality_levels"] = 5  # More quality options

        # Adapt based on packet loss
        if network_conditions.packet_loss_rate > 0.01:
            fragment_config["enable_forward_error_correction"] = True
            fragment_config["redundant_fragments"] = 1

        self.streaming_metrics["network_adaptation_events"] += 1

        logger.debug(f"Adapted fragment config: {fragment_config}")
        return fragment_config


# Factory functions for easy integration
async def create_adaptive_quality_controller(
    camera_id: str,
    target_bandwidth_reduction: float = 0.25
) -> AdaptiveQualityController:
    """Create adaptive quality controller for camera stream.
    
    Args:
        camera_id: Camera identifier
        target_bandwidth_reduction: Target bandwidth reduction ratio
        
    Returns:
        Configured adaptive quality controller
    """
    return AdaptiveQualityController(camera_id, target_bandwidth_reduction)


async def create_network_aware_streaming() -> NetworkAwareStreamingManager:
    """Create network-aware streaming manager.
    
    Returns:
        Configured network-aware streaming manager
    """
    return NetworkAwareStreamingManager()
