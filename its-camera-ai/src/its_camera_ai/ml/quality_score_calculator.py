"""ML Quality Score Calculator for ITS Camera AI System.

This module provides comprehensive quality assessment for ML detections,
replacing placeholder quality_score=0.0 with actual calculation based on
detection confidence, image quality, model uncertainty, and temporal consistency.

Quality Score Calculation:
- Detection confidence: 40% weight (from model outputs)
- Image quality: 30% weight (blur, brightness, contrast)
- Model uncertainty: 20% weight (entropy-based confidence)
- Temporal consistency: 10% weight (tracking smoothness)

Performance Requirements:
- Quality calculation: <5ms per detection
- Batch processing: 100+ detections/sec
- Memory efficient: <50MB working set
- Cache optimization: 95% hit rate for recent calculations
"""

import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from ..core.logging import get_logger
from ..services.cache import CacheService

logger = get_logger(__name__)


@dataclass
class QualityFactors:
    """Individual quality assessment factors."""

    detection_confidence: float  # 0.0-1.0
    image_quality: float  # 0.0-1.0
    model_uncertainty: float  # 0.0-1.0 (lower is better)
    temporal_consistency: float  # 0.0-1.0


@dataclass
class ImageQualityMetrics:
    """Detailed image quality metrics."""

    blur_score: float  # 0.0-1.0 (Laplacian variance based)
    brightness_score: float  # 0.0-1.0 (optimal lighting)
    contrast_score: float  # 0.0-1.0 (dynamic range)
    noise_score: float  # 0.0-1.0 (signal-to-noise ratio)
    sharpness_score: float  # 0.0-1.0 (edge definition)


@dataclass
class TemporalQualityMetrics:
    """Temporal consistency metrics for tracking."""

    position_consistency: float  # 0.0-1.0 (smooth movement)
    size_consistency: float  # 0.0-1.0 (stable bounding box)
    confidence_consistency: float  # 0.0-1.0 (stable confidence)
    track_length: int  # Number of frames tracked
    track_age_seconds: float  # Age of current track


class QualityScoreCalculator:
    """Calculate comprehensive quality scores for ML detections.

    Implements multi-factor quality assessment with configurable weights
    and performance optimization through caching and batch processing.
    """

    # Quality calculation weights
    WEIGHTS = {
        "detection_confidence": 0.4,
        "image_quality": 0.3,
        "model_uncertainty": 0.2,
        "temporal_consistency": 0.1,
    }

    # Image quality thresholds (empirically determined)
    BLUR_THRESHOLD = 1000.0  # Laplacian variance threshold
    BRIGHTNESS_OPTIMAL = 128.0  # Optimal brightness level
    CONTRAST_MIN = 50.0  # Minimum acceptable contrast
    NOISE_THRESHOLD = 0.1  # Noise level threshold

    # Temporal consistency parameters
    HISTORY_WINDOW = 5  # Number of frames for consistency check
    MAX_POSITION_DELTA = 50.0  # Max pixel movement between frames
    MAX_SIZE_DELTA = 0.2  # Max size change ratio

    def __init__(self, cache_service: CacheService, enable_compression: bool = True):
        """Initialize quality score calculator.

        Args:
            cache_service: Cache service for performance optimization
        """
        self.cache = cache_service
        self.history_window = self.HISTORY_WINDOW

        # Performance optimization
        self.calculation_cache = {}  # In-memory cache for recent calculations
        self.cache_ttl = 300  # 5 minutes cache TTL
        self.last_cleanup = time.time()

        # Statistics tracking
        self.stats = {
            "calculations": 0,
            "cache_hits": 0,
            "avg_calculation_time_ms": 0.0,
        }

        logger.info("QualityScoreCalculator initialized")

    async def calculate_quality_score(
        self,
        detection: dict[str, Any],
        frame: np.ndarray,
        model_output: dict[str, Any],
        historical_detections: list[dict] | None = None,
    ) -> float:
        """Calculate comprehensive quality score for a detection.

        Args:
            detection: Detection result dictionary
            frame: Input frame as numpy array
            model_output: Raw model output with confidence distributions
            historical_detections: Previous detections for temporal analysis

        Returns:
            Quality score between 0.0 and 1.0 (higher is better)
        """
        start_time = time.time()

        try:
            # Check cache first
            cache_key = self._generate_cache_key(detection, frame.shape)
            cached_score = await self._get_cached_score(cache_key)

            if cached_score is not None:
                self.stats["cache_hits"] += 1
                return cached_score

            # 1. Detection confidence (from model)
            detection_confidence = self._calculate_detection_confidence(
                detection, model_output
            )

            # 2. Image quality metrics (most expensive calculation)
            # Use compressed caching for image features to reduce memory usage
            image_quality = await self._calculate_image_quality_with_caching(
                frame, detection, cache_key
            )

            # 3. Model uncertainty (entropy-based)
            model_uncertainty = self._calculate_model_uncertainty(model_output)

            # 4. Temporal consistency (tracking smoothness)
            temporal_consistency = self._calculate_temporal_consistency(
                detection, historical_detections
            )

            # Calculate weighted score
            factors = QualityFactors(
                detection_confidence=detection_confidence,
                image_quality=image_quality,
                model_uncertainty=1.0 - model_uncertainty,  # Invert uncertainty
                temporal_consistency=temporal_consistency,
            )

            quality_score = self._weighted_average(factors)

            # Cache the result
            await self._cache_quality_score(cache_key, quality_score)

            # Update statistics
            calculation_time_ms = (time.time() - start_time) * 1000
            self._update_stats(calculation_time_ms)

            return quality_score

        except Exception as e:
            logger.error(f"Quality score calculation failed: {e}")
            return 0.5  # Return neutral score on error

    def _calculate_detection_confidence(
        self, detection: dict[str, Any], model_output: dict[str, Any]
    ) -> float:
        """Calculate detection confidence from model outputs.

        Args:
            detection: Detection result
            model_output: Raw model output with distributions

        Returns:
            Normalized confidence score (0.0-1.0)
        """
        try:
            # Get base confidence from detection
            base_conf = float(detection.get("confidence", 0.0))

            # Get class probability distribution if available
            class_probs = model_output.get("class_probabilities", {})

            if class_probs and len(class_probs) > 1:
                # Calculate entropy-based confidence
                probs = list(class_probs.values())
                entropy = -sum(p * np.log(p + 1e-10) for p in probs if p > 0)
                max_entropy = np.log(len(probs))
                entropy_confidence = (
                    1.0 - (entropy / max_entropy) if max_entropy > 0 else 0.0
                )

                # Combine base and entropy confidence
                return 0.7 * base_conf + 0.3 * entropy_confidence

            return base_conf

        except Exception as e:
            logger.warning(f"Detection confidence calculation failed: {e}")
            return detection.get("confidence", 0.0)

    async def _calculate_image_quality(
        self, frame: np.ndarray, detection: dict[str, Any]
    ) -> float:
        """Calculate comprehensive image quality metrics.

        Args:
            frame: Input frame
            detection: Detection with bounding box

        Returns:
            Combined image quality score (0.0-1.0)
        """
        try:
            # Extract region of interest from detection bbox
            bbox = detection.get("bbox", detection.get("bounding_box", {}))
            roi = self._extract_roi(frame, bbox)

            if roi is None or roi.size == 0:
                return 0.5  # Default score for invalid ROI

            # Calculate individual quality metrics
            metrics = await self._calculate_detailed_image_metrics(roi)

            # Combine metrics with weights
            quality_score = (
                metrics.blur_score * 0.3
                + metrics.brightness_score * 0.25
                + metrics.contrast_score * 0.2
                + metrics.sharpness_score * 0.15
                + metrics.noise_score * 0.1
            )

            return np.clip(quality_score, 0.0, 1.0)

        except Exception as e:
            logger.warning(f"Image quality calculation failed: {e}")
            return 0.5

    async def _calculate_detailed_image_metrics(
        self, roi: np.ndarray
    ) -> ImageQualityMetrics:
        """Calculate detailed image quality metrics for ROI.

        Args:
            roi: Region of interest from detection

        Returns:
            Detailed image quality metrics
        """
        # Convert to grayscale for analysis
        gray: np.ndarray = (
            cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        )

        # 1. Blur detection using Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_score = min(1.0, laplacian_var / self.BLUR_THRESHOLD)

        # 2. Brightness assessment
        mean_brightness = np.mean(gray).astype(float)
        brightness_score = 1.0 - abs(mean_brightness - self.BRIGHTNESS_OPTIMAL) / 128.0
        brightness_score = max(0.0, brightness_score)

        # 3. Contrast measurement
        contrast = np.std(gray).astype(float)
        contrast_score = min(1.0, contrast / self.CONTRAST_MIN)

        # 4. Sharpness using gradient magnitude
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        sharpness_score = min(1.0, np.mean(gradient_magnitude).astype(float) / 50.0)

        # 5. Noise estimation using high-frequency content
        noise_estimate = self._estimate_noise(gray)
        noise_score = max(0.0, 1.0 - noise_estimate / self.NOISE_THRESHOLD)

        return ImageQualityMetrics(
            blur_score=blur_score,
            brightness_score=brightness_score,
            contrast_score=contrast_score,
            sharpness_score=sharpness_score,
            noise_score=noise_score,
        )

    def _calculate_model_uncertainty(self, model_output: dict[str, Any]) -> float:
        """Calculate model uncertainty using entropy and confidence distributions.

        Args:
            model_output: Raw model output with distributions

        Returns:
            Uncertainty score (0.0-1.0, lower is better)
        """
        try:
            # Method 1: Entropy from class probabilities
            class_probs = model_output.get("class_probabilities", {})
            if class_probs:
                probs = [p for p in class_probs.values() if p > 0]
                if len(probs) > 1:
                    entropy = -sum(p * np.log(p) for p in probs)
                    max_entropy = np.log(len(probs))
                    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
                    return normalized_entropy

            # Method 2: Confidence spread analysis
            confidence_values = model_output.get("confidence_scores", [])
            if confidence_values and len(confidence_values) > 1:
                confidence_std = np.std(confidence_values).astype(float)
                return min(1.0, confidence_std * 2)  # Normalize to 0-1

            # Method 3: Fallback - inverse of max confidence
            max_confidence = model_output.get("max_confidence", 0.5)
            return 1.0 - max_confidence

        except Exception as e:
            logger.warning(f"Model uncertainty calculation failed: {e}")
            return 0.5  # Default moderate uncertainty

    def _calculate_temporal_consistency(
        self,
        detection: dict[str, Any],
        historical_detections: list[dict] | None = None,
    ) -> float:
        """Calculate temporal consistency for tracking quality.

        Args:
            detection: Current detection
            historical_detections: Previous detections for same track

        Returns:
            Temporal consistency score (0.0-1.0)
        """
        if not historical_detections or len(historical_detections) < 2:
            return 0.5  # Default score for insufficient history

        try:
            # Get current detection properties
            current_bbox = detection.get("bbox", {})
            current_conf = detection.get("confidence", 0.0)
            track_id = detection.get("track_id")

            # Filter history for same track
            track_history = [
                h
                for h in historical_detections[-self.history_window :]
                if h.get("track_id") == track_id
            ]

            if len(track_history) < 2:
                return 0.5

            # Calculate position consistency
            position_consistency = self._calculate_position_consistency(
                current_bbox, track_history
            )

            # Calculate size consistency
            size_consistency = self._calculate_size_consistency(
                current_bbox, track_history
            )

            # Calculate confidence consistency
            confidence_consistency = self._calculate_confidence_consistency(
                current_conf, track_history
            )

            # Combine temporal metrics
            temporal_score = (
                position_consistency * 0.5
                + size_consistency * 0.3
                + confidence_consistency * 0.2
            )

            return np.clip(temporal_score, 0.0, 1.0)

        except Exception as e:
            logger.warning(f"Temporal consistency calculation failed: {e}")
            return 0.5

    def _calculate_position_consistency(
        self, current_bbox: dict[str, Any], track_history: list[dict]
    ) -> float:
        """Calculate position movement consistency.

        Args:
            current_bbox: Current bounding box
            track_history: Historical bounding boxes

        Returns:
            Position consistency score (0.0-1.0)
        """
        try:
            # Get center points
            current_center = self._get_bbox_center(current_bbox)

            movements = []
            for i in range(1, len(track_history)):
                prev_center = self._get_bbox_center(
                    track_history[i - 1].get("bbox", {})
                )
                curr_center = self._get_bbox_center(track_history[i].get("bbox", {}))

                if prev_center and curr_center:
                    movement = np.sqrt(
                        (curr_center[0] - prev_center[0]) ** 2
                        + (curr_center[1] - prev_center[1]) ** 2
                    )
                    movements.append(movement)

            if not movements:
                return 0.5

            # Calculate movement consistency (lower variance is better)
            movement_std = np.std(movements)
            consistency = max(0.0, 1.0 - movement_std / self.MAX_POSITION_DELTA)

            return consistency

        except Exception:
            return 0.5

    def _calculate_size_consistency(
        self, current_bbox: dict[str, Any], track_history: list[dict]
    ) -> float:
        """Calculate bounding box size consistency.

        Args:
            current_bbox: Current bounding box
            track_history: Historical bounding boxes

        Returns:
            Size consistency score (0.0-1.0)
        """
        try:
            current_area = self._get_bbox_area(current_bbox)

            areas = []
            for hist_det in track_history:
                area = self._get_bbox_area(hist_det.get("bbox", {}))
                if area > 0:
                    areas.append(area)

            if not areas or current_area <= 0:
                return 0.5

            # Calculate area ratios
            area_ratios = [abs(area / current_area - 1.0) for area in areas]
            avg_ratio_change = np.mean(area_ratios)

            # Convert to consistency score
            consistency = max(0.0, 1.0 - avg_ratio_change / self.MAX_SIZE_DELTA)

            return consistency

        except Exception:
            return 0.5

    def _calculate_confidence_consistency(
        self, current_conf: float, track_history: list[dict]
    ) -> float:
        """Calculate detection confidence consistency.

        Args:
            current_conf: Current detection confidence
            track_history: Historical detections

        Returns:
            Confidence consistency score (0.0-1.0)
        """
        try:
            confidences = [current_conf]
            for hist_det in track_history:
                conf = hist_det.get("confidence", 0.0)
                confidences.append(conf)

            if len(confidences) < 2:
                return 0.5

            # Calculate confidence standard deviation
            conf_std = np.std(confidences)

            # Convert to consistency score (lower std is better)
            consistency = max(0.0, 1.0 - conf_std * 2)  # Scale to 0-1

            return consistency

        except Exception:
            return 0.5

    def _weighted_average(self, factors: QualityFactors) -> float:
        """Calculate weighted average of quality factors.

        Args:
            factors: Quality factors to combine

        Returns:
            Combined quality score (0.0-1.0)
        """
        weighted_sum = (
            factors.detection_confidence * self.WEIGHTS["detection_confidence"]
            + factors.image_quality * self.WEIGHTS["image_quality"]
            + (1.0 - factors.model_uncertainty) * self.WEIGHTS["model_uncertainty"]
            + factors.temporal_consistency * self.WEIGHTS["temporal_consistency"]
        )

        return np.clip(weighted_sum, 0.0, 1.0)

    # Utility methods

    def _extract_roi(
        self, frame: np.ndarray, bbox: dict[str, Any]
    ) -> np.ndarray | None:
        """Extract region of interest from frame using bounding box.

        Args:
            frame: Input frame
            bbox: Bounding box coordinates

        Returns:
            Extracted ROI or None if invalid
        """
        try:
            if isinstance(bbox, list) and len(bbox) >= 4:
                x1, y1, x2, y2 = bbox[:4]
            else:
                x1 = bbox.get("x1", bbox.get("left", 0))
                y1 = bbox.get("y1", bbox.get("top", 0))
                x2 = bbox.get("x2", bbox.get("right", x1 + bbox.get("width", 0)))
                y2 = bbox.get("y2", bbox.get("bottom", y1 + bbox.get("height", 0)))

            # Convert to integers and validate
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            h, w = frame.shape[:2]

            # Clip to frame boundaries
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(x1 + 1, min(x2, w))
            y2 = max(y1 + 1, min(y2, h))

            return frame[y1:y2, x1:x2]

        except Exception:
            return None

    def _get_bbox_center(self, bbox: dict[str, Any]) -> tuple[float, float] | None:
        """Get center point of bounding box.

        Args:
            bbox: Bounding box coordinates

        Returns:
            Center point (x, y) or None if invalid
        """
        try:
            if isinstance(bbox, list) and len(bbox) >= 4:
                x1, y1, x2, y2 = bbox[:4]
            else:
                x1 = bbox.get("x1", bbox.get("left", 0))
                y1 = bbox.get("y1", bbox.get("top", 0))
                x2 = bbox.get("x2", bbox.get("right", x1 + bbox.get("width", 0)))
                y2 = bbox.get("y2", bbox.get("bottom", y1 + bbox.get("height", 0)))

            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            return (center_x, center_y)

        except Exception:
            return None

    def _get_bbox_area(self, bbox: dict[str, Any]) -> float:
        """Calculate bounding box area.

        Args:
            bbox: Bounding box coordinates

        Returns:
            Area in pixels
        """
        try:
            if isinstance(bbox, list) and len(bbox) >= 4:
                x1, y1, x2, y2 = bbox[:4]
            else:
                x1 = bbox.get("x1", bbox.get("left", 0))
                y1 = bbox.get("y1", bbox.get("top", 0))
                x2 = bbox.get("x2", bbox.get("right", x1 + bbox.get("width", 0)))
                y2 = bbox.get("y2", bbox.get("bottom", y1 + bbox.get("height", 0)))

            width = max(0, x2 - x1)
            height = max(0, y2 - y1)

            return width * height

        except Exception:
            return 0.0

    def _estimate_noise(self, gray_image: np.ndarray) -> float:
        """Estimate image noise level.

        Args:
            gray_image: Grayscale image

        Returns:
            Noise estimate (higher values indicate more noise)
        """
        try:
            # Use high-pass filter to isolate noise
            kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
            high_pass = cv2.filter2D(gray_image, -1, kernel)

            # Calculate noise as standard deviation of high-frequency content
            noise_estimate = np.std(high_pass) / 255.0  # Normalize to 0-1

            return noise_estimate

        except Exception:
            return 0.1  # Default moderate noise level

    def _generate_cache_key(self, detection: dict[str, Any], frame_shape: tuple) -> str:
        """Generate cache key for quality score.

        Args:
            detection: Detection data
            frame_shape: Frame dimensions

        Returns:
            Cache key string
        """
        try:
            # Use track_id and frame info for caching
            track_id = detection.get("track_id", "unknown")
            confidence = detection.get("confidence", 0.0)
            frame_key = f"{frame_shape[0]}x{frame_shape[1]}"

            return f"quality:{track_id}:{confidence:.3f}:{frame_key}"

        except Exception:
            return f"quality:fallback:{time.time()}"

    async def _get_cached_score(self, cache_key: str) -> float | None:
        """Get cached quality score.

        Args:
            cache_key: Cache key

        Returns:
            Cached score or None if not found
        """
        try:
            # Check in-memory cache first
            if cache_key in self.calculation_cache:
                entry_time, score = self.calculation_cache[cache_key]
                if time.time() - entry_time < 60:  # 1 minute in-memory cache
                    return score
                else:
                    del self.calculation_cache[cache_key]

            # Check Redis cache
            cached_data = await self.cache.get_json(cache_key)
            if cached_data and "score" in cached_data:
                score = float(cached_data["score"])
                # Update in-memory cache
                self.calculation_cache[cache_key] = (time.time(), score)
                return score

        except Exception:
            pass

        return None

    async def _cache_quality_score(self, cache_key: str, score: float) -> None:
        """Cache quality score result.

        Args:
            cache_key: Cache key
            score: Quality score to cache
        """
        try:
            # Update in-memory cache
            self.calculation_cache[cache_key] = (time.time(), score)

            # Cache in Redis
            cache_data = {"score": score, "timestamp": time.time()}
            await self.cache.set_json(cache_key, cache_data, ttl=self.cache_ttl)

            # Periodic cleanup of in-memory cache
            if time.time() - self.last_cleanup > 300:  # 5 minutes
                self._cleanup_memory_cache()

        except Exception:
            pass  # Don't fail on caching errors

    def _cleanup_memory_cache(self) -> None:
        """Clean up expired entries from in-memory cache."""
        try:
            current_time = time.time()
            expired_keys = [
                key
                for key, (entry_time, _) in self.calculation_cache.items()
                if current_time - entry_time > 60
            ]

            for key in expired_keys:
                del self.calculation_cache[key]

            self.last_cleanup = current_time

        except Exception:
            pass

    def _update_stats(self, calculation_time_ms: float) -> None:
        """Update calculation statistics.

        Args:
            calculation_time_ms: Time taken for calculation
        """
        self.stats["calculations"] += 1

        # Update rolling average
        count = self.stats["calculations"]
        current_avg = self.stats["avg_calculation_time_ms"]
        self.stats["avg_calculation_time_ms"] = (
            current_avg * (count - 1) + calculation_time_ms
        ) / count

    def get_stats(self) -> dict[str, Any]:
        """Get calculator performance statistics.

        Returns:
            Statistics dictionary
        """
        cache_hit_rate = self.stats["cache_hits"] / max(1, self.stats["calculations"])

        stats = {
            "quality_calculator": {
                "total_calculations": self.stats["calculations"],
                "cache_hits": self.stats["cache_hits"],
                "cache_hit_rate": cache_hit_rate,
                "avg_calculation_time_ms": self.stats["avg_calculation_time_ms"],
                "memory_cache_size": len(self.calculation_cache),
                "weights": self.WEIGHTS,
            }
        }

        # Add compression-specific stats if enabled
        if self.enable_compression:
            stats["quality_calculator"].update({
                "compression_enabled": True,
                "compressed_cache_hits": self.stats["compressed_cache_hits"],
                "compressed_cache_hit_rate": (
                    self.stats["compressed_cache_hits"] / max(1, self.stats["calculations"])
                ),
                "avg_compression_ratio": self.stats["compression_ratio"],
                "memory_saved_mb": self.stats["memory_saved_bytes"] / (1024 * 1024),
                "feature_cache_size": len(self.feature_cache),
            })

        return stats

    def get_performance_metrics(self) -> dict[str, Any]:
        """Get comprehensive performance metrics including compression stats."""
        total_requests = self.stats["calculations"]
        cache_hit_rate = (
            (self.stats["cache_hits"] + self.stats["compressed_cache_hits"]) / total_requests
            if total_requests > 0 else 0
        )

        metrics = {
            "total_calculations": total_requests,
            "cache_hit_rate": cache_hit_rate,
            "compressed_cache_hit_rate": (
                self.stats["compressed_cache_hits"] / total_requests
                if total_requests > 0 else 0
            ),
            "avg_calculation_time_ms": self.stats["avg_calculation_time_ms"],
            "compression_enabled": self.enable_compression,
        }

        if self.enable_compression:
            metrics.update({
                "avg_compression_ratio": self.stats["compression_ratio"],
                "total_memory_saved_bytes": self.stats["memory_saved_bytes"],
                "total_memory_saved_mb": self.stats["memory_saved_bytes"] / (1024 * 1024),
                "feature_cache_size": len(self.feature_cache),
            })

        return metrics

    async def _calculate_image_quality_with_caching(
        self, frame: np.ndarray, detection: dict[str, Any], cache_key: str
    ) -> float:
        """Calculate image quality with compressed feature caching.
        
        Args:
            frame: Input frame
            detection: Detection data
            cache_key: Cache key for compressed features
            
        Returns:
            Image quality score (0.0-1.0)
        """
        try:
            # Check for compressed cached features
            feature_cache_key = f"features_{cache_key}"
            cached_features = await self._get_compressed_features(feature_cache_key)

            if cached_features is not None:
                self.stats["compressed_cache_hits"] += 1
                return self._compute_image_quality_from_features(cached_features)

            # Calculate image quality and extract cacheable features
            image_quality, features = await self._calculate_image_quality_detailed(
                frame, detection
            )

            # Cache compressed features for reuse
            if self.enable_compression and features is not None:
                await self._cache_compressed_features(feature_cache_key, features)

            return image_quality

        except Exception as e:
            logger.error(f"Image quality calculation with caching failed: {e}")
            return await self._calculate_image_quality(frame, detection)

    async def _calculate_image_quality_detailed(
        self, frame: np.ndarray, detection: dict[str, Any]
    ) -> tuple[float, dict[str, np.ndarray] | None]:
        """Calculate detailed image quality with extractable features.
        
        Args:
            frame: Input frame
            detection: Detection data
            
        Returns:
            Tuple of (quality_score, cacheable_features)
        """
        try:
            # Extract ROI from frame
            bbox = detection.get("bbox", [])
            if len(bbox) >= 4:
                x1, y1, x2, y2 = map(int, bbox[:4])
                roi = frame[y1:y2, x1:x2]
            else:
                roi = frame

            if roi.size == 0:
                return 0.5, None

            # Convert to grayscale for feature extraction
            if len(roi.shape) == 3:
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            else:
                gray_roi = roi

            # Calculate detailed metrics with feature extraction
            blur_score, blur_features = self._calculate_blur_with_features(gray_roi)
            brightness_score, brightness_features = self._calculate_brightness_with_features(gray_roi)
            contrast_score, contrast_features = self._calculate_contrast_with_features(gray_roi)
            noise_score, noise_features = self._calculate_noise_with_features(gray_roi)
            sharpness_score, sharpness_features = self._calculate_sharpness_with_features(gray_roi)

            # Combine metrics
            metrics = ImageQualityMetrics(
                blur_score=blur_score,
                brightness_score=brightness_score,
                contrast_score=contrast_score,
                noise_score=noise_score,
                sharpness_score=sharpness_score,
            )

            # Weighted average of quality metrics
            quality_score = (
                metrics.blur_score * 0.25 +
                metrics.brightness_score * 0.20 +
                metrics.contrast_score * 0.20 +
                metrics.noise_score * 0.15 +
                metrics.sharpness_score * 0.20
            )

            # Collect features for caching
            features = {
                "blur_features": blur_features,
                "brightness_features": brightness_features,
                "contrast_features": contrast_features,
                "noise_features": noise_features,
                "sharpness_features": sharpness_features,
                "roi_shape": np.array(roi.shape, dtype=np.int32),
                "quality_metrics": np.array([
                    metrics.blur_score, metrics.brightness_score,
                    metrics.contrast_score, metrics.noise_score,
                    metrics.sharpness_score
                ], dtype=np.float32)
            }

            return quality_score, features

        except Exception as e:
            logger.error(f"Detailed image quality calculation failed: {e}")
            return 0.5, None

    def _calculate_blur_with_features(self, gray_roi: np.ndarray) -> tuple[float, np.ndarray]:
        """Calculate blur score with extractable features."""
        # Laplacian variance for blur detection
        laplacian = cv2.Laplacian(gray_roi, cv2.CV_64F)
        variance = laplacian.var()
        blur_score = min(1.0, variance / self.BLUR_THRESHOLD)

        # Extract gradient features for caching
        grad_x = cv2.Sobel(gray_roi, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_roi, cv2.CV_32F, 0, 1, ksize=3)
        grad_features = np.array([
            np.mean(np.abs(grad_x)), np.std(grad_x),
            np.mean(np.abs(grad_y)), np.std(grad_y),
            variance
        ], dtype=np.float32)

        return blur_score, grad_features

    def _calculate_brightness_with_features(self, gray_roi: np.ndarray) -> tuple[float, np.ndarray]:
        """Calculate brightness score with extractable features."""
        mean_brightness = np.mean(gray_roi)
        brightness_deviation = abs(mean_brightness - self.BRIGHTNESS_OPTIMAL)
        brightness_score = max(0.0, 1.0 - brightness_deviation / 128.0)

        # Extract histogram features for caching
        hist = cv2.calcHist([gray_roi], [0], None, [16], [0, 256])
        hist_features = hist.flatten().astype(np.float32)

        brightness_features = np.concatenate([
            [mean_brightness, np.std(gray_roi), brightness_score],
            hist_features
        ])

        return brightness_score, brightness_features

    def _calculate_contrast_with_features(self, gray_roi: np.ndarray) -> tuple[float, np.ndarray]:
        """Calculate contrast score with extractable features."""
        contrast = np.std(gray_roi)
        contrast_score = min(1.0, contrast / 100.0)

        # Dynamic range features
        min_val, max_val = np.min(gray_roi), np.max(gray_roi)
        dynamic_range = max_val - min_val

        contrast_features = np.array([
            contrast, dynamic_range, min_val, max_val,
            np.percentile(gray_roi, 25), np.percentile(gray_roi, 75)
        ], dtype=np.float32)

        return contrast_score, contrast_features

    def _calculate_noise_with_features(self, gray_roi: np.ndarray) -> tuple[float, np.ndarray]:
        """Calculate noise score with extractable features."""
        # Simple noise estimation using high-pass filter
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        noise_map = cv2.filter2D(gray_roi.astype(np.float32), -1, kernel)
        noise_level = np.std(noise_map) / 255.0
        noise_score = max(0.0, 1.0 - noise_level / self.NOISE_THRESHOLD)

        noise_features = np.array([
            noise_level, np.mean(np.abs(noise_map)),
            np.percentile(np.abs(noise_map), 90)
        ], dtype=np.float32)

        return noise_score, noise_features

    def _calculate_sharpness_with_features(self, gray_roi: np.ndarray) -> tuple[float, np.ndarray]:
        """Calculate sharpness score with extractable features."""
        # Sobel edge detection for sharpness
        sobelx = cv2.Sobel(gray_roi, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray_roi, cv2.CV_64F, 0, 1, ksize=3)
        edge_magnitude = np.sqrt(sobelx**2 + sobely**2)
        sharpness = np.mean(edge_magnitude)
        sharpness_score = min(1.0, sharpness / 100.0)

        sharpness_features = np.array([
            sharpness, np.std(edge_magnitude),
            np.percentile(edge_magnitude, 95)
        ], dtype=np.float32)

        return sharpness_score, sharpness_features

    def _compute_image_quality_from_features(self, features: dict[str, np.ndarray]) -> float:
        """Compute image quality score from cached features."""
        try:
            quality_metrics = features.get("quality_metrics")
            if quality_metrics is not None and len(quality_metrics) >= 5:
                # Weighted average of cached quality metrics
                return (
                    quality_metrics[0] * 0.25 +  # blur_score
                    quality_metrics[1] * 0.20 +  # brightness_score
                    quality_metrics[2] * 0.20 +  # contrast_score
                    quality_metrics[3] * 0.15 +  # noise_score
                    quality_metrics[4] * 0.20     # sharpness_score
                )
            return 0.5
        except Exception as e:
            logger.error(f"Failed to compute quality from cached features: {e}")
            return 0.5

    async def _get_compressed_features(self, cache_key: str) -> dict[str, np.ndarray] | None:
        """Retrieve and decompress cached features."""
        if not self.enable_compression or not self.compressor:
            return None

        try:
            # Check in-memory feature cache first
            if cache_key in self.feature_cache:
                cached_data, timestamp = self.feature_cache[cache_key]
                if time.time() - timestamp < self.cache_ttl:
                    return self._decompress_features(cached_data)
                else:
                    del self.feature_cache[cache_key]

            # Check Redis cache
            cached_compressed = await self.cache.get_bytes(cache_key)
            if cached_compressed:
                features = self._decompress_features(cached_compressed)
                # Update in-memory cache
                self.feature_cache[cache_key] = (cached_compressed, time.time())
                return features

            return None

        except Exception as e:
            logger.error(f"Failed to get compressed features: {e}")
            return None

    async def _cache_compressed_features(self, cache_key: str, features: dict[str, np.ndarray]) -> None:
        """Compress and cache features."""
        if not self.enable_compression or not self.compressor:
            return

        try:
            compressed_data = self._compress_features(features)

            # Calculate compression statistics
            original_size = sum(arr.nbytes for arr in features.values())
            compressed_size = len(compressed_data)
            compression_ratio = compressed_size / original_size if original_size > 0 else 0

            # Update statistics
            self.stats["compression_ratio"] = (
                self.stats["compression_ratio"] + compression_ratio
            ) / 2  # Running average
            self.stats["memory_saved_bytes"] += original_size - compressed_size

            # Cache in memory and Redis
            self.feature_cache[cache_key] = (compressed_data, time.time())
            await self.cache.set_bytes(cache_key, compressed_data, ttl=self.cache_ttl)

            logger.debug(
                f"Cached compressed features: {original_size} -> {compressed_size} bytes "
                f"({compression_ratio:.3f} ratio)"
            )

        except Exception as e:
            logger.error(f"Failed to cache compressed features: {e}")

    def _compress_features(self, features: dict[str, np.ndarray]) -> bytes:
        """Compress feature dictionary using blosc."""
        # Serialize features to single array for compression
        feature_data = []
        feature_metadata = []

        for key, array in features.items():
            feature_data.append(array.flatten())
            feature_metadata.append({
                "key": key,
                "shape": array.shape,
                "dtype": str(array.dtype),
                "size": array.size
            })

        # Combine all features into single array
        combined_array = np.concatenate(feature_data).astype(np.float32)

        # Compress with metadata
        compressed_features = self.compressor.compress_with_metadata(combined_array)

        # Combine with metadata (simple JSON serialization)
        import json
        metadata_json = json.dumps(feature_metadata).encode('utf-8')
        metadata_size = len(metadata_json)

        # Pack: [metadata_size:4][metadata][compressed_features]
        result = metadata_size.to_bytes(4, 'little') + metadata_json + compressed_features

        return result

    def _decompress_features(self, compressed_data: bytes) -> dict[str, np.ndarray]:
        """Decompress feature dictionary."""
        # Unpack metadata
        metadata_size = int.from_bytes(compressed_data[:4], 'little')
        metadata_json = compressed_data[4:4+metadata_size]
        compressed_features = compressed_data[4+metadata_size:]

        import json
        feature_metadata = json.loads(metadata_json.decode('utf-8'))

        # Decompress features
        combined_array = self.compressor.decompress_with_metadata(compressed_features)

        # Reconstruct individual feature arrays
        features = {}
        offset = 0

        for meta in feature_metadata:
            key = meta["key"]
            shape = tuple(meta["shape"])
            dtype = np.dtype(meta["dtype"])
            size = meta["size"]

            # Extract array data
            array_data = combined_array[offset:offset+size]
            features[key] = array_data.reshape(shape).astype(dtype)
            offset += size

        return features
