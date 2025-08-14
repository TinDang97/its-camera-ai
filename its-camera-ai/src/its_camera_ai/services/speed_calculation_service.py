"""Speed Calculation Service with proper DI and camera calibration.

This service calculates vehicle speeds from trajectory data using
camera calibration parameters and geometric transformations.
"""

import math
import statistics
from typing import Any

import cv2
import numpy as np

from ..core.config import Settings
from ..core.logging import get_logger
from ..repositories.analytics_repository import AnalyticsRepository
from .analytics_dtos import (
    CameraCalibration,
    Position,
    SpeedMeasurement,
)
from .cache import CacheService

logger = get_logger(__name__)


class SpeedCalculationService:
    """Speed calculation service with camera calibration support.

    Calculates vehicle speeds from trajectory data using geometric
    transformations and camera calibration parameters.
    """

    def __init__(
        self,
        analytics_repository: AnalyticsRepository,
        cache_service: CacheService,
        settings: Settings,
    ):
        """Initialize with injected dependencies.

        Args:
            analytics_repository: Repository for analytics data access
            cache_service: Redis cache service
            settings: Application settings
        """
        self.analytics_repository = analytics_repository
        self.cache_service = cache_service
        self.settings = settings

        # Speed calculation parameters
        self.min_distance_threshold = 5.0  # meters
        self.max_speed_threshold = 200.0  # km/h
        self.smoothing_window = 3  # frames

        # Cache for camera calibrations
        self.calibration_cache: dict[str, CameraCalibration] = {}

    async def calculate_speed(
        self,
        positions: list[Position],
        camera_id: str,
        calibration: CameraCalibration | None = None,
    ) -> SpeedMeasurement | None:
        """Calculate speed from trajectory positions.

        Args:
            positions: List of vehicle positions
            camera_id: Camera identifier
            calibration: Optional camera calibration data

        Returns:
            Speed measurement result or None if calculation fails
        """
        if len(positions) < 2:
            logger.warning("Need at least 2 positions for speed calculation")
            return None

        try:
            # Get or load camera calibration
            if not calibration:
                calibration = await self._get_camera_calibration(camera_id)

            # Sort positions by timestamp
            sorted_positions = sorted(positions, key=lambda p: p.timestamp)

            # Convert pixel coordinates to world coordinates
            world_positions = []
            for pos in sorted_positions:
                world_pos = self._pixel_to_world(pos, calibration)
                if world_pos:
                    world_positions.append(world_pos)

            if len(world_positions) < 2:
                logger.warning("Not enough valid world positions for speed calculation")
                return None

            # Calculate distances and time intervals
            distances = []
            time_intervals = []

            for i in range(1, len(world_positions)):
                prev_pos = world_positions[i - 1]
                curr_pos = world_positions[i]

                # Calculate distance
                distance = self._euclidean_distance(prev_pos, curr_pos)
                distances.append(distance)

                # Calculate time interval
                time_interval = curr_pos.timestamp - prev_pos.timestamp
                time_intervals.append(time_interval)

            # Calculate instantaneous speeds
            speeds = []
            for distance, time_interval in zip(distances, time_intervals, strict=False):
                if time_interval > 0:
                    # Speed in m/s
                    speed_ms = distance / time_interval
                    # Convert to km/h
                    speed_kmh = speed_ms * 3.6
                    speeds.append(speed_kmh)

            if not speeds:
                return None

            # Calculate final speed with smoothing
            smoothed_speed = self._smooth_speed(speeds)

            # Validate speed measurement
            if not self._validate_speed(smoothed_speed, sum(distances)):
                logger.warning(f"Invalid speed measurement: {smoothed_speed} km/h")
                return None

            # Calculate confidence based on data quality
            confidence = self._calculate_confidence(
                sorted_positions, world_positions, speeds
            )

            return SpeedMeasurement(
                speed_kmh=smoothed_speed,
                measurement_method="trajectory_analysis",
                confidence=confidence,
                distance_traveled=sum(distances),
                time_elapsed=sum(time_intervals),
                positions_used=sorted_positions,
            )

        except Exception as e:
            logger.error(f"Speed calculation failed: {e}")
            return None

    async def calculate_speed_batch(
        self,
        trajectory_data: list[dict[str, Any]],
        camera_id: str,
    ) -> list[SpeedMeasurement]:
        """Calculate speeds for multiple vehicle trajectories.

        Args:
            trajectory_data: List of trajectory data
            camera_id: Camera identifier

        Returns:
            List of speed measurements
        """
        results = []

        # Get camera calibration once for batch
        calibration = await self._get_camera_calibration(camera_id)

        for trajectory in trajectory_data:
            try:
                # Extract positions from trajectory
                positions = []
                for point in trajectory.get("points", []):
                    position = Position(
                        x=point["x"],
                        y=point["y"],
                        timestamp=point["timestamp"],
                        frame_id=point.get("frame_id"),
                        confidence=point.get("confidence", 1.0),
                    )
                    positions.append(position)

                # Calculate speed
                speed_measurement = await self.calculate_speed(
                    positions, camera_id, calibration
                )

                if speed_measurement:
                    results.append(speed_measurement)

            except Exception as e:
                logger.error(f"Failed to process trajectory: {e}")
                continue

        logger.info(
            f"Processed {len(trajectory_data)} trajectories, got {len(results)} speed measurements"
        )
        return results

    def _pixel_to_world(
        self, position: Position, calibration: CameraCalibration
    ) -> Position | None:
        """Convert pixel coordinates to world coordinates.

        Args:
            position: Position in pixel coordinates
            calibration: Camera calibration data

        Returns:
            Position in world coordinates or None if conversion fails
        """
        try:
            if calibration.homography_matrix is not None:
                # Use homography transformation
                pixel_point = np.array([[position.x, position.y]], dtype=np.float32)
                world_point = cv2.perspectiveTransform(
                    pixel_point.reshape(1, 1, 2), calibration.homography_matrix
                )[0][0]

                return Position(
                    x=world_point[0],
                    y=world_point[1],
                    timestamp=position.timestamp,
                    frame_id=position.frame_id,
                    confidence=position.confidence,
                )

            else:
                # Use simple pixel-to-meter ratio
                world_x = position.x * calibration.pixel_to_meter_ratio
                world_y = position.y * calibration.pixel_to_meter_ratio

                return Position(
                    x=world_x,
                    y=world_y,
                    timestamp=position.timestamp,
                    frame_id=position.frame_id,
                    confidence=position.confidence,
                )

        except Exception as e:
            logger.warning(f"Pixel to world conversion failed: {e}")
            return None

    def _euclidean_distance(self, pos1: Position, pos2: Position) -> float:
        """Calculate Euclidean distance between two positions.

        Args:
            pos1: First position
            pos2: Second position

        Returns:
            Distance in meters
        """
        dx = pos2.x - pos1.x
        dy = pos2.y - pos1.y
        return math.sqrt(dx * dx + dy * dy)

    def _smooth_speed(self, speeds: list[float]) -> float:
        """Apply smoothing to speed measurements.

        Args:
            speeds: List of instantaneous speeds

        Returns:
            Smoothed speed value
        """
        if len(speeds) <= self.smoothing_window:
            return statistics.mean(speeds)

        # Use moving average for smoothing
        smoothed_speeds = []
        half_window = self.smoothing_window // 2

        for i in range(len(speeds)):
            start_idx = max(0, i - half_window)
            end_idx = min(len(speeds), i + half_window + 1)
            window_speeds = speeds[start_idx:end_idx]
            smoothed_speeds.append(statistics.mean(window_speeds))

        # Return the median of smoothed speeds for robustness
        return statistics.median(smoothed_speeds)

    def _validate_speed(self, speed: float, distance: float) -> bool:
        """Validate speed measurement for reasonableness.

        Args:
            speed: Calculated speed in km/h
            distance: Total distance traveled

        Returns:
            True if speed is valid
        """
        # Check speed limits
        if speed < 0 or speed > self.max_speed_threshold:
            return False

        # Check minimum distance
        if distance < self.min_distance_threshold:
            return False

        return True

    def _calculate_confidence(
        self,
        pixel_positions: list[Position],
        world_positions: list[Position],
        speeds: list[float],
    ) -> float:
        """Calculate confidence score for speed measurement.

        Args:
            pixel_positions: Original pixel positions
            world_positions: Converted world positions
            speeds: Calculated speeds

        Returns:
            Confidence score (0.0 to 1.0)
        """
        confidence_factors = []

        # Factor 1: Position confidence
        avg_position_confidence = statistics.mean(
            [p.confidence for p in pixel_positions]
        )
        confidence_factors.append(avg_position_confidence)

        # Factor 2: Conversion success rate
        conversion_rate = len(world_positions) / len(pixel_positions)
        confidence_factors.append(conversion_rate)

        # Factor 3: Speed consistency
        if len(speeds) > 1:
            speed_variance = statistics.variance(speeds)
            mean_speed = statistics.mean(speeds)

            # Normalized variance (lower is better)
            if mean_speed > 0:
                cv = math.sqrt(speed_variance) / mean_speed
                consistency_score = max(0, 1 - cv)
            else:
                consistency_score = 0
        else:
            consistency_score = 0.5

        confidence_factors.append(consistency_score)

        # Factor 4: Trajectory smoothness
        if len(world_positions) >= 3:
            # Calculate trajectory smoothness based on direction changes
            direction_changes = 0
            for i in range(1, len(world_positions) - 1):
                prev_pos = world_positions[i - 1]
                curr_pos = world_positions[i]
                next_pos = world_positions[i + 1]

                # Calculate angle change
                vec1 = np.array([curr_pos.x - prev_pos.x, curr_pos.y - prev_pos.y])
                vec2 = np.array([next_pos.x - curr_pos.x, next_pos.y - curr_pos.y])

                # Avoid division by zero
                norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
                if norm1 > 0 and norm2 > 0:
                    cos_angle = np.dot(vec1, vec2) / (norm1 * norm2)
                    cos_angle = np.clip(cos_angle, -1, 1)
                    angle_change = math.acos(cos_angle)

                    # Significant direction change (> 30 degrees)
                    if angle_change > math.pi / 6:
                        direction_changes += 1

            # Smoothness score (fewer direction changes is better)
            max_changes = len(world_positions) - 2
            smoothness_score = (
                1 - (direction_changes / max_changes) if max_changes > 0 else 1
            )
        else:
            smoothness_score = 0.5

        confidence_factors.append(smoothness_score)

        # Calculate overall confidence as weighted average
        weights = [0.3, 0.2, 0.3, 0.2]  # Position, conversion, consistency, smoothness
        overall_confidence = sum(
            factor * weight
            for factor, weight in zip(confidence_factors, weights, strict=False)
        )

        return round(overall_confidence, 3)

    async def _get_camera_calibration(self, camera_id: str) -> CameraCalibration:
        """Get camera calibration data.

        Args:
            camera_id: Camera identifier

        Returns:
            Camera calibration data
        """
        # Check cache first
        if camera_id in self.calibration_cache:
            return self.calibration_cache[camera_id]

        # Try to load from cache service
        cache_key = f"camera_calibration:{camera_id}"
        cached_data = await self.cache_service.get_json(cache_key)

        if cached_data:
            calibration = self._deserialize_calibration(cached_data)
            self.calibration_cache[camera_id] = calibration
            return calibration

        # Load from repository (would query database in production)
        try:
            # For now, return default calibration
            default_calibration = CameraCalibration(
                camera_id=camera_id,
                pixel_to_meter_ratio=0.05,  # 5cm per pixel
            )

            # Cache the calibration
            serialized = self._serialize_calibration(default_calibration)
            await self.cache_service.set_json(cache_key, serialized, ttl=3600)

            self.calibration_cache[camera_id] = default_calibration
            return default_calibration

        except Exception as e:
            logger.error(f"Failed to load camera calibration: {e}")
            # Return minimal default
            return CameraCalibration(
                camera_id=camera_id,
                pixel_to_meter_ratio=0.05,
            )

    def _serialize_calibration(self, calibration: CameraCalibration) -> dict[str, Any]:
        """Serialize calibration data for caching.

        Args:
            calibration: Camera calibration

        Returns:
            Serialized data
        """
        data = {
            "camera_id": calibration.camera_id,
            "pixel_to_meter_ratio": calibration.pixel_to_meter_ratio,
            "focal_length": calibration.focal_length,
            "sensor_size": calibration.sensor_size,
        }

        # Handle numpy arrays
        if calibration.homography_matrix is not None:
            data["homography_matrix"] = calibration.homography_matrix.tolist()

        if calibration.distortion_coefficients is not None:
            data["distortion_coefficients"] = (
                calibration.distortion_coefficients.tolist()
            )

        return data

    def _deserialize_calibration(self, data: dict[str, Any]) -> CameraCalibration:
        """Deserialize calibration data from cache.

        Args:
            data: Serialized data

        Returns:
            Camera calibration
        """
        calibration = CameraCalibration(
            camera_id=data["camera_id"],
            pixel_to_meter_ratio=data.get("pixel_to_meter_ratio", 0.05),
            focal_length=data.get("focal_length"),
            sensor_size=tuple(data["sensor_size"]) if data.get("sensor_size") else None,
        )

        # Handle numpy arrays
        if "homography_matrix" in data and data["homography_matrix"]:
            calibration.homography_matrix = np.array(data["homography_matrix"])

        if "distortion_coefficients" in data and data["distortion_coefficients"]:
            calibration.distortion_coefficients = np.array(
                data["distortion_coefficients"]
            )

        return calibration

    async def calibrate_camera(
        self,
        camera_id: str,
        calibration_points: list[dict[str, Any]],
        world_coordinates: list[dict[str, Any]],
    ) -> bool:
        """Calibrate camera using known reference points.

        Args:
            camera_id: Camera identifier
            calibration_points: Pixel coordinates of known points
            world_coordinates: Real-world coordinates of known points

        Returns:
            True if calibration successful
        """
        try:
            if (
                len(calibration_points) != len(world_coordinates)
                or len(calibration_points) < 4
            ):
                logger.error("Need at least 4 corresponding points for calibration")
                return False

            # Prepare points for homography calculation
            pixel_points = np.array(
                [[p["x"], p["y"]] for p in calibration_points], dtype=np.float32
            )
            world_points = np.array(
                [[w["x"], w["y"]] for w in world_coordinates], dtype=np.float32
            )

            # Calculate homography matrix
            homography_matrix, mask = cv2.findHomography(
                pixel_points, world_points, cv2.RANSAC
            )

            if homography_matrix is None:
                logger.error("Failed to calculate homography matrix")
                return False

            # Create calibration object
            calibration = CameraCalibration(
                camera_id=camera_id,
                homography_matrix=homography_matrix,
                pixel_to_meter_ratio=0.05,  # Fallback value
            )

            # Cache the calibration
            cache_key = f"camera_calibration:{camera_id}"
            serialized = self._serialize_calibration(calibration)
            await self.cache_service.set_json(
                cache_key, serialized, ttl=86400
            )  # 24 hours

            # Update local cache
            self.calibration_cache[camera_id] = calibration

            logger.info(f"Camera {camera_id} calibrated successfully")
            return True

        except Exception as e:
            logger.error(f"Camera calibration failed: {e}")
            return False

    async def get_camera_calibration_status(self, camera_id: str) -> dict[str, Any]:
        """Get calibration status for a camera.

        Args:
            camera_id: Camera identifier

        Returns:
            Calibration status information
        """
        calibration = await self._get_camera_calibration(camera_id)

        return {
            "camera_id": camera_id,
            "is_calibrated": calibration.homography_matrix is not None,
            "pixel_to_meter_ratio": calibration.pixel_to_meter_ratio,
            "has_homography": calibration.homography_matrix is not None,
            "has_distortion_correction": calibration.distortion_coefficients
            is not None,
            "calibration_method": (
                "homography"
                if calibration.homography_matrix is not None
                else "pixel_ratio"
            ),
        }
