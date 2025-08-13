"""Comprehensive tests for the Analytics Service.

Tests all components of the analytics service including:
- Rule engine for traffic violations
- Speed calculations with homography
- Trajectory analysis
- Anomaly detection using ML
- Real-time metrics calculation
- Alert generation and delivery
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock

import numpy as np
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.its_camera_ai.core.config import Settings
from src.its_camera_ai.models.analytics import (
    AlertNotification,
    RuleViolation,
    TrafficAnomaly,
    TrafficMetrics,
    VehicleTrajectory,
    ViolationType,
)
from src.its_camera_ai.models.detection_result import DetectionResult
from src.its_camera_ai.services.alert_service import (
    AlertRuleEngine,
    AlertService,
    EmailNotificationChannel,
    WebhookNotificationChannel,
)
from src.its_camera_ai.services.analytics_service import (
    AnalyticsService,
    AnomalyDetector,
    RuleEngine,
    SpeedCalculator,
)


@pytest.fixture
def settings():
    """Test settings fixture."""
    return Settings(
        environment="testing",
        database={"url": "postgresql+asyncpg://test:test@localhost/test_db"},
        debug=True,
    )


@pytest.fixture
def sample_detections():
    """Sample vehicle detection data for testing."""
    base_time = datetime.utcnow()

    return [
        DetectionResult(
            id="det1",
            detection_id=1,
            frame_metadata_id="frame1",
            bbox_x1=100,
            bbox_y1=200,
            bbox_x2=150,
            bbox_y2=250,
            bbox_width=50,
            bbox_height=50,
            bbox_area=2500,
            class_name="car",
            class_confidence=0.9,
            vehicle_type="sedan",
            track_id=1001,
            velocity_x=5.0,
            velocity_y=0.0,
            velocity_magnitude=5.0,
            direction=0.0,
            detection_quality=0.95,
            created_at=base_time,
        ),
        DetectionResult(
            id="det2",
            detection_id=2,
            frame_metadata_id="frame1",
            bbox_x1=200,
            bbox_y1=300,
            bbox_x2=280,
            bbox_y2=380,
            bbox_width=80,
            bbox_height=80,
            bbox_area=6400,
            class_name="truck",
            class_confidence=0.85,
            vehicle_type="truck_large",
            track_id=1002,
            velocity_x=3.0,
            velocity_y=1.0,
            velocity_magnitude=3.16,
            direction=18.4,
            detection_quality=0.88,
            license_plate="ABC123",
            created_at=base_time,
        ),
        DetectionResult(
            id="det3",
            detection_id=3,
            frame_metadata_id="frame1",
            bbox_x1=50,
            bbox_y1=100,
            bbox_x2=90,
            bbox_y2=140,
            bbox_width=40,
            bbox_height=40,
            bbox_area=1600,
            class_name="motorcycle",
            class_confidence=0.75,
            vehicle_type="motorcycle_sport",
            track_id=1003,
            velocity_x=8.0,
            velocity_y=-2.0,
            velocity_magnitude=8.25,
            direction=346.0,
            detection_quality=0.82,
            created_at=base_time,
        ),
    ]


class TestRuleEngine:
    """Test cases for the traffic rule evaluation engine."""

    def test_rule_engine_initialization(self, settings):
        """Test rule engine initialization with default rules."""
        engine = RuleEngine(settings)

        assert "speed_limit" in engine._rules
        assert "wrong_way" in engine._rules
        assert "red_light" in engine._rules
        assert "illegal_parking" in engine._rules

        speed_rule = engine._rules["speed_limit"]
        assert speed_rule["default_limit"] == 50.0
        assert speed_rule["tolerance"] == 5.0
        assert "severity_thresholds" in speed_rule

    def test_speed_limit_evaluation_no_violation(self, settings):
        """Test speed limit evaluation with no violation."""
        engine = RuleEngine(settings)

        # Speed within limit + tolerance
        result = engine.evaluate_speed_rule(54.0)  # 50 + 5 - 1
        assert result is None

        # Exactly at limit
        result = engine.evaluate_speed_rule(50.0)
        assert result is None

        # At tolerance threshold
        result = engine.evaluate_speed_rule(55.0)
        assert result is None

    def test_speed_limit_evaluation_with_violation(self, settings):
        """Test speed limit evaluation with violations of different severities."""
        engine = RuleEngine(settings)

        # Medium severity violation (15+ km/h over)
        result = engine.evaluate_speed_rule(70.0)  # 15 over + tolerance
        assert result is not None
        assert result["violation_type"] == ViolationType.SPEEDING
        assert result["severity"] == "medium"
        assert result["measured_value"] == 70.0
        assert result["threshold_value"] == 50.0
        assert result["excess_amount"] == 20.0

        # High severity violation (25+ km/h over)
        result = engine.evaluate_speed_rule(80.0)  # 25 over + tolerance
        assert result is not None
        assert result["severity"] == "high"
        assert result["excess_amount"] == 30.0

        # Critical severity violation (40+ km/h over)
        result = engine.evaluate_speed_rule(95.0)  # 40 over + tolerance
        assert result is not None
        assert result["severity"] == "critical"
        assert result["excess_amount"] == 45.0

    def test_wrong_way_evaluation_no_violation(self, settings):
        """Test wrong way evaluation with correct direction."""
        engine = RuleEngine(settings)

        # Trajectory moving in correct direction (east)
        trajectory = [
            {"x": 100, "y": 200, "timestamp": 1000},
            {"x": 150, "y": 205, "timestamp": 1001},
            {"x": 200, "y": 210, "timestamp": 1002},
        ]

        result = engine.evaluate_wrong_way_rule(trajectory, "east")
        assert result is None

    def test_wrong_way_evaluation_with_violation(self, settings):
        """Test wrong way evaluation with wrong direction."""
        engine = RuleEngine(settings)

        # Trajectory moving in wrong direction (west when expected east)
        trajectory = [
            {"x": 200, "y": 200, "timestamp": 1000},
            {"x": 150, "y": 205, "timestamp": 1001},
            {"x": 100, "y": 210, "timestamp": 1002},
        ]

        result = engine.evaluate_wrong_way_rule(trajectory, "east")
        assert result is not None
        assert result["violation_type"] == ViolationType.WRONG_WAY
        assert result["confidence"] >= 0.8
        assert "rule_definition" in result

    def test_wrong_way_evaluation_insufficient_distance(self, settings):
        """Test wrong way evaluation with insufficient travel distance."""
        engine = RuleEngine(settings)

        # Very short trajectory (less than minimum distance)
        trajectory = [
            {"x": 100, "y": 200, "timestamp": 1000},
            {"x": 105, "y": 202, "timestamp": 1001},
        ]

        result = engine.evaluate_wrong_way_rule(trajectory, "east")
        assert result is None


class TestSpeedCalculator:
    """Test cases for speed calculation with homography."""

    def test_speed_calculator_initialization(self):
        """Test speed calculator initialization."""
        calculator = SpeedCalculator()
        assert calculator.pixel_to_meter_ratio == 0.05
        assert calculator.homography_matrix is None

        # Test with calibration data
        calibration = {
            "pixel_to_meter_ratio": 0.03,
            "homography": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        }
        calculator = SpeedCalculator(calibration)
        assert calculator.pixel_to_meter_ratio == 0.03
        assert calculator.homography_matrix is not None

    def test_speed_calculation_simple(self):
        """Test speed calculation without homography."""
        calculator = SpeedCalculator()

        # Positions: (x, y, timestamp)
        positions = [
            (100, 200, 1000.0),
            (150, 200, 1001.0),  # 50 pixels in 1 second
            (200, 200, 1002.0),  # Another 50 pixels in 1 second
        ]

        speed = calculator.calculate_speed_from_positions(positions, 1.0)
        assert speed is not None
        assert speed > 0

        # Expected: 50 pixels * 0.05 m/pixel = 2.5 m/s = 9 km/h
        expected_speed = 50 * 0.05 * 3.6  # Convert to km/h
        assert abs(speed - expected_speed) < 1.0

    def test_speed_calculation_no_movement(self):
        """Test speed calculation with no movement."""
        calculator = SpeedCalculator()

        # Stationary positions
        positions = [(100, 200, 1000.0), (100, 200, 1001.0), (100, 200, 1002.0)]

        speed = calculator.calculate_speed_from_positions(positions, 1.0)
        assert speed is not None
        assert speed == 0.0

    def test_speed_calculation_insufficient_positions(self):
        """Test speed calculation with insufficient position data."""
        calculator = SpeedCalculator()

        # Only one position
        positions = [(100, 200, 1000.0)]

        speed = calculator.calculate_speed_from_positions(positions, 1.0)
        assert speed is None

        # Empty positions
        speed = calculator.calculate_speed_from_positions([], 1.0)
        assert speed is None


class TestAnomalyDetector:
    """Test cases for ML-based anomaly detection."""

    def test_anomaly_detector_initialization(self):
        """Test anomaly detector initialization."""
        detector = AnomalyDetector(contamination_rate=0.1)
        assert detector.contamination_rate == 0.1
        assert not detector.is_trained

    def test_feature_extraction(self):
        """Test feature extraction from data points."""
        detector = AnomalyDetector()

        data_points = [
            {
                "vehicle_count": 10,
                "average_speed": 45.0,
                "traffic_density": 20.0,
                "flow_rate": 600.0,
                "occupancy_rate": 0.3,
                "queue_length": 15.0,
                "timestamp": datetime(2024, 1, 1, 10, 30, 0),
            },
            {
                "vehicle_count": 15,
                "average_speed": 40.0,
                "traffic_density": 25.0,
                "timestamp": datetime(2024, 1, 1, 14, 15, 30),
            },
        ]

        features = detector._extract_features(data_points)
        assert features.shape[0] == 2  # Two data points
        assert features.shape[1] == 9  # Expected number of features

        # Check feature values are normalized
        assert np.all(features[:, 6] <= 1.0)  # Normalized hour
        assert np.all(features[:, 7] <= 1.0)  # Normalized day of week

    def test_anomaly_training(self):
        """Test anomaly detector training."""
        detector = AnomalyDetector()

        # Generate training data
        training_data = []
        for i in range(100):
            training_data.append(
                {
                    "vehicle_count": 10 + np.random.normal(0, 2),
                    "average_speed": 50 + np.random.normal(0, 5),
                    "traffic_density": 20 + np.random.normal(0, 3),
                    "flow_rate": 600 + np.random.normal(0, 50),
                    "occupancy_rate": 0.3 + np.random.normal(0, 0.1),
                    "queue_length": 10 + np.random.normal(0, 5),
                    "timestamp": datetime(2024, 1, 1, 10, 0, 0) + timedelta(minutes=i),
                }
            )

        detector.train(training_data)
        assert detector.is_trained

    def test_anomaly_detection(self):
        """Test anomaly detection on trained model."""
        detector = AnomalyDetector()

        # Train with normal data
        normal_data = []
        for i in range(50):
            normal_data.append(
                {
                    "vehicle_count": 10,
                    "average_speed": 50.0,
                    "traffic_density": 20.0,
                    "flow_rate": 600.0,
                    "occupancy_rate": 0.3,
                    "queue_length": 10.0,
                    "timestamp": datetime(2024, 1, 1, 10, 0, 0) + timedelta(minutes=i),
                }
            )

        detector.train(normal_data)

        # Test with anomalous data
        test_data = [
            # Normal point
            {
                "vehicle_count": 10,
                "average_speed": 50.0,
                "traffic_density": 20.0,
                "flow_rate": 600.0,
                "occupancy_rate": 0.3,
                "queue_length": 10.0,
                "timestamp": datetime(2024, 1, 1, 12, 0, 0),
            },
            # Anomalous point (very high traffic)
            {
                "vehicle_count": 100,  # Significantly higher
                "average_speed": 10.0,  # Much slower
                "traffic_density": 200.0,  # Much higher
                "flow_rate": 100.0,  # Much lower
                "occupancy_rate": 0.9,  # Much higher
                "queue_length": 100.0,  # Much longer
                "timestamp": datetime(2024, 1, 1, 12, 1, 0),
            },
        ]

        anomalies = detector.detect_anomalies(test_data)

        # Should detect at least one anomaly (the second data point)
        assert len(anomalies) >= 1

        for anomaly in anomalies:
            assert "anomaly_score" in anomaly
            assert "severity" in anomaly
            assert "probable_cause" in anomaly
            assert anomaly["anomaly_score"] > 0


@pytest.mark.asyncio
class TestAnalyticsService:
    """Test cases for the main Analytics Service."""

    async def test_analytics_service_initialization(
        self, async_session: AsyncSession, settings
    ):
        """Test analytics service initialization."""
        service = AnalyticsService(async_session, settings)

        assert service.session == async_session
        assert service.settings == settings
        assert service.rule_engine is not None
        assert service.speed_calculator is not None
        assert service.anomaly_detector is not None

    async def test_process_detections(
        self, async_session: AsyncSession, settings, sample_detections
    ):
        """Test processing of vehicle detections."""
        service = AnalyticsService(async_session, settings)

        camera_id = "cam001"
        frame_timestamp = datetime.utcnow()

        result = await service.process_detections(
            sample_detections, camera_id, frame_timestamp
        )

        assert result["camera_id"] == camera_id
        assert result["timestamp"] == frame_timestamp
        assert result["vehicle_count"] == 3
        assert "metrics" in result
        assert "violations" in result
        assert "anomalies" in result
        assert "processing_time_ms" in result
        assert result["processing_time_ms"] > 0

    async def test_real_time_metrics_calculation(
        self, async_session: AsyncSession, settings, sample_detections
    ):
        """Test real-time traffic metrics calculation."""
        service = AnalyticsService(async_session, settings)

        camera_id = "cam001"
        timestamp = datetime.utcnow()

        metrics = await service._calculate_real_time_metrics(
            sample_detections, camera_id, timestamp
        )

        assert metrics["total_vehicles"] == 3
        assert "vehicle_breakdown" in metrics
        assert "average_speed" in metrics
        assert "traffic_density" in metrics
        assert "congestion_level" in metrics
        assert metrics["timestamp"] == timestamp

        # Check vehicle breakdown
        breakdown = metrics["vehicle_breakdown"]
        assert breakdown.get("sedan", 0) >= 1  # From first detection
        assert breakdown.get("truck_large", 0) >= 1  # From second detection
        assert breakdown.get("motorcycle_sport", 0) >= 1  # From third detection

    async def test_traffic_rule_evaluation(
        self, async_session: AsyncSession, settings, sample_detections
    ):
        """Test traffic rule evaluation for violations."""
        service = AnalyticsService(async_session, settings)

        # Modify a detection to have high speed (violation)
        sample_detections[0].velocity_magnitude = 25.0  # Very high speed

        camera_id = "cam001"
        violations = await service._evaluate_traffic_rules(sample_detections, camera_id)

        # Should detect at least one violation (speeding)
        assert len(violations) >= 1

        violation = violations[0]
        assert violation["violation_type"] == ViolationType.SPEEDING
        assert violation["camera_id"] == camera_id
        assert violation["detection_id"] == sample_detections[0].id
        assert "severity" in violation
        assert "measured_value" in violation
        assert "threshold_value" in violation

    async def test_trajectory_analysis(
        self, async_session: AsyncSession, settings, sample_detections
    ):
        """Test vehicle trajectory analysis."""
        service = AnalyticsService(async_session, settings)

        camera_id = "cam001"

        # Simulate trajectory update
        await service._update_trajectory_analysis(sample_detections, camera_id)

        # Check that trajectories were created/updated in database
        query = select(VehicleTrajectory).where(
            VehicleTrajectory.camera_id == camera_id
        )
        result = await async_session.execute(query)
        trajectories = result.scalars().all()

        # Should have trajectories for tracked vehicles
        tracked_vehicles = {det.track_id for det in sample_detections if det.track_id}
        assert len(trajectories) >= len(tracked_vehicles)

        for trajectory in trajectories:
            assert trajectory.camera_id == camera_id
            assert trajectory.vehicle_track_id in tracked_vehicles
            assert trajectory.path_points is not None
            assert len(trajectory.path_points) > 0

    async def test_calculate_traffic_metrics(
        self, async_session: AsyncSession, settings
    ):
        """Test aggregated traffic metrics calculation."""
        service = AnalyticsService(async_session, settings)

        # First, insert some test metrics data
        camera_id = "cam001"
        base_time = datetime.utcnow().replace(minute=0, second=0, microsecond=0)

        test_metrics = []
        for i in range(3):
            metric = TrafficMetrics(
                timestamp=base_time + timedelta(hours=i),
                camera_id=camera_id,
                aggregation_period="1hour",
                total_vehicles=10 + i * 5,
                vehicle_cars=8 + i * 3,
                vehicle_trucks=2 + i,
                average_speed=45.0 + i * 5,
                traffic_density=20.0 + i * 10,
                congestion_level="light" if i == 0 else "moderate",
                occupancy_rate=0.3 + i * 0.1,
            )
            test_metrics.append(metric)

        async_session.add_all(test_metrics)
        await async_session.commit()

        # Test metrics calculation
        time_range = (base_time, base_time + timedelta(hours=3))
        metrics = await service.calculate_traffic_metrics(
            camera_id, time_range, "1hour"
        )

        assert len(metrics) == 3
        for i, metric in enumerate(metrics):
            assert metric.camera_id == camera_id
            assert metric.total_vehicles == 10 + i * 5
            assert metric.avg_speed == 45.0 + i * 5

    async def test_get_active_violations(self, async_session: AsyncSession, settings):
        """Test retrieval of active violations."""
        service = AnalyticsService(async_session, settings)

        camera_id = "cam001"

        # Create test violations
        violations = [
            RuleViolation(
                violation_type=ViolationType.SPEEDING,
                severity="high",
                detection_time=datetime.utcnow(),
                camera_id=camera_id,
                vehicle_track_id=1001,
                measured_value=80.0,
                threshold_value=50.0,
                detection_confidence=0.9,
                rule_definition={"speed_limit": 50.0},
                status="active",
            ),
            RuleViolation(
                violation_type=ViolationType.RED_LIGHT,
                severity="critical",
                detection_time=datetime.utcnow() - timedelta(minutes=5),
                camera_id=camera_id,
                vehicle_track_id=1002,
                detection_confidence=0.95,
                rule_definition={"intersection": "main_st"},
                status="active",
            ),
        ]

        async_session.add_all(violations)
        await async_session.commit()

        # Test retrieval
        active_violations = await service.get_active_violations(camera_id=camera_id)

        assert len(active_violations) == 2
        assert all(v["camera_id"] == camera_id for v in active_violations)
        assert any(v["type"] == ViolationType.SPEEDING for v in active_violations)
        assert any(v["type"] == ViolationType.RED_LIGHT for v in active_violations)

    async def test_get_traffic_anomalies(self, async_session: AsyncSession, settings):
        """Test retrieval of traffic anomalies."""
        service = AnalyticsService(async_session, settings)

        camera_id = "cam001"

        # Create test anomalies
        anomalies = [
            TrafficAnomaly(
                anomaly_type="traffic_pattern",
                severity="high",
                detection_time=datetime.utcnow(),
                camera_id=camera_id,
                anomaly_score=0.8,
                confidence=0.9,
                detection_method="isolation_forest",
                model_name="IsolationForest",
                model_version="1.0",
                model_confidence=0.8,
                probable_cause="unusual_congestion",
            ),
            TrafficAnomaly(
                anomaly_type="speed_anomaly",
                severity="medium",
                detection_time=datetime.utcnow() - timedelta(minutes=10),
                camera_id=camera_id,
                anomaly_score=0.6,
                confidence=0.7,
                detection_method="statistical",
                model_name="ZScore",
                model_version="1.0",
                model_confidence=0.6,
                probable_cause="speed_drop",
            ),
        ]

        async_session.add_all(anomalies)
        await async_session.commit()

        # Test retrieval
        detected_anomalies = await service.get_traffic_anomalies(
            camera_id=camera_id, min_score=0.5
        )

        assert len(detected_anomalies) == 2
        assert all(a["camera_id"] == camera_id for a in detected_anomalies)
        assert all(a["score"] >= 0.5 for a in detected_anomalies)

    async def test_generate_analytics_report(
        self, async_session: AsyncSession, settings
    ):
        """Test analytics report generation."""
        service = AnalyticsService(async_session, settings)

        camera_ids = ["cam001", "cam002"]
        time_range = (datetime.utcnow() - timedelta(hours=24), datetime.utcnow())

        # Add some test data for report generation
        for camera_id in camera_ids:
            # Add metrics
            metric = TrafficMetrics(
                timestamp=datetime.utcnow() - timedelta(hours=1),
                camera_id=camera_id,
                aggregation_period="1hour",
                total_vehicles=50,
                average_speed=45.0,
            )
            async_session.add(metric)

            # Add violation
            violation = RuleViolation(
                violation_type=ViolationType.SPEEDING,
                severity="medium",
                detection_time=datetime.utcnow() - timedelta(minutes=30),
                camera_id=camera_id,
                detection_confidence=0.8,
                rule_definition={},
            )
            async_session.add(violation)

        await async_session.commit()

        # Generate report
        report = await service.generate_analytics_report(
            camera_ids, time_range, "traffic_summary"
        )

        assert report["report_type"] == "traffic_summary"
        assert report["cameras"] == camera_ids
        assert "summary" in report
        assert "camera_summaries" in report

        summary = report["summary"]
        assert "total_vehicles" in summary
        assert "total_violations" in summary
        assert "total_anomalies" in summary
        assert summary["cameras_analyzed"] == 2


@pytest.mark.asyncio
class TestAlertService:
    """Test cases for the Alert Service."""

    async def test_alert_service_initialization(
        self, async_session: AsyncSession, settings
    ):
        """Test alert service initialization."""
        service = AlertService(async_session, settings)

        assert service.session == async_session
        assert service.settings == settings
        assert service.rule_engine is not None
        assert "email" in service.notification_channels
        assert "webhook" in service.notification_channels

    async def test_process_violation_alert(self, async_session: AsyncSession, settings):
        """Test processing violation alerts."""
        service = AlertService(async_session, settings)

        # Create test violation
        violation = RuleViolation(
            violation_type=ViolationType.SPEEDING,
            severity="high",
            detection_time=datetime.utcnow(),
            camera_id="cam001",
            vehicle_track_id=1001,
            measured_value=80.0,
            threshold_value=50.0,
            detection_confidence=0.9,
            rule_definition={"speed_limit": 50.0},
        )

        async_session.add(violation)
        await async_session.commit()
        await async_session.refresh(violation)

        # Process alert
        recipients = ["admin@example.com", "operator@example.com"]
        results = await service.process_violation_alert(
            violation, recipients, ["email"]
        )

        assert len(results) == 2  # Two recipients
        for result in results:
            assert result["alert_type"] == "violation"
            assert result["channel"] == "email"
            assert result["recipient"] in recipients

    async def test_process_anomaly_alert(self, async_session: AsyncSession, settings):
        """Test processing anomaly alerts."""
        service = AlertService(async_session, settings)

        # Create test anomaly
        anomaly = TrafficAnomaly(
            anomaly_type="traffic_pattern",
            severity="critical",
            detection_time=datetime.utcnow(),
            camera_id="cam001",
            anomaly_score=0.9,
            confidence=0.95,
            detection_method="isolation_forest",
            model_name="IsolationForest",
            model_version="1.0",
            model_confidence=0.9,
            probable_cause="severe_congestion",
        )

        async_session.add(anomaly)
        await async_session.commit()
        await async_session.refresh(anomaly)

        # Process alert
        recipients = ["admin@example.com"]
        results = await service.process_anomaly_alert(anomaly, recipients, ["webhook"])

        assert len(results) == 1
        result = results[0]
        assert result["alert_type"] == "anomaly"
        assert result["channel"] == "webhook"
        assert result["recipient"] == "admin@example.com"

    async def test_alert_content_generation(
        self, async_session: AsyncSession, settings
    ):
        """Test alert content generation for different types."""
        service = AlertService(async_session, settings)

        # Test violation alert content
        violation = RuleViolation(
            violation_type=ViolationType.SPEEDING,
            severity="high",
            detection_time=datetime(2024, 1, 15, 10, 30, 0),
            camera_id="cam001",
            license_plate="ABC123",
            measured_value=80.0,
            threshold_value=50.0,
            detection_confidence=0.9,
        )

        subject, message = service._generate_violation_alert_content(violation)

        assert "Speed Limit Violation" in subject
        assert "High" in subject
        assert "Speed: 80.0 km/h (limit: 50.0 km/h)" in message
        assert "Vehicle: ABC123" in message
        assert "Camera cam001" in message
        assert "90.0%" in message  # Confidence as percentage

        # Test anomaly alert content
        anomaly = TrafficAnomaly(
            anomaly_type="traffic_pattern",
            severity="critical",
            detection_time=datetime(2024, 1, 15, 14, 45, 0),
            camera_id="cam002",
            anomaly_score=0.85,
            confidence=0.92,
            probable_cause="unusual_congestion",
            baseline_value=20.0,
            observed_value=80.0,
            deviation_magnitude=60.0,
            detection_method="isolation_forest",
        )

        subject, message = service._generate_anomaly_alert_content(anomaly)

        assert "Traffic Pattern Anomaly" in subject
        assert "Critical" in subject
        assert "Score: 0.85" in message
        assert "Probable Cause: unusual_congestion" in message
        assert "Baseline: 20.00" in message
        assert "Observed: 80.00" in message
        assert "Deviation: 60.00" in message
        assert "92.0%" in message  # Confidence as percentage

    async def test_get_alert_statistics(self, async_session: AsyncSession, settings):
        """Test alert statistics generation."""
        service = AlertService(async_session, settings)

        # Create test notifications
        base_time = datetime.utcnow()
        notifications = [
            AlertNotification(
                alert_type="violation",
                reference_id="viol1",
                notification_channel="email",
                recipient="user1@example.com",
                priority="high",
                subject="Test Alert",
                message_content="Test message",
                status="delivered",
                created_time=base_time - timedelta(hours=1),
                sent_time=base_time - timedelta(hours=1) + timedelta(minutes=1),
                delivery_details={"camera_id": "cam001"},
            ),
            AlertNotification(
                alert_type="anomaly",
                reference_id="anom1",
                notification_channel="webhook",
                recipient="system",
                priority="critical",
                subject="Anomaly Alert",
                message_content="Anomaly detected",
                status="failed",
                created_time=base_time - timedelta(minutes=30),
                error_message="Connection timeout",
                delivery_details={"camera_id": "cam001"},
            ),
        ]

        async_session.add_all(notifications)
        await async_session.commit()

        # Get statistics
        stats = await service.get_alert_statistics(
            time_range=(base_time - timedelta(hours=2), base_time)
        )

        assert stats["summary"]["total_alerts"] == 2
        assert stats["summary"]["delivered_alerts"] == 1
        assert stats["summary"]["failed_alerts"] == 1
        assert stats["summary"]["delivery_rate"] == 0.5

        # Check channel stats
        assert "email" in stats["by_channel"]
        assert "webhook" in stats["by_channel"]
        assert stats["by_channel"]["email"]["delivered"] == 1
        assert stats["by_channel"]["webhook"]["failed"] == 1

        # Check priority stats
        assert "high" in stats["by_priority"]
        assert "critical" in stats["by_priority"]

    async def test_acknowledge_alert(self, async_session: AsyncSession, settings):
        """Test alert acknowledgment."""
        service = AlertService(async_session, settings)

        # Create test notification
        notification = AlertNotification(
            alert_type="violation",
            reference_id="viol1",
            notification_channel="email",
            recipient="user@example.com",
            priority="medium",
            subject="Test Alert",
            message_content="Test message",
            status="delivered",
        )

        async_session.add(notification)
        await async_session.commit()
        await async_session.refresh(notification)

        # Acknowledge alert
        acknowledged = await service.acknowledge_alert(
            notification.id,
            acknowledged_by="operator123",
            response_action="investigated",
            notes="False positive - construction work",
        )

        assert acknowledged is True

        # Verify acknowledgment was recorded
        await async_session.refresh(notification)
        assert notification.acknowledged_time is not None
        assert notification.acknowledged_by == "operator123"
        assert notification.response_action == "investigated"
        assert notification.response_notes == "False positive - construction work"


class TestNotificationChannels:
    """Test cases for notification channels."""

    @pytest.mark.asyncio
    async def test_email_notification_channel(self):
        """Test email notification channel."""
        config = {
            "smtp_server": "smtp.example.com",
            "smtp_port": 587,
            "username": "test@example.com",
            "password": "password",
            "from_email": "alerts@example.com",
        }

        channel = EmailNotificationChannel(config)

        result = await channel.send_notification(
            recipient="user@example.com",
            subject="Test Alert",
            message="This is a test alert message",
            priority="high",
        )

        assert result["status"] == "delivered"
        assert result["channel"] == "email"
        assert result["recipient"] == "user@example.com"
        assert "delivery_id" in result
        assert "delivered_at" in result

    @pytest.mark.asyncio
    async def test_webhook_notification_channel(self):
        """Test webhook notification channel."""
        config = {
            "url": "https://api.example.com/webhooks/alerts",
            "headers": {"Authorization": "Bearer token123"},
            "timeout": 30,
        }

        channel = WebhookNotificationChannel(config)

        result = await channel.send_notification(
            recipient="system_monitor",
            subject="Anomaly Detected",
            message="Traffic anomaly detected in zone A",
            priority="critical",
            metadata={"camera_id": "cam001", "zone": "A"},
        )

        assert result["status"] == "delivered"
        assert result["channel"] == "webhook"
        assert result["recipient"] == "system_monitor"
        assert "delivery_id" in result


class TestAlertRuleEngine:
    """Test cases for alert rule evaluation engine."""

    def test_alert_rule_engine_initialization(self, settings):
        """Test alert rule engine initialization."""
        engine = AlertRuleEngine(settings)

        assert "violation_alerts" in engine._alert_rules
        assert "anomaly_alerts" in engine._alert_rules
        assert "pattern_alerts" in engine._alert_rules

    def test_violation_alert_evaluation(self, settings):
        """Test violation alert rule evaluation."""
        engine = AlertRuleEngine(settings)

        # High severity violation should trigger alert
        violation = MagicMock()
        violation.severity = "high"

        should_alert = engine.should_alert_for_violation(violation)
        assert should_alert is True

        # Low severity violation should not trigger alert (by default config)
        violation.severity = "low"
        should_alert = engine.should_alert_for_violation(violation)
        assert should_alert is False

    def test_anomaly_alert_evaluation(self, settings):
        """Test anomaly alert rule evaluation."""
        engine = AlertRuleEngine(settings)

        # High severity, high score anomaly should trigger alert
        anomaly = MagicMock()
        anomaly.severity = "high"
        anomaly.anomaly_score = 0.8

        should_alert = engine.should_alert_for_anomaly(anomaly)
        assert should_alert is True

        # Low score anomaly should not trigger alert
        anomaly.anomaly_score = 0.5  # Below threshold
        should_alert = engine.should_alert_for_anomaly(anomaly)
        assert should_alert is False

    def test_get_alert_priority(self, settings):
        """Test alert priority determination."""
        engine = AlertRuleEngine(settings)

        entity = MagicMock()

        entity.severity = "critical"
        assert engine.get_alert_priority(entity) == "critical"

        entity.severity = "high"
        assert engine.get_alert_priority(entity) == "high"

        entity.severity = "medium"
        assert engine.get_alert_priority(entity) == "medium"

        entity.severity = "low"
        assert engine.get_alert_priority(entity) == "low"

    def test_get_cooldown_period(self, settings):
        """Test cooldown period calculation."""
        engine = AlertRuleEngine(settings)

        violation_cooldown = engine.get_cooldown_period("violation")
        assert violation_cooldown == timedelta(minutes=5)

        anomaly_cooldown = engine.get_cooldown_period("anomaly")
        assert anomaly_cooldown == timedelta(minutes=15)

        default_cooldown = engine.get_cooldown_period("unknown")
        assert default_cooldown == timedelta(minutes=5)


# Performance and integration tests
@pytest.mark.asyncio
@pytest.mark.slow
class TestAnalyticsServicePerformance:
    """Performance tests for analytics service."""

    async def test_high_volume_detection_processing(
        self, async_session: AsyncSession, settings
    ):
        """Test processing high volume of detections."""
        service = AnalyticsService(async_session, settings)

        # Generate large number of detections
        detections = []
        base_time = datetime.utcnow()

        for i in range(1000):  # 1000 detections
            detection = DetectionResult(
                id=f"det_{i}",
                detection_id=i,
                frame_metadata_id=f"frame_{i // 50}",  # 50 detections per frame
                bbox_x1=100 + i % 100,
                bbox_y1=200 + i % 100,
                bbox_x2=150 + i % 100,
                bbox_y2=250 + i % 100,
                bbox_width=50,
                bbox_height=50,
                bbox_area=2500,
                class_name="car",
                class_confidence=0.8 + (i % 20) * 0.01,  # Vary confidence
                track_id=1000 + i % 100,  # 100 unique tracks
                velocity_magnitude=5.0 + (i % 50) * 0.5,  # Vary speed
                detection_quality=0.7 + (i % 30) * 0.01,
                created_at=base_time + timedelta(milliseconds=i * 100),
            )
            detections.append(detection)

        # Process detections in batches
        batch_size = 50
        total_processing_time = 0

        for i in range(0, len(detections), batch_size):
            batch = detections[i : i + batch_size]

            start_time = datetime.utcnow()
            result = await service.process_detections(
                batch, "cam001", base_time + timedelta(seconds=i // batch_size)
            )
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            total_processing_time += processing_time

            # Verify processing time meets performance requirements
            assert processing_time < 100  # < 100ms per batch (50 detections)
            assert result["vehicle_count"] == len(batch)

        avg_processing_time = total_processing_time / (len(detections) // batch_size)
        logger.info(f"Average batch processing time: {avg_processing_time:.2f}ms")

        # Verify overall performance requirement
        assert avg_processing_time < 50  # < 50ms average per batch

    async def test_rule_evaluation_throughput(
        self, async_session: AsyncSession, settings
    ):
        """Test rule evaluation throughput meets requirements."""
        service = AnalyticsService(async_session, settings)

        # Generate detections with various speeds for rule evaluation
        detections = []
        speeds = [30, 45, 60, 75, 90, 105]  # Mix of legal and illegal speeds

        for i in range(10000):  # 10,000 detections for throughput test
            detection = DetectionResult(
                id=f"det_{i}",
                detection_id=i,
                frame_metadata_id=f"frame_{i}",
                bbox_x1=100,
                bbox_y1=200,
                bbox_x2=150,
                bbox_y2=250,
                bbox_width=50,
                bbox_height=50,
                bbox_area=2500,
                class_name="car",
                class_confidence=0.9,
                track_id=i,
                velocity_magnitude=speeds[i % len(speeds)] / 3.6,  # Convert km/h to m/s
                detection_quality=0.9,
                created_at=datetime.utcnow(),
            )
            detections.append(detection)

        start_time = datetime.utcnow()
        violations = await service._evaluate_traffic_rules(detections, "cam001")
        end_time = datetime.utcnow()

        processing_time = (end_time - start_time).total_seconds()
        throughput = len(detections) / processing_time

        logger.info(f"Rule evaluation throughput: {throughput:.0f} events/second")

        # Verify throughput meets requirement (> 10,000 events/second)
        assert throughput > 10000

        # Verify violations were detected for high speeds
        violation_count = len(violations)
        expected_violations = sum(
            1 for speed in speeds if speed > 55
        )  # Above limit + tolerance
        expected_total = (len(detections) // len(speeds)) * expected_violations

        # Should be close to expected (allowing for some variance)
        assert violation_count >= expected_total * 0.8
