"""
License Plate Recognition Pipeline for ITS Camera AI.

Comprehensive LPR system that integrates vehicle detection, plate localization,
OCR recognition, and validation with TensorRT optimization and caching.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Enhanced memory manager for tensor pooling
from .ocr_engine import (
    OCREngine,
    PlateRegion,
    create_ocr_engine,
)

logger = logging.getLogger(__name__)


class PlateDetectionStatus(Enum):
    """Status of license plate detection."""

    SUCCESS = "success"
    NO_PLATE_FOUND = "no_plate_found"
    LOW_CONFIDENCE = "low_confidence"
    OCR_FAILED = "ocr_failed"
    VALIDATION_FAILED = "validation_failed"
    ERROR = "error"


@dataclass
class PlateDetectionResult:
    """Result of license plate detection and recognition."""

    # Detection info
    status: PlateDetectionStatus
    plate_text: str | None = None
    confidence: float = 0.0

    # Bounding boxes (relative to original image)
    vehicle_bbox: tuple[int, int, int, int] | None = None
    plate_bbox: tuple[int, int, int, int] | None = None

    # Processing metrics
    processing_time_ms: float = 0.0
    ocr_time_ms: float = 0.0
    detection_time_ms: float = 0.0

    # Quality metrics
    plate_quality_score: float = 0.0
    ocr_confidence: float = 0.0
    character_confidences: list[float] = None

    # Metadata
    region: PlateRegion = PlateRegion.AUTO
    engine_used: str = "unknown"

    def __post_init__(self):
        if self.character_confidences is None:
            self.character_confidences = []

    @property
    def is_reliable(self) -> bool:
        """Check if detection is reliable."""
        return (
            self.status == PlateDetectionStatus.SUCCESS and
            self.confidence >= 0.7 and
            self.ocr_confidence >= 0.7 and
            self.plate_quality_score >= 0.6
        )

    @property
    def is_high_confidence(self) -> bool:
        """Check if detection has high confidence."""
        return (
            self.is_reliable and
            self.confidence >= 0.85 and
            self.ocr_confidence >= 0.85
        )


@dataclass
class LPRConfig:
    """Configuration for License Plate Recognition pipeline."""

    # Model paths
    plate_detection_model: str = "models/plate_detector.pt"
    fallback_to_yolo_detection: bool = True

    # Performance settings
    use_gpu: bool = True
    device_ids: list[int] = None
    max_batch_size: int = 16
    target_latency_ms: float = 15.0

    # Detection thresholds
    vehicle_confidence_threshold: float = 0.7
    plate_confidence_threshold: float = 0.5
    min_plate_area: int = 400  # Minimum plate area in pixels
    max_plate_area: int = 20000  # Maximum plate area in pixels

    # OCR settings
    ocr_region: PlateRegion = PlateRegion.AUTO
    ocr_min_confidence: float = 0.6
    enable_ocr_preprocessing: bool = True

    # Caching
    enable_caching: bool = True
    cache_ttl_seconds: float = 5.0
    max_cache_size: int = 1000

    # Quality filtering
    min_plate_width: int = 60
    min_plate_height: int = 20
    max_aspect_ratio: float = 8.0
    min_aspect_ratio: float = 1.5

    def __post_init__(self):
        if self.device_ids is None:
            self.device_ids = [0] if torch.cuda.is_available() else []


class PlateLocalizer:
    """Localize license plates within vehicle bounding boxes."""

    def __init__(self, config: LPRConfig):
        self.config = config
        self.plate_detector = None

        # Initialize plate detection model
        self._initialize_plate_detector()

        # Fallback cascade classifier for plate detection
        self.cascade_classifier = None
        self._initialize_cascade_classifier()

    def _initialize_plate_detector(self):
        """Initialize YOLO-based plate detector."""
        try:
            # Try to load custom plate detection model
            if self.config.plate_detection_model:
                self.plate_detector = YOLO(self.config.plate_detection_model)
                logger.info("Loaded custom plate detection model")
        except Exception as e:
            logger.warning(f"Failed to load plate detector: {e}")

            # Use general object detection and filter for license plates
            if self.config.fallback_to_yolo_detection:
                try:
                    self.plate_detector = YOLO("yolo11n.pt")
                    logger.info("Using YOLO11 as fallback plate detector")
                except Exception as e2:
                    logger.error(f"Failed to load fallback detector: {e2}")

    def _initialize_cascade_classifier(self):
        """Initialize OpenCV cascade classifier as backup."""
        try:
            # Try to load Haar cascade for license plates
            cascade_path = "models/haarcascade_license_plate.xml"
            self.cascade_classifier = cv2.CascadeClassifier(cascade_path)

            if self.cascade_classifier.empty():
                self.cascade_classifier = None
            else:
                logger.info("Loaded Haar cascade plate detector")

        except Exception as e:
            logger.warning(f"Cascade classifier not available: {e}")

    async def localize_plates(
        self,
        image: np.ndarray,
        vehicle_bbox: tuple[int, int, int, int]
    ) -> list[tuple[int, int, int, int]]:
        """Localize license plates within vehicle bounding box.
        
        Args:
            image: Full image
            vehicle_bbox: Vehicle bounding box (x1, y1, x2, y2)
            
        Returns:
            List of plate bounding boxes relative to full image
        """
        x1, y1, x2, y2 = vehicle_bbox

        # Extract vehicle region
        vehicle_crop = image[y1:y2, x1:x2]

        if vehicle_crop.size == 0:
            return []

        plate_bboxes = []

        # Method 1: YOLO-based detection
        if self.plate_detector:
            yolo_plates = await self._detect_with_yolo(vehicle_crop)

            # Convert relative coordinates to absolute
            for px1, py1, px2, py2 in yolo_plates:
                abs_x1 = x1 + px1
                abs_y1 = y1 + py1
                abs_x2 = x1 + px2
                abs_y2 = y1 + py2

                if self._is_valid_plate_bbox(abs_x2 - abs_x1, abs_y2 - abs_y1):
                    plate_bboxes.append((abs_x1, abs_y1, abs_x2, abs_y2))

        # Method 2: Cascade classifier (if YOLO fails)
        if not plate_bboxes and self.cascade_classifier:
            cascade_plates = self._detect_with_cascade(vehicle_crop)

            # Convert relative coordinates to absolute
            for px1, py1, px2, py2 in cascade_plates:
                abs_x1 = x1 + px1
                abs_y1 = y1 + py1
                abs_x2 = x1 + px2
                abs_y2 = y1 + py2

                if self._is_valid_plate_bbox(abs_x2 - abs_x1, abs_y2 - abs_y1):
                    plate_bboxes.append((abs_x1, abs_y1, abs_x2, abs_y2))

        # Method 3: Geometric heuristics (last resort)
        if not plate_bboxes:
            heuristic_plates = self._detect_with_heuristics(vehicle_crop)

            # Convert relative coordinates to absolute
            for px1, py1, px2, py2 in heuristic_plates:
                abs_x1 = x1 + px1
                abs_y1 = y1 + py1
                abs_x2 = x1 + px2
                abs_y2 = y1 + py2

                if self._is_valid_plate_bbox(abs_x2 - abs_x1, abs_y2 - abs_y1):
                    plate_bboxes.append((abs_x1, abs_y1, abs_x2, abs_y2))

        return plate_bboxes

    async def _detect_with_yolo(self, vehicle_crop: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Detect plates using YOLO model."""
        try:
            results = self.plate_detector(vehicle_crop, verbose=False)

            plate_bboxes = []
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        confidence = float(box.conf)

                        if confidence >= self.config.plate_confidence_threshold:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                            plate_bboxes.append((x1, y1, x2, y2))

            return plate_bboxes

        except Exception as e:
            logger.warning(f"YOLO plate detection failed: {e}")
            return []

    def _detect_with_cascade(self, vehicle_crop: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Detect plates using Haar cascade."""
        try:
            gray = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2GRAY)

            plates = self.cascade_classifier.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(50, 15),
                maxSize=(300, 100)
            )

            plate_bboxes = []
            for (x, y, w, h) in plates:
                plate_bboxes.append((x, y, x + w, y + h))

            return plate_bboxes

        except Exception as e:
            logger.warning(f"Cascade plate detection failed: {e}")
            return []

    def _detect_with_heuristics(self, vehicle_crop: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Detect plates using geometric heuristics."""
        try:
            gray = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape

            # Focus on lower portion of vehicle where plates typically are
            lower_portion = gray[int(h * 0.6):, :]

            # Edge detection
            edges = cv2.Canny(lower_portion, 50, 150)

            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            plate_candidates = []

            for contour in contours:
                # Get bounding rectangle
                x, y, cw, ch = cv2.boundingRect(contour)

                # Adjust y coordinate for full image
                y += int(h * 0.6)

                # Check aspect ratio and size
                aspect_ratio = cw / ch if ch > 0 else 0
                area = cw * ch

                if (self.config.min_aspect_ratio <= aspect_ratio <= self.config.max_aspect_ratio and
                    self.config.min_plate_area <= area <= self.config.max_plate_area and
                    cw >= self.config.min_plate_width and
                    ch >= self.config.min_plate_height):

                    plate_candidates.append((x, y, x + cw, y + ch))

            # Sort by area (larger plates first) and return top candidates
            plate_candidates.sort(key=lambda bbox: (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]), reverse=True)

            return plate_candidates[:3]  # Return top 3 candidates

        except Exception as e:
            logger.warning(f"Heuristic plate detection failed: {e}")
            return []

    def _is_valid_plate_bbox(self, width: int, height: int) -> bool:
        """Check if bounding box dimensions are valid for a license plate."""
        if width <= 0 or height <= 0:
            return False

        aspect_ratio = width / height
        area = width * height

        return (
            self.config.min_aspect_ratio <= aspect_ratio <= self.config.max_aspect_ratio and
            self.config.min_plate_area <= area <= self.config.max_plate_area and
            width >= self.config.min_plate_width and
            height >= self.config.min_plate_height
        )


class PlateQualityAssessor:
    """Assess license plate image quality for OCR reliability."""

    def __init__(self, config: LPRConfig):
        self.config = config

    def assess_quality(self, plate_image: np.ndarray) -> float:
        """Assess plate image quality score (0.0 - 1.0).
        
        Args:
            plate_image: License plate image
            
        Returns:
            Quality score from 0.0 (poor) to 1.0 (excellent)
        """
        if plate_image.size == 0:
            return 0.0

        scores = []

        # 1. Sharpness (Laplacian variance)
        sharpness_score = self._assess_sharpness(plate_image)
        scores.append(sharpness_score * 0.3)

        # 2. Contrast
        contrast_score = self._assess_contrast(plate_image)
        scores.append(contrast_score * 0.25)

        # 3. Brightness
        brightness_score = self._assess_brightness(plate_image)
        scores.append(brightness_score * 0.15)

        # 4. Aspect ratio validity
        aspect_ratio_score = self._assess_aspect_ratio(plate_image)
        scores.append(aspect_ratio_score * 0.15)

        # 5. Size adequacy
        size_score = self._assess_size(plate_image)
        scores.append(size_score * 0.15)

        return sum(scores)

    def _assess_sharpness(self, image: np.ndarray) -> float:
        """Assess image sharpness using Laplacian variance."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Normalize: typical values 0-2000, good quality > 100
        return min(1.0, laplacian_var / 1000.0)

    def _assess_contrast(self, image: np.ndarray) -> float:
        """Assess image contrast."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        contrast = gray.std()

        # Normalize: typical values 0-100, good quality > 30
        return min(1.0, contrast / 60.0)

    def _assess_brightness(self, image: np.ndarray) -> float:
        """Assess image brightness (not too dark, not too bright)."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        mean_brightness = gray.mean()

        # Optimal range: 80-200, penalize extremes
        if 80 <= mean_brightness <= 200:
            return 1.0
        elif mean_brightness < 80:
            return max(0.0, mean_brightness / 80.0)
        else:  # > 200
            return max(0.0, 1.0 - (mean_brightness - 200) / 55.0)

    def _assess_aspect_ratio(self, image: np.ndarray) -> float:
        """Assess if aspect ratio is reasonable for license plates."""
        h, w = image.shape[:2]
        if h == 0:
            return 0.0

        aspect_ratio = w / h

        # Typical license plate ratios: 2.0 - 7.0
        if 2.0 <= aspect_ratio <= 7.0:
            return 1.0
        elif 1.5 <= aspect_ratio < 2.0 or 7.0 < aspect_ratio <= 8.0:
            return 0.7
        else:
            return 0.3

    def _assess_size(self, image: np.ndarray) -> float:
        """Assess if image size is adequate for OCR."""
        h, w = image.shape[:2]
        area = h * w

        # Minimum area for reliable OCR
        if area >= 3000:  # ~75x40 pixels
            return 1.0
        elif area >= 1500:  # ~50x30 pixels
            return 0.7
        elif area >= 800:   # ~40x20 pixels
            return 0.4
        else:
            return 0.1


class LPRCache:
    """Simple LRU cache for license plate recognition results."""

    def __init__(self, max_size: int = 1000, ttl_seconds: float = 5.0):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: dict[str, tuple[PlateDetectionResult, float]] = {}
        self._access_order: list[str] = []

    def _make_key(self, image: np.ndarray, vehicle_bbox: tuple[int, int, int, int]) -> str:
        """Create cache key from image and bbox."""
        # Simple hash based on image content and bbox
        x1, y1, x2, y2 = vehicle_bbox
        crop = image[y1:y2, x1:x2]

        if crop.size > 0:
            # Use mean and std as simple image fingerprint
            mean_val = crop.mean()
            std_val = crop.std()
            return f"{x1}_{y1}_{x2}_{y2}_{mean_val:.1f}_{std_val:.1f}"

        return f"{x1}_{y1}_{x2}_{y2}"

    def get(self, image: np.ndarray, vehicle_bbox: tuple[int, int, int, int]) -> PlateDetectionResult | None:
        """Get cached result if available and not expired."""
        key = self._make_key(image, vehicle_bbox)

        if key in self._cache:
            result, timestamp = self._cache[key]

            # Check if expired
            if time.time() - timestamp <= self.ttl_seconds:
                # Move to front (LRU)
                if key in self._access_order:
                    self._access_order.remove(key)
                self._access_order.append(key)

                return result
            else:
                # Expired, remove
                del self._cache[key]
                if key in self._access_order:
                    self._access_order.remove(key)

        return None

    def put(self, image: np.ndarray, vehicle_bbox: tuple[int, int, int, int], result: PlateDetectionResult):
        """Cache a result."""
        key = self._make_key(image, vehicle_bbox)

        # Remove if already exists
        if key in self._cache:
            if key in self._access_order:
                self._access_order.remove(key)

        # Add new entry
        self._cache[key] = (result, time.time())
        self._access_order.append(key)

        # Evict oldest if over capacity
        while len(self._cache) > self.max_size:
            oldest_key = self._access_order.pop(0)
            if oldest_key in self._cache:
                del self._cache[oldest_key]

    def clear(self):
        """Clear the cache."""
        self._cache.clear()
        self._access_order.clear()


class LicensePlateRecognitionPipeline:
    """Complete License Plate Recognition pipeline."""

    def __init__(self, config: LPRConfig | None = None):
        self.config = config or LPRConfig()

        # Initialize components with performance optimizations
        self.plate_localizer = PlateLocalizer(self.config)
        self.quality_assessor = PlateQualityAssessor(self.config)

        # Create optimized OCR engine with TensorRT prioritization
        self.ocr_engine = create_ocr_engine(
            region=self.config.ocr_region,
            use_gpu=self.config.use_gpu,
            primary_engine=OCREngine.TENSORRT  # Prioritize TensorRT for performance
        )

        # Initialize high-performance cache
        self.cache = None
        if self.config.enable_caching:
            self.cache = LPRCache(
                max_size=self.config.max_cache_size,
                ttl_seconds=self.config.cache_ttl_seconds
            )

        # Initialize enhanced memory manager for optimal GPU utilization
        self.memory_manager = None
        if self.config.device_ids:
            try:
                from .enhanced_memory_manager import (
                    MultiGPUMemoryManager,
                    TensorPoolConfig,
                )
                memory_config = TensorPoolConfig(
                    max_tensors_per_shape=12,  # Increased for LPR workload
                    max_total_tensors=150,     # Increased pool size
                    enable_memory_profiling=True,
                    enable_pinned_memory=True  # Enable for faster transfers
                )
                self.memory_manager = MultiGPUMemoryManager(self.config.device_ids, memory_config)
                logger.info("Enhanced memory manager initialized for LPR")
            except ImportError:
                logger.warning("Enhanced memory manager not available, using basic memory management")

        # Performance tracking with additional metrics
        self.stats = {
            "total_detections": 0,
            "successful_detections": 0,
            "cached_results": 0,
            "avg_processing_time_ms": 0.0,
            "avg_ocr_time_ms": 0.0,
            "sub_15ms_detections": 0,  # Track sub-15ms performance
            "tensorrt_usage_count": 0  # Track TensorRT usage
        }

        logger.info("Optimized License Plate Recognition pipeline initialized with TensorRT priority")

    async def recognize_plate(
        self,
        image: np.ndarray,
        vehicle_bbox: tuple[int, int, int, int],
        vehicle_confidence: float = 1.0
    ) -> PlateDetectionResult:
        """Optimized license plate recognition with sub-15ms target latency.
        
        Args:
            image: Full camera frame
            vehicle_bbox: Vehicle bounding box (x1, y1, x2, y2)
            vehicle_confidence: Confidence of vehicle detection
            
        Returns:
            License plate detection result
        """
        start_time = time.perf_counter()

        try:
            # Check cache first for sub-1ms cache hits
            if self.cache:
                cached_result = self.cache.get(image, vehicle_bbox)
                if cached_result:
                    self.stats["cached_results"] += 1
                    return cached_result

            # Fast confidence validation
            if vehicle_confidence < self.config.vehicle_confidence_threshold:
                return PlateDetectionResult(
                    status=PlateDetectionStatus.LOW_CONFIDENCE,
                    vehicle_bbox=vehicle_bbox,
                    processing_time_ms=(time.perf_counter() - start_time) * 1000
                )

            # Optimized plate localization with early termination
            detection_start = time.perf_counter()
            plate_bboxes = await self._fast_plate_localization(image, vehicle_bbox)
            detection_time = (time.perf_counter() - detection_start) * 1000

            if not plate_bboxes:
                result = PlateDetectionResult(
                    status=PlateDetectionStatus.NO_PLATE_FOUND,
                    vehicle_bbox=vehicle_bbox,
                    detection_time_ms=detection_time,
                    processing_time_ms=(time.perf_counter() - start_time) * 1000
                )

                if self.cache:
                    self.cache.put(image, vehicle_bbox, result)

                return result

            # Process only the best plate candidate for speed
            best_result = await self._process_best_plate_candidate(
                image, plate_bboxes, vehicle_bbox, vehicle_confidence, detection_time, start_time
            )

            # Cache result
            if self.cache:
                self.cache.put(image, vehicle_bbox, best_result)

            # Update stats with engine info
            engine_used = best_result.engine_used if hasattr(best_result, 'engine_used') else ""
            self._update_stats(best_result.processing_time_ms, best_result.ocr_time_ms, engine_used)

            return best_result

        except Exception as e:
            logger.error(f"License plate recognition failed: {e}")
            return PlateDetectionResult(
                status=PlateDetectionStatus.ERROR,
                vehicle_bbox=vehicle_bbox,
                processing_time_ms=(time.perf_counter() - start_time) * 1000
            )

    async def _fast_plate_localization(
        self,
        image: np.ndarray,
        vehicle_bbox: tuple[int, int, int, int]
    ) -> list[tuple[int, int, int, int]]:
        """Optimized plate localization with early termination for performance."""
        x1, y1, x2, y2 = vehicle_bbox
        vehicle_crop = image[y1:y2, x1:x2]

        if vehicle_crop.size == 0:
            return []

        plate_bboxes = []

        # Method 1: Fast YOLO-based detection (prioritized for speed)
        if self.plate_localizer.plate_detector:
            try:
                # Use optimized inference with smaller input size for speed
                yolo_plates = await self.plate_localizer._detect_with_yolo(vehicle_crop)

                # Convert and validate only best candidates
                for px1, py1, px2, py2 in yolo_plates[:2]:  # Limit to top 2 for speed
                    abs_x1, abs_y1 = x1 + px1, y1 + py1
                    abs_x2, abs_y2 = x1 + px2, y1 + py2

                    if self.plate_localizer._is_valid_plate_bbox(abs_x2 - abs_x1, abs_y2 - abs_y1):
                        plate_bboxes.append((abs_x1, abs_y1, abs_x2, abs_y2))

                # Early termination if we found good candidates
                if plate_bboxes:
                    return plate_bboxes

            except Exception as e:
                logger.debug(f"Fast YOLO detection failed: {e}")

        # Method 2: Geometric heuristics (fast fallback)
        try:
            heuristic_plates = self.plate_localizer._detect_with_heuristics(vehicle_crop)

            for px1, py1, px2, py2 in heuristic_plates[:1]:  # Only best candidate
                abs_x1, abs_y1 = x1 + px1, y1 + py1
                abs_x2, abs_y2 = x1 + px2, y1 + py2

                if self.plate_localizer._is_valid_plate_bbox(abs_x2 - abs_x1, abs_y2 - abs_y1):
                    plate_bboxes.append((abs_x1, abs_y1, abs_x2, abs_y2))
                    break  # Take first valid candidate for speed

        except Exception as e:
            logger.debug(f"Heuristic detection failed: {e}")

        return plate_bboxes

    async def _process_best_plate_candidate(
        self,
        image: np.ndarray,
        plate_bboxes: list[tuple[int, int, int, int]],
        vehicle_bbox: tuple[int, int, int, int],
        vehicle_confidence: float,
        detection_time: float,
        start_time: float
    ) -> PlateDetectionResult:
        """Process the best plate candidate with optimized OCR."""

        # Process only the first (best) candidate for maximum speed
        plate_bbox = plate_bboxes[0]
        x1, y1, x2, y2 = plate_bbox
        plate_crop = image[y1:y2, x1:x2]

        if plate_crop.size == 0:
            return PlateDetectionResult(
                status=PlateDetectionStatus.OCR_FAILED,
                vehicle_bbox=vehicle_bbox,
                detection_time_ms=detection_time,
                processing_time_ms=(time.perf_counter() - start_time) * 1000
            )

        # Fast quality assessment (simplified for speed)
        quality_score = self._fast_quality_assessment(plate_crop)

        if quality_score < 0.2:  # Very low threshold for speed
            return PlateDetectionResult(
                status=PlateDetectionStatus.OCR_FAILED,
                vehicle_bbox=vehicle_bbox,
                plate_bbox=plate_bbox,
                detection_time_ms=detection_time,
                plate_quality_score=quality_score,
                processing_time_ms=(time.perf_counter() - start_time) * 1000
            )

        # Optimized OCR with TensorRT prioritization
        ocr_start = time.perf_counter()
        ocr_result = await self.ocr_engine.recognize(plate_crop, self.config.ocr_region)
        ocr_time = (time.perf_counter() - ocr_start) * 1000

        if ocr_result and ocr_result.confidence >= self.config.ocr_min_confidence:
            # Optimized confidence calculation
            overall_confidence = (
                vehicle_confidence * 0.25 +
                quality_score * 0.25 +
                ocr_result.confidence * 0.5  # Weight OCR more heavily for speed
            )

            result = PlateDetectionResult(
                status=PlateDetectionStatus.SUCCESS,
                plate_text=ocr_result.text,
                confidence=overall_confidence,
                vehicle_bbox=vehicle_bbox,
                plate_bbox=plate_bbox,
                processing_time_ms=(time.perf_counter() - start_time) * 1000,
                ocr_time_ms=ocr_time,
                detection_time_ms=detection_time,
                plate_quality_score=quality_score,
                ocr_confidence=ocr_result.confidence,
                character_confidences=ocr_result.character_confidences,
                region=self.config.ocr_region,
                engine_used=ocr_result.engine_used
            )

            self.stats["successful_detections"] += 1
            return result
        else:
            return PlateDetectionResult(
                status=PlateDetectionStatus.OCR_FAILED,
                vehicle_bbox=vehicle_bbox,
                plate_bbox=plate_bbox,
                detection_time_ms=detection_time,
                ocr_time_ms=ocr_time,
                plate_quality_score=quality_score,
                processing_time_ms=(time.perf_counter() - start_time) * 1000
            )

    def _fast_quality_assessment(self, plate_image: np.ndarray) -> float:
        """Fast quality assessment optimized for sub-1ms execution."""
        if plate_image.size == 0:
            return 0.0

        try:
            # Simplified quality metrics for speed
            gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY) if len(plate_image.shape) == 3 else plate_image

            # Fast sharpness assessment (Laplacian variance)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_score = min(1.0, laplacian_var / 800.0)  # Adjusted threshold

            # Fast contrast assessment
            contrast = gray.std()
            contrast_score = min(1.0, contrast / 50.0)  # Adjusted threshold

            # Fast size assessment
            area = plate_image.shape[0] * plate_image.shape[1]
            size_score = 1.0 if area >= 1200 else max(0.3, area / 1200.0)

            # Weighted combination (optimized weights)
            return sharpness_score * 0.5 + contrast_score * 0.3 + size_score * 0.2

        except Exception:
            return 0.5  # Default moderate quality

    async def batch_recognize(
        self,
        image: np.ndarray,
        vehicle_detections: list[tuple[tuple[int, int, int, int], float]]
    ) -> list[PlateDetectionResult]:
        """Recognize license plates for multiple vehicles concurrently.
        
        Args:
            image: Full camera frame
            vehicle_detections: List of (bbox, confidence) tuples
            
        Returns:
            List of plate detection results
        """
        tasks = [
            self.recognize_plate(image, bbox, confidence)
            for bbox, confidence in vehicle_detections
        ]

        return await asyncio.gather(*tasks, return_exceptions=False)

    def _update_stats(self, processing_time: float, ocr_time: float, engine_used: str = ""):
        """Update performance statistics with enhanced tracking."""
        self.stats["total_detections"] += 1

        # Track sub-15ms performance
        if processing_time <= 15.0:
            self.stats["sub_15ms_detections"] += 1

        # Track TensorRT usage
        if "tensorrt" in engine_used.lower():
            self.stats["tensorrt_usage_count"] += 1

        # Update averages
        count = self.stats["total_detections"]
        old_avg_processing = self.stats["avg_processing_time_ms"]
        old_avg_ocr = self.stats["avg_ocr_time_ms"]

        self.stats["avg_processing_time_ms"] = (
            old_avg_processing * (count - 1) + processing_time
        ) / count

        self.stats["avg_ocr_time_ms"] = (
            old_avg_ocr * (count - 1) + ocr_time
        ) / count

    def get_stats(self) -> dict[str, Any]:
        """Get performance statistics."""
        success_rate = (
            self.stats["successful_detections"] / max(1, self.stats["total_detections"])
        ) * 100

        cache_hit_rate = (
            self.stats["cached_results"] / max(1, self.stats["total_detections"])
        ) * 100 if self.cache else 0.0

        sub_15ms_rate = (
            self.stats["sub_15ms_detections"] / max(1, self.stats["total_detections"])
        ) * 100

        tensorrt_usage_rate = (
            self.stats["tensorrt_usage_count"] / max(1, self.stats["total_detections"])
        ) * 100

        return {
            **self.stats,
            "success_rate_percent": success_rate,
            "cache_hit_rate_percent": cache_hit_rate,
            "sub_15ms_rate_percent": sub_15ms_rate,
            "tensorrt_usage_rate_percent": tensorrt_usage_rate,
            "performance_target_met": sub_15ms_rate >= 90.0,  # 90% of requests under 15ms
            "ocr_stats": self.ocr_engine.get_stats() if self.ocr_engine else {}
        }

    def cleanup(self):
        """Clean up resources."""
        if self.ocr_engine:
            self.ocr_engine.cleanup()

        if self.memory_manager:
            self.memory_manager.cleanup()

        if self.cache:
            self.cache.clear()

        logger.info("LPR pipeline cleanup completed")


# Factory function for easy initialization
def create_lpr_pipeline(
    region: PlateRegion = PlateRegion.AUTO,
    use_gpu: bool = True,
    enable_caching: bool = True,
    target_latency_ms: float = 15.0
) -> LicensePlateRecognitionPipeline:
    """Create an LPR pipeline with common configurations.
    
    Args:
        region: Target license plate region
        use_gpu: Whether to use GPU acceleration
        enable_caching: Whether to enable result caching
        target_latency_ms: Target processing latency
        
    Returns:
        Configured LPR pipeline
    """
    config = LPRConfig(
        use_gpu=use_gpu,
        device_ids=[0] if use_gpu and torch.cuda.is_available() else [],
        target_latency_ms=target_latency_ms,
        ocr_region=region,
        enable_caching=enable_caching,
        vehicle_confidence_threshold=0.7,
        plate_confidence_threshold=0.5,
        ocr_min_confidence=0.6
    )

    return LicensePlateRecognitionPipeline(config)


# Example usage and testing
async def test_lpr_pipeline():
    """Test the LPR pipeline with a sample image."""
    # Create LPR pipeline
    lpr = create_lpr_pipeline(region=PlateRegion.US, use_gpu=True)

    # Create a sample image with a vehicle
    sample_image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

    # Add a vehicle-like rectangle
    cv2.rectangle(sample_image, (400, 300), (800, 500), (100, 100, 200), -1)

    # Add a license plate-like rectangle
    cv2.rectangle(sample_image, (550, 420), (650, 460), (255, 255, 255), -1)
    cv2.putText(sample_image, "ABC-1234", (555, 445), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    # Define vehicle bounding box
    vehicle_bbox = (400, 300, 800, 500)

    # Recognize license plate
    result = await lpr.recognize_plate(sample_image, vehicle_bbox, vehicle_confidence=0.9)

    print(f"Status: {result.status.value}")
    if result.plate_text:
        print(f"Plate text: {result.plate_text}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"OCR confidence: {result.ocr_confidence:.2f}")
        print(f"Quality score: {result.plate_quality_score:.2f}")
    print(f"Processing time: {result.processing_time_ms:.1f}ms")
    print(f"Reliable: {result.is_reliable}")

    # Get pipeline stats
    stats = lpr.get_stats()
    print(f"Success rate: {stats['success_rate_percent']:.1f}%")

    # Cleanup
    lpr.cleanup()


if __name__ == "__main__":
    asyncio.run(test_lpr_pipeline())
