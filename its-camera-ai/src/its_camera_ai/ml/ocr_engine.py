"""
OCR Engine for License Plate Recognition.

High-performance optical character recognition optimized for license plates
with TensorRT acceleration, multi-regional support, and fallback engines.
"""

import asyncio
import logging
import re
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# OCR engines with graceful fallbacks
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    easyocr = None

try:
    import paddleocr
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False
    paddleocr = None

logger = logging.getLogger(__name__)


class OCREngine(Enum):
    """Available OCR engines."""

    TENSORRT = "tensorrt"
    EASYOCR = "easyocr"
    PADDLEOCR = "paddleocr"
    AUTO = "auto"


class PlateRegion(Enum):
    """Supported license plate regions."""

    US = "us"
    EU = "eu"
    ASIA = "asia"
    AUTO = "auto"


@dataclass
class OCRResult:
    """OCR recognition result."""

    text: str
    confidence: float
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    character_confidences: list[float]
    processing_time_ms: float
    engine_used: str

    @property
    def is_high_confidence(self) -> bool:
        """Check if result has high confidence."""
        return self.confidence >= 0.8

    @property
    def is_reliable(self) -> bool:
        """Check if result is reliable for license plate."""
        return (
            self.confidence >= 0.7 and
            len(self.text) >= 4 and
            all(c >= 0.6 for c in self.character_confidences)
        )


@dataclass
class OCRConfig:
    """Configuration for OCR engines."""

    # Engine selection
    primary_engine: OCREngine = OCREngine.AUTO
    fallback_engines: list[OCREngine] = None

    # Regional settings
    region: PlateRegion = PlateRegion.AUTO
    languages: list[str] = None

    # Performance settings
    use_gpu: bool = True
    batch_size: int = 8
    max_workers: int = 4

    # Quality settings
    min_confidence: float = 0.5
    enable_preprocessing: bool = True
    enable_postprocessing: bool = True

    # TensorRT settings
    tensorrt_precision: str = "fp16"
    workspace_size_mb: int = 1024

    def __post_init__(self):
        if self.fallback_engines is None:
            self.fallback_engines = [OCREngine.EASYOCR, OCREngine.PADDLEOCR]

        if self.languages is None:
            self.languages = ["en"]


class PlatePreprocessor:
    """Preprocessor for license plate images."""

    def __init__(self, config: OCRConfig):
        self.config = config
        self.target_height = 64
        self.target_width = 256

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess license plate image for OCR.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Preprocessed image optimized for OCR
        """
        if not self.config.enable_preprocessing:
            return image

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Resize with aspect ratio preservation
        processed = self._resize_with_padding(gray)

        # Enhance contrast
        processed = self._enhance_contrast(processed)

        # Denoise
        processed = self._denoise(processed)

        # Binarize
        processed = self._binarize(processed)

        # Morphological operations
        processed = self._morphological_cleanup(processed)

        return processed

    def _resize_with_padding(self, image: np.ndarray) -> np.ndarray:
        """Resize image with letterboxing to target dimensions."""
        h, w = image.shape
        scale = min(self.target_width / w, self.target_height / h)

        new_w = int(w * scale)
        new_h = int(h * scale)

        # Resize
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        # Add padding
        pad_w = (self.target_width - new_w) // 2
        pad_h = (self.target_height - new_h) // 2

        padded = np.full((self.target_height, self.target_width), 128, dtype=np.uint8)
        padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

        return padded

    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance image contrast using CLAHE."""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)

    def _denoise(self, image: np.ndarray) -> np.ndarray:
        """Remove noise from image."""
        # Gaussian blur to remove noise
        denoised = cv2.GaussianBlur(image, (3, 3), 0)

        # Bilateral filter for edge preservation
        denoised = cv2.bilateralFilter(denoised, 9, 75, 75)

        return denoised

    def _binarize(self, image: np.ndarray) -> np.ndarray:
        """Convert to binary image."""
        # Adaptive thresholding
        binary = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        # Also try Otsu's method and combine
        _, otsu = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Combine both methods
        combined = cv2.bitwise_and(binary, otsu)

        return combined

    def _morphological_cleanup(self, image: np.ndarray) -> np.ndarray:
        """Clean up binary image using morphological operations."""
        # Remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

        # Fill gaps in characters
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

        return cleaned


class TextPostProcessor:
    """Post-processor for OCR text results."""

    def __init__(self, config: OCRConfig):
        self.config = config

        # Regional patterns
        self.patterns = {
            PlateRegion.US: [
                r"^[A-Z0-9]{2,3}[-\s]?[A-Z0-9]{3,4}$",  # AAA-1234, AB-123
                r"^[0-9]{3}[-\s]?[A-Z]{3}$",             # 123-ABC
                r"^[A-Z]{3}[-\s]?[0-9]{4}$",             # ABC-1234
            ],
            PlateRegion.EU: [
                r"^[A-Z]{1,3}[-\s]?[0-9]{1,4}[-\s]?[A-Z]{1,3}$",  # AB-123-CD
                r"^[0-9]{3,4}[-\s]?[A-Z]{2,3}[-\s]?[0-9]{1,2}$",  # 1234-AB-56
            ],
            PlateRegion.ASIA: [
                r"^[A-Z0-9]{4,8}$",  # Various formats
            ]
        }

    def postprocess(self, text: str, region: PlateRegion = None) -> str:
        """Post-process OCR text for license plates.
        
        Args:
            text: Raw OCR text
            region: License plate region
            
        Returns:
            Cleaned and formatted text
        """
        if not self.config.enable_postprocessing:
            return text

        # Clean text
        cleaned = self._clean_text(text)

        # Format according to region
        if region:
            formatted = self._format_for_region(cleaned, region)
        else:
            formatted = self._auto_format(cleaned)

        return formatted

    def _clean_text(self, text: str) -> str:
        """Clean raw OCR text."""
        # Remove non-alphanumeric characters except hyphens and spaces
        cleaned = re.sub(r"[^A-Z0-9\-\s]", "", text.upper())

        # Fix common OCR mistakes
        replacements = {
            "0": "O",  # Context-dependent
            "1": "I",  # Context-dependent
            "8": "B",  # Context-dependent
            "5": "S",  # Context-dependent
        }

        # Apply replacements based on position and context
        cleaned = self._apply_contextual_fixes(cleaned, replacements)

        # Remove extra spaces and normalize
        cleaned = re.sub(r"\s+", " ", cleaned.strip())

        return cleaned

    def _apply_contextual_fixes(self, text: str, replacements: dict[str, str]) -> str:
        """Apply OCR fixes based on context."""
        # Simple heuristic: letters usually at start/end, numbers in middle
        result = ""

        for i, char in enumerate(text):
            if char in replacements:
                # First and last positions are usually letters
                if i == 0 or i == len(text) - 1:
                    if char in "01":
                        result += "O" if char == "0" else "I"
                    else:
                        result += char
                else:
                    result += char
            else:
                result += char

        return result

    def _format_for_region(self, text: str, region: PlateRegion) -> str:
        """Format text according to regional patterns."""
        patterns = self.patterns.get(region, [])

        for pattern in patterns:
            if re.match(pattern, text):
                return text

        # Try to auto-format if no pattern matches
        return self._auto_format(text)

    def _auto_format(self, text: str) -> str:
        """Auto-format text based on common patterns."""
        # Remove all spaces and hyphens
        clean = re.sub(r"[\s\-]", "", text)

        # Common US format: 3 letters + 4 numbers
        if len(clean) == 7 and clean[:3].isalpha() and clean[3:].isdigit():
            return f"{clean[:3]}-{clean[3:]}"

        # Common US format: 3 numbers + 3 letters
        if len(clean) == 6 and clean[:3].isdigit() and clean[3:].isalpha():
            return f"{clean[:3]}-{clean[3:]}"

        # Default: return as-is
        return text

    def validate_format(self, text: str, region: PlateRegion = None) -> bool:
        """Validate if text matches expected plate format."""
        if region:
            patterns = self.patterns.get(region, [])
            return any(re.match(pattern, text) for pattern in patterns)

        # Check against all patterns
        all_patterns = []
        for region_patterns in self.patterns.values():
            all_patterns.extend(region_patterns)

        return any(re.match(pattern, text) for pattern in all_patterns)


class EasyOCRWrapper:
    """Wrapper for EasyOCR engine."""

    def __init__(self, config: OCRConfig):
        self.config = config
        self.reader = None

        if EASYOCR_AVAILABLE:
            self._initialize_reader()

    def _initialize_reader(self):
        """Initialize EasyOCR reader."""
        try:
            self.reader = easyocr.Reader(
                self.config.languages,
                gpu=self.config.use_gpu
            )
            logger.info("EasyOCR initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR: {e}")
            self.reader = None

    async def recognize(self, image: np.ndarray) -> list[OCRResult]:
        """Recognize text in image using EasyOCR."""
        if not self.reader:
            raise RuntimeError("EasyOCR not available")

        start_time = time.perf_counter()

        try:
            # EasyOCR expects PIL Image or numpy array
            results = self.reader.readtext(image)

            processing_time = (time.perf_counter() - start_time) * 1000

            ocr_results = []
            for result in results:
                bbox_points, text, confidence = result

                # Convert bbox points to x1, y1, x2, y2
                bbox_array = np.array(bbox_points)
                x1, y1 = bbox_array.min(axis=0).astype(int)
                x2, y2 = bbox_array.max(axis=0).astype(int)

                ocr_result = OCRResult(
                    text=text,
                    confidence=confidence,
                    bbox=(x1, y1, x2, y2),
                    character_confidences=[confidence],  # EasyOCR doesn't provide per-char
                    processing_time_ms=processing_time,
                    engine_used="easyocr"
                )

                ocr_results.append(ocr_result)

            return ocr_results

        except Exception as e:
            logger.error(f"EasyOCR recognition failed: {e}")
            return []


class PaddleOCRWrapper:
    """Wrapper for PaddleOCR engine."""

    def __init__(self, config: OCRConfig):
        self.config = config
        self.reader = None

        if PADDLEOCR_AVAILABLE:
            self._initialize_reader()

    def _initialize_reader(self):
        """Initialize PaddleOCR reader."""
        try:
            self.reader = paddleocr.PaddleOCR(
                use_angle_cls=True,
                lang='en',
                use_gpu=self.config.use_gpu
            )
            logger.info("PaddleOCR initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize PaddleOCR: {e}")
            self.reader = None

    async def recognize(self, image: np.ndarray) -> list[OCRResult]:
        """Recognize text in image using PaddleOCR."""
        if not self.reader:
            raise RuntimeError("PaddleOCR not available")

        start_time = time.perf_counter()

        try:
            results = self.reader.ocr(image, cls=True)

            processing_time = (time.perf_counter() - start_time) * 1000

            ocr_results = []
            for line in results:
                if line:
                    for result in line:
                        bbox_points, (text, confidence) = result

                        # Convert bbox points to x1, y1, x2, y2
                        bbox_array = np.array(bbox_points)
                        x1, y1 = bbox_array.min(axis=0).astype(int)
                        x2, y2 = bbox_array.max(axis=0).astype(int)

                        ocr_result = OCRResult(
                            text=text,
                            confidence=confidence,
                            bbox=(x1, y1, x2, y2),
                            character_confidences=[confidence],  # PaddleOCR doesn't provide per-char
                            processing_time_ms=processing_time,
                            engine_used="paddleocr"
                        )

                        ocr_results.append(ocr_result)

            return ocr_results

        except Exception as e:
            logger.error(f"PaddleOCR recognition failed: {e}")
            return []


class TensorRTOCR:
    """Production TensorRT-optimized OCR engine for license plates."""

    def __init__(self, config: OCRConfig, model_path: str | None = None):
        self.config = config
        self.model_path = model_path or "models/ocr_crnn_tensorrt.trt"

        # TensorRT components
        self.engine: Any | None = None  # trt.ICudaEngine when available
        self.context: Any | None = None  # trt.IExecutionContext when available
        self.stream: Any | None = None  # cuda.Stream when available

        # Memory management
        self.input_buffer = None
        self.output_buffer = None
        self.host_input = None
        self.host_output = None

        # Character mapping for license plates
        self.char_map = self._create_char_map()

        # Performance tracking
        self.inference_times: deque = deque(maxlen=1000)

        self._initialize_engine()

    def _create_char_map(self) -> dict[int, str]:
        """Create character mapping for OCR output."""
        chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-"
        return {i: char for i, char in enumerate(chars)}

    def _initialize_engine(self) -> None:
        """Initialize TensorRT OCR engine."""
        try:
            # Import TensorRT components
            import pycuda.autoinit  # noqa: F401
            import pycuda.driver as cuda
            import tensorrt as trt

            if not Path(self.model_path).exists():
                logger.warning(f"TensorRT OCR model not found at {self.model_path}, falling back")
                return

            # Load TensorRT engine
            trt_logger = trt.Logger(trt.Logger.WARNING)
            with open(self.model_path, 'rb') as f:
                runtime = trt.Runtime(trt_logger)
                self.engine = runtime.deserialize_cuda_engine(f.read())

            if self.engine is None:
                logger.error("Failed to load TensorRT OCR engine")
                return

            self.context = self.engine.create_execution_context()
            self.stream = cuda.Stream()

            # Allocate memory buffers
            self._allocate_buffers()

            logger.info("TensorRT OCR engine initialized successfully")

        except ImportError:
            logger.warning("TensorRT not available, TensorRT OCR disabled")
        except Exception as e:
            logger.error(f"Failed to initialize TensorRT OCR: {e}")

    def _allocate_buffers(self) -> None:
        """Allocate input/output buffers for TensorRT inference."""
        import pycuda.driver as cuda

        # Get binding dimensions
        input_shape = self.engine.get_binding_shape(0)
        output_shape = self.engine.get_binding_shape(1)

        # Calculate buffer sizes
        input_size = np.prod(input_shape) * np.dtype(np.float32).itemsize
        output_size = np.prod(output_shape) * np.dtype(np.float32).itemsize

        # Allocate device memory
        self.input_buffer = cuda.mem_alloc(input_size)
        self.output_buffer = cuda.mem_alloc(output_size)

        # Allocate pinned host memory for async transfers
        self.host_input = cuda.pagelocked_empty(input_shape, np.float32)
        self.host_output = cuda.pagelocked_empty(output_shape, np.float32)

        logger.debug(f"Allocated TensorRT OCR buffers: input {input_shape}, output {output_shape}")

    async def recognize(self, image: np.ndarray) -> list[OCRResult]:
        """Recognize text using optimized TensorRT engine."""
        if self.engine is None or self.context is None:
            # Fall back to other engines
            return []

        start_time = time.perf_counter()

        try:
            # Preprocess image for CRNN input
            processed_input = self._preprocess_for_crnn(image)

            # Copy to host input buffer
            np.copyto(self.host_input, processed_input.ravel())

            # Async memory transfer and inference
            import pycuda.driver as cuda

            # Copy input to device
            cuda.memcpy_htod_async(self.input_buffer, self.host_input, self.stream)

            # Execute inference
            bindings = [int(self.input_buffer), int(self.output_buffer)]
            self.context.execute_async_v2(bindings, self.stream.handle)

            # Copy output back to host
            cuda.memcpy_dtoh_async(self.host_output, self.output_buffer, self.stream)

            # Synchronize stream
            self.stream.synchronize()

            # Post-process output
            text, confidence = self._postprocess_crnn_output(self.host_output)

            processing_time = (time.perf_counter() - start_time) * 1000
            self.inference_times.append(processing_time)

            if text and len(text) >= 4:  # Valid license plate length
                return [OCRResult(
                    text=text,
                    confidence=confidence,
                    bbox=(0, 0, image.shape[1], image.shape[0]),
                    character_confidences=[confidence] * len(text),
                    processing_time_ms=processing_time,
                    engine_used="tensorrt_crnn"
                )]

        except Exception as e:
            logger.error(f"TensorRT OCR inference failed: {e}")

        return []

    def _preprocess_for_crnn(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for CRNN model input."""
        # Target size for CRNN (height=32, width=128)
        target_height, target_width = 32, 128

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Resize with aspect ratio preservation
        h, w = gray.shape
        scale = min(target_height / h, target_width / w)

        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Pad to target size
        pad_h = (target_height - new_h) // 2
        pad_w = (target_width - new_w) // 2

        padded = np.full((target_height, target_width), 128, dtype=np.uint8)
        padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

        # Normalize to [0, 1] and add batch dimension
        normalized = padded.astype(np.float32) / 255.0

        # Convert to CHW format: (1, 1, 32, 128)
        return normalized.reshape(1, 1, target_height, target_width)

    def _postprocess_crnn_output(self, output: np.ndarray) -> tuple[str, float]:
        """Postprocess CRNN output to text and confidence."""
        # CRNN output is typically (seq_len, num_classes)
        # Apply CTC decoding

        # Get predicted classes (greedy decoding)
        predictions = np.argmax(output, axis=-1)

        # Remove duplicates and blanks (CTC decoding)
        decoded_chars = []
        prev_char = None

        confidences = []

        for pred in predictions.flatten():
            if pred != prev_char and pred < len(self.char_map):
                char = self.char_map.get(pred, '')
                if char and char != '-':  # '-' is often used as blank
                    decoded_chars.append(char)
                    # Get confidence from softmax output
                    char_conf = np.max(output[len(decoded_chars)-1]) if len(output) > len(decoded_chars)-1 else 0.5
                    confidences.append(float(char_conf))
            prev_char = pred

        text = ''.join(decoded_chars)
        avg_confidence = np.mean(confidences) if confidences else 0.0

        return text, avg_confidence

    def get_performance_stats(self) -> dict[str, float]:
        """Get TensorRT OCR performance statistics."""
        if not self.inference_times:
            return {}

        times = list(self.inference_times)
        return {
            'avg_latency_ms': float(np.mean(times)),
            'min_latency_ms': float(np.min(times)),
            'max_latency_ms': float(np.max(times)),
            'p95_latency_ms': float(np.percentile(times, 95)),
            'p99_latency_ms': float(np.percentile(times, 99)),
        }

    def cleanup(self) -> None:
        """Clean up TensorRT resources."""
        try:
            if hasattr(self, 'input_buffer') and self.input_buffer:
                self.input_buffer.free()
            if hasattr(self, 'output_buffer') and self.output_buffer:
                self.output_buffer.free()

            if self.context:
                del self.context
            if self.engine:
                del self.engine

            logger.debug("TensorRT OCR cleanup completed")
        except Exception as e:
            logger.error(f"Error during TensorRT OCR cleanup: {e}")


class AdvancedOCREngine:
    """Advanced OCR engine with multiple backends and optimization."""

    def __init__(self, config: OCRConfig | None = None):
        self.config = config or OCRConfig()

        # Initialize components
        self.preprocessor = PlatePreprocessor(self.config)
        self.postprocessor = TextPostProcessor(self.config)

        # Initialize engines
        self.engines = {}
        self._initialize_engines()

        # Performance stats
        self.stats = {
            "total_recognitions": 0,
            "successful_recognitions": 0,
            "avg_processing_time_ms": 0.0,
            "engine_usage": {}
        }

    def _initialize_engines(self):
        """Initialize available OCR engines."""
        # TensorRT engine (if available)
        try:
            self.engines[OCREngine.TENSORRT] = TensorRTOCR(self.config)
        except Exception as e:
            logger.warning(f"TensorRT OCR not available: {e}")

        # EasyOCR engine
        if EASYOCR_AVAILABLE:
            try:
                self.engines[OCREngine.EASYOCR] = EasyOCRWrapper(self.config)
            except Exception as e:
                logger.warning(f"EasyOCR not available: {e}")

        # PaddleOCR engine
        if PADDLEOCR_AVAILABLE:
            try:
                self.engines[OCREngine.PADDLEOCR] = PaddleOCRWrapper(self.config)
            except Exception as e:
                logger.warning(f"PaddleOCR not available: {e}")

        if not self.engines:
            raise RuntimeError("No OCR engines available")

        logger.info(f"Initialized OCR engines: {list(self.engines.keys())}")

    async def recognize(
        self,
        image: np.ndarray,
        region: PlateRegion = PlateRegion.AUTO
    ) -> OCRResult | None:
        """Recognize license plate text in image.
        
        Args:
            image: Input image containing license plate
            region: Expected plate region for validation
            
        Returns:
            Best OCR result or None if recognition failed
        """
        start_time = time.perf_counter()

        try:
            # Preprocess image
            processed_image = self.preprocessor.preprocess(image)

            # Determine engine order
            engine_order = self._get_engine_order()

            best_result = None

            # Try each engine in order
            for engine_type in engine_order:
                if engine_type not in self.engines:
                    continue

                try:
                    engine = self.engines[engine_type]
                    results = await engine.recognize(processed_image)

                    if results:
                        # Post-process and validate results
                        for result in results:
                            processed_text = self.postprocessor.postprocess(
                                result.text, region
                            )

                            # Update result with processed text
                            result.text = processed_text

                            # Check if this is a good result
                            if self._is_valid_result(result, region):
                                best_result = result
                                break

                    # Update engine usage stats
                    self._update_engine_stats(engine_type)

                    # If we found a good result, stop trying other engines
                    if best_result and best_result.is_reliable:
                        break

                except Exception as e:
                    logger.warning(f"Engine {engine_type} failed: {e}")
                    continue

            # Update performance stats
            processing_time = (time.perf_counter() - start_time) * 1000
            self._update_performance_stats(processing_time, best_result is not None)

            return best_result

        except Exception as e:
            logger.error(f"OCR recognition failed: {e}")
            return None

    def _get_engine_order(self) -> list[OCREngine]:
        """Get ordered list of engines to try with TensorRT prioritization."""
        if self.config.primary_engine == OCREngine.AUTO:
            # Prioritize TensorRT for production performance
            order = [OCREngine.TENSORRT, OCREngine.EASYOCR, OCREngine.PADDLEOCR]
        else:
            # Start with primary, then fallbacks
            order = [self.config.primary_engine] + self.config.fallback_engines

        # Filter to only available engines and prioritize TensorRT
        available_engines = [engine for engine in order if engine in self.engines]

        # If TensorRT is available and working, use it first for performance
        if OCREngine.TENSORRT in available_engines and OCREngine.TENSORRT in self.engines:
            tensorrt_engine = self.engines[OCREngine.TENSORRT]
            if hasattr(tensorrt_engine, 'engine') and tensorrt_engine.engine is not None:
                # Move TensorRT to front if it's properly initialized
                if OCREngine.TENSORRT in available_engines:
                    available_engines.remove(OCREngine.TENSORRT)
                available_engines.insert(0, OCREngine.TENSORRT)

        return available_engines

    def _is_valid_result(self, result: OCRResult, region: PlateRegion) -> bool:
        """Check if OCR result is valid for license plate."""
        # Basic confidence check
        if result.confidence < self.config.min_confidence:
            return False

        # Length check (license plates are typically 4-8 characters)
        if len(result.text) < 4 or len(result.text) > 10:
            return False

        # Format validation
        if region != PlateRegion.AUTO:
            return self.postprocessor.validate_format(result.text, region)

        return True

    def _update_engine_stats(self, engine: OCREngine):
        """Update engine usage statistics."""
        engine_name = engine.value
        if engine_name not in self.stats["engine_usage"]:
            self.stats["engine_usage"][engine_name] = 0
        self.stats["engine_usage"][engine_name] += 1

    def _update_performance_stats(self, processing_time: float, success: bool):
        """Update performance statistics."""
        self.stats["total_recognitions"] += 1

        if success:
            self.stats["successful_recognitions"] += 1

        # Update average processing time
        old_avg = self.stats["avg_processing_time_ms"]
        count = self.stats["total_recognitions"]
        self.stats["avg_processing_time_ms"] = (old_avg * (count - 1) + processing_time) / count

    def get_stats(self) -> dict[str, Any]:
        """Get performance statistics."""
        success_rate = (
            self.stats["successful_recognitions"] / max(1, self.stats["total_recognitions"])
        ) * 100

        return {
            **self.stats,
            "success_rate_percent": success_rate
        }

    async def batch_recognize(
        self,
        images: list[np.ndarray],
        region: PlateRegion = PlateRegion.AUTO
    ) -> list[OCRResult | None]:
        """Recognize text in multiple images concurrently."""
        tasks = [self.recognize(image, region) for image in images]
        return await asyncio.gather(*tasks, return_exceptions=False)

    def cleanup(self):
        """Clean up resources."""
        for engine in self.engines.values():
            if hasattr(engine, 'cleanup'):
                engine.cleanup()

        self.engines.clear()
        logger.info("OCR engine cleanup completed")


# Factory function for easy initialization
def create_ocr_engine(
    region: PlateRegion = PlateRegion.AUTO,
    use_gpu: bool = True,
    primary_engine: OCREngine = OCREngine.AUTO
) -> AdvancedOCREngine:
    """Create an OCR engine with common configurations.
    
    Args:
        region: Target license plate region
        use_gpu: Whether to use GPU acceleration
        primary_engine: Primary OCR engine to use
        
    Returns:
        Configured OCR engine
    """
    config = OCRConfig(
        primary_engine=primary_engine,
        region=region,
        use_gpu=use_gpu,
        min_confidence=0.6,
        enable_preprocessing=True,
        enable_postprocessing=True
    )

    return AdvancedOCREngine(config)


# Example usage and testing
async def test_ocr_engine():
    """Test the OCR engine with a sample image."""
    # Create OCR engine
    ocr = create_ocr_engine(region=PlateRegion.US, use_gpu=True)

    # Create a sample license plate image (placeholder)
    sample_image = np.random.randint(0, 255, (100, 300, 3), dtype=np.uint8)

    # Add some text-like patterns
    cv2.rectangle(sample_image, (50, 30), (250, 70), (255, 255, 255), -1)
    cv2.putText(sample_image, "ABC-1234", (60, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # Recognize text
    result = await ocr.recognize(sample_image)

    if result:
        print(f"Recognized: {result.text}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Engine: {result.engine_used}")
        print(f"Processing time: {result.processing_time_ms:.1f}ms")
    else:
        print("No text recognized")

    # Get stats
    stats = ocr.get_stats()
    print(f"Success rate: {stats['success_rate_percent']:.1f}%")

    # Cleanup
    ocr.cleanup()


if __name__ == "__main__":
    asyncio.run(test_ocr_engine())
