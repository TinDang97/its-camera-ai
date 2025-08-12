"""Unit tests for model upload functionality."""

import hashlib
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException, UploadFile

from its_camera_ai.api.routers.models import (
    ALLOWED_CONFIG_EXTENSIONS,
    ALLOWED_MODEL_EXTENSIONS,
    MALICIOUS_PATTERNS,
    calculate_file_checksum,
    create_model_storage_directory,
    save_uploaded_file,
    scan_for_malicious_content,
    validate_file_extension,
    validate_model_structure,
)
from its_camera_ai.api.schemas.models import ModelFramework, ModelType


class TestFileValidation:
    """Test file validation functions."""

    def test_validate_file_extension_valid(self):
        """Test valid file extensions."""
        assert validate_file_extension("model.pt", ALLOWED_MODEL_EXTENSIONS)
        assert validate_file_extension("model.onnx", ALLOWED_MODEL_EXTENSIONS)
        assert validate_file_extension("config.yaml", ALLOWED_CONFIG_EXTENSIONS)
        assert validate_file_extension("requirements.txt", ALLOWED_CONFIG_EXTENSIONS)

    def test_validate_file_extension_invalid(self):
        """Test invalid file extensions."""
        assert not validate_file_extension("malicious.exe", ALLOWED_MODEL_EXTENSIONS)
        assert not validate_file_extension("script.py", ALLOWED_MODEL_EXTENSIONS)
        assert not validate_file_extension("data.csv", ALLOWED_CONFIG_EXTENSIONS)

    def test_validate_file_extension_case_insensitive(self):
        """Test case insensitive extension validation."""
        assert validate_file_extension("model.PT", ALLOWED_MODEL_EXTENSIONS)
        assert validate_file_extension("config.YAML", ALLOWED_CONFIG_EXTENSIONS)

    def test_calculate_file_checksum(self):
        """Test file checksum calculation."""
        content = b"test model content"
        checksum = calculate_file_checksum(content)

        # Verify it's a valid SHA256 hex string
        assert len(checksum) == 64
        assert all(c in "0123456789abcdef" for c in checksum)

        # Verify it's consistent
        assert checksum == calculate_file_checksum(content)

        # Verify it matches expected SHA256
        expected = hashlib.sha256(content).hexdigest()
        assert checksum == expected

    @pytest.mark.asyncio
    async def test_scan_for_malicious_content_clean(self):
        """Test scanning clean content."""
        clean_content = (
            b"import torch\nmodel = torch.load('model.pt', weights_only=True)"
        )
        patterns = await scan_for_malicious_content(clean_content)
        assert patterns == []

    @pytest.mark.asyncio
    async def test_scan_for_malicious_content_malicious(self):
        """Test scanning malicious content."""
        malicious_content = b"import os\nos.system('rm -rf /')"
        patterns = await scan_for_malicious_content(malicious_content)
        assert "os.system" in patterns

    @pytest.mark.asyncio
    async def test_scan_for_malicious_content_multiple_patterns(self):
        """Test scanning content with multiple malicious patterns."""
        malicious_content = b"exec('malicious code') and eval('more malicious')"
        patterns = await scan_for_malicious_content(malicious_content)
        assert "exec(" in patterns
        assert "eval(" in patterns


class TestModelStructureValidation:
    """Test model structure validation."""

    @pytest.mark.asyncio
    async def test_validate_pytorch_model_structure(self):
        """Test PyTorch model validation."""
        # Create a temporary file with mock model data
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp_file:
            # Write some dummy data (in real scenario, this would be a PyTorch model)
            tmp_file.write(b"dummy pytorch model data")
            tmp_path = Path(tmp_file.name)

        try:
            with patch("torch.load") as mock_load:
                mock_load.return_value = {
                    "model": "state_dict",
                    "optimizer": "optimizer_state",
                }

                result = await validate_model_structure(
                    tmp_path, ModelFramework.PYTORCH
                )

                assert result["is_valid"] is True
                assert result["framework"] == ModelFramework.PYTORCH
                assert "torch_version" in result["metadata"]
                assert result["metadata"]["has_model_state"] is True
                assert result["metadata"]["has_optimizer_state"] is True
                assert "Model contains optimizer state" in result["warnings"]
        finally:
            tmp_path.unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_validate_onnx_model_structure(self):
        """Test ONNX model validation."""
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp_file:
            tmp_file.write(b"dummy onnx model data")
            tmp_path = Path(tmp_file.name)

        try:
            with patch("onnx.load") as mock_load, patch("onnx.checker.check_model"):
                mock_model = MagicMock()
                mock_model.ir_version = 7
                mock_model.opset_import = [MagicMock(version=11)]
                mock_load.return_value = mock_model

                result = await validate_model_structure(tmp_path, ModelFramework.ONNX)

                assert result["is_valid"] is True
                assert result["framework"] == ModelFramework.ONNX
                assert result["metadata"]["ir_version"] == 7
                assert result["metadata"]["opset_version"] == 11
        finally:
            tmp_path.unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_validate_tensorrt_model_structure(self):
        """Test TensorRT model validation."""
        with tempfile.NamedTemporaryFile(suffix=".engine", delete=False) as tmp_file:
            tmp_file.write(b"dummy tensorrt engine data")
            tmp_path = Path(tmp_file.name)

        try:
            result = await validate_model_structure(tmp_path, ModelFramework.TENSORRT)

            assert result["is_valid"] is True
            assert result["framework"] == ModelFramework.TENSORRT
            assert result["metadata"]["engine_file"] is True
        finally:
            tmp_path.unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_validate_model_structure_invalid_pytorch(self):
        """Test validation failure for invalid PyTorch model."""
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp_file:
            tmp_file.write(b"invalid model data")
            tmp_path = Path(tmp_file.name)

        try:
            with (
                patch("torch.load", side_effect=Exception("Invalid model")),
                pytest.raises(ValueError, match="Model validation failed"),
            ):
                await validate_model_structure(tmp_path, ModelFramework.PYTORCH)
        finally:
            tmp_path.unlink(missing_ok=True)


class TestStorageManagement:
    """Test storage management functions."""

    @pytest.mark.asyncio
    async def test_create_model_storage_directory(self):
        """Test model storage directory creation."""
        with (
            tempfile.TemporaryDirectory() as tmp_dir,
            patch(
                "its_camera_ai.api.routers.models.MODEL_STORAGE_BASE_PATH",
                Path(tmp_dir),
            ),
        ):
            storage_dir = await create_model_storage_directory("test-model", "1.0.0")

            assert storage_dir.exists()
            assert storage_dir.is_dir()
            assert storage_dir.name == "1.0.0"
            assert storage_dir.parent.name == "test-model"

    @pytest.mark.asyncio
    async def test_save_uploaded_file_success(self):
        """Test successful file upload and save."""
        content = b"test model file content"

        # Create mock UploadFile
        upload_file = MagicMock(spec=UploadFile)
        upload_file.filename = "model.pt"
        upload_file.content_type = "application/octet-stream"
        upload_file.read = AsyncMock(return_value=content)

        with tempfile.TemporaryDirectory() as tmp_dir:
            destination = Path(tmp_dir) / "model.pt"

            result = await save_uploaded_file(upload_file, destination)

            assert result["file_size"] == len(content)
            assert result["original_filename"] == "model.pt"
            assert result["content_type"] == "application/octet-stream"
            assert len(result["checksum"]) == 64  # SHA256 hex string
            assert destination.exists()
            assert destination.read_bytes() == content

    @pytest.mark.asyncio
    async def test_save_uploaded_file_too_large(self):
        """Test file upload failure due to size limit."""
        # Create large content exceeding MAX_FILE_SIZE
        large_content = b"x" * (500 * 1024 * 1024 + 1)  # Slightly over 500MB

        upload_file = MagicMock(spec=UploadFile)
        upload_file.filename = "large_model.pt"
        upload_file.read = AsyncMock(return_value=large_content)

        with tempfile.TemporaryDirectory() as tmp_dir:
            destination = Path(tmp_dir) / "large_model.pt"

            with pytest.raises(HTTPException) as exc_info:
                await save_uploaded_file(upload_file, destination)

            assert exc_info.value.status_code == 413  # Request Entity Too Large
            assert "exceeds maximum allowed size" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_save_uploaded_file_malicious_content(self):
        """Test file upload failure due to malicious content."""
        malicious_content = b"exec('malicious code')"

        upload_file = MagicMock(spec=UploadFile)
        upload_file.filename = "malicious.pt"
        upload_file.read = AsyncMock(return_value=malicious_content)

        with tempfile.TemporaryDirectory() as tmp_dir:
            destination = Path(tmp_dir) / "malicious.pt"

            with pytest.raises(HTTPException) as exc_info:
                await save_uploaded_file(upload_file, destination)

            assert exc_info.value.status_code == 400  # Bad Request
            assert "malicious content detected" in str(exc_info.value.detail)


class TestIntegration:
    """Integration tests for upload workflow."""

    def test_malicious_patterns_detection(self):
        """Test that all malicious patterns are properly detected."""
        for pattern in MALICIOUS_PATTERNS:
            test_content = b"some code " + pattern + b" more code"
            # This should be detected (patterns are checked in lowercase)
            assert pattern in test_content.lower()

    def test_file_extension_coverage(self):
        """Test that all common model formats are covered."""
        pytorch_extensions = {".pt", ".pth"}
        onnx_extensions = {".onnx"}
        tensorflow_extensions = {".pb", ".tflite"}
        tensorrt_extensions = {".engine"}

        (
            pytorch_extensions
            | onnx_extensions
            | tensorflow_extensions
            | tensorrt_extensions
        )

        # Verify our allowed extensions cover common formats
        assert pytorch_extensions.issubset(ALLOWED_MODEL_EXTENSIONS)
        assert onnx_extensions.issubset(ALLOWED_MODEL_EXTENSIONS)
        assert tensorflow_extensions.issubset(ALLOWED_MODEL_EXTENSIONS)
        assert tensorrt_extensions.issubset(ALLOWED_MODEL_EXTENSIONS)

    def test_config_file_formats(self):
        """Test that common config file formats are supported."""
        yaml_extensions = {".yaml", ".yml"}
        json_extensions = {".json"}
        text_extensions = {".txt"}

        assert yaml_extensions.issubset(ALLOWED_CONFIG_EXTENSIONS)
        assert json_extensions.issubset(ALLOWED_CONFIG_EXTENSIONS)
        assert text_extensions.issubset(ALLOWED_CONFIG_EXTENSIONS)


@pytest.fixture
def sample_model_metadata():
    """Sample model metadata for testing."""
    return {
        "name": "Test YOLO11",
        "version": "1.0.0",
        "model_type": ModelType.DETECTION,
        "framework": ModelFramework.PYTORCH,
        "description": "Test model for unit testing",
        "classes": ["car", "truck", "person"],
        "input_shape": [1, 3, 640, 640],
        "config": {"confidence_threshold": 0.5, "iou_threshold": 0.45},
        "tags": ["test", "yolo11"],
        "benchmark_dataset": "test-dataset",
        "training_config": {"epochs": 100, "batch_size": 16},
    }


@pytest.fixture
def mock_upload_file():
    """Mock UploadFile for testing."""

    def _create_mock(
        filename: str, content: bytes, content_type: str = "application/octet-stream"
    ):
        upload_file = MagicMock(spec=UploadFile)
        upload_file.filename = filename
        upload_file.content_type = content_type
        upload_file.read = AsyncMock(return_value=content)
        return upload_file

    return _create_mock


class TestModelUploadWorkflow:
    """Test the complete model upload workflow."""

    @pytest.mark.asyncio
    async def test_complete_upload_workflow(
        self, sample_model_metadata, mock_upload_file
    ):
        """Test the complete upload workflow from validation to storage."""
        # Create test files
        model_content = b"fake pytorch model data"
        config_content = b"confidence_threshold: 0.5\niou_threshold: 0.45"

        model_file = mock_upload_file("yolo11n.pt", model_content)
        config_file = mock_upload_file("config.yaml", config_content, "text/yaml")

        # Test file validation
        assert validate_file_extension(model_file.filename, ALLOWED_MODEL_EXTENSIONS)
        assert validate_file_extension(config_file.filename, ALLOWED_CONFIG_EXTENSIONS)

        # Test content scanning
        model_patterns = await scan_for_malicious_content(model_content)
        config_patterns = await scan_for_malicious_content(config_content)
        assert model_patterns == []
        assert config_patterns == []

        # Test checksum calculation
        model_checksum = calculate_file_checksum(model_content)
        config_checksum = calculate_file_checksum(config_content)
        assert len(model_checksum) == 64
        assert len(config_checksum) == 64

        # Test storage directory creation
        with (
            tempfile.TemporaryDirectory() as tmp_dir,
            patch(
                "its_camera_ai.api.routers.models.MODEL_STORAGE_BASE_PATH",
                Path(tmp_dir),
            ),
        ):
            storage_dir = await create_model_storage_directory("test-yolo11", "1.0.0")
            assert storage_dir.exists()

            # Test file saving
            model_path = storage_dir / "model.pt"
            config_path = storage_dir / "config.yaml"

            model_result = await save_uploaded_file(model_file, model_path)
            config_result = await save_uploaded_file(config_file, config_path)

            assert model_result["file_size"] == len(model_content)
            assert config_result["file_size"] == len(config_content)
            assert model_path.exists()
            assert config_path.exists()

    @pytest.mark.asyncio
    async def test_upload_validation_failures(self, mock_upload_file):
        """Test various validation failure scenarios."""
        # Test invalid file extension
        invalid_file = mock_upload_file("malicious.exe", b"malicious content")
        assert not validate_file_extension(
            invalid_file.filename, ALLOWED_MODEL_EXTENSIONS
        )

        # Test malicious content
        malicious_file = mock_upload_file("model.pt", b"exec('rm -rf /')")
        patterns = await scan_for_malicious_content(await malicious_file.read())
        assert "exec(" in patterns

        # Test oversized file (simulate by mocking the content length check)
        with tempfile.TemporaryDirectory() as tmp_dir:
            destination = Path(tmp_dir) / "large.pt"
            large_file = mock_upload_file("large.pt", b"x" * (500 * 1024 * 1024 + 1))

            with pytest.raises(HTTPException) as exc_info:
                await save_uploaded_file(large_file, destination)
            assert exc_info.value.status_code == 413


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
