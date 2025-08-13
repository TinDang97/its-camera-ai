"""Basic tests for dependency injection containers.

Simple tests that verify container structure without requiring
all services to be fully importable.
"""

import pytest
from unittest.mock import Mock

from src.its_camera_ai.core.config import Settings


@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
    return {
        "database": {
            "url": "postgresql+asyncpg://test:test@localhost/test",
            "echo": False,
            "pool_size": 5,
            "max_overflow": 10,
        },
        "redis": {
            "url": "redis://localhost:6379/0",
            "max_connections": 10,
            "timeout": 5,
            "retry_on_timeout": True,
        },
        "security": {
            "secret_key": "test-secret-key-for-testing-only",
            "algorithm": "HS256",
            "access_token_expire_minutes": 30,
        },
    }


def test_container_import():
    """Test that container modules can be imported."""
    try:
        from src.its_camera_ai.containers import ApplicationContainer

        assert ApplicationContainer is not None
    except ImportError as e:
        pytest.fail(f"Failed to import container: {e}")


def test_container_structure():
    """Test basic container structure."""
    from src.its_camera_ai.containers import ApplicationContainer

    container = ApplicationContainer()

    # Test that sub-containers exist
    assert hasattr(container, "infrastructure")
    assert hasattr(container, "repositories")
    assert hasattr(container, "services")
    assert hasattr(container, "config")


def test_infrastructure_container_providers():
    """Test infrastructure container has required providers."""
    from src.its_camera_ai.containers import InfrastructureContainer

    container = InfrastructureContainer()

    # Test required providers exist
    assert hasattr(container, "database_engine")
    assert hasattr(container, "session_factory")
    assert hasattr(container, "database_session")
    assert hasattr(container, "redis_client")


def test_repository_container_providers():
    """Test repository container has required providers."""
    from src.its_camera_ai.containers import RepositoryContainer

    container = RepositoryContainer()

    # Test required providers exist
    assert hasattr(container, "user_repository")
    assert hasattr(container, "camera_repository")
    assert hasattr(container, "frame_repository")
    assert hasattr(container, "detection_repository")
    assert hasattr(container, "metrics_repository")


def test_service_container_providers():
    """Test service container has required providers."""
    from src.its_camera_ai.containers import ServiceContainer

    container = ServiceContainer()

    # Test core service providers exist
    assert hasattr(container, "auth_service")
    assert hasattr(container, "camera_service")
    assert hasattr(container, "frame_service")
    assert hasattr(container, "detection_service")
    assert hasattr(container, "metrics_service")


def test_container_configuration(mock_settings):
    """Test container configuration."""
    from src.its_camera_ai.containers import ApplicationContainer

    container = ApplicationContainer()

    # Should not raise an exception
    container.config.from_dict(mock_settings)

    # Test configuration is accessible
    assert container.config.database.url() == mock_settings["database"]["url"]
    assert container.config.redis.url() == mock_settings["redis"]["url"]


def test_base_repository():
    """Test base repository can be imported and instantiated."""
    from src.its_camera_ai.repositories.base_repository import BaseRepository
    from src.its_camera_ai.models.base import BaseTableModel
    from unittest.mock import Mock

    # Mock session factory
    mock_session_factory = Mock()
    mock_model = Mock(spec=BaseTableModel)

    # Should not raise an exception
    repo = BaseRepository(mock_session_factory, mock_model)
    assert repo.session_factory == mock_session_factory
    assert repo.model == mock_model


def test_repository_imports():
    """Test that repositories can be imported."""
    repositories = [
        "src.its_camera_ai.repositories.user_repository.UserRepository",
        "src.its_camera_ai.repositories.camera_repository.CameraRepository",
        "src.its_camera_ai.repositories.frame_repository.FrameRepository",
        "src.its_camera_ai.repositories.detection_repository.DetectionRepository",
        "src.its_camera_ai.repositories.metrics_repository.MetricsRepository",
    ]

    for repo_path in repositories:
        module_path, class_name = repo_path.rsplit(".", 1)
        try:
            import importlib

            module = importlib.import_module(module_path)
            repo_class = getattr(module, class_name)
            assert repo_class is not None
        except ImportError as e:
            pytest.fail(f"Failed to import {repo_path}: {e}")


def test_dependencies_v2_import():
    """Test that new dependencies module can be imported."""
    try:
        from its_camera_ai.api.dependencies import (
            get_cache_service,
            get_auth_service,
            get_camera_service,
        )

        assert get_cache_service is not None
        assert get_auth_service is not None
        assert get_camera_service is not None
    except ImportError as e:
        pytest.fail(f"Failed to import dependencies_v2: {e}")


def test_container_lifecycle_functions():
    """Test container lifecycle management functions."""
    try:
        from src.its_camera_ai.containers import (
            init_container,
            shutdown_container,
            wire_container,
            unwire_container,
        )

        assert init_container is not None
        assert shutdown_container is not None
        assert wire_container is not None
        assert unwire_container is not None
    except ImportError as e:
        pytest.fail(f"Failed to import lifecycle functions: {e}")


@pytest.mark.asyncio
async def test_container_lifecycle_basic():
    """Test basic container lifecycle without real connections."""
    from src.its_camera_ai.containers import shutdown_container

    # Shutdown should work even if nothing is initialized
    try:
        await shutdown_container()
    except Exception as e:
        pytest.fail(f"Container shutdown failed: {e}")


def test_app_di_import():
    """Test that DI app can be imported."""
    try:
        from its_camera_ai.api.app import create_app_with_di

        assert create_app_with_di is not None
    except ImportError as e:
        pytest.fail(f"Failed to import DI app: {e}")


def test_cli_di_import():
    """Test that DI CLI can be imported."""
    try:
        from src.its_camera_ai.cli.commands.auth_di import app

        assert app is not None
    except ImportError as e:
        pytest.fail(f"Failed to import DI CLI: {e}")


def test_service_di_example():
    """Test that DI service example can be imported."""
    try:
        from src.its_camera_ai.services.camera_service_di import CameraService

        assert CameraService is not None
    except ImportError as e:
        pytest.fail(f"Failed to import DI service: {e}")
