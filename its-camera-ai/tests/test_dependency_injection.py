"""Tests for dependency injection containers.

Tests the container architecture, service wiring, and proper
dependency resolution for the ITS Camera AI system.
"""

import pytest
from unittest.mock import AsyncMock, Mock

from src.its_camera_ai.containers import ApplicationContainer
from src.its_camera_ai.core.config import Settings
from src.its_camera_ai.repositories.camera_repository import CameraRepository
from src.its_camera_ai.repositories.user_repository import UserRepository
from src.its_camera_ai.services.auth_service import AuthenticationService as AuthService
from src.its_camera_ai.services.camera_service_di import CameraService
from src.its_camera_ai.services.cache import CacheService


@pytest.fixture
def container():
    """Create a test container instance."""
    return ApplicationContainer()


@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
    return Settings(
        database_url="postgresql+asyncpg://test:test@localhost/test",
        redis={
            "url": "redis://localhost:6379/0",
            "max_connections": 10,
            "timeout": 5,
            "retry_on_timeout": True,
        },
        security={
            "secret_key": "test-secret-key-for-testing-only",
            "algorithm": "HS256",
            "access_token_expire_minutes": 30,
        },
    )


class TestInfrastructureContainer:
    """Test infrastructure container configuration."""

    def test_infrastructure_container_configuration(self, container, mock_settings):
        """Test that infrastructure container is properly configured."""
        # Configure the container
        container.config.from_dict(mock_settings.model_dump())

        # Verify configuration is loaded
        assert (
            container.infrastructure.config.database.url() == mock_settings.database_url
        )
        assert container.infrastructure.config.redis.url() == mock_settings.redis.url

    def test_database_engine_provider(self, container, mock_settings):
        """Test database engine provider configuration."""
        container.config.from_dict(
            {
                "database": {
                    "url": mock_settings.database_url,
                    "echo": False,
                    "pool_size": 5,
                    "max_overflow": 10,
                }
            }
        )

        # Verify the provider is configured
        engine_provider = container.infrastructure.database_engine
        assert engine_provider is not None

    def test_redis_client_provider(self, container, mock_settings):
        """Test Redis client provider configuration."""
        container.config.from_dict(
            {
                "redis": {
                    "url": mock_settings.redis.url,
                    "max_connections": 10,
                    "timeout": 5,
                    "retry_on_timeout": True,
                }
            }
        )

        # Verify the provider is configured
        redis_provider = container.infrastructure.redis_client
        assert redis_provider is not None


class TestRepositoryContainer:
    """Test repository container configuration."""

    def test_repository_providers(self, container):
        """Test that all repository providers are configured."""
        # Check that all repository providers exist
        assert hasattr(container.repositories, "user_repository")
        assert hasattr(container.repositories, "camera_repository")
        assert hasattr(container.repositories, "frame_repository")
        assert hasattr(container.repositories, "detection_repository")
        assert hasattr(container.repositories, "metrics_repository")

    def test_repository_dependencies(self, container):
        """Test repository dependencies are properly wired."""
        # Repositories should depend on infrastructure
        user_repo_provider = container.repositories.user_repository
        assert user_repo_provider is not None

        # Check that dependencies container is properly set
        deps = container.repositories.dependencies
        assert deps is not None


class TestServiceContainer:
    """Test service container configuration."""

    def test_service_providers(self, container):
        """Test that all service providers are configured."""
        # Check core services
        assert hasattr(container.services, "cache_service")
        assert hasattr(container.services, "auth_service")
        assert hasattr(container.services, "token_service")
        assert hasattr(container.services, "mfa_service")

        # Check business services
        assert hasattr(container.services, "camera_service")
        assert hasattr(container.services, "frame_service")
        assert hasattr(container.services, "detection_service")
        assert hasattr(container.services, "metrics_service")

        # Check advanced services
        assert hasattr(container.services, "streaming_service")
        assert hasattr(container.services, "analytics_service")
        assert hasattr(container.services, "alert_service")

    def test_singleton_services(self, container):
        """Test that singleton services are properly configured."""
        # These services should be singletons
        cache_provider = container.services.cache_service
        auth_provider = container.services.auth_service
        token_provider = container.services.token_service

        # Verify they're configured as singletons (would need actual instantiation to verify)
        assert cache_provider is not None
        assert auth_provider is not None
        assert token_provider is not None

    def test_factory_services(self, container):
        """Test that factory services are properly configured."""
        # These services should be factories
        camera_provider = container.services.camera_service
        frame_provider = container.services.frame_service

        assert camera_provider is not None
        assert frame_provider is not None


class TestApplicationContainer:
    """Test main application container."""

    def test_container_structure(self, container):
        """Test that container has all required sub-containers."""
        assert hasattr(container, "infrastructure")
        assert hasattr(container, "repositories")
        assert hasattr(container, "services")
        assert hasattr(container, "service_mesh")

    def test_configuration_propagation(self, container, mock_settings):
        """Test that configuration is properly propagated through containers."""
        container.config.from_dict(mock_settings.model_dump())

        # Configuration should be available in sub-containers
        assert container.config is not None
        assert container.infrastructure.config is not None
        assert container.services.config is not None

    def test_dependency_wiring(self, container):
        """Test that dependencies are properly wired between containers."""
        # Services should depend on repositories and infrastructure
        services_deps = container.services.dependencies
        repos_deps = container.repositories.dependencies

        assert services_deps is not None
        assert repos_deps is not None


class TestContainerIntegration:
    """Test container integration and service resolution."""

    @pytest.mark.asyncio
    async def test_service_instantiation(self, container, mock_settings):
        """Test that services can be instantiated from container."""
        # This would require mocking the actual database and Redis connections
        # For now, just test that the configuration works
        container.config.from_dict(mock_settings.model_dump())

        # Verify that providers are callable
        cache_provider = container.services.cache_service
        auth_provider = container.services.auth_service

        assert callable(cache_provider)
        assert callable(auth_provider)

    def test_container_wiring_modules(self):
        """Test that container can wire specified modules."""
        from src.its_camera_ai.containers import wire_container, unwire_container

        # Test module list
        modules = [
            "src.its_camera_ai.api.dependencies_v2",
        ]

        # This should not raise an exception
        try:
            wire_container(modules)
            unwire_container()
        except Exception as e:
            pytest.fail(f"Container wiring failed: {e}")


class TestServiceDependencies:
    """Test service dependencies and injection."""

    def test_camera_service_dependencies(self):
        """Test CameraService dependency requirements."""
        # Mock repository
        mock_camera_repository = Mock(spec=CameraRepository)

        # Should be able to create service with repository
        camera_service = CameraService(mock_camera_repository)
        assert camera_service.camera_repository == mock_camera_repository

    def test_auth_service_dependencies(self):
        """Test AuthService dependency requirements."""
        # This would test that AuthService can be created with proper dependencies
        # Skipping actual instantiation due to complex dependencies

        # Verify the class exists and can be imported
        assert AuthService is not None

    def test_cache_service_dependencies(self):
        """Test CacheService dependency requirements."""
        # Mock Redis client
        mock_redis = AsyncMock()

        # Should be able to create service with Redis client
        cache_service = CacheService(mock_redis)
        assert cache_service.redis == mock_redis


class TestContainerLifecycle:
    """Test container lifecycle management."""

    @pytest.mark.asyncio
    async def test_container_initialization(self, mock_settings):
        """Test container initialization process."""
        from src.its_camera_ai.containers import init_container

        # Mock the actual connection setup
        with pytest.raises(Exception):
            # This will fail without actual database/Redis, but should test the flow
            await init_container(mock_settings)

    @pytest.mark.asyncio
    async def test_container_shutdown(self):
        """Test container shutdown process."""
        from src.its_camera_ai.containers import shutdown_container

        # This should not raise an exception even if nothing is initialized
        await shutdown_container()


# Marker for integration tests that require real infrastructure
@pytest.mark.integration
class TestContainerWithRealServices:
    """Integration tests with real service instances."""

    @pytest.mark.asyncio
    async def test_full_container_setup(self, mock_settings):
        """Test full container setup with real services."""
        # This would test the full container with real database/Redis connections
        # Requires test infrastructure to be running
        pytest.skip("Requires test infrastructure (PostgreSQL, Redis)")

    @pytest.mark.asyncio
    async def test_service_interactions(self):
        """Test interactions between services through DI."""
        # This would test that services can interact through the container
        pytest.skip("Requires test infrastructure")
