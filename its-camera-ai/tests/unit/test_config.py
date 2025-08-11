"""Unit tests for configuration management."""

import os
import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from its_camera_ai.core.config import Settings, get_settings_for_testing


class TestSettings:
    """Test configuration settings."""

    def test_default_settings(self):
        """Test default configuration values."""
        settings = Settings()

        assert settings.app_name == "ITS Camera AI"
        assert settings.environment == "development"
        assert settings.log_level == "INFO"
        assert settings.api_port == 8080
        assert not settings.debug

    def test_environment_validation(self):
        """Test environment validation."""
        # Valid environments
        for env in ["development", "testing", "staging", "production"]:
            settings = Settings(environment=env)
            assert settings.environment == env

        # Invalid environment
        with pytest.raises(ValidationError):
            Settings(environment="invalid")

    def test_log_level_validation(self):
        """Test log level validation."""
        # Valid log levels
        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            settings = Settings(log_level=level)
            assert settings.log_level == level

        # Case insensitive
        settings = Settings(log_level="debug")
        assert settings.log_level == "DEBUG"

        # Invalid log level
        with pytest.raises(ValidationError):
            Settings(log_level="INVALID")

    def test_database_url_generation(self):
        """Test database URL generation for sync/async drivers."""
        settings = Settings()

        # Async URL (default)
        async_url = settings.get_database_url(async_driver=True)
        assert "postgresql+asyncpg://" in async_url

        # Sync URL for migrations
        sync_url = settings.get_database_url(async_driver=False)
        assert "postgresql://" in sync_url
        assert "postgresql+asyncpg://" not in sync_url

    def test_environment_detection(self):
        """Test environment detection methods."""
        # Development
        dev_settings = Settings(environment="development")
        assert dev_settings.is_development()
        assert not dev_settings.is_production()

        # Production
        prod_settings = Settings(environment="production")
        assert prod_settings.is_production()
        assert not prod_settings.is_development()

    def test_directory_creation(self):
        """Test automatic directory creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            settings = Settings(
                data_dir=temp_path / "data",
                logs_dir=temp_path / "logs",
                temp_dir=temp_path / "temp",
            )
            settings.ml.model_path = temp_path / "models"

            # Directories shouldn't exist yet
            assert not (temp_path / "data").exists()
            assert not (temp_path / "logs").exists()
            assert not (temp_path / "temp").exists()
            assert not (temp_path / "models").exists()

            # Create directories
            settings.create_directories()

            # Directories should now exist
            assert (temp_path / "data").exists()
            assert (temp_path / "logs").exists()
            assert (temp_path / "temp").exists()
            assert (temp_path / "models").exists()

    def test_nested_config_from_env(self):
        """Test nested configuration from environment variables."""
        # Set environment variables
        env_vars = {
            "DATABASE__URL": "postgresql://test:test@localhost/test",
            "DATABASE__POOL_SIZE": "20",
            "REDIS__URL": "redis://localhost:6380/0",
            "ML__BATCH_SIZE": "64",
        }

        # Store original values
        original_env = {}
        for key, value in env_vars.items():
            original_env[key] = os.environ.get(key)
            os.environ[key] = value

        try:
            settings = Settings()

            # Check nested values were set correctly
            assert settings.database.url == "postgresql://test:test@localhost/test"
            assert settings.database.pool_size == 20
            assert settings.redis.url == "redis://localhost:6380/0"
            assert settings.ml.batch_size == 64

        finally:
            # Restore original environment
            for key, original_value in original_env.items():
                if original_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = original_value

    def test_security_config(self):
        """Test security configuration."""
        settings = Settings()

        # Check default values
        assert settings.security.algorithm == "HS256"
        assert settings.security.access_token_expire_minutes == 30
        assert settings.security.enable_cors
        assert settings.security.rate_limit_per_minute == 100

        # Check secret key validation
        with pytest.raises(ValidationError, match="at least 32 characters"):
            Settings(security={"secret_key": "short"})


class TestTestingSettings:
    """Test configuration for testing environment."""

    def test_testing_settings_override(self):
        """Test that testing settings properly override defaults."""
        settings = get_settings_for_testing(
            log_level="debug", database__url="postgresql://test:test@localhost/test_db"
        )

        assert settings.environment == "testing"
        assert settings.log_level == "DEBUG"
        assert "test_db" in settings.database.url
        assert settings.debug

    def test_environment_restoration(self):
        """Test that environment variables are restored after testing."""
        # Set a test environment variable
        original_value = os.environ.get("TEST_VAR")
        os.environ["TEST_VAR"] = "original"

        try:
            # Get testing settings (this modifies environment)
            get_settings_for_testing(test_var="modified")

            # Environment should be restored
            assert os.environ.get("TEST_VAR") == "original"

        finally:
            # Cleanup
            if original_value is None:
                os.environ.pop("TEST_VAR", None)
            else:
                os.environ["TEST_VAR"] = original_value
