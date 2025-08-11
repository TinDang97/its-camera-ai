"""Pytest configuration and shared fixtures.

Provides common test fixtures for database, Redis, authentication,
and FastAPI test client.
"""

import asyncio
from typing import AsyncGenerator, Generator

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession
import redis.asyncio as redis

from its_camera_ai.api.app import create_app
from its_camera_ai.core.config import get_settings_for_testing
from its_camera_ai.models.database import create_database_engine


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_settings():
    """Test application settings."""
    return get_settings_for_testing()


@pytest_asyncio.fixture(scope="session")
async def test_db_manager(test_settings):
    """Database manager for tests."""
    db_manager = await create_database_engine(test_settings)
    
    # Create tables
    await db_manager.create_tables()
    
    yield db_manager
    
    # Cleanup
    await db_manager.close()


@pytest_asyncio.fixture
async def db_session(test_db_manager) -> AsyncGenerator[AsyncSession, None]:
    """Database session for tests."""
    async with test_db_manager.get_session() as session:
        yield session
        await session.rollback()


@pytest_asyncio.fixture
async def redis_client(test_settings) -> AsyncGenerator[redis.Redis, None]:
    """Redis client for tests."""
    client = redis.from_url(test_settings.redis.url)
    
    # Clear test database
    await client.flushdb()
    
    yield client
    
    # Cleanup
    await client.flushdb()
    await client.close()


@pytest.fixture
def test_app(test_settings):
    """FastAPI test application."""
    return create_app(test_settings)


@pytest.fixture
def test_client(test_app) -> TestClient:
    """FastAPI test client."""
    with TestClient(test_app) as client:
        yield client


@pytest_asyncio.fixture
async def auth_headers(test_client) -> dict[str, str]:
    """Authentication headers for testing."""
    # TODO: Implement test user creation and authentication
    return {"Authorization": "Bearer test-token"}


# Test data fixtures
@pytest.fixture
def sample_user_data() -> dict:
    """Sample user data for testing."""
    return {
        "username": "testuser",
        "email": "test@example.com",
        "full_name": "Test User",
        "hashed_password": "hashed_password",
    }


@pytest.fixture
def sample_camera_data() -> dict:
    """Sample camera data for testing."""
    return {
        "name": "Test Camera",
        "location": "Test Location",
        "stream_url": "rtsp://test-camera:554/stream",
        "is_active": True,
    }


# Markers for different test types
pytest.main.add_option(
    "--integration",
    action="store_true",
    help="Run integration tests"
)

pytest.main.add_option(
    "--performance",
    action="store_true",
    help="Run performance tests"
)
