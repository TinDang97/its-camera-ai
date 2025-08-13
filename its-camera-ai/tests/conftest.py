"""Pytest configuration and shared fixtures.

Provides common test fixtures for database, Redis, authentication,
and FastAPI test client.
"""

import asyncio
from collections.abc import AsyncGenerator, Generator
from datetime import datetime, timedelta
from uuid import uuid4

import bcrypt
import pytest
import pytest_asyncio
import redis.asyncio as redis
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from its_camera_ai.api.app import create_app
from its_camera_ai.core.config import get_settings_for_testing
from its_camera_ai.models.database import create_database_engine
from its_camera_ai.models.user import Role, User
from its_camera_ai.services.auth_service import (
    AuthenticationService,
    JWTManager,
    SecurityConfig,
    UserCredentials,
)
from its_camera_ai.services.base_service import BaseAsyncService


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


# User service mock for testing
class MockUserService(BaseAsyncService[User]):
    """Mock user service for testing."""

    def __init__(self, session: AsyncSession):
        super().__init__(session, User)
        self._users = {}

    async def get_by_username(self, username: str) -> User | None:
        """Get user by username."""
        for user in self._users.values():
            if user.username == username:
                return user
        return None

    async def get_by_id(self, user_id: str) -> User | None:
        """Get user by ID."""
        return self._users.get(user_id)

    async def create_test_user(self, **kwargs) -> User:
        """Create a test user."""
        user = User(**kwargs)
        self._users[user.id] = user
        return user


@pytest_asyncio.fixture
async def test_user(db_session: AsyncSession) -> User:
    """Create a test user for authentication tests."""
    return User(
        id=str(uuid4()),
        username="testuser",
        email="test@example.com",
        hashed_password=bcrypt.hashpw(b"password123", bcrypt.gensalt()).decode(),
        full_name="Test User",
        is_active=True,
        is_verified=True,
        mfa_enabled=False,
        roles=[],
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )


@pytest_asyncio.fixture
async def test_admin_user(db_session: AsyncSession) -> User:
    """Create a test admin user for authentication tests."""
    admin_role = Role(
        id=str(uuid4()),
        name="admin",
        description="Administrator role",
        permissions=[],
    )

    return User(
        id=str(uuid4()),
        username="adminuser",
        email="admin@example.com",
        hashed_password=bcrypt.hashpw(b"adminpass123", bcrypt.gensalt()).decode(),
        full_name="Admin User",
        is_active=True,
        is_verified=True,
        is_superuser=True,
        mfa_enabled=False,
        roles=[admin_role],
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )


@pytest_asyncio.fixture
async def mock_user_service(db_session: AsyncSession) -> MockUserService:
    """Mock user service for testing."""
    return MockUserService(db_session)


@pytest_asyncio.fixture
async def auth_service(db_session: AsyncSession, redis_client: redis.Redis, test_settings) -> AuthenticationService:
    """Authentication service for testing."""
    security_config = SecurityConfig()
    service = AuthenticationService(db_session, redis_client, security_config)

    # Mock the user service dependency
    service.user_service = MockUserService(db_session)

    return service


@pytest_asyncio.fixture
async def jwt_manager(test_settings) -> JWTManager:
    """JWT manager for testing."""
    security_config = SecurityConfig()
    return JWTManager(security_config)


@pytest_asyncio.fixture
async def auth_headers(auth_service: AuthenticationService, test_user: User) -> dict[str, str]:
    """Authentication headers for testing."""
    # Create test user in the mock service
    await auth_service.user_service.create_test_user(
        id=test_user.id,
        username=test_user.username,
        email=test_user.email,
        hashed_password=test_user.hashed_password,
        full_name=test_user.full_name,
        is_active=test_user.is_active,
        is_verified=test_user.is_verified,
        mfa_enabled=test_user.mfa_enabled,
        roles=test_user.roles,
        created_at=test_user.created_at,
        updated_at=test_user.updated_at,
    )

    # Authenticate and get tokens
    credentials = UserCredentials(
        username="testuser",
        password="password123",
        ip_address="127.0.0.1"
    )

    # Mock the authentication flow to return a valid token
    token_data = {
        "sub": test_user.id,
        "username": test_user.username,
        "session_id": str(uuid4()),
        "roles": [],
        "permissions": [],
    }

    access_token = auth_service.jwt_manager.create_access_token(token_data)

    return {"Authorization": f"Bearer {access_token}"}


@pytest_asyncio.fixture
async def admin_auth_headers(auth_service: AuthenticationService, test_admin_user: User) -> dict[str, str]:
    """Admin authentication headers for testing."""
    # Create admin user in the mock service
    await auth_service.user_service.create_test_user(
        id=test_admin_user.id,
        username=test_admin_user.username,
        email=test_admin_user.email,
        hashed_password=test_admin_user.hashed_password,
        full_name=test_admin_user.full_name,
        is_active=test_admin_user.is_active,
        is_verified=test_admin_user.is_verified,
        is_superuser=test_admin_user.is_superuser,
        mfa_enabled=test_admin_user.mfa_enabled,
        roles=test_admin_user.roles,
        created_at=test_admin_user.created_at,
        updated_at=test_admin_user.updated_at,
    )

    # Create admin token
    token_data = {
        "sub": test_admin_user.id,
        "username": test_admin_user.username,
        "session_id": str(uuid4()),
        "roles": ["admin"],
        "permissions": ["*"],  # All permissions for admin
    }

    access_token = auth_service.jwt_manager.create_access_token(token_data)

    return {"Authorization": f"Bearer {access_token}"}


# Test data fixtures
@pytest.fixture
def sample_user_data() -> dict:
    """Sample user data for testing."""
    return {
        "username": "testuser",
        "email": "test@example.com",
        "full_name": "Test User",
        "hashed_password": bcrypt.hashpw(b"password123", bcrypt.gensalt()).decode(),
        "is_active": True,
        "is_verified": True,
        "mfa_enabled": False,
    }


@pytest.fixture
def mfa_user_data() -> dict:
    """Sample MFA-enabled user data for testing."""
    return {
        "username": "mfauser",
        "email": "mfa@example.com",
        "full_name": "MFA User",
        "hashed_password": bcrypt.hashpw(b"mfapass123", bcrypt.gensalt()).decode(),
        "is_active": True,
        "is_verified": True,
        "mfa_enabled": True,
        "mfa_secret": "JBSWY3DPEHPK3PXP",
    }


@pytest.fixture
def test_credentials() -> dict:
    """Test authentication credentials."""
    return {
        "username": "testuser",
        "password": "password123",
        "ip_address": "127.0.0.1",
    }


@pytest.fixture
def invalid_credentials() -> dict:
    """Invalid authentication credentials for testing failures."""
    return {
        "username": "testuser",
        "password": "wrongpassword",
        "ip_address": "127.0.0.1",
    }


@pytest_asyncio.fixture
async def expired_session_data() -> dict:
    """Expired session data for testing."""
    return {
        "session_id": str(uuid4()),
        "user_id": str(uuid4()),
        "username": "testuser",
        "roles": "viewer",
        "permissions": "cameras:read",
        "created_at": (datetime.utcnow() - timedelta(hours=10)).isoformat(),
        "last_activity": (datetime.utcnow() - timedelta(hours=2)).isoformat(),
        "expires_at": (datetime.utcnow() - timedelta(hours=1)).isoformat(),  # Expired
        "mfa_verified": "False",
    }


@pytest_asyncio.fixture
async def valid_session_data() -> dict:
    """Valid session data for testing."""
    return {
        "session_id": str(uuid4()),
        "user_id": str(uuid4()),
        "username": "testuser",
        "roles": "viewer",
        "permissions": "cameras:read",
        "created_at": datetime.utcnow().isoformat(),
        "last_activity": datetime.utcnow().isoformat(),
        "expires_at": (datetime.utcnow() + timedelta(hours=8)).isoformat(),
        "mfa_verified": "False",
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


# Performance testing fixtures
@pytest.fixture
def concurrent_users_count() -> int:
    """Number of concurrent users for performance testing."""
    return 10


@pytest.fixture
def stress_test_duration() -> int:
    """Duration in seconds for stress testing."""
    return 30


# Mock fixtures for external dependencies
@pytest_asyncio.fixture
async def mock_redis_with_data(redis_client: redis.Redis, valid_session_data: dict) -> redis.Redis:
    """Redis client with pre-populated test data."""
    # Pre-populate with valid session data
    session_key = f"session:{valid_session_data['session_id']}"
    await redis_client.hset(session_key, mapping=valid_session_data)
    await redis_client.expire(session_key, 28800)  # 8 hours

    # Add to user sessions set
    user_sessions_key = f"user_sessions:{valid_session_data['user_id']}"
    await redis_client.sadd(user_sessions_key, valid_session_data['session_id'])

    return redis_client


@pytest_asyncio.fixture
async def clean_redis(redis_client: redis.Redis) -> redis.Redis:
    """Ensure Redis is clean before each test."""
    await redis_client.flushdb()
    return redis_client


# Stress testing fixtures
@pytest_asyncio.fixture
async def stress_test_users(db_session: AsyncSession, concurrent_users_count: int) -> list[User]:
    """Create multiple test users for stress testing."""
    users = []
    for i in range(concurrent_users_count):
        user = User(
            id=str(uuid4()),
            username=f"stressuser_{i}",
            email=f"stress{i}@example.com",
            hashed_password=bcrypt.hashpw(f"stresspass{i}".encode(), bcrypt.gensalt()).decode(),
            full_name=f"Stress User {i}",
            is_active=True,
            is_verified=True,
            mfa_enabled=False,
            roles=[],
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        users.append(user)
    return users


@pytest.fixture
def performance_thresholds() -> dict[str, float]:
    """Performance thresholds for authentication tests."""
    return {
        "auth_max_time": 1.0,  # Max 1 second per authentication
        "token_verify_max_time": 0.1,  # Max 100ms per token verification
        "concurrent_success_rate": 0.95,  # At least 95% success rate
        "memory_growth_mb": 50,  # Max 50MB memory growth during stress test
    }


# Test utilities
class AuthTestHelper:
    """Helper class for authentication testing."""

    @staticmethod
    def create_test_token_data(user_id: str, username: str, roles: list[str] = None, permissions: list[str] = None) -> dict:
        """Create test token data."""
        return {
            "sub": user_id,
            "username": username,
            "session_id": str(uuid4()),
            "roles": roles or [],
            "permissions": permissions or [],
        }

    @staticmethod
    def create_session_info(user_id: str, username: str, session_id: str = None, expired: bool = False):
        """Create SessionInfo for testing."""
        from its_camera_ai.services.auth_service import SessionInfo

        if expired:
            created_at = datetime.utcnow() - timedelta(hours=10)
            expires_at = datetime.utcnow() - timedelta(hours=1)
        else:
            created_at = datetime.utcnow()
            expires_at = datetime.utcnow() + timedelta(hours=8)

        return SessionInfo(
            session_id=session_id or str(uuid4()),
            user_id=user_id,
            username=username,
            roles=[],
            permissions=[],
            created_at=created_at,
            last_activity=datetime.utcnow(),
            expires_at=expires_at,
            mfa_verified=False,
        )


@pytest.fixture
def auth_test_helper() -> AuthTestHelper:
    """Authentication test helper."""
    return AuthTestHelper()


# Test cleanup and lifecycle hooks
@pytest_asyncio.fixture(autouse=True, scope="function")
async def cleanup_test_data(redis_client: redis.Redis, db_session: AsyncSession):
    """Automatically clean up test data after each test."""
    yield  # Run the test

    # Cleanup Redis
    await redis_client.flushdb()

    # Cleanup database session (rollback any changes)
    await db_session.rollback()


# Mock factories for complex test scenarios
class TestDataFactory:
    """Factory for creating test data."""

    @staticmethod
    def create_bulk_users(count: int, base_username: str = "bulkuser") -> list[dict]:
        """Create bulk user data for testing."""
        users = []
        for i in range(count):
            users.append({
                "id": str(uuid4()),
                "username": f"{base_username}_{i}",
                "email": f"{base_username}{i}@example.com",
                "hashed_password": bcrypt.hashpw(f"password{i}".encode(), bcrypt.gensalt()).decode(),
                "full_name": f"Bulk User {i}",
                "is_active": True,
                "is_verified": True,
                "mfa_enabled": i % 5 == 0,  # Every 5th user has MFA
                "roles": [],
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
            })
        return users

    @staticmethod
    def create_session_scenarios() -> dict[str, dict]:
        """Create various session scenarios for testing."""
        base_time = datetime.utcnow()
        return {
            "valid_session": {
                "session_id": str(uuid4()),
                "user_id": str(uuid4()),
                "username": "validuser",
                "roles": "viewer",
                "permissions": "cameras:read",
                "created_at": base_time.isoformat(),
                "last_activity": base_time.isoformat(),
                "expires_at": (base_time + timedelta(hours=8)).isoformat(),
                "mfa_verified": "True",
            },
            "expired_session": {
                "session_id": str(uuid4()),
                "user_id": str(uuid4()),
                "username": "expireduser",
                "roles": "viewer",
                "permissions": "cameras:read",
                "created_at": (base_time - timedelta(hours=10)).isoformat(),
                "last_activity": (base_time - timedelta(hours=2)).isoformat(),
                "expires_at": (base_time - timedelta(hours=1)).isoformat(),
                "mfa_verified": "False",
            },
            "soon_to_expire_session": {
                "session_id": str(uuid4()),
                "user_id": str(uuid4()),
                "username": "soonexpireduser",
                "roles": "admin",
                "permissions": "*",
                "created_at": (base_time - timedelta(hours=7, minutes=55)).isoformat(),
                "last_activity": (base_time - timedelta(minutes=5)).isoformat(),
                "expires_at": (base_time + timedelta(minutes=5)).isoformat(),
                "mfa_verified": "True",
            },
        }


@pytest.fixture
def test_data_factory() -> TestDataFactory:
    """Test data factory instance."""
    return TestDataFactory()


# Test markers are defined in pyproject.toml
# This ensures consistent marker definitions across the project

def pytest_configure(config):
    """Configure pytest markers and settings."""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )
    config.addinivalue_line(
        "markers", "ml: marks tests for ML models"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow"
    )
    config.addinivalue_line(
        "markers", "e2e: marks tests as end-to-end tests"
    )
    config.addinivalue_line(
        "markers", "benchmark: marks performance benchmark tests"
    )

def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="Run integration tests"
    )
    parser.addoption(
        "--performance",
        action="store_true",
        default=False,
        help="Run performance tests"
    )
    parser.addoption(
        "--auth-only",
        action="store_true",
        default=False,
        help="Run authentication tests only"
    )
    parser.addoption(
        "--concurrent-users",
        action="store",
        default="10",
        type=int,
        help="Number of concurrent users for performance testing"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on command line options."""
    if config.getoption("--auth-only"):
        # Only run authentication-related tests
        selected_items = []
        for item in items:
            if "auth" in item.nodeid.lower() or "test_auth" in item.name:
                selected_items.append(item)
        items[:] = selected_items

    # Add markers based on test patterns
    for item in items:
        if "concurrent" in item.name or "performance" in item.name:
            item.add_marker(pytest.mark.performance)
        if "integration" in item.name or "flow" in item.name:
            item.add_marker(pytest.mark.integration)
        if "auth" in item.nodeid.lower():
            item.add_marker(pytest.mark.unit)
