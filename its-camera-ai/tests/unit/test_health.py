"""Unit tests for health check endpoints."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch


class TestHealthEndpoints:
    """Test health check endpoints."""
    
    def test_basic_health_check(self, test_client: TestClient):
        """Test basic health check endpoint."""
        response = test_client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
        assert "environment" in data
        assert "uptime_seconds" in data
        assert isinstance(data["uptime_seconds"], (int, float))
    
    def test_liveness_check(self, test_client: TestClient):
        """Test Kubernetes liveness probe endpoint."""
        response = test_client.get("/health/live")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "alive"
        assert "timestamp" in data
    
    @pytest.mark.asyncio
    async def test_readiness_check_healthy(self, test_client: TestClient):
        """Test readiness check when all services are healthy."""
        with patch("its_camera_ai.api.routers.health.get_db") as mock_get_db, \
             patch("its_camera_ai.api.routers.health.get_redis") as mock_get_redis:
            
            # Mock successful database connection
            mock_db = AsyncMock()
            mock_get_db.return_value.__aenter__.return_value = mock_db
            
            # Mock successful Redis connection
            mock_redis = AsyncMock()
            mock_get_redis.return_value = mock_redis
            
            response = test_client.get("/health/ready")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["ready"] is True
            assert "checks" in data
            assert "database" in data["checks"]
            assert "redis" in data["checks"]
    
    @pytest.mark.asyncio
    async def test_readiness_check_unhealthy(self, test_client: TestClient):
        """Test readiness check when services are unhealthy."""
        with patch("its_camera_ai.api.routers.health.get_db") as mock_get_db:
            
            # Mock database connection failure
            mock_get_db.side_effect = Exception("Database connection failed")
            
            response = test_client.get("/health/ready")
            
            assert response.status_code == 503
            data = response.json()
            
            assert "Database connection failed" in str(data)
    
    @pytest.mark.asyncio
    async def test_detailed_health_check(self, test_client: TestClient):
        """Test detailed health check endpoint."""
        with patch("its_camera_ai.api.routers.health._check_database_health") as mock_db, \
             patch("its_camera_ai.api.routers.health._check_redis_health") as mock_redis, \
             patch("its_camera_ai.api.routers.health._check_ml_models_health") as mock_ml, \
             patch("its_camera_ai.api.routers.health._check_disk_space") as mock_disk, \
             patch("its_camera_ai.api.routers.health._check_memory_usage") as mock_memory:
            
            # Mock all health checks as successful
            mock_db.return_value = {"status": "healthy", "message": "Connected"}
            mock_redis.return_value = {"status": "healthy", "message": "Connected"}
            mock_ml.return_value = {"status": "healthy", "message": "Models loaded"}
            mock_disk.return_value = {"status": "healthy", "usage_percent": 50.0}
            mock_memory.return_value = {"status": "healthy", "usage_percent": 60.0}
            
            response = test_client.get("/health/detailed")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["status"] == "healthy"
            assert "checks" in data
            assert len(data["checks"]) == 5  # All health checks
            assert all(
                check["status"] == "healthy" 
                for check in data["checks"].values()
            )
    
    @pytest.mark.asyncio
    async def test_detailed_health_check_degraded(self, test_client: TestClient):
        """Test detailed health check with some failures."""
        with patch("its_camera_ai.api.routers.health._check_database_health") as mock_db, \
             patch("its_camera_ai.api.routers.health._check_redis_health") as mock_redis, \
             patch("its_camera_ai.api.routers.health._check_ml_models_health") as mock_ml, \
             patch("its_camera_ai.api.routers.health._check_disk_space") as mock_disk, \
             patch("its_camera_ai.api.routers.health._check_memory_usage") as mock_memory:
            
            # Mock mixed health check results
            mock_db.return_value = {"status": "healthy", "message": "Connected"}
            mock_redis.return_value = {"status": "unhealthy", "message": "Connection failed"}
            mock_ml.return_value = {"status": "healthy", "message": "Models loaded"}
            mock_disk.return_value = {"status": "warning", "usage_percent": 85.0}
            mock_memory.return_value = {"status": "healthy", "usage_percent": 60.0}
            
            response = test_client.get("/health/detailed")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["status"] == "degraded"
            assert data["checks"]["redis"]["status"] == "unhealthy"
            assert data["checks"]["disk_space"]["status"] == "warning"


class TestHealthCheckHelpers:
    """Test health check helper functions."""
    
    @pytest.mark.asyncio
    async def test_check_disk_space(self):
        """Test disk space health check."""
        from its_camera_ai.api.routers.health import _check_disk_space
        
        result = await _check_disk_space()
        
        assert "status" in result
        assert "total_gb" in result
        assert "used_gb" in result
        assert "free_gb" in result
        assert "usage_percent" in result
        
        # Check status logic
        if result["usage_percent"] > 90:
            assert result["status"] == "critical"
        elif result["usage_percent"] > 80:
            assert result["status"] == "warning"
        else:
            assert result["status"] == "healthy"
    
    @pytest.mark.asyncio
    async def test_check_memory_usage(self):
        """Test memory usage health check."""
        from its_camera_ai.api.routers.health import _check_memory_usage
        
        result = await _check_memory_usage()
        
        assert "status" in result
        assert "total_gb" in result
        assert "used_gb" in result
        assert "available_gb" in result
        assert "usage_percent" in result
        
        # Check status logic
        if result["usage_percent"] > 90:
            assert result["status"] == "critical"
        elif result["usage_percent"] > 80:
            assert result["status"] == "warning"
        else:
            assert result["status"] == "healthy"
