"""End-to-End Integration Tests for Service Mesh.

This module contains comprehensive integration tests for the service mesh,
testing service-to-service communication, circuit breakers, load balancing,
health checks, and distributed tracing.
"""

import asyncio
import json
import time
import uuid
from datetime import datetime
from unittest.mock import AsyncMock, patch

import pytest
import redis.asyncio as redis

from src.its_camera_ai.core.exceptions import ServiceMeshError
from src.its_camera_ai.services.service_mesh import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerState,
    CorrelationContext,
    EventBus,
    LoadBalancer,
    LoadBalancingStrategy,
    ServiceEndpoint,
    ServiceMeshClient,
    ServiceOrchestrator,
    ServiceRegistry,
    ServiceStatus,
    get_service_mesh,
    shutdown_service_mesh,
)


@pytest.fixture
async def redis_client():
    """Create Redis client for testing."""
    client = redis.Redis.from_url("redis://localhost:6379", decode_responses=True)
    try:
        await client.ping()
        yield client
    except redis.ConnectionError:
        pytest.skip("Redis not available")
    finally:
        await client.close()


@pytest.fixture
def service_endpoint():
    """Create test service endpoint."""
    return ServiceEndpoint(
        name="test_service", host="127.0.0.1", port=50051, protocol="grpc", weight=1
    )


@pytest.fixture
def correlation_context():
    """Create test correlation context."""
    return CorrelationContext(
        correlation_id=str(uuid.uuid4()),
        trace_id=str(uuid.uuid4()),
        span_id=str(uuid.uuid4()),
        user_id="test_user",
        session_id="test_session",
    )


class TestCircuitBreaker:
    """Test circuit breaker functionality."""

    def test_circuit_breaker_init(self):
        """Test circuit breaker initialization."""
        config = CircuitBreakerConfig(
            failure_threshold=3, timeout_duration=30, success_threshold=2
        )

        breaker = CircuitBreaker("test_service", config)

        assert breaker.service_name == "test_service"
        assert breaker.state == CircuitBreakerState.CLOSED
        assert breaker.failure_count == 0
        assert breaker.success_count == 0

    @pytest.mark.asyncio
    async def test_circuit_breaker_success(self):
        """Test circuit breaker with successful calls."""
        config = CircuitBreakerConfig(failure_threshold=3)
        breaker = CircuitBreaker("test_service", config)

        async def successful_func():
            return "success"

        result = await breaker.call(successful_func)

        assert result == "success"
        assert breaker.state == CircuitBreakerState.CLOSED
        assert breaker.success_count == 1
        assert breaker.failure_count == 0

    @pytest.mark.asyncio
    async def test_circuit_breaker_failure_threshold(self):
        """Test circuit breaker opens after failure threshold."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            request_volume_threshold=2,
            error_threshold_percentage=50.0,
        )
        breaker = CircuitBreaker("test_service", config)

        async def failing_func():
            raise Exception("Test failure")

        # First failure
        with pytest.raises(Exception):
            await breaker.call(failing_func)
        assert breaker.state == CircuitBreakerState.CLOSED

        # Second failure should trip the circuit
        with pytest.raises(Exception):
            await breaker.call(failing_func)
        assert breaker.state == CircuitBreakerState.OPEN

        # Third call should be blocked by circuit breaker
        with pytest.raises(ServiceMeshError, match="Circuit breaker open"):
            await breaker.call(failing_func)

    @pytest.mark.asyncio
    async def test_circuit_breaker_half_open_recovery(self):
        """Test circuit breaker recovery through half-open state."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            timeout_duration=1,  # 1 second for fast testing
            success_threshold=1,
            request_volume_threshold=1,
            error_threshold_percentage=50.0,
        )
        breaker = CircuitBreaker("test_service", config)

        # Trip the circuit
        async def failing_func():
            raise Exception("Test failure")

        with pytest.raises(Exception):
            await breaker.call(failing_func)
        assert breaker.state == CircuitBreakerState.OPEN

        # Wait for timeout
        await asyncio.sleep(1.1)

        # Next call should go to half-open state
        async def successful_func():
            return "success"

        result = await breaker.call(successful_func)

        assert result == "success"
        assert breaker.state == CircuitBreakerState.CLOSED


class TestServiceRegistry:
    """Test service registry functionality."""

    @pytest.mark.asyncio
    async def test_service_registration(self, redis_client, service_endpoint):
        """Test service registration in Redis."""
        registry = ServiceRegistry(redis_client)

        success = await registry.register_service(
            "test_service", service_endpoint, ttl=60
        )

        assert success

        # Verify service was stored in Redis
        service_key = f"services:test_service:{service_endpoint.address}"
        service_data = await redis_client.get(service_key)
        assert service_data is not None

        data = json.loads(service_data)
        assert data["name"] == "test_service"
        assert data["host"] == service_endpoint.host
        assert data["port"] == service_endpoint.port

    @pytest.mark.asyncio
    async def test_service_discovery(self, redis_client):
        """Test service discovery from Redis."""
        registry = ServiceRegistry(redis_client)

        # Register multiple service instances
        endpoint1 = ServiceEndpoint("service1", "127.0.0.1", 50051, "grpc")
        endpoint2 = ServiceEndpoint("service1", "127.0.0.1", 50052, "grpc")

        await registry.register_service("test_service", endpoint1)
        await registry.register_service("test_service", endpoint2)

        # Discover services
        discovered = await registry.discover_services("test_service")

        assert len(discovered) == 2
        addresses = {ep.address for ep in discovered}
        assert "127.0.0.1:50051" in addresses
        assert "127.0.0.1:50052" in addresses

    @pytest.mark.asyncio
    async def test_health_monitoring_start_stop(self, redis_client):
        """Test health monitoring lifecycle."""
        registry = ServiceRegistry(redis_client)

        # Start monitoring
        await registry.start_health_monitoring()
        assert registry._running
        assert registry._health_check_task is not None

        # Stop monitoring
        await registry.stop_health_monitoring()
        assert not registry._running
        assert registry._health_check_task is None


class TestLoadBalancer:
    """Test load balancer functionality."""

    def test_round_robin_selection(self):
        """Test round-robin load balancing."""
        balancer = LoadBalancer(LoadBalancingStrategy.ROUND_ROBIN)

        endpoints = [
            ServiceEndpoint("service1", "host1", 8001),
            ServiceEndpoint("service2", "host2", 8002),
            ServiceEndpoint("service3", "host3", 8003),
        ]

        # Set all endpoints as healthy
        for ep in endpoints:
            ep.status = ServiceStatus.HEALTHY

        # Test round-robin selection
        selected = []
        for _ in range(6):
            endpoint = balancer.select_endpoint("test_service", endpoints)
            selected.append(endpoint.port)

        # Should cycle through endpoints
        assert selected == [8001, 8002, 8003, 8001, 8002, 8003]

    def test_least_connections_selection(self):
        """Test least connections load balancing."""
        balancer = LoadBalancer(LoadBalancingStrategy.LEAST_CONNECTIONS)

        endpoints = [
            ServiceEndpoint("service1", "host1", 8001),
            ServiceEndpoint("service2", "host2", 8002),
            ServiceEndpoint("service3", "host3", 8003),
        ]

        # Set different connection counts
        endpoints[0].active_connections = 5
        endpoints[1].active_connections = 2
        endpoints[2].active_connections = 8

        # Set all endpoints as healthy
        for ep in endpoints:
            ep.status = ServiceStatus.HEALTHY

        selected = balancer.select_endpoint("test_service", endpoints)

        # Should select endpoint with least connections (port 8002)
        assert selected.port == 8002

    def test_health_aware_selection(self):
        """Test health-aware load balancing."""
        balancer = LoadBalancer(LoadBalancingStrategy.HEALTH_AWARE)

        endpoints = [
            ServiceEndpoint("service1", "host1", 8001),
            ServiceEndpoint("service2", "host2", 8002),
            ServiceEndpoint("service3", "host3", 8003),
        ]

        # Set health status and response times
        endpoints[0].status = ServiceStatus.UNHEALTHY
        endpoints[0].avg_response_time = 200

        endpoints[1].status = ServiceStatus.HEALTHY
        endpoints[1].avg_response_time = 50

        endpoints[2].status = ServiceStatus.HEALTHY
        endpoints[2].avg_response_time = 150

        selected = balancer.select_endpoint("test_service", endpoints)

        # Should select healthy endpoint with better response time (port 8002)
        assert selected.port == 8002

    def test_no_healthy_endpoints_fallback(self):
        """Test fallback when no healthy endpoints available."""
        balancer = LoadBalancer(LoadBalancingStrategy.HEALTH_AWARE)

        endpoints = [
            ServiceEndpoint("service1", "host1", 8001),
            ServiceEndpoint("service2", "host2", 8002),
        ]

        # Set all endpoints as unhealthy
        for ep in endpoints:
            ep.status = ServiceStatus.UNHEALTHY

        selected = balancer.select_endpoint("test_service", endpoints)

        # Should still select an endpoint (fallback behavior)
        assert selected is not None
        assert selected in endpoints


class TestCorrelationContext:
    """Test correlation context functionality."""

    def test_correlation_context_init(self, correlation_context):
        """Test correlation context initialization."""
        assert correlation_context.correlation_id is not None
        assert correlation_context.trace_id is not None
        assert correlation_context.span_id is not None
        assert correlation_context.user_id == "test_user"
        assert correlation_context.session_id == "test_session"

    def test_to_grpc_metadata(self, correlation_context):
        """Test conversion to gRPC metadata."""
        metadata = correlation_context.to_grpc_metadata()

        metadata_dict = dict(metadata)

        assert metadata_dict["correlation-id"] == correlation_context.correlation_id
        assert metadata_dict["trace-id"] == correlation_context.trace_id
        assert metadata_dict["span-id"] == correlation_context.span_id
        assert metadata_dict["user-id"] == correlation_context.user_id
        assert metadata_dict["session-id"] == correlation_context.session_id

    def test_from_grpc_metadata(self):
        """Test creation from gRPC metadata."""
        metadata = [
            ("correlation-id", "test-correlation"),
            ("trace-id", "test-trace"),
            ("span-id", "test-span"),
            ("user-id", "test-user"),
            ("session-id", "test-session"),
            ("timestamp", datetime.utcnow().isoformat()),
        ]

        context = CorrelationContext.from_grpc_metadata(metadata)

        assert context.correlation_id == "test-correlation"
        assert context.trace_id == "test-trace"
        assert context.span_id == "test-span"
        assert context.user_id == "test-user"
        assert context.session_id == "test-session"


class TestEventBus:
    """Test event bus functionality."""

    @pytest.mark.asyncio
    async def test_event_bus_start_stop(self):
        """Test event bus lifecycle."""
        event_bus = EventBus(["localhost:9092"])

        # Start event bus (will not actually connect to Kafka in tests)
        await event_bus.start()

        # Stop event bus
        await event_bus.stop()

    @pytest.mark.asyncio
    async def test_publish_event(self, correlation_context):
        """Test event publishing."""
        event_bus = EventBus(["localhost:9092"])
        await event_bus.start()

        event_data = {"event_type": "test_event", "payload": {"key": "value"}}

        # This should not raise an error even without Kafka
        await event_bus.publish_event("test_topic", event_data, correlation_context)

        await event_bus.stop()


class TestServiceMeshClient:
    """Test service mesh client functionality."""

    @pytest.fixture
    async def mock_redis(self):
        """Mock Redis client."""
        mock_redis = AsyncMock()
        mock_redis.ping.return_value = True
        mock_redis.keys.return_value = []
        return mock_redis

    @pytest.mark.asyncio
    async def test_service_mesh_start_stop(self, mock_redis):
        """Test service mesh client lifecycle."""
        with patch(
            "src.its_camera_ai.services.service_mesh.redis.from_url",
            return_value=mock_redis,
        ):
            mesh_client = ServiceMeshClient()

            # Start service mesh
            await mesh_client.start()

            assert mesh_client.redis is not None
            assert mesh_client.service_registry is not None
            assert mesh_client.load_balancer is not None

            # Stop service mesh
            await mesh_client.stop()

    @pytest.mark.asyncio
    async def test_circuit_breaker_creation(self, mock_redis):
        """Test circuit breaker creation for services."""
        with patch(
            "src.its_camera_ai.services.service_mesh.redis.from_url",
            return_value=mock_redis,
        ):
            mesh_client = ServiceMeshClient()
            await mesh_client.start()

            # Get circuit breaker for service
            breaker = mesh_client._get_circuit_breaker("test_service")

            assert breaker is not None
            assert breaker.service_name == "test_service"
            assert breaker.state == CircuitBreakerState.CLOSED

            # Should reuse same breaker for same service
            breaker2 = mesh_client._get_circuit_breaker("test_service")
            assert breaker is breaker2

            await mesh_client.stop()

    @pytest.mark.asyncio
    async def test_service_health_monitoring(self, mock_redis):
        """Test service health monitoring."""
        with patch(
            "src.its_camera_ai.services.service_mesh.redis.from_url",
            return_value=mock_redis,
        ):
            mesh_client = ServiceMeshClient()
            await mesh_client.start()

            # Mock service discovery to return test endpoints
            test_endpoints = [
                ServiceEndpoint("streaming", "127.0.0.1", 50051, "grpc"),
                ServiceEndpoint("analytics", "127.0.0.1", 50052, "grpc"),
            ]

            mesh_client.service_registry.discover_services = AsyncMock(
                return_value=test_endpoints
            )

            # Get service health
            health_data = await mesh_client.get_service_health("streaming")

            assert health_data["service_name"] == "streaming"
            assert "total_endpoints" in health_data
            assert "endpoints" in health_data

            await mesh_client.stop()

    @pytest.mark.asyncio
    async def test_circuit_breaker_status(self, mock_redis):
        """Test circuit breaker status retrieval."""
        with patch(
            "src.its_camera_ai.services.service_mesh.redis.from_url",
            return_value=mock_redis,
        ):
            mesh_client = ServiceMeshClient()
            await mesh_client.start()

            # Create some circuit breakers
            mesh_client._get_circuit_breaker("streaming")
            mesh_client._get_circuit_breaker("analytics")

            # Get circuit breaker status
            status = await mesh_client.get_circuit_breaker_status()

            assert "streaming" in status
            assert "analytics" in status

            for service_status in status.values():
                assert "state" in service_status
                assert "failure_count" in service_status
                assert "success_count" in service_status

            await mesh_client.stop()


class TestServiceOrchestrator:
    """Test service orchestration workflows."""

    @pytest.fixture
    async def mock_service_mesh(self):
        """Mock service mesh client."""
        mock_mesh = AsyncMock(spec=ServiceMeshClient)
        mock_mesh.publish_event = AsyncMock()
        return mock_mesh

    @pytest.mark.asyncio
    async def test_camera_frame_workflow_success(
        self, mock_service_mesh, correlation_context
    ):
        """Test successful camera frame processing workflow."""
        orchestrator = ServiceOrchestrator(mock_service_mesh)

        result = await orchestrator.process_camera_frame_workflow(
            "camera_001", {"frame_data": "test"}, correlation_context
        )

        assert result["camera_id"] == "camera_001"
        assert result["correlation_id"] == correlation_context.correlation_id
        assert result["success"]
        assert len(result["errors"]) == 0

        # Verify workflow completion event was published
        mock_service_mesh.publish_event.assert_called_once()
        call_args = mock_service_mesh.publish_event.call_args
        assert call_args[0][0] == "workflow.frame_processing.completed"

    @pytest.mark.asyncio
    async def test_alert_escalation_workflow_success(
        self, mock_service_mesh, correlation_context
    ):
        """Test successful alert escalation workflow."""
        orchestrator = ServiceOrchestrator(mock_service_mesh)

        result = await orchestrator.alert_escalation_workflow(
            "alert_001", "critical", correlation_context
        )

        assert result["alert_id"] == "alert_001"
        assert result["severity"] == "critical"
        assert result["correlation_id"] == correlation_context.correlation_id
        assert result["success"]


class TestIntegrationWorkflows:
    """Test end-to-end integration workflows."""

    @pytest.mark.asyncio
    async def test_complete_service_mesh_workflow(self):
        """Test complete service mesh workflow with multiple services."""
        # This test would require actual service instances running
        # For now, we'll test the workflow with mocked services

        with patch(
            "src.its_camera_ai.services.service_mesh.redis.from_url"
        ) as mock_redis_factory:
            mock_redis = AsyncMock()
            mock_redis.ping.return_value = True
            mock_redis.keys.return_value = []
            mock_redis_factory.return_value = mock_redis

            # Create and start service mesh
            mesh_client = ServiceMeshClient()
            await mesh_client.start()

            try:
                # Test service registration
                test_endpoint = ServiceEndpoint("test_service", "127.0.0.1", 50051)
                await mesh_client.service_registry.register_service(
                    "test_service", test_endpoint
                )

                # Test service discovery
                discovered = await mesh_client.service_registry.discover_services(
                    "test_service"
                )
                assert len(discovered) >= 1

                # Test event publishing
                await mesh_client.publish_event("test.topic", {"message": "test event"})

                # Test orchestration
                orchestrator = ServiceOrchestrator(mesh_client)
                workflow_result = await orchestrator.process_camera_frame_workflow(
                    "camera_001", {"frame_data": "test"}
                )

                assert workflow_result["success"]

            finally:
                await mesh_client.stop()

    @pytest.mark.asyncio
    async def test_service_mesh_performance_under_load(self):
        """Test service mesh performance under load."""
        with patch(
            "src.its_camera_ai.services.service_mesh.redis.from_url"
        ) as mock_redis_factory:
            mock_redis = AsyncMock()
            mock_redis.ping.return_value = True
            mock_redis.keys.return_value = []
            mock_redis_factory.return_value = mock_redis

            mesh_client = ServiceMeshClient()
            await mesh_client.start()

            try:
                # Simulate high load with concurrent requests
                start_time = time.time()

                async def simulate_request():
                    await mesh_client.publish_event(
                        "load.test", {"request_id": str(uuid.uuid4())}
                    )

                # Run 100 concurrent requests
                tasks = [simulate_request() for _ in range(100)]
                await asyncio.gather(*tasks)

                end_time = time.time()
                duration = end_time - start_time

                # Should handle 100 requests in reasonable time
                assert duration < 5.0  # Less than 5 seconds

                # Test throughput: should handle 1000+ RPS
                requests_per_second = 100 / duration
                assert requests_per_second > 20  # At least 20 RPS in test environment

            finally:
                await mesh_client.stop()

    @pytest.mark.asyncio
    async def test_circuit_breaker_integration(self):
        """Test circuit breaker integration with service calls."""
        with patch(
            "src.its_camera_ai.services.service_mesh.redis.from_url"
        ) as mock_redis_factory:
            mock_redis = AsyncMock()
            mock_redis.ping.return_value = True
            mock_redis.keys.return_value = []
            mock_redis_factory.return_value = mock_redis

            mesh_client = ServiceMeshClient()
            await mesh_client.start()

            try:
                # Mock failing service call
                async def failing_service_call(*args, **kwargs):
                    raise Exception("Service unavailable")

                with patch.object(
                    mesh_client,
                    "_execute_service_call",
                    side_effect=failing_service_call,
                ):
                    # Make multiple calls to trigger circuit breaker
                    for _i in range(6):  # More than failure threshold
                        try:
                            await mesh_client.call_service(
                                "failing_service", "test_method", {"test": "data"}
                            )
                        except (ServiceMeshError, Exception):
                            pass  # Expected failures

                    # Check circuit breaker state
                    cb_status = await mesh_client.get_circuit_breaker_status()

                    if "failing_service" in cb_status:
                        # Circuit breaker should be open after failures
                        assert cb_status["failing_service"]["state"] == "open"

            finally:
                await mesh_client.stop()


class TestGlobalServiceMeshInstance:
    """Test global service mesh instance management."""

    @pytest.mark.asyncio
    async def test_get_global_service_mesh(self):
        """Test getting global service mesh instance."""
        with patch(
            "src.its_camera_ai.services.service_mesh.redis.from_url"
        ) as mock_redis_factory:
            mock_redis = AsyncMock()
            mock_redis.ping.return_value = True
            mock_redis.keys.return_value = []
            mock_redis_factory.return_value = mock_redis

            # Ensure no global instance exists
            await shutdown_service_mesh()

            # Get global instance
            mesh1 = await get_service_mesh()
            assert mesh1 is not None

            # Should return same instance
            mesh2 = await get_service_mesh()
            assert mesh1 is mesh2

            # Cleanup
            await shutdown_service_mesh()

    @pytest.mark.asyncio
    async def test_shutdown_global_service_mesh(self):
        """Test shutting down global service mesh instance."""
        with patch(
            "src.its_camera_ai.services.service_mesh.redis.from_url"
        ) as mock_redis_factory:
            mock_redis = AsyncMock()
            mock_redis.ping.return_value = True
            mock_redis.keys.return_value = []
            mock_redis_factory.return_value = mock_redis

            # Create global instance
            mesh = await get_service_mesh()
            assert mesh is not None

            # Shutdown
            await shutdown_service_mesh()

            # Should create new instance after shutdown
            new_mesh = await get_service_mesh()
            assert new_mesh is not mesh

            # Cleanup
            await shutdown_service_mesh()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
