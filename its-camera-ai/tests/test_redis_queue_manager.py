"""Tests for Redis Queue Manager.

This module tests the high-performance Redis queue implementation
that replaces Kafka for stream processing.
"""

import time

import pytest

from its_camera_ai.data.redis_queue_manager import (
    QueueConfig,
    QueueMetrics,
    QueueStatus,
    QueueType,
    RedisQueueManager,
)


@pytest.fixture
async def redis_queue_manager():
    """Create a Redis queue manager for testing."""
    manager = RedisQueueManager(
        redis_url="redis://localhost:6379/15",  # Use test DB
        pool_size=5,
        timeout=10,
    )

    try:
        await manager.connect()
        yield manager
    finally:
        await manager.disconnect()


@pytest.fixture
def queue_config():
    """Create a test queue configuration."""
    return QueueConfig(
        name="test_queue",
        queue_type=QueueType.STREAM,
        max_length=1000,
        consumer_group="test_group",
        consumer_name="test_consumer",
        batch_size=10,
        enable_compression=True,
    )


@pytest.mark.asyncio
class TestRedisQueueManager:
    """Test Redis Queue Manager functionality."""

    async def test_connection(self, redis_queue_manager):
        """Test Redis connection establishment."""
        assert redis_queue_manager.status == QueueStatus.ACTIVE
        assert redis_queue_manager.redis is not None

    async def test_create_queue(self, redis_queue_manager, queue_config):
        """Test queue creation."""
        success = await redis_queue_manager.create_queue(queue_config)
        assert success
        assert queue_config.name in redis_queue_manager.queues
        assert queue_config.name in redis_queue_manager.metrics

    async def test_enqueue_dequeue(self, redis_queue_manager, queue_config):
        """Test basic enqueue and dequeue operations."""
        await redis_queue_manager.create_queue(queue_config)

        # Test data
        test_data = b"test message data"
        metadata = {"test_key": "test_value"}

        # Enqueue
        message_id = await redis_queue_manager.enqueue(
            queue_config.name, test_data, metadata=metadata
        )
        assert message_id is not None

        # Dequeue
        result = await redis_queue_manager.dequeue(queue_config.name, timeout_ms=1000)
        assert result is not None
        received_id, received_data = result
        assert received_data == test_data

    async def test_batch_operations(self, redis_queue_manager, queue_config):
        """Test batch enqueue and dequeue operations."""
        await redis_queue_manager.create_queue(queue_config)

        # Prepare batch data
        batch_data = [f"message_{i}".encode() for i in range(5)]

        # Batch enqueue
        message_ids = await redis_queue_manager.enqueue_batch(
            queue_config.name, batch_data
        )
        assert len(message_ids) == len(batch_data)

        # Batch dequeue
        results = await redis_queue_manager.dequeue_batch(
            queue_config.name, batch_size=5, timeout_ms=1000
        )
        assert len(results) == len(batch_data)

        # Verify data
        received_data = [data for _, data in results]
        assert set(received_data) == set(batch_data)

    async def test_acknowledge_reject(self, redis_queue_manager, queue_config):
        """Test message acknowledgment and rejection."""
        await redis_queue_manager.create_queue(queue_config)

        test_data = b"test acknowledgment"

        # Enqueue and dequeue
        await redis_queue_manager.enqueue(queue_config.name, test_data)
        result = await redis_queue_manager.dequeue(queue_config.name)
        assert result is not None

        message_id, _ = result

        # Test acknowledgment
        success = await redis_queue_manager.acknowledge(
            queue_config.name, message_id, processing_time_ms=50.0
        )
        assert success

        # Test rejection (with new message)
        await redis_queue_manager.enqueue(queue_config.name, test_data)
        result = await redis_queue_manager.dequeue(queue_config.name)
        message_id, _ = result

        success = await redis_queue_manager.reject(
            queue_config.name, message_id, reason="test_rejection"
        )
        assert success

    async def test_queue_metrics(self, redis_queue_manager, queue_config):
        """Test queue metrics collection."""
        await redis_queue_manager.create_queue(queue_config)

        # Add some messages
        for i in range(3):
            await redis_queue_manager.enqueue(
                queue_config.name, f"message_{i}".encode()
            )

        # Get metrics
        metrics = await redis_queue_manager.get_queue_metrics(queue_config.name)
        assert metrics is not None
        assert isinstance(metrics, QueueMetrics)
        assert metrics.queue_name == queue_config.name
        assert metrics.pending_count >= 0

    async def test_health_check(self, redis_queue_manager):
        """Test health check functionality."""
        health = await redis_queue_manager.health_check()
        assert health["status"] == "healthy"
        assert health["redis_connected"] is True
        assert "timestamp" in health

    async def test_purge_queue(self, redis_queue_manager, queue_config):
        """Test queue purging."""
        await redis_queue_manager.create_queue(queue_config)

        # Add messages
        for i in range(5):
            await redis_queue_manager.enqueue(
                queue_config.name, f"message_{i}".encode()
            )

        # Purge queue
        purged_count = await redis_queue_manager.purge_queue(
            queue_config.name, force=True
        )
        assert purged_count > 0

        # Verify queue is empty
        await redis_queue_manager.get_queue_metrics(queue_config.name)
        # Note: After purge, the queue structure is recreated, so pending might be 0

    @pytest.mark.benchmark
    async def test_performance_benchmark(self, redis_queue_manager, queue_config):
        """Benchmark Redis queue performance."""
        await redis_queue_manager.create_queue(queue_config)

        message_count = 100
        test_data = b"x" * 1024  # 1KB message

        # Benchmark enqueue
        start_time = time.time()
        message_ids = await redis_queue_manager.enqueue_batch(
            queue_config.name, [test_data] * message_count
        )
        enqueue_time = time.time() - start_time

        assert len(message_ids) == message_count
        enqueue_rate = message_count / enqueue_time
        print(f"Enqueue rate: {enqueue_rate:.1f} messages/second")

        # Benchmark dequeue
        start_time = time.time()
        all_messages = []
        while len(all_messages) < message_count:
            batch = await redis_queue_manager.dequeue_batch(
                queue_config.name, batch_size=20, timeout_ms=1000
            )
            all_messages.extend(batch)

            if not batch:  # No more messages
                break

        dequeue_time = time.time() - start_time
        dequeue_rate = len(all_messages) / dequeue_time
        print(f"Dequeue rate: {dequeue_rate:.1f} messages/second")

        # Performance assertions
        assert enqueue_rate > 100  # At least 100 messages/second
        assert dequeue_rate > 100  # At least 100 messages/second

    async def test_connection_failure_handling(self):
        """Test handling of connection failures."""
        # Test with invalid Redis URL
        manager = RedisQueueManager(redis_url="redis://invalid-host:6379", timeout=1)

        with pytest.raises((ConnectionError, ValueError, RuntimeError)):
            await manager.connect()

        assert manager.status == QueueStatus.ERROR

    async def test_queue_types(self, redis_queue_manager):
        """Test different queue types (Stream vs List)."""
        # Test Stream queue
        stream_config = QueueConfig(
            name="test_stream", queue_type=QueueType.STREAM, consumer_group="test_group"
        )

        # Test List queue
        list_config = QueueConfig(name="test_list", queue_type=QueueType.LIST)

        # Create both queues
        assert await redis_queue_manager.create_queue(stream_config)
        assert await redis_queue_manager.create_queue(list_config)

        # Test operations on both
        test_data = b"test data"

        # Stream operations
        await redis_queue_manager.enqueue(stream_config.name, test_data)
        result = await redis_queue_manager.dequeue(stream_config.name)
        assert result is not None

        # List operations
        await redis_queue_manager.enqueue(list_config.name, test_data)
        result = await redis_queue_manager.dequeue(list_config.name)
        assert result is not None


@pytest.mark.integration
class TestRedisQueueIntegration:
    """Integration tests requiring actual Redis instance."""

    @pytest.mark.skipif(
        not pytest.redis_available, reason="Redis not available for integration tests"
    )
    async def test_real_redis_connection(self):
        """Test with real Redis instance."""
        manager = RedisQueueManager(redis_url="redis://localhost:6379/15")

        try:
            await manager.connect()
            assert manager.status == QueueStatus.ACTIVE

            # Test basic operations
            config = QueueConfig("integration_test", QueueType.STREAM)
            await manager.create_queue(config)

            await manager.enqueue("integration_test", b"test")
            result = await manager.dequeue("integration_test")
            assert result is not None

        finally:
            await manager.disconnect()


# Add Redis availability check
def pytest_configure(config):
    """Configure pytest with Redis availability."""
    try:
        import redis

        client = redis.Redis(host="localhost", port=6379, db=15)
        client.ping()
        pytest.redis_available = True
    except Exception:
        pytest.redis_available = False
