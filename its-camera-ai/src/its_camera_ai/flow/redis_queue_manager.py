"""Redis Queue Manager for High-Performance Stream Processing.

This module provides Redis-based queue management to replace Kafka,
optimized for high-throughput video frame processing with gRPC serialization.

Key Features:
- Redis Streams for ordered message processing
- Connection pooling and automatic reconnection
- Batch processing for improved throughput
- Dead letter queue for failed messages
- Comprehensive monitoring and metrics
- gRPC-optimized serialization
"""

import logging
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import redis.asyncio as aioredis
from redis.exceptions import ConnectionError, ResponseError

# Import blosc compression for high-performance queue serialization
try:
    from ..core.blosc_numpy_compressor import (
        BloscNumpyCompressor,
        CompressionAlgorithm,
        CompressionLevel,
        get_global_compressor,
    )
    BLOSC_AVAILABLE = True
except ImportError:
    BLOSC_AVAILABLE = False

if TYPE_CHECKING:
    from redis.asyncio import Redis

logger = logging.getLogger(__name__)


class QueueType(Enum):
    """Redis queue implementation types."""

    STREAM = "stream"  # Redis Streams (ordered, persistent)
    LIST = "list"  # Redis Lists (FIFO, simple)
    PUBSUB = "pubsub"  # Redis Pub/Sub (fire-and-forget)


class QueueStatus(Enum):
    """Queue processing status."""

    ACTIVE = "active"
    PAUSED = "paused"
    DRAINING = "draining"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class QueueConfig:
    """Configuration for Redis queue with advanced optimization settings."""

    name: str
    queue_type: QueueType = QueueType.STREAM
    max_length: int = 50000  # Increased for high throughput
    consumer_group: str = "default"
    consumer_name: str = "worker"
    batch_size: int = 100  # Larger batches for better throughput
    block_time_ms: int = 500  # Reduced for lower latency
    retry_attempts: int = 3
    dead_letter_queue: bool = True
    enable_compression: bool = True
    compression_level: int = 6

    # Advanced optimization settings
    enable_blosc_compression: bool = True
    blosc_algorithm: str = "zstd"  # Best balance of speed/compression
    blosc_level: int = 5  # Balanced compression level
    adaptive_batch_sizing: bool = True  # Dynamic batch size optimization
    pipeline_size: int = 1000  # Redis pipeline batch size
    connection_pool_size: int = 50  # Increased pool size
    prefetch_multiplier: int = 3  # Prefetch 3x batch size for better throughput


@dataclass
class QueueMetrics:
    """Queue performance metrics."""

    queue_name: str
    pending_count: int = 0
    processing_count: int = 0
    completed_count: int = 0
    failed_count: int = 0
    avg_processing_time_ms: float = 0.0
    throughput_fps: float = 0.0
    last_updated: float = field(default_factory=time.time)

    # Consumer group metrics (for streams)
    consumer_count: int = 0
    lag: int = 0

    # Memory and performance
    memory_usage_bytes: int = 0
    cpu_usage_percent: float = 0.0


class RedisQueueManager:
    """High-performance Redis queue manager for stream processing."""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        pool_size: int = 100,  # Further increased for ultra-high throughput
        timeout: int = 30,
        retry_on_failure: bool = True,
        enable_clustering: bool = False,
        cluster_nodes: list[str] | None = None,
        enable_blosc_compression: bool = True,
        compression_algorithm: str = "zstd",
        enable_adaptive_batching: bool = True,
    ):
        """Initialize Redis queue manager optimized for 15,000+ messages/second.

        Args:
            redis_url: Redis connection URL
            pool_size: Maximum connections in pool (increased for ultra-high throughput)
            timeout: Connection timeout in seconds
            retry_on_failure: Enable automatic retry on connection failure
            enable_clustering: Enable Redis cluster support
            cluster_nodes: List of cluster node addresses
            enable_blosc_compression: Enable blosc compression for queue data
            compression_algorithm: Compression algorithm (zstd, lz4, zlib)
            enable_adaptive_batching: Enable adaptive batch size optimization
        """
        self.redis_url = redis_url
        self.pool_size = pool_size
        self.timeout = timeout
        self.retry_on_failure = retry_on_failure
        self.enable_clustering = enable_clustering
        self.cluster_nodes = cluster_nodes or []

        # Connection management
        self.redis: Redis[bytes] | None = None
        self._connection_pool: aioredis.ConnectionPool | None = None  # type: ignore
        self._cluster_client: Any | None = None

        # Queue management
        self.queues: dict[str, QueueConfig] = {}
        self.metrics: dict[str, QueueMetrics] = {}
        self.status: QueueStatus = QueueStatus.STOPPED

        # Advanced performance tracking
        self.processing_times: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=2000)  # Increased for better statistics
        )
        self.last_metrics_update = time.time()
        self._throughput_tracker = deque(maxlen=200)  # Increased tracking samples
        self._batch_size_optimizer = BatchSizeOptimizer()  # Dynamic batch sizing

        # Blosc compression integration
        self.enable_blosc_compression = enable_blosc_compression and BLOSC_AVAILABLE
        self.compression_algorithm = compression_algorithm
        self.enable_adaptive_batching = enable_adaptive_batching

        if self.enable_blosc_compression:
            try:
                self.blosc_compressor = get_global_compressor()
                logger.info(f"Blosc compression enabled with {compression_algorithm} algorithm")
            except Exception as e:
                logger.warning(f"Failed to initialize blosc compressor: {e}")
                self.enable_blosc_compression = False
                self.blosc_compressor = None
        else:
            self.blosc_compressor = None

        # Advanced metrics
        self.compression_metrics = {
            "total_compressed_bytes": 0,
            "total_original_bytes": 0,
            "compression_time_ms": 0.0,
            "decompression_time_ms": 0.0,
            "compression_ratio_avg": 0.0
        }

        # Error handling
        self.error_count = 0
        self.last_error: Exception | None = None

        logger.info(
            f"Redis queue manager initialized with URL: {redis_url}, clustering: {enable_clustering}"
        )

    async def connect(self) -> None:
        """Establish Redis connection with pool."""
        try:
            # Optimized connection pool settings for ultra-high throughput
            self._connection_pool = aioredis.ConnectionPool.from_url(
                self.redis_url,
                max_connections=self.pool_size,
                socket_timeout=self.timeout,
                socket_connect_timeout=self.timeout,
                retry_on_timeout=True,
                health_check_interval=15,  # More frequent health checks
                socket_keepalive=True,
                socket_keepalive_options={},
                retry_on_error=[ConnectionError],
            )

            self.redis = aioredis.Redis(
                connection_pool=self._connection_pool,
                decode_responses=False,  # Keep bytes for binary data
            )

            # Test connection
            await self.redis.ping()
            self.status = QueueStatus.ACTIVE

            logger.info("Redis connection established successfully")

        except Exception as e:
            self.status = QueueStatus.ERROR
            self.last_error = e
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    async def disconnect(self) -> None:
        """Close Redis connection and cleanup."""
        try:
            if self.redis:
                await self.redis.close()

            if self._connection_pool:
                await self._connection_pool.disconnect()

            self.status = QueueStatus.STOPPED
            logger.info("Redis connection closed")

        except Exception as e:
            logger.error(f"Error closing Redis connection: {e}")

    async def create_queue(
        self, config: QueueConfig, initialize_consumer_group: bool = True
    ) -> bool:
        """Create or configure a Redis queue.

        Args:
            config: Queue configuration
            initialize_consumer_group: Whether to create consumer group for streams

        Returns:
            bool: Success status
        """
        try:
            if not self.redis:
                raise ConnectionError("Redis not connected")

            self.queues[config.name] = config
            self.metrics[config.name] = QueueMetrics(queue_name=config.name)

            # Initialize based on queue type
            if config.queue_type == QueueType.STREAM:
                await self._initialize_stream(config, initialize_consumer_group)
            elif config.queue_type == QueueType.LIST:
                await self._initialize_list(config)

            logger.info(
                f"Queue '{config.name}' created with type {config.queue_type.value}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to create queue '{config.name}': {e}")
            return False

    async def _initialize_stream(
        self, config: QueueConfig, create_consumer_group: bool
    ) -> None:
        """Initialize Redis Stream with consumer group."""
        if not self.redis:
            raise ConnectionError("Redis not connected")

        try:
            # Create consumer group if it doesn't exist
            if create_consumer_group:
                try:
                    await self.redis.xgroup_create(
                        config.name, config.consumer_group, id="0", mkstream=True
                    )
                    logger.info(
                        f"Created consumer group '{config.consumer_group}' for stream '{config.name}'"
                    )
                except ResponseError as e:
                    if "BUSYGROUP" not in str(e):
                        raise
                    # Consumer group already exists
                    pass

            # Set max length if specified
            if config.max_length > 0:
                await self.redis.xtrim(
                    config.name, maxlen=config.max_length, approximate=True
                )

        except Exception as e:
            logger.error(f"Failed to initialize stream '{config.name}': {e}")
            raise

    async def _initialize_list(self, config: QueueConfig) -> None:
        """Initialize Redis List queue."""
        if not self.redis:
            raise ConnectionError("Redis not connected")

        # Lists don't require special initialization
        # Just ensure the key exists
        await self.redis.lpush(config.name, "__init__")
        await self.redis.lpop(config.name)

    async def enqueue(
        self,
        queue_name: str,
        data: bytes,
        priority: int = 0,
        metadata: dict[str, Any] | None = None,
        enable_compression: bool | None = None,
    ) -> str:
        """Enqueue data to specified queue with optimized compression.

        Args:
            queue_name: Name of the queue
            data: Binary data to enqueue
            priority: Message priority (higher = more important)
            metadata: Additional metadata
            enable_compression: Override global compression setting

        Returns:
            str: Message ID
        """
        if not self.redis or queue_name not in self.queues:
            raise ValueError(f"Queue '{queue_name}' not found or Redis not connected")

        config = self.queues[queue_name]
        message_id = str(uuid.uuid4())

        try:
            # Apply blosc compression if enabled and beneficial
            compressed_data = data
            compression_applied = False

            if enable_compression is None:
                enable_compression = self.enable_blosc_compression

            if enable_compression and self.blosc_compressor and len(data) > 1000:  # Compress if >1KB
                try:
                    start_time = time.perf_counter()

                    # Try to compress as numpy array if possible, otherwise as raw bytes
                    try:
                        import numpy as np
                        # Assume data might be numpy array bytes
                        array = np.frombuffer(data, dtype=np.uint8)
                        if array.size > 0:
                            compressed_data = self.blosc_compressor.compress_array(array)
                        else:
                            raise ValueError("Empty array")
                    except:
                        # Fallback to raw byte compression
                        import blosc
                        compressed_data = blosc.compress(data, cname=self.compression_algorithm)

                    compression_time = (time.perf_counter() - start_time) * 1000
                    compression_applied = True

                    # Update compression metrics
                    self.compression_metrics["total_original_bytes"] += len(data)
                    self.compression_metrics["total_compressed_bytes"] += len(compressed_data)
                    self.compression_metrics["compression_time_ms"] += compression_time

                    compression_ratio = len(compressed_data) / len(data)
                    self.compression_metrics["compression_ratio_avg"] = (
                        (self.compression_metrics["compression_ratio_avg"] + compression_ratio) / 2
                    )

                    logger.debug(f"Queue compression: {len(data)} -> {len(compressed_data)} bytes ({compression_ratio:.3f} ratio, {compression_time:.2f}ms)")

                except Exception as e:
                    logger.warning(f"Queue compression failed, using raw data: {e}")
                    compressed_data = data

            # Prepare message with metadata and compression info
            message_data = {
                "id": message_id,
                "data": compressed_data,
                "timestamp": time.time(),
                "priority": priority,
                "attempts": 0,
                "compressed": compression_applied,
                "original_size": len(data) if compression_applied else len(compressed_data)
            }

            if metadata:
                message_data.update(metadata)

            # Enqueue based on queue type
            if config.queue_type == QueueType.STREAM:
                stream_id = await self.redis.xadd(
                    queue_name, message_data, maxlen=config.max_length
                )
                message_id = stream_id.decode()

            elif config.queue_type == QueueType.LIST:
                # Use priority scoring for lists
                if priority > 0:
                    await self.redis.lpush(queue_name, message_id)
                    await self.redis.hset(f"{queue_name}:data", message_id, data)
                else:
                    await self.redis.rpush(queue_name, message_id)
                    await self.redis.hset(f"{queue_name}:data", message_id, data)

            elif config.queue_type == QueueType.PUBSUB:
                await self.redis.publish(queue_name, data)

            # Update metrics
            self.metrics[queue_name].pending_count += 1

            return message_id

        except Exception as e:
            self.error_count += 1
            self.last_error = e
            logger.error(f"Failed to enqueue to '{queue_name}': {e}")
            raise

    async def enqueue_batch(
        self,
        queue_name: str,
        data_batch: list[bytes],
        priorities: list[int] | None = None,
        metadata_batch: list[dict[str, Any]] | None = None,
        compression: bool = True,
    ) -> list[str]:
        """Enqueue multiple messages in a batch for efficiency.

        Optimized for 10,000+ messages/second throughput requirement.

        Args:
            queue_name: Name of the queue
            data_batch: List of binary data to enqueue
            priorities: Optional list of priorities
            metadata_batch: Optional list of metadata dicts
            compression: Enable data compression for large payloads

        Returns:
            list[str]: List of message IDs
        """
        if not data_batch:
            return []

        message_ids = []

        if not self.redis:
            raise ConnectionError("Redis not connected")

        # Optimize batch size for maximum throughput
        max_pipeline_size = 1000
        all_message_ids = []

        # Process in optimized chunks
        for chunk_start in range(0, len(data_batch), max_pipeline_size):
            chunk_end = min(chunk_start + max_pipeline_size, len(data_batch))
            chunk_data = data_batch[chunk_start:chunk_end]
            chunk_priorities = priorities[chunk_start:chunk_end] if priorities else None
            chunk_metadata = (
                metadata_batch[chunk_start:chunk_end] if metadata_batch else None
            )

            chunk_ids = await self._enqueue_chunk(
                queue_name, chunk_data, chunk_priorities, chunk_metadata, compression
            )
            all_message_ids.extend(chunk_ids)

        # Update metrics
        self.metrics[queue_name].pending_count += len(data_batch)

        return all_message_ids

    async def _enqueue_chunk(
        self,
        queue_name: str,
        data_batch: list[bytes],
        priorities: list[int] | None,
        metadata_batch: list[dict[str, Any]] | None,
        compression: bool,
    ) -> list[str]:
        """Enqueue a single chunk using Redis pipeline for maximum efficiency."""
        message_ids = []
        config = self.queues[queue_name]

        # Use Redis pipeline for batch operations
        async with self.redis.pipeline() as pipe:
            for i, data in enumerate(data_batch):
                priority = priorities[i] if priorities else 0
                metadata = metadata_batch[i] if metadata_batch else None

                message_id = str(uuid.uuid4())
                message_ids.append(message_id)

                # Compress large payloads for efficiency
                if compression and len(data) > 1024:  # Compress if > 1KB
                    import gzip

                    data = gzip.compress(data)
                    compressed = True
                else:
                    compressed = False

                if config.queue_type == QueueType.STREAM:
                    message_data = {
                        "id": message_id,
                        "data": data,
                        "timestamp": time.time(),
                        "priority": priority,
                        "attempts": 0,
                        "compressed": compressed,
                    }
                    if metadata:
                        message_data.update(metadata)

                    pipe.xadd(queue_name, message_data, maxlen=config.max_length)

                elif config.queue_type == QueueType.LIST:
                    if priority > 0:
                        pipe.lpush(queue_name, message_id)
                    else:
                        pipe.rpush(queue_name, message_id)
                    pipe.hset(f"{queue_name}:data", message_id, data)
                    if compressed:
                        pipe.hset(f"{queue_name}:meta", message_id, "compressed:true")

            # Execute pipeline
            await pipe.execute()

        return message_ids

    async def dequeue(
        self, queue_name: str, timeout_ms: int | None = None, auto_decompress: bool = True
    ) -> tuple[str, bytes] | None:
        """Dequeue single message from queue with automatic decompression.

        Args:
            queue_name: Name of the queue
            timeout_ms: Block timeout in milliseconds
            auto_decompress: Automatically decompress data if compressed

        Returns:
            tuple[str, bytes]: Message ID and data, or None if timeout
        """
        if not self.redis or queue_name not in self.queues:
            return None

        config = self.queues[queue_name]

        try:
            if config.queue_type == QueueType.STREAM:
                return await self._dequeue_stream(config, timeout_ms)
            elif config.queue_type == QueueType.LIST:
                return await self._dequeue_list(config, timeout_ms)
            else:
                raise ValueError(f"Dequeue not supported for {config.queue_type}")

        except Exception as e:
            logger.error(f"Failed to dequeue from '{queue_name}': {e}")
            return None

    async def _dequeue_stream(
        self, config: QueueConfig, timeout_ms: int | None
    ) -> tuple[str, bytes] | None:
        """Dequeue from Redis Stream."""
        if not self.redis:
            raise ConnectionError("Redis not connected")

        try:
            timeout = timeout_ms or config.block_time_ms

            # Read from consumer group
            messages = await self.redis.xreadgroup(
                config.consumer_group,
                config.consumer_name,
                {config.name: ">"},
                count=1,
                block=timeout,
            )

            if not messages or not messages[0][1]:
                return None

            stream_name, stream_messages = messages[0]
            message_id, fields = stream_messages[0]

            # Extract data with automatic decompression
            raw_data = fields.get(b"data", b"")

            # Check if data is compressed and decompress if needed
            decompressed_data = raw_data
            if auto_decompress and fields.get(b"compressed", b"false") == b"true":
                try:
                    start_time = time.perf_counter()

                    # Try numpy array decompression first
                    try:
                        decompressed_array = self.blosc_compressor.decompress_array(
                            raw_data, None, None
                        )
                        decompressed_data = decompressed_array.tobytes()
                    except:
                        # Fallback to raw byte decompression
                        import blosc
                        decompressed_data = blosc.decompress(raw_data)

                    decompression_time = (time.perf_counter() - start_time) * 1000
                    self.compression_metrics["decompression_time_ms"] += decompression_time

                    logger.debug(f"Queue decompression: {len(raw_data)} -> {len(decompressed_data)} bytes in {decompression_time:.2f}ms")

                except Exception as e:
                    logger.warning(f"Queue decompression failed, using raw data: {e}")
                    decompressed_data = raw_data

            # Update metrics
            self.metrics[config.name].pending_count = max(
                0, self.metrics[config.name].pending_count - 1
            )
            self.metrics[config.name].processing_count += 1

            return message_id.decode(), decompressed_data

        except Exception as e:
            logger.error(f"Stream dequeue error: {e}")
            return None

    async def _dequeue_list(
        self, config: QueueConfig, timeout_ms: int | None
    ) -> tuple[str, bytes] | None:
        """Dequeue from Redis List."""
        if not self.redis:
            raise ConnectionError("Redis not connected")

        try:
            timeout = (timeout_ms or config.block_time_ms) / 1000

            # Blocking left pop
            result = await self.redis.blpop([config.name], timeout=timeout)

            if not result:
                return None

            queue_name, message_id = result
            message_id_str = (
                message_id.decode()
                if isinstance(message_id, bytes)
                else str(message_id)
            )

            # Get data
            data = await self.redis.hget(f"{config.name}:data", message_id_str)

            if data:
                # Clean up
                await self.redis.hdel(f"{config.name}:data", message_id_str)

                # Update metrics
                self.metrics[config.name].pending_count = max(
                    0, self.metrics[config.name].pending_count - 1
                )
                self.metrics[config.name].processing_count += 1

                return message_id_str, data

            return None

        except Exception as e:
            logger.error(f"List dequeue error: {e}")
            return None

    async def dequeue_batch(
        self,
        queue_name: str,
        batch_size: int | None = None,
        timeout_ms: int | None = None,
    ) -> list[tuple[str, bytes]]:
        """Dequeue multiple messages in a batch.

        Args:
            queue_name: Name of the queue
            batch_size: Number of messages to dequeue
            timeout_ms: Block timeout in milliseconds

        Returns:
            list[tuple[str, bytes]]: List of (message_id, data) tuples
        """
        if not self.redis or queue_name not in self.queues:
            return []

        config = self.queues[queue_name]
        batch_size = batch_size or config.batch_size

        messages = []

        try:
            if config.queue_type == QueueType.STREAM:
                timeout = timeout_ms or config.block_time_ms

                # Read batch from consumer group
                stream_messages = await self.redis.xreadgroup(
                    config.consumer_group,
                    config.consumer_name,
                    {config.name: ">"},
                    count=batch_size,
                    block=timeout,
                )

                if stream_messages and stream_messages[0][1]:
                    for message_id, fields in stream_messages[0][1]:
                        data = fields.get(b"data", b"")
                        messages.append((message_id.decode(), data))

            elif config.queue_type == QueueType.LIST:
                # Dequeue multiple from list
                for _ in range(batch_size):
                    result = await self.dequeue(
                        queue_name, timeout_ms=100
                    )  # Short timeout for batch
                    if result:
                        messages.append(result)
                    else:
                        break

            # Update metrics
            if messages:
                self.metrics[config.name].pending_count = max(
                    0, self.metrics[config.name].pending_count - len(messages)
                )
                self.metrics[config.name].processing_count += len(messages)

            return messages

        except Exception as e:
            logger.error(f"Batch dequeue error from '{queue_name}': {e}")
            return []

    async def acknowledge(
        self, queue_name: str, message_id: str, processing_time_ms: float | None = None
    ) -> bool:
        """Acknowledge successful message processing.

        Args:
            queue_name: Name of the queue
            message_id: Message ID to acknowledge
            processing_time_ms: Processing time for metrics

        Returns:
            bool: Success status
        """
        if not self.redis or queue_name not in self.queues:
            return False

        config = self.queues[queue_name]

        try:
            if config.queue_type == QueueType.STREAM:
                if not self.redis:
                    return False
                # Acknowledge in consumer group
                await self.redis.xack(config.name, config.consumer_group, message_id)  # type: ignore

            # Update metrics
            metrics = self.metrics[queue_name]
            metrics.processing_count = max(0, metrics.processing_count - 1)
            metrics.completed_count += 1

            if processing_time_ms:
                self.processing_times[queue_name].append(processing_time_ms)
                # Update average processing time
                times = list(self.processing_times[queue_name])
                metrics.avg_processing_time_ms = sum(times) / len(times)

            return True

        except Exception as e:
            logger.error(f"Failed to acknowledge message '{message_id}': {e}")
            return False

    async def reject(
        self,
        queue_name: str,
        message_id: str,
        reason: str = "processing_failed",
        send_to_dlq: bool = True,
    ) -> bool:
        """Reject message and optionally send to dead letter queue.

        Args:
            queue_name: Name of the queue
            message_id: Message ID to reject
            reason: Rejection reason
            send_to_dlq: Whether to send to dead letter queue

        Returns:
            bool: Success status
        """
        if not self.redis or queue_name not in self.queues:
            return False

        config = self.queues[queue_name]

        try:
            # Update metrics
            metrics = self.metrics[queue_name]
            metrics.processing_count = max(0, metrics.processing_count - 1)
            metrics.failed_count += 1

            if config.queue_type == QueueType.STREAM:
                # For streams, we need to handle failed messages differently
                if send_to_dlq and config.dead_letter_queue:
                    # Move to dead letter queue
                    dlq_name = f"{queue_name}:dlq"
                    dlq_data = {
                        "original_id": message_id,
                        "failed_at": time.time(),
                        "reason": reason,
                        "queue": queue_name,
                    }
                    await self.redis.xadd(dlq_name, dlq_data)

                # Acknowledge to remove from pending
                if self.redis:
                    await self.redis.xack(
                        config.name, config.consumer_group, message_id
                    )  # type: ignore

            return True

        except Exception as e:
            logger.error(f"Failed to reject message '{message_id}': {e}")
            return False

    async def get_queue_metrics(self, queue_name: str) -> QueueMetrics | None:
        """Get current metrics for a queue.

        Args:
            queue_name: Name of the queue

        Returns:
            QueueMetrics: Current queue metrics
        """
        if queue_name not in self.queues:
            return None

        if not self.redis:
            return None

        try:
            config = self.queues[queue_name]
            metrics = self.metrics[queue_name]

            # Update real-time metrics
            if config.queue_type == QueueType.STREAM:
                # Get stream length
                await self.redis.xlen(queue_name)

                # Get consumer group info
                try:
                    groups = await self.redis.xinfo_groups(queue_name)  # type: ignore
                    for group in groups:
                        if group[b"name"].decode() == config.consumer_group:
                            metrics.pending_count = int(group[b"pending"])
                            metrics.consumer_count = int(group[b"consumers"])
                            break
                except ResponseError:
                    # Consumer group might not exist
                    pass

            elif config.queue_type == QueueType.LIST:
                # Get list length
                metrics.pending_count = await self.redis.llen(queue_name)

            # Calculate throughput
            current_time = time.time()
            time_diff = current_time - metrics.last_updated
            if time_diff > 0:
                # Estimate throughput based on completed count
                metrics.throughput_fps = metrics.completed_count / time_diff

            metrics.last_updated = current_time

            # Get memory usage
            try:
                memory_info = await self.redis.memory_usage(queue_name)
                metrics.memory_usage_bytes = memory_info or 0
            except (ResponseError, AttributeError):
                # Memory usage command might not be available
                pass

            return metrics

        except Exception as e:
            logger.error(f"Failed to get metrics for '{queue_name}': {e}")
            return None

    async def get_all_metrics(self) -> dict[str, QueueMetrics]:
        """Get metrics for all queues.

        Returns:
            dict[str, QueueMetrics]: Metrics for all queues
        """
        all_metrics = {}

        for queue_name in self.queues:
            metrics = await self.get_queue_metrics(queue_name)
            if metrics:
                all_metrics[queue_name] = metrics

        return all_metrics


class BatchSizeOptimizer:
    """Dynamic batch size optimization for maximum throughput."""

    def __init__(self, initial_size: int = 100):
        self.current_size = initial_size
        self.min_size = 10
        self.max_size = 1000
        self.performance_history = deque(maxlen=20)
        self.adjustment_factor = 1.1

    def update_performance(
        self, batch_size: int, throughput: float, latency: float
    ) -> None:
        """Update performance metrics and adjust batch size."""
        score = throughput / max(latency, 0.001)  # Throughput/latency ratio
        self.performance_history.append((batch_size, score))

        if len(self.performance_history) >= 5:
            # Analyze recent performance
            recent_scores = [score for _, score in list(self.performance_history)[-5:]]
            avg_recent_score = sum(recent_scores) / len(recent_scores)

            # Get baseline performance
            if len(self.performance_history) >= 10:
                baseline_scores = [
                    score for _, score in list(self.performance_history)[-10:-5]
                ]
                avg_baseline_score = sum(baseline_scores) / len(baseline_scores)

                if avg_recent_score > avg_baseline_score:
                    # Performance improving, try larger batches
                    self.current_size = min(
                        int(self.current_size * self.adjustment_factor), self.max_size
                    )
                elif avg_recent_score < avg_baseline_score * 0.9:
                    # Performance degrading, reduce batch size
                    self.current_size = max(
                        int(self.current_size / self.adjustment_factor), self.min_size
                    )

    def get_optimal_batch_size(self) -> int:
        """Get current optimal batch size."""
        return self.current_size

    async def purge_queue(self, queue_name: str, force: bool = False) -> int:
        """Purge all messages from a queue.

        Args:
            queue_name: Name of the queue to purge
            force: Force purge even if queue is active

        Returns:
            int: Number of messages purged
        """
        if not self.redis or queue_name not in self.queues:
            return 0

        if not force and self.status == QueueStatus.ACTIVE:
            logger.warning(
                f"Cannot purge active queue '{queue_name}' without force=True"
            )
            return 0

        config = self.queues[queue_name]
        purged_count = 0

        try:
            if config.queue_type == QueueType.STREAM:
                # Get current length
                length = await self.redis.xlen(queue_name)
                # Delete the stream
                await self.redis.delete(queue_name)
                purged_count = length

            elif config.queue_type == QueueType.LIST:
                # Get current length
                length = await self.redis.llen(queue_name)
                # Delete the list and data hash
                await self.redis.delete(queue_name, f"{queue_name}:data")
                purged_count = length

            # Reset metrics
            self.metrics[queue_name] = QueueMetrics(queue_name=queue_name)

            logger.info(f"Purged {purged_count} messages from queue '{queue_name}'")
            return purged_count

        except Exception as e:
            logger.error(f"Failed to purge queue '{queue_name}': {e}")
            return 0

    async def health_check(self) -> dict[str, Any]:
        """Perform health check on Redis connection and queues.

        Returns:
            dict[str, Any]: Health status information
        """
        health_info = {
            "status": "unknown",
            "timestamp": time.time(),
            "redis_connected": False,
            "queue_count": len(self.queues),
            "total_pending": 0,
            "total_processing": 0,
            "error_count": self.error_count,
            "last_error": str(self.last_error) if self.last_error else None,
        }

        try:
            if self.redis:
                # Test Redis connection
                await self.redis.ping()
                health_info["redis_connected"] = True

                # Get Redis info
                redis_info = await self.redis.info()
                health_info["redis_version"] = redis_info.get("redis_version")
                health_info["used_memory"] = redis_info.get("used_memory")
                health_info["connected_clients"] = redis_info.get("connected_clients")

                # Calculate total pending/processing
                for queue_name in self.queues:
                    metrics = await self.get_queue_metrics(queue_name)
                    if metrics:
                        # Handle pending/processing count types
                        pending = getattr(metrics, "pending_count", 0)
                        processing = getattr(metrics, "processing_count", 0)
                        health_info["total_pending"] += (
                            int(pending) if pending is not None else 0
                        )  # type: ignore
                        health_info["total_processing"] += (
                            int(processing) if processing is not None else 0
                        )  # type: ignore

                health_info["status"] = "healthy"

        except Exception as e:
            health_info["status"] = "unhealthy"
            health_info["error"] = str(e)
            logger.error(f"Health check failed: {e}")

        return health_info

    def get_status(self) -> QueueStatus:
        """Get current queue manager status."""
        return self.status

    async def pause(self) -> None:
        """Pause queue processing."""
        self.status = QueueStatus.PAUSED
        logger.info("Queue processing paused")

    async def resume(self) -> None:
        """Resume queue processing."""
        if self.redis and self.status == QueueStatus.PAUSED:
            self.status = QueueStatus.ACTIVE
            logger.info("Queue processing resumed")

    async def __aenter__(self) -> "RedisQueueManager":
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.disconnect()
