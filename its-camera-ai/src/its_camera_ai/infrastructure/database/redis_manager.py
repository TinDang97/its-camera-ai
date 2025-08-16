"""Redis connection manager with cluster and streaming support.

Provides Redis connection management with clustering, Sentinel support,
pub/sub operations, stream processing, and caching functionality.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import redis.asyncio as redis
from redis.asyncio import RedisCluster
from redis.asyncio.sentinel import Sentinel

from ...core.exceptions import DatabaseError
from ...core.logging import get_logger
from .models import ConnectionHealth, ConnectionStatus, DatabaseType, RedisConfig

logger = get_logger(__name__)


class RedisManager:
    """Redis connection manager with advanced features.
    
    Supports standalone Redis, Redis Cluster, and Redis Sentinel configurations
    with connection pooling, pub/sub, streams, and caching operations.
    """

    def __init__(self, config: RedisConfig) -> None:
        """Initialize Redis manager.
        
        Args:
            config: Redis configuration
        """
        self.config = config
        self._client: Any = None  # Redis client (can be Redis or RedisCluster)
        self._sentinel: Sentinel | None = None
        self._is_connected = False
        self._health_status = ConnectionStatus.DISCONNECTED

        # Pub/Sub and streaming
        self._pubsub_client: Any = None
        self._active_subscriptions: set[str] = set()
        self._consumer_groups: dict[str, str] = {}

    async def connect(self) -> None:
        """Establish Redis connection based on configuration."""
        try:
            self._health_status = ConnectionStatus.CONNECTING
            logger.info("Connecting to Redis", host=self.config.host, port=self.config.port)

            # Choose connection type based on configuration
            if self.config.sentinel_mode:
                await self._connect_sentinel()
            elif self.config.cluster_mode:
                await self._connect_cluster()
            else:
                await self._connect_standalone()

            # Test connection
            await self._test_connection()

            # Create dedicated pub/sub client
            await self._create_pubsub_client()

            self._is_connected = True
            self._health_status = ConnectionStatus.HEALTHY
            logger.info("Redis connection established successfully")

        except Exception as e:
            self._health_status = ConnectionStatus.UNHEALTHY
            logger.error("Failed to connect to Redis", error=str(e))
            raise DatabaseError(
                "Failed to establish Redis connection",
                operation="connect",
                details={"host": self.config.host, "port": self.config.port},
                cause=e,
            ) from e

    async def disconnect(self) -> None:
        """Close Redis connections."""
        try:
            logger.info("Disconnecting from Redis")

            # Close pub/sub client
            if self._pubsub_client:
                await self._pubsub_client.close()
                self._pubsub_client = None

            # Close main client
            if self._client:
                await self._client.close()
                self._client = None

            # Close sentinel
            if self._sentinel:
                await self._sentinel.close()
                self._sentinel = None

            self._is_connected = False
            self._health_status = ConnectionStatus.DISCONNECTED
            self._active_subscriptions.clear()
            self._consumer_groups.clear()

            logger.info("Redis disconnection completed")

        except Exception as e:
            logger.error("Error during Redis disconnection", error=str(e))
            raise DatabaseError("Failed to disconnect from Redis", cause=e) from e

    async def _connect_standalone(self) -> None:
        """Connect to standalone Redis instance."""
        connection_params = self.config.get_connection_params()

        self._client = redis.Redis(
            connection_pool=redis.ConnectionPool(**connection_params)
        )

    async def _connect_cluster(self) -> None:
        """Connect to Redis Cluster."""
        if not self.config.cluster_nodes:
            raise DatabaseError("Cluster nodes not configured for cluster mode")

        # Parse cluster nodes
        from redis.asyncio.cluster import ClusterNode
        startup_nodes = []
        for node in self.config.cluster_nodes:
            if ":" in node:
                host, port = node.split(":", 1)
                startup_nodes.append(ClusterNode(host, int(port)))
            else:
                startup_nodes.append(ClusterNode(node, 6379))

        connection_params = self.config.get_connection_params()
        connection_params.pop("host", None)
        connection_params.pop("port", None)

        self._client = RedisCluster(
            startup_nodes=startup_nodes,
            require_full_coverage=self.config.cluster_require_full_coverage,
            **connection_params,
        )

    async def _connect_sentinel(self) -> None:
        """Connect to Redis via Sentinel."""
        if not self.config.sentinel_hosts:
            raise DatabaseError("Sentinel hosts not configured for sentinel mode")

        self._sentinel = Sentinel(
            self.config.sentinel_hosts,
            socket_timeout=self.config.socket_timeout,
        )

        # Get master connection
        self._client = self._sentinel.master_for(
            self.config.sentinel_service_name,
            **self.config.get_connection_params()
        )

    async def _test_connection(self) -> None:
        """Test Redis connection."""
        if not self._client:
            raise DatabaseError("Redis client not initialized")

        # Handle different Redis client types
        try:
            if hasattr(self._client, 'ping'):
                await self._client.ping()
            else:
                # Fallback for cluster
                await self._client.execute_command('PING')
        except Exception as e:
            raise DatabaseError("Connection test failed") from e

    async def _create_pubsub_client(self) -> None:
        """Create dedicated pub/sub client."""
        if self.config.cluster_mode or self.config.sentinel_mode:
            # Use main client for cluster/sentinel
            self._pubsub_client = self._client
        else:
            # Create separate connection for pub/sub
            connection_params = self.config.get_connection_params()
            self._pubsub_client = redis.Redis(
                connection_pool=redis.ConnectionPool(**connection_params)
            )

    # Cache Operations
    async def get(self, key: str, default: Any = None) -> Any:
        """Get value from Redis with JSON deserialization.
        
        Args:
            key: Redis key
            default: Default value if key doesn't exist
            
        Returns:
            Deserialized value or default
        """
        if not self._is_connected:
            raise DatabaseError("Not connected to Redis")

        try:
            value = await self._client.get(key)
            if value is None:
                return default

            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value.decode() if isinstance(value, bytes) else value

        except Exception as e:
            logger.error("Redis GET failed", key=key, error=str(e))
            raise DatabaseError(f"Failed to get key {key}", operation="get", cause=e) from e

    async def set(
        self,
        key: str,
        value: Any,
        ex: int | None = None,
        px: int | None = None,
        nx: bool = False,
        xx: bool = False,
    ) -> bool:
        """Set value in Redis with JSON serialization.
        
        Args:
            key: Redis key
            value: Value to store
            ex: Expiration in seconds
            px: Expiration in milliseconds
            nx: Only set if key doesn't exist
            xx: Only set if key exists
            
        Returns:
            True if operation succeeded
        """
        if not self._is_connected:
            raise DatabaseError("Not connected to Redis")

        try:
            # Serialize complex objects to JSON
            if isinstance(value, (dict, list)):
                serialized_value = json.dumps(value)
            else:
                serialized_value = value

            result = await self._client.set(key, serialized_value, ex=ex, px=px, nx=nx, xx=xx)
            return bool(result)

        except Exception as e:
            logger.error("Redis SET failed", key=key, error=str(e))
            raise DatabaseError(f"Failed to set key {key}", operation="set", cause=e) from e

    async def delete(self, *keys: str) -> int:
        """Delete keys from Redis.
        
        Args:
            *keys: Keys to delete
            
        Returns:
            Number of keys deleted
        """
        if not self._is_connected:
            raise DatabaseError("Not connected to Redis")

        try:
            return await self._client.delete(*keys)

        except Exception as e:
            logger.error("Redis DELETE failed", keys=keys, error=str(e))
            raise DatabaseError("Failed to delete keys", operation="delete", cause=e) from e

    async def exists(self, *keys: str) -> int:
        """Check if keys exist in Redis.
        
        Args:
            *keys: Keys to check
            
        Returns:
            Number of existing keys
        """
        if not self._is_connected:
            raise DatabaseError("Not connected to Redis")

        try:
            return await self._client.exists(*keys)

        except Exception as e:
            logger.error("Redis EXISTS failed", keys=keys, error=str(e))
            raise DatabaseError("Failed to check key existence", operation="exists", cause=e) from e

    async def expire(self, key: str, seconds: int) -> bool:
        """Set expiration for a key.
        
        Args:
            key: Redis key
            seconds: Expiration in seconds
            
        Returns:
            True if expiration was set
        """
        if not self._is_connected:
            raise DatabaseError("Not connected to Redis")

        try:
            return await self._client.expire(key, seconds)

        except Exception as e:
            logger.error("Redis EXPIRE failed", key=key, error=str(e))
            raise DatabaseError(f"Failed to set expiration for key {key}", operation="expire", cause=e) from e

    # Hash Operations
    async def hget(self, name: str, key: str) -> Any:
        """Get hash field value.
        
        Args:
            name: Hash name
            key: Field key
            
        Returns:
            Field value
        """
        if not self._is_connected:
            raise DatabaseError("Not connected to Redis")

        try:
            value = await self._client.hget(name, key)
            if value is None:
                return None

            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value.decode() if isinstance(value, bytes) else value

        except Exception as e:
            logger.error("Redis HGET failed", hash=name, key=key, error=str(e))
            raise DatabaseError("Failed to get hash field", operation="hget", cause=e) from e

    async def hset(self, name: str, mapping: dict[str, Any]) -> int:
        """Set hash fields.
        
        Args:
            name: Hash name
            mapping: Field-value mapping
            
        Returns:
            Number of fields added
        """
        if not self._is_connected:
            raise DatabaseError("Not connected to Redis")

        try:
            # Serialize complex values
            serialized_mapping = {}
            for key, value in mapping.items():
                if isinstance(value, (dict, list)):
                    serialized_mapping[key] = json.dumps(value)
                else:
                    serialized_mapping[key] = value

            return await self._client.hset(name, mapping=serialized_mapping)

        except Exception as e:
            logger.error("Redis HSET failed", hash=name, error=str(e))
            raise DatabaseError("Failed to set hash fields", operation="hset", cause=e) from e

    async def hgetall(self, name: str) -> dict[str, Any]:
        """Get all hash fields and values.
        
        Args:
            name: Hash name
            
        Returns:
            Dictionary of field-value pairs
        """
        if not self._is_connected:
            raise DatabaseError("Not connected to Redis")

        try:
            result = await self._client.hgetall(name)

            # Deserialize values
            deserialized_result = {}
            for key, value in result.items():
                key_str = key.decode() if isinstance(key, bytes) else key

                try:
                    deserialized_result[key_str] = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    deserialized_result[key_str] = value.decode() if isinstance(value, bytes) else value

            return deserialized_result

        except Exception as e:
            logger.error("Redis HGETALL failed", hash=name, error=str(e))
            raise DatabaseError("Failed to get hash fields", operation="hgetall", cause=e) from e

    # List Operations
    async def lpush(self, name: str, *values: Any) -> int:
        """Push values to the left of a list.
        
        Args:
            name: List name
            *values: Values to push
            
        Returns:
            Length of list after push
        """
        if not self._is_connected:
            raise DatabaseError("Not connected to Redis")

        try:
            # Serialize complex objects
            serialized_values = []
            for value in values:
                if isinstance(value, (dict, list)):
                    serialized_values.append(json.dumps(value))
                else:
                    serialized_values.append(value)

            return await self._client.lpush(name, *serialized_values)

        except Exception as e:
            logger.error("Redis LPUSH failed", list=name, error=str(e))
            raise DatabaseError(f"Failed to push to list {name}", operation="lpush", cause=e) from e

    async def rpop(self, name: str, count: int | None = None) -> Any:
        """Pop values from the right of a list.
        
        Args:
            name: List name
            count: Number of values to pop
            
        Returns:
            Popped value(s)
        """
        if not self._is_connected:
            raise DatabaseError("Not connected to Redis")

        try:
            if count is not None:
                values = await self._client.rpop(name, count)
                if not values:
                    return None

                # Deserialize values
                return [self._deserialize_value(v) for v in values]
            else:
                value = await self._client.rpop(name)
                return self._deserialize_value(value) if value is not None else None

        except Exception as e:
            logger.error("Redis RPOP failed", list=name, error=str(e))
            raise DatabaseError(f"Failed to pop from list {name}", operation="rpop", cause=e) from e

    def _deserialize_value(self, value: Any) -> Any:
        """Deserialize a Redis value.
        
        Args:
            value: Raw value from Redis
            
        Returns:
            Deserialized value
        """
        if value is None:
            return None

        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value.decode() if isinstance(value, bytes) else value

    # Pub/Sub Operations
    async def publish(self, channel: str, message: Any) -> int:
        """Publish message to a channel.
        
        Args:
            channel: Channel name
            message: Message to publish
            
        Returns:
            Number of subscribers that received the message
        """
        if not self._is_connected:
            raise DatabaseError("Not connected to Redis")

        try:
            if isinstance(message, (dict, list)):
                serialized_message = json.dumps(message)
            else:
                serialized_message = message

            return await self._client.publish(channel, serialized_message)

        except Exception as e:
            logger.error("Redis PUBLISH failed", channel=channel, error=str(e))
            raise DatabaseError(f"Failed to publish to channel {channel}", operation="publish", cause=e) from e

    @asynccontextmanager
    async def subscribe(self, *channels: str) -> AsyncGenerator[Any, None]:
        """Subscribe to channels and return pub/sub context manager.
        
        Args:
            *channels: Channels to subscribe to
            
        Yields:
            PubSub object for receiving messages
        """
        if not self._is_connected:
            raise DatabaseError("Not connected to Redis")

        if not self._pubsub_client:
            raise DatabaseError("Pub/Sub client not initialized")

        pubsub = self._pubsub_client.pubsub(
            ignore_subscribe_messages=self.config.pubsub_ignore_subscribe_messages
        )

        try:
            await pubsub.subscribe(*channels)
            self._active_subscriptions.update(channels)
            logger.debug("Subscribed to channels", channels=channels)

            yield pubsub

        finally:
            try:
                await pubsub.unsubscribe(*channels)
                self._active_subscriptions.difference_update(channels)
                logger.debug("Unsubscribed from channels", channels=channels)
            except Exception as e:
                logger.warning("Error during unsubscribe", error=str(e))
            finally:
                await pubsub.close()

    # Stream Operations
    async def xadd(
        self,
        stream: str,
        fields: dict[str, Any],
        message_id: str = "*",
        maxlen: int | None = None,
        approximate: bool = True,
    ) -> str:
        """Add message to a stream.
        
        Args:
            stream: Stream name
            fields: Message fields
            message_id: Message ID (default: auto-generate)
            maxlen: Maximum stream length
            approximate: Use approximate maxlen
            
        Returns:
            Generated message ID
        """
        if not self._is_connected:
            raise DatabaseError("Not connected to Redis")

        try:
            # Serialize complex field values
            serialized_fields = {}
            for key, value in fields.items():
                if isinstance(value, (dict, list)):
                    serialized_fields[key] = json.dumps(value)
                else:
                    serialized_fields[key] = str(value)

            maxlen = maxlen or self.config.stream_maxlen

            return await self._client.xadd(
                stream,
                serialized_fields,
                id=message_id,
                maxlen=maxlen,
                approximate=approximate,
            )

        except Exception as e:
            logger.error("Redis XADD failed", stream=stream, error=str(e))
            raise DatabaseError(f"Failed to add to stream {stream}", operation="xadd", cause=e) from e

    async def xread(
        self,
        streams: dict[str, str],
        count: int | None = None,
        block: int | None = None,
    ) -> dict[str, list[tuple[str, dict[str, Any]]]]:
        """Read messages from streams.
        
        Args:
            streams: Stream name to last ID mapping
            count: Maximum number of messages per stream
            block: Block for milliseconds if no messages
            
        Returns:
            Stream messages grouped by stream name
        """
        if not self._is_connected:
            raise DatabaseError("Not connected to Redis")

        try:
            result = await self._client.xread(streams, count=count, block=block)

            # Process result to deserialize field values
            processed_result = {}
            for stream_name, messages in result.items():
                stream_key = stream_name.decode() if isinstance(stream_name, bytes) else stream_name
                processed_messages = []

                for message_id, fields in messages:
                    msg_id = message_id.decode() if isinstance(message_id, bytes) else message_id

                    # Deserialize field values
                    deserialized_fields = {}
                    for key, value in fields.items():
                        field_key = key.decode() if isinstance(key, bytes) else key
                        deserialized_fields[field_key] = self._deserialize_value(value)

                    processed_messages.append((msg_id, deserialized_fields))

                processed_result[stream_key] = processed_messages

            return processed_result

        except Exception as e:
            logger.error("Redis XREAD failed", streams=streams, error=str(e))
            raise DatabaseError("Failed to read from streams", operation="xread", cause=e) from e

    async def create_consumer_group(
        self,
        stream: str,
        group_name: str,
        consumer_name: str,
        start_id: str = "0",
    ) -> bool:
        """Create consumer group for stream processing.
        
        Args:
            stream: Stream name
            group_name: Consumer group name
            consumer_name: Consumer name
            start_id: Starting message ID
            
        Returns:
            True if group was created
        """
        if not self._is_connected:
            raise DatabaseError("Not connected to Redis")

        try:
            await self._client.xgroup_create(stream, group_name, id=start_id, mkstream=True)
            self._consumer_groups[stream] = group_name
            logger.info(
                "Consumer group created",
                stream=stream,
                group=group_name,
                consumer=consumer_name,
            )
            return True

        except redis.ResponseError as e:
            if "BUSYGROUP" in str(e):
                # Group already exists
                logger.debug("Consumer group already exists", stream=stream, group=group_name)
                return False
            raise DatabaseError(
                f"Failed to create consumer group {group_name}",
                operation="create_consumer_group",
                cause=e,
            ) from e
        except Exception as e:
            logger.error("Consumer group creation failed", stream=stream, error=str(e))
            raise DatabaseError(
                f"Failed to create consumer group {group_name}",
                operation="create_consumer_group",
                cause=e,
            ) from e

    async def health_check(self) -> ConnectionHealth:
        """Check Redis connection health.
        
        Returns:
            Connection health status
        """
        start_time = time.time()

        try:
            if not self._is_connected:
                return ConnectionHealth(
                    status=ConnectionStatus.DISCONNECTED,
                    database_type=DatabaseType.REDIS,
                    host=self.config.host,
                    port=self.config.port,
                    database=str(self.config.database),
                    error_message="Not connected",
                )

            # Test connection with PING
            try:
                if hasattr(self._client, 'ping'):
                    await self._client.ping()
                else:
                    await self._client.execute_command('PING')
            except Exception as e:
                raise DatabaseError("Health check ping failed") from e

            # Get Redis info
            try:
                if hasattr(self._client, 'info'):
                    info = await self._client.info()
                else:
                    info = {}
            except Exception:
                info = {}

            response_time = (time.time() - start_time) * 1000

            return ConnectionHealth(
                status=ConnectionStatus.HEALTHY,
                database_type=DatabaseType.REDIS,
                host=self.config.host,
                port=self.config.port,
                database=str(self.config.database),
                response_time_ms=response_time,
                active_connections=info.get("connected_clients"),
                version=info.get("redis_version"),
                uptime_seconds=info.get("uptime_in_seconds"),
            )

        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            logger.warning("Redis health check failed", error=str(e))

            return ConnectionHealth(
                status=ConnectionStatus.UNHEALTHY,
                database_type=DatabaseType.REDIS,
                host=self.config.host,
                port=self.config.port,
                database=str(self.config.database),
                response_time_ms=response_time,
                error_message=str(e),
            )

    @property
    def is_connected(self) -> bool:
        """Check if service is connected."""
        return self._is_connected

    @property
    def health_status(self) -> ConnectionStatus:
        """Get current health status."""
        return self._health_status

    @property
    def active_subscriptions(self) -> set[str]:
        """Get active pub/sub subscriptions."""
        return self._active_subscriptions.copy()

    @property
    def consumer_groups(self) -> dict[str, str]:
        """Get active consumer groups."""
        return self._consumer_groups.copy()

    async def __aenter__(self) -> RedisManager:
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.disconnect()
