"""Cache service for Redis operations."""

import json
from typing import Any, Optional

import redis.asyncio as redis

from ..core.logging import get_logger


logger = get_logger(__name__)


class CacheService:
    """Redis cache service for key-value operations."""
    
    def __init__(self, redis_client: redis.Redis) -> None:
        self.redis = redis_client
    
    async def get(self, key: str) -> Optional[str]:
        """Get value by key.
        
        Args:
            key: Cache key
            
        Returns:
            Value or None if not found
        """
        try:
            return await self.redis.get(key)
        except Exception as e:
            logger.error("Cache get error", key=key, error=str(e))
            return None
    
    async def set(
        self,
        key: str,
        value: str,
        ttl: Optional[int] = None
    ) -> bool:
        """Set value by key.
        
        Args:
            key: Cache key
            value: Value to store
            ttl: Time to live in seconds
            
        Returns:
            True if successful
        """
        try:
            return await self.redis.set(key, value, ex=ttl)
        except Exception as e:
            logger.error("Cache set error", key=key, error=str(e))
            return False
    
    async def get_json(self, key: str) -> Optional[Any]:
        """Get JSON value by key.
        
        Args:
            key: Cache key
            
        Returns:
            Parsed JSON value or None
        """
        value = await self.get(key)
        if value:
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                logger.warning("Invalid JSON in cache", key=key)
        return None
    
    async def set_json(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """Set JSON value by key.
        
        Args:
            key: Cache key
            value: Value to serialize and store
            ttl: Time to live in seconds
            
        Returns:
            True if successful
        """
        try:
            json_value = json.dumps(value, default=str)
            return await self.set(key, json_value, ttl)
        except (TypeError, ValueError) as e:
            logger.error("JSON serialization error", key=key, error=str(e))
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key.
        
        Args:
            key: Cache key
            
        Returns:
            True if key was deleted
        """
        try:
            result = await self.redis.delete(key)
            return result > 0
        except Exception as e:
            logger.error("Cache delete error", key=key, error=str(e))
            return False
    
    async def get_counter(self, key: str, window: int) -> int:
        """Get counter value for rate limiting.
        
        Args:
            key: Counter key
            window: Time window in seconds
            
        Returns:
            Current counter value
        """
        try:
            value = await self.redis.get(key)
            return int(value) if value else 0
        except Exception:
            return 0
    
    async def increment_counter(self, key: str, window: int) -> int:
        """Increment counter for rate limiting.
        
        Args:
            key: Counter key
            window: Time window in seconds
            
        Returns:
            New counter value
        """
        try:
            pipe = self.redis.pipeline()
            pipe.incr(key)
            pipe.expire(key, window)
            results = await pipe.execute()
            return results[0] if results else 1
        except Exception:
            return 1
