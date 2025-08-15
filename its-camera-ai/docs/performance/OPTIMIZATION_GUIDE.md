# ITS Camera AI Analytics Service - Performance Optimization Guide

## Overview

This guide provides comprehensive strategies for optimizing the ITS Camera AI Analytics Service to achieve and maintain **sub-100ms latency** with **maximum throughput**. It covers GPU optimization, memory management, caching strategies, and production deployment optimizations.

## GPU Optimization Strategies

### Memory Management Optimization

#### GPU Memory Pool Implementation
```python
import torch
from typing import Dict, Tuple, Optional
from contextlib import contextmanager

class GPUMemoryPoolManager:
    """Advanced GPU memory pool with device affinity and optimization."""
    
    def __init__(self, device_ids: list[int], pool_size_gb: int = 4):
        self.device_ids = device_ids
        self.pools: Dict[int, torch.cuda.mempool_stats] = {}
        self.allocators: Dict[int, torch.cuda.memory.MemoryAllocator] = {}
        
        for device_id in device_ids:
            # Reserve memory pool per device
            with torch.cuda.device(device_id):
                torch.cuda.empty_cache()
                self.pools[device_id] = torch.cuda.memory_pool_reserved_memory()
                
                # Configure memory allocator for optimal performance
                torch.cuda.memory.set_per_process_memory_fraction(0.8, device_id)
    
    @contextmanager
    def allocate_batch_memory(self, device_id: int, batch_size: int, 
                              input_shape: Tuple[int, ...]):
        """Context manager for batch memory allocation with automatic cleanup."""
        try:
            with torch.cuda.device(device_id):
                # Pre-allocate tensor with optimal memory layout
                batch_tensor = torch.empty(
                    (batch_size, *input_shape),
                    dtype=torch.float16,  # Mixed precision for speed
                    device=device_id,
                    memory_format=torch.channels_last  # Optimized for convolutions
                )
                yield batch_tensor
        finally:
            # Automatic cleanup handled by context manager
            if 'batch_tensor' in locals():
                del batch_tensor
                torch.cuda.empty_cache()
    
    def get_memory_stats(self) -> Dict[int, Dict[str, float]]:
        """Get detailed memory statistics for all GPUs."""
        stats = {}
        for device_id in self.device_ids:
            with torch.cuda.device(device_id):
                stats[device_id] = {
                    'allocated_gb': torch.cuda.memory_allocated() / (1024**3),
                    'reserved_gb': torch.cuda.memory_reserved() / (1024**3),
                    'utilization_pct': (
                        torch.cuda.memory_allocated() / 
                        torch.cuda.memory_reserved() * 100
                    ) if torch.cuda.memory_reserved() > 0 else 0
                }
        return stats
```

#### CUDA Stream Optimization
```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class OptimizedStreamProcessor:
    """Multi-stream GPU processing with optimal pipeline parallelization."""
    
    def __init__(self, num_streams: int = 4, device_id: int = 0):
        self.device_id = device_id
        self.streams = []
        
        with torch.cuda.device(device_id):
            # Create high-priority streams for critical operations
            self.streams = [
                torch.cuda.Stream(priority=-1) for _ in range(num_streams)
            ]
            
        # Thread pool for CPU-GPU coordination
        self.executor = ThreadPoolExecutor(max_workers=num_streams)
    
    async def process_batches_parallel(self, batches: list[torch.Tensor], 
                                     model: torch.nn.Module) -> list[torch.Tensor]:
        """Process multiple batches in parallel using CUDA streams."""
        
        async def process_single_batch(batch: torch.Tensor, stream: torch.cuda.Stream):
            """Process single batch on specific CUDA stream."""
            with torch.cuda.stream(stream):
                # Async memory copy
                batch_gpu = batch.cuda(device=self.device_id, non_blocking=True)
                
                # Inference with mixed precision
                with torch.cuda.amp.autocast():
                    with torch.no_grad():
                        result = model(batch_gpu)
                
                # Async copy back to CPU if needed
                return result.cpu(non_blocking=True)
        
        # Create tasks for parallel processing
        tasks = []
        for i, batch in enumerate(batches):
            stream = self.streams[i % len(self.streams)]
            task = process_single_batch(batch, stream)
            tasks.append(task)
        
        # Wait for all batches to complete
        results = await asyncio.gather(*tasks)
        
        # Synchronize all streams
        for stream in self.streams:
            stream.synchronize()
            
        return results
```

### TensorRT Optimization

#### Automatic TensorRT Engine Generation
```python
import tensorrt as trt
import torch_tensorrt
from pathlib import Path

class TensorRTOptimizer:
    """Automated TensorRT optimization with fallback mechanisms."""
    
    def __init__(self, model_path: str, device_id: int = 0):
        self.model_path = Path(model_path)
        self.device_id = device_id
        self.engine_cache_dir = Path("./tensorrt_engines")
        self.engine_cache_dir.mkdir(exist_ok=True)
        
        # TensorRT configuration
        self.precision_mode = torch_tensorrt.dtype.half  # FP16 for speed
        self.max_batch_size = 32
        self.workspace_size = 2 << 30  # 2GB workspace
    
    def optimize_model(self, input_shape: Tuple[int, ...]) -> torch.nn.Module:
        """Optimize model with TensorRT, fallback to PyTorch if failed."""
        
        # Check for cached TensorRT engine
        engine_path = self.engine_cache_dir / f"model_{self.device_id}_{input_shape}.engine"
        
        if engine_path.exists():
            logger.info(f"Loading cached TensorRT engine: {engine_path}")
            return self._load_tensorrt_engine(engine_path)
        
        try:
            # Load PyTorch model
            model = torch.jit.load(self.model_path, map_location=f"cuda:{self.device_id}")
            model.eval()
            
            # Create example input for TensorRT compilation
            example_input = torch.randn(1, *input_shape, device=f"cuda:{self.device_id}")
            
            # TensorRT optimization with dynamic shapes
            trt_model = torch_tensorrt.compile(
                model,
                inputs=[
                    torch_tensorrt.Input(
                        min_shape=(1, *input_shape),
                        opt_shape=(16, *input_shape),  # Optimal batch size
                        max_shape=(self.max_batch_size, *input_shape),
                        dtype=self.precision_mode
                    )
                ],
                enabled_precisions={self.precision_mode},
                workspace_size=self.workspace_size,
                max_batch_size=self.max_batch_size,
                use_python_runtime=False  # C++ runtime for maximum performance
            )
            
            # Save compiled engine
            torch.jit.save(trt_model, engine_path)
            logger.info(f"TensorRT engine saved: {engine_path}")
            
            return trt_model
            
        except Exception as e:
            logger.warning(f"TensorRT optimization failed: {e}. Using PyTorch fallback.")
            # Fallback to optimized PyTorch model
            return self._optimize_pytorch_model()
    
    def _optimize_pytorch_model(self) -> torch.nn.Module:
        """Optimize PyTorch model when TensorRT is not available."""
        model = torch.jit.load(self.model_path, map_location=f"cuda:{self.device_id}")
        model.eval()
        
        # PyTorch optimization
        model = torch.jit.optimize_for_inference(model)
        
        # Enable CUDNN benchmarking for consistent input sizes
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        return model
```

## Advanced Caching Strategies

### Multi-Level Intelligent Cache
```python
import asyncio
import time
import hashlib
from typing import Any, Optional, Union
from dataclasses import dataclass
from collections import OrderedDict

@dataclass
class CacheEntry:
    """Cache entry with metadata for intelligent management."""
    value: Any
    created_at: float
    access_count: int
    last_accessed: float
    ttl: Optional[float] = None
    freshness_score: float = 1.0
    
    @property
    def is_expired(self) -> bool:
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl
    
    @property
    def age_seconds(self) -> float:
        return time.time() - self.created_at

class IntelligentCacheService:
    """Advanced multi-level cache with predictive warming and adaptive TTL."""
    
    def __init__(self, l1_size: int = 1000, redis_client=None):
        # L1: In-memory LRU cache
        self.l1_cache = OrderedDict()
        self.l1_size = l1_size
        
        # L2: Redis distributed cache
        self.redis_client = redis_client
        
        # Cache statistics
        self.stats = {
            'l1_hits': 0, 'l1_misses': 0,
            'l2_hits': 0, 'l2_misses': 0,
            'warming_operations': 0
        }
        
        # Background tasks
        self._warmup_task = None
        self._cleanup_task = None
        self._start_background_tasks()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value with intelligent cache hierarchy."""
        
        # Try L1 cache first (microsecond access)
        if l1_entry := self._get_l1(key):
            if not l1_entry.is_expired:
                l1_entry.access_count += 1
                l1_entry.last_accessed = time.time()
                self.stats['l1_hits'] += 1
                return l1_entry.value
            else:
                # Remove expired entry
                self._remove_l1(key)
        
        self.stats['l1_misses'] += 1
        
        # Try L2 cache (millisecond access)
        if self.redis_client:
            try:
                l2_value = await self.redis_client.get(key)
                if l2_value:
                    # Parse cached metadata
                    cached_data = json.loads(l2_value)
                    
                    # Check expiration
                    if not self._is_l2_expired(cached_data):
                        self.stats['l2_hits'] += 1
                        
                        # Warm L1 cache asynchronously
                        asyncio.create_task(
                            self._warm_l1_async(key, cached_data['value'])
                        )
                        
                        return cached_data['value']
            
            except Exception as e:
                logger.warning(f"L2 cache error for key {key}: {e}")
        
        self.stats['l2_misses'] += 1
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None,
                  freshness_score: float = 1.0) -> None:
        """Set value in cache with intelligent TTL and warming."""
        
        # Create cache entry
        entry = CacheEntry(
            value=value,
            created_at=time.time(),
            access_count=1,
            last_accessed=time.time(),
            ttl=ttl,
            freshness_score=freshness_score
        )
        
        # Store in L1 cache
        self._set_l1(key, entry)
        
        # Store in L2 cache with metadata
        if self.redis_client and ttl:
            cached_data = {
                'value': value,
                'created_at': entry.created_at,
                'freshness_score': freshness_score
            }
            
            # Adaptive TTL based on freshness score
            adaptive_ttl = int(ttl * freshness_score)
            
            try:
                await self.redis_client.setex(
                    key, adaptive_ttl, json.dumps(cached_data, default=str)
                )
            except Exception as e:
                logger.warning(f"L2 cache set error for key {key}: {e}")
    
    def _get_l1(self, key: str) -> Optional[CacheEntry]:
        """Get entry from L1 cache and update LRU order."""
        if key in self.l1_cache:
            # Move to end (most recently used)
            entry = self.l1_cache.pop(key)
            self.l1_cache[key] = entry
            return entry
        return None
    
    def _set_l1(self, key: str, entry: CacheEntry) -> None:
        """Set entry in L1 cache with LRU eviction."""
        # Remove if already exists
        if key in self.l1_cache:
            del self.l1_cache[key]
        
        # Add new entry
        self.l1_cache[key] = entry
        
        # Evict least recently used if over capacity
        while len(self.l1_cache) > self.l1_size:
            oldest_key = next(iter(self.l1_cache))
            del self.l1_cache[oldest_key]
    
    async def _warm_l1_async(self, key: str, value: Any) -> None:
        """Asynchronously warm L1 cache."""
        entry = CacheEntry(
            value=value,
            created_at=time.time(),
            access_count=1,
            last_accessed=time.time()
        )
        self._set_l1(key, entry)
        self.stats['warming_operations'] += 1
    
    def get_cache_stats(self) -> dict:
        """Get comprehensive cache statistics."""
        total_l1_requests = self.stats['l1_hits'] + self.stats['l1_misses']
        total_l2_requests = self.stats['l2_hits'] + self.stats['l2_misses']
        
        return {
            'l1_hit_rate': (
                self.stats['l1_hits'] / max(1, total_l1_requests)
            ),
            'l2_hit_rate': (
                self.stats['l2_hits'] / max(1, total_l2_requests)
            ),
            'l1_size': len(self.l1_cache),
            'l1_capacity': self.l1_size,
            'warming_operations': self.stats['warming_operations'],
            'total_requests': total_l1_requests
        }
```

### Predictive Cache Warming
```python
import numpy as np
from sklearn.ensemble import IsolationForest
from typing import List, Tuple

class PredictiveCacheWarmer:
    """Machine learning-based cache warming for performance optimization."""
    
    def __init__(self, cache_service: IntelligentCacheService):
        self.cache_service = cache_service
        self.access_patterns = []  # Store access pattern history
        self.ml_model = IsolationForest(contamination=0.1, random_state=42)
        self.is_trained = False
        
        # Pattern tracking
        self.pattern_window = 3600  # 1 hour window
        self.min_training_samples = 100
    
    def record_access(self, key: str, timestamp: float, 
                     processing_time: float) -> None:
        """Record cache access pattern for ML analysis."""
        
        # Extract time-based features
        hour = int(timestamp % 86400) // 3600  # Hour of day
        minute = int(timestamp % 3600) // 60   # Minute of hour
        weekday = int((timestamp // 86400) % 7)  # Day of week
        
        pattern = {
            'key': key,
            'timestamp': timestamp,
            'hour': hour,
            'minute': minute,
            'weekday': weekday,
            'processing_time': processing_time,
            'key_hash': hashlib.md5(key.encode()).hexdigest()[:8]
        }
        
        self.access_patterns.append(pattern)
        
        # Keep only recent patterns
        cutoff_time = timestamp - self.pattern_window
        self.access_patterns = [
            p for p in self.access_patterns 
            if p['timestamp'] > cutoff_time
        ]
        
        # Retrain model periodically
        if len(self.access_patterns) >= self.min_training_samples:
            if len(self.access_patterns) % 50 == 0:  # Retrain every 50 samples
                asyncio.create_task(self._retrain_model())
    
    async def _retrain_model(self) -> None:
        """Retrain ML model with recent access patterns."""
        if len(self.access_patterns) < self.min_training_samples:
            return
        
        try:
            # Prepare feature matrix
            features = []
            for pattern in self.access_patterns:
                feature_vector = [
                    pattern['hour'],
                    pattern['minute'],
                    pattern['weekday'],
                    pattern['processing_time'],
                    int(pattern['key_hash'], 16) % 1000  # Hash-based feature
                ]
                features.append(feature_vector)
            
            # Train isolation forest for anomaly detection
            X = np.array(features)
            self.ml_model.fit(X)
            self.is_trained = True
            
            logger.info(f"Cache warming model retrained with {len(features)} samples")
            
        except Exception as e:
            logger.error(f"Model retraining failed: {e}")
    
    def predict_access_probability(self, key: str, 
                                 current_time: float) -> float:
        """Predict probability of cache key access in near future."""
        
        if not self.is_trained:
            return 0.5  # Default moderate probability
        
        try:
            # Extract features for current context
            hour = int(current_time % 86400) // 3600
            minute = int(current_time % 3600) // 60
            weekday = int((current_time // 86400) % 7)
            key_hash = int(hashlib.md5(key.encode()).hexdigest()[:8], 16) % 1000
            
            # Average processing time for this key type
            avg_processing_time = np.mean([
                p['processing_time'] for p in self.access_patterns
                if p['key'].startswith(key.split(':')[0])
            ] or [0.05])  # Default 50ms
            
            feature_vector = np.array([[
                hour, minute, weekday, avg_processing_time, key_hash
            ]])
            
            # Use isolation forest score as probability indicator
            anomaly_score = self.ml_model.score_samples(feature_vector)[0]
            
            # Convert to probability (higher score = more likely to be accessed)
            probability = max(0.0, min(1.0, (anomaly_score + 0.5)))
            
            return probability
            
        except Exception as e:
            logger.warning(f"Prediction failed for key {key}: {e}")
            return 0.5
    
    async def warm_predicted_keys(self, candidate_keys: List[str]) -> None:
        """Warm cache for keys likely to be accessed soon."""
        
        current_time = time.time()
        warming_candidates = []
        
        # Score all candidate keys
        for key in candidate_keys:
            probability = self.predict_access_probability(key, current_time)
            
            # Only warm high-probability keys
            if probability > 0.7:  # 70% probability threshold
                warming_candidates.append((key, probability))
        
        # Sort by probability and warm top candidates
        warming_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Limit warming to prevent resource exhaustion
        max_warming = min(10, len(warming_candidates))
        
        for key, probability in warming_candidates[:max_warming]:
            try:
                # Generate value if not in cache (simulation)
                await self._generate_and_cache_value(key, probability)
                
                logger.debug(f"Warmed cache for key {key} (probability: {probability:.2f})")
                
            except Exception as e:
                logger.warning(f"Cache warming failed for key {key}: {e}")
    
    async def _generate_and_cache_value(self, key: str, probability: float) -> None:
        """Generate and cache value based on key pattern."""
        # This would be replaced with actual value generation logic
        # For example, pre-computing analytics results, predictions, etc.
        
        # Adaptive TTL based on access probability
        base_ttl = 300  # 5 minutes
        adaptive_ttl = int(base_ttl * probability)
        
        # Simulate value generation (replace with actual logic)
        simulated_value = f"precomputed_value_for_{key}"
        
        await self.cache_service.set(
            key, 
            simulated_value, 
            ttl=adaptive_ttl,
            freshness_score=probability
        )
```

## Database Connection Optimization

### Advanced Connection Pool Management
```python
import asyncio
import asyncpg
from typing import Optional, Dict, Any
import time
from contextlib import asynccontextmanager

class OptimizedConnectionPool:
    """Advanced database connection pool with monitoring and optimization."""
    
    def __init__(self, database_url: str, **pool_kwargs):
        self.database_url = database_url
        self.pool: Optional[asyncpg.Pool] = None
        
        # Pool configuration with optimization
        self.pool_config = {
            'min_size': 10,           # Minimum connections
            'max_size': 50,           # Maximum connections
            'max_queries': 50000,     # Max queries per connection
            'max_inactive_connection_lifetime': 300,  # 5 minutes
            'setup': self._connection_setup,
            'command_timeout': 10,    # 10 second timeout
            **pool_kwargs
        }
        
        # Connection statistics
        self.stats = {
            'total_connections': 0,
            'active_connections': 0,
            'queries_executed': 0,
            'connection_errors': 0,
            'avg_query_time': 0.0,
            'last_reset': time.time()
        }
    
    async def initialize(self) -> None:
        """Initialize connection pool with optimization."""
        try:
            self.pool = await asyncpg.create_pool(
                self.database_url, 
                **self.pool_config
            )
            logger.info(f"Database pool initialized: {self.pool_config['min_size']}-{self.pool_config['max_size']} connections")
            
        except Exception as e:
            logger.error(f"Failed to initialize database pool: {e}")
            raise
    
    async def _connection_setup(self, connection: asyncpg.Connection) -> None:
        """Optimize individual database connections."""
        
        # Set optimal connection parameters
        await connection.execute("SET statement_timeout = '10s'")
        await connection.execute("SET lock_timeout = '5s'")
        await connection.execute("SET idle_in_transaction_session_timeout = '30s'")
        
        # Enable query optimization
        await connection.execute("SET enable_hashjoin = on")
        await connection.execute("SET enable_mergejoin = on")
        await connection.execute("SET enable_nestloop = off")  # Disable for large datasets
        
        # Optimize for analytics workloads
        await connection.execute("SET work_mem = '64MB'")
        await connection.execute("SET maintenance_work_mem = '256MB'")
        
        # Connection-level statistics
        self.stats['total_connections'] += 1
    
    @asynccontextmanager
    async def get_connection(self):
        """Get database connection with automatic cleanup and monitoring."""
        
        if not self.pool:
            await self.initialize()
        
        connection = None
        start_time = time.time()
        
        try:
            # Acquire connection with timeout
            connection = await asyncio.wait_for(
                self.pool.acquire(), 
                timeout=2.0  # 2 second acquire timeout
            )
            
            self.stats['active_connections'] += 1
            
            yield connection
            
        except asyncio.TimeoutError:
            self.stats['connection_errors'] += 1
            logger.error("Database connection acquire timeout")
            raise
            
        except Exception as e:
            self.stats['connection_errors'] += 1
            logger.error(f"Database connection error: {e}")
            raise
            
        finally:
            # Release connection and update statistics
            if connection:
                try:
                    await self.pool.release(connection)
                    self.stats['active_connections'] -= 1
                    
                    # Update query time statistics
                    query_time = time.time() - start_time
                    self._update_query_stats(query_time)
                    
                except Exception as e:
                    logger.warning(f"Connection release error: {e}")
    
    async def execute_optimized_query(self, query: str, *args, **kwargs) -> Any:
        """Execute query with optimization and monitoring."""
        
        async with self.get_connection() as conn:
            try:
                # Execute with timeout
                result = await asyncio.wait_for(
                    conn.fetch(query, *args, **kwargs),
                    timeout=self.pool_config['command_timeout']
                )
                
                self.stats['queries_executed'] += 1
                return result
                
            except asyncio.TimeoutError:
                logger.error(f"Query timeout: {query[:100]}...")
                raise
            except Exception as e:
                logger.error(f"Query execution error: {e}")
                raise
    
    def _update_query_stats(self, query_time: float) -> None:
        """Update rolling average query time."""
        current_avg = self.stats['avg_query_time']
        total_queries = self.stats['queries_executed']
        
        if total_queries > 0:
            self.stats['avg_query_time'] = (
                (current_avg * (total_queries - 1) + query_time) / total_queries
            )
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get comprehensive pool statistics."""
        uptime = time.time() - self.stats['last_reset']
        
        return {
            'pool_size': len(self.pool._holders) if self.pool else 0,
            'active_connections': self.stats['active_connections'],
            'total_connections_created': self.stats['total_connections'],
            'queries_per_second': self.stats['queries_executed'] / max(1, uptime),
            'avg_query_time_ms': self.stats['avg_query_time'] * 1000,
            'connection_error_rate': (
                self.stats['connection_errors'] / max(1, self.stats['total_connections'])
            ),
            'uptime_seconds': uptime
        }
```

## Redis Optimization

### High-Performance Redis Configuration
```python
import redis.asyncio as redis
from redis.asyncio import ConnectionPool
from typing import Optional, List, Dict, Any
import json
import time

class OptimizedRedisService:
    """High-performance Redis service with cluster support and optimization."""
    
    def __init__(self, redis_urls: List[str], cluster_mode: bool = False):
        self.redis_urls = redis_urls
        self.cluster_mode = cluster_mode
        self.connections: Dict[str, redis.Redis] = {}
        
        # Connection pool configuration
        self.pool_config = {
            'max_connections': 50,        # Per node
            'retry_on_timeout': True,
            'socket_keepalive': True,
            'socket_keepalive_options': {
                1: 1,    # TCP_KEEPIDLE
                2: 3,    # TCP_KEEPINTVL  
                3: 5     # TCP_KEEPCNT
            },
            'health_check_interval': 30,   # 30 seconds
            'socket_connect_timeout': 2,   # 2 second connect timeout
            'socket_timeout': 1            # 1 second socket timeout
        }
        
        # Performance tracking
        self.stats = {
            'operations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'errors': 0,
            'avg_response_time': 0.0
        }
    
    async def initialize(self) -> None:
        """Initialize Redis connections with optimization."""
        
        if self.cluster_mode:
            await self._initialize_cluster()
        else:
            await self._initialize_standalone()
        
        logger.info(f"Redis service initialized: {len(self.connections)} connections")
    
    async def _initialize_standalone(self) -> None:
        """Initialize standalone Redis connections with pooling."""
        
        for i, url in enumerate(self.redis_urls):
            try:
                # Create connection pool
                pool = ConnectionPool.from_url(
                    url,
                    **self.pool_config,
                    decode_responses=True,
                    encoding='utf-8'
                )
                
                # Create Redis client
                client = redis.Redis(connection_pool=pool)
                
                # Test connection
                await client.ping()
                
                # Store connection
                connection_name = f"redis_{i}"
                self.connections[connection_name] = client
                
                logger.info(f"Redis connection established: {connection_name}")
                
            except Exception as e:
                logger.error(f"Failed to connect to Redis {url}: {e}")
                raise
    
    async def set_optimized(self, key: str, value: Any, ttl: Optional[int] = None,
                           pipeline: Optional[redis.client.Pipeline] = None) -> bool:
        """Optimized Redis SET operation with serialization optimization."""
        
        start_time = time.time()
        
        try:
            # Optimize serialization
            if isinstance(value, (dict, list)):
                serialized_value = json.dumps(value, separators=(',', ':'))
            else:
                serialized_value = str(value)
            
            # Select optimal connection (load balancing)
            client = self._get_optimal_client()
            
            # Execute SET operation
            if pipeline:
                # Use pipeline for batching
                if ttl:
                    pipeline.setex(key, ttl, serialized_value)
                else:
                    pipeline.set(key, serialized_value)
                return True
            else:
                # Direct operation
                if ttl:
                    result = await client.setex(key, ttl, serialized_value)
                else:
                    result = await client.set(key, serialized_value)
                
                self._update_stats(time.time() - start_time)
                return bool(result)
                
        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"Redis SET error for key {key}: {e}")
            return False
    
    async def get_optimized(self, key: str,
                           pipeline: Optional[redis.client.Pipeline] = None) -> Optional[Any]:
        """Optimized Redis GET operation with deserialization."""
        
        start_time = time.time()
        
        try:
            # Select optimal connection
            client = self._get_optimal_client()
            
            # Execute GET operation
            if pipeline:
                pipeline.get(key)
                return None  # Result will be in pipeline execution
            else:
                result = await client.get(key)
                
                if result is None:
                    self.stats['cache_misses'] += 1
                    return None
                
                # Optimize deserialization
                try:
                    # Try JSON first (most common)
                    deserialized = json.loads(result)
                    self.stats['cache_hits'] += 1
                    self._update_stats(time.time() - start_time)
                    return deserialized
                    
                except (json.JSONDecodeError, TypeError):
                    # Fallback to string
                    self.stats['cache_hits'] += 1
                    self._update_stats(time.time() - start_time)
                    return result
                
        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"Redis GET error for key {key}: {e}")
            return None
    
    async def pipeline_batch_operation(self, operations: List[Dict[str, Any]]) -> List[Any]:
        """Execute batch operations using Redis pipeline for optimal performance."""
        
        client = self._get_optimal_client()
        
        try:
            async with client.pipeline(transaction=False) as pipe:
                # Add operations to pipeline
                for op in operations:
                    operation_type = op['type']
                    key = op['key']
                    
                    if operation_type == 'get':
                        pipe.get(key)
                    elif operation_type == 'set':
                        value = op['value']
                        ttl = op.get('ttl')
                        await self.set_optimized(key, value, ttl, pipeline=pipe)
                    elif operation_type == 'delete':
                        pipe.delete(key)
                    elif operation_type == 'exists':
                        pipe.exists(key)
                
                # Execute all operations in batch
                results = await pipe.execute()
                
                logger.debug(f"Executed pipeline batch: {len(operations)} operations")
                return results
                
        except Exception as e:
            logger.error(f"Pipeline batch operation failed: {e}")
            return []
    
    def _get_optimal_client(self) -> redis.Redis:
        """Select optimal Redis client based on load balancing."""
        
        if not self.connections:
            raise RuntimeError("No Redis connections available")
        
        # Simple round-robin selection
        # In production, could implement more sophisticated load balancing
        connection_names = list(self.connections.keys())
        selected_name = connection_names[self.stats['operations'] % len(connection_names)]
        
        return self.connections[selected_name]
    
    def _update_stats(self, response_time: float) -> None:
        """Update performance statistics."""
        self.stats['operations'] += 1
        
        # Rolling average response time
        current_avg = self.stats['avg_response_time']
        total_ops = self.stats['operations']
        
        self.stats['avg_response_time'] = (
            (current_avg * (total_ops - 1) + response_time) / total_ops
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive Redis performance statistics."""
        total_requests = self.stats['cache_hits'] + self.stats['cache_misses']
        
        return {
            'total_operations': self.stats['operations'],
            'cache_hit_rate': (
                self.stats['cache_hits'] / max(1, total_requests)
            ),
            'avg_response_time_ms': self.stats['avg_response_time'] * 1000,
            'error_rate': (
                self.stats['errors'] / max(1, self.stats['operations'])
            ),
            'connections_active': len(self.connections)
        }
```

## Performance Monitoring Implementation

### Real-time Performance Tracker
```python
import asyncio
import time
from typing import Dict, List, Callable, Any
from dataclasses import dataclass, field
from collections import deque
import numpy as np

@dataclass
class PerformanceMetric:
    """Performance metric with statistical analysis."""
    name: str
    values: deque = field(default_factory=lambda: deque(maxlen=1000))
    timestamps: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    def add_value(self, value: float) -> None:
        """Add new metric value."""
        self.values.append(value)
        self.timestamps.append(time.time())
    
    @property
    def current_value(self) -> float:
        """Get most recent value."""
        return self.values[-1] if self.values else 0.0
    
    @property
    def average(self) -> float:
        """Get average value."""
        return np.mean(self.values) if self.values else 0.0
    
    @property
    def percentile_95(self) -> float:
        """Get 95th percentile value."""
        return np.percentile(self.values, 95) if self.values else 0.0
    
    @property
    def percentile_99(self) -> float:
        """Get 99th percentile value."""
        return np.percentile(self.values, 99) if self.values else 0.0

class RealTimePerformanceMonitor:
    """Comprehensive real-time performance monitoring system."""
    
    def __init__(self):
        self.metrics: Dict[str, PerformanceMetric] = {}
        self.alert_thresholds: Dict[str, float] = {}
        self.alert_callbacks: List[Callable] = []
        
        # System monitoring
        self.monitoring_active = False
        self.monitoring_task: Optional[asyncio.Task] = None
        
        # Performance targets
        self.performance_targets = {
            'latency_ms': 100.0,        # <100ms target
            'throughput_fps': 30.0,     # >30 FPS target
            'gpu_utilization': 95.0,    # >95% target
            'error_rate': 0.1,          # <0.1% target
            'queue_depth': 600.0        # <600 items target
        }
    
    async def start_monitoring(self) -> None:
        """Start real-time performance monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Real-time performance monitoring started")
    
    async def stop_monitoring(self) -> None:
        """Stop performance monitoring."""
        self.monitoring_active = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Real-time performance monitoring stopped")
    
    def record_metric(self, name: str, value: float) -> None:
        """Record performance metric value."""
        
        if name not in self.metrics:
            self.metrics[name] = PerformanceMetric(name)
        
        self.metrics[name].add_value(value)
        
        # Check alert thresholds
        self._check_alerts(name, value)
    
    def set_alert_threshold(self, metric_name: str, threshold: float,
                           callback: Optional[Callable] = None) -> None:
        """Set alert threshold for metric."""
        self.alert_thresholds[metric_name] = threshold
        
        if callback:
            self.alert_callbacks.append(callback)
    
    def _check_alerts(self, metric_name: str, value: float) -> None:
        """Check if metric value exceeds alert threshold."""
        
        threshold = self.alert_thresholds.get(metric_name)
        
        if threshold and value > threshold:
            alert_data = {
                'metric_name': metric_name,
                'current_value': value,
                'threshold': threshold,
                'timestamp': time.time(),
                'severity': 'critical' if value > threshold * 1.2 else 'warning'
            }
            
            # Trigger alert callbacks
            for callback in self.alert_callbacks:
                try:
                    asyncio.create_task(callback(alert_data))
                except Exception as e:
                    logger.error(f"Alert callback failed: {e}")
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop for system metrics."""
        
        while self.monitoring_active:
            try:
                # System resource monitoring
                await self._collect_system_metrics()
                
                # Performance analysis
                await self._analyze_performance_trends()
                
                # Sleep for monitoring interval
                await asyncio.sleep(1.0)  # 1 second intervals
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(5.0)  # Error recovery delay
    
    async def _collect_system_metrics(self) -> None:
        """Collect system-level performance metrics."""
        
        try:
            import psutil
            
            # CPU utilization
            cpu_percent = psutil.cpu_percent(interval=None)
            self.record_metric('cpu_utilization_percent', cpu_percent)
            
            # Memory utilization
            memory = psutil.virtual_memory()
            self.record_metric('memory_utilization_percent', memory.percent)
            
            # GPU utilization (if available)
            try:
                import pynvml
                pynvml.nvmlInit()
                
                device_count = pynvml.nvmlDeviceGetCount()
                total_gpu_util = 0
                
                for i in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    total_gpu_util += utilization.gpu
                
                avg_gpu_util = total_gpu_util / device_count if device_count > 0 else 0
                self.record_metric('gpu_utilization_percent', avg_gpu_util)
                
            except ImportError:
                # GPU monitoring not available
                pass
                
        except Exception as e:
            logger.warning(f"System metrics collection failed: {e}")
    
    async def _analyze_performance_trends(self) -> None:
        """Analyze performance trends and detect degradation."""
        
        for metric_name, metric in self.metrics.items():
            if len(metric.values) < 10:  # Need minimum samples
                continue
            
            try:
                # Detect performance degradation
                recent_values = list(metric.values)[-10:]  # Last 10 values
                older_values = list(metric.values)[-50:-10] if len(metric.values) >= 50 else []
                
                if older_values:
                    recent_avg = np.mean(recent_values)
                    older_avg = np.mean(older_values)
                    
                    # Check for significant degradation (>20% increase)
                    if recent_avg > older_avg * 1.2:
                        logger.warning(
                            f"Performance degradation detected in {metric_name}: "
                            f"{older_avg:.2f} -> {recent_avg:.2f}"
                        )
                
            except Exception as e:
                logger.warning(f"Trend analysis failed for {metric_name}: {e}")
    
    def get_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        
        report = {
            'timestamp': time.time(),
            'monitoring_duration_seconds': time.time() - min(
                min(metric.timestamps) for metric in self.metrics.values()
                if metric.timestamps
            ) if self.metrics else 0,
            'metrics_summary': {},
            'performance_analysis': {},
            'alerts_triggered': 0
        }
        
        # Metrics summary
        for name, metric in self.metrics.items():
            if not metric.values:
                continue
                
            report['metrics_summary'][name] = {
                'current': metric.current_value,
                'average': metric.average,
                'p95': metric.percentile_95,
                'p99': metric.percentile_99,
                'samples': len(metric.values),
                'target': self.performance_targets.get(name, 'N/A')
            }
        
        # Performance analysis
        latency_metric = self.metrics.get('latency_ms')
        if latency_metric and latency_metric.values:
            report['performance_analysis']['sla_compliance'] = {
                'latency_under_100ms': sum(1 for v in latency_metric.values if v < 100) / len(latency_metric.values),
                'p99_latency_ms': latency_metric.percentile_99,
                'target_met': latency_metric.percentile_99 < 100
            }
        
        return report
```

## Production Deployment Optimizations

### Kubernetes Resource Optimization
```yaml
# High-performance Kubernetes deployment configuration
apiVersion: apps/v1
kind: Deployment
metadata:
  name: its-analytics-service
  labels:
    app: its-analytics
    component: analytics-service
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0  # Zero-downtime deployment
  selector:
    matchLabels:
      app: its-analytics
      component: analytics-service
  template:
    metadata:
      labels:
        app: its-analytics
        component: analytics-service
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "9090"
        prometheus.io/path: "/metrics"
    spec:
      # Performance optimizations
      nodeSelector:
        gpu: "true"
        instance-type: "gpu-optimized"
      
      # Resource allocation
      containers:
      - name: analytics-service
        image: its-camera-ai/analytics:latest
        
        resources:
          requests:
            cpu: "2000m"        # 2 CPU cores
            memory: "8Gi"       # 8GB RAM
            nvidia.com/gpu: 1   # 1 GPU
          limits:
            cpu: "4000m"        # 4 CPU cores max
            memory: "16Gi"      # 16GB RAM max
            nvidia.com/gpu: 1   # 1 GPU max
        
        # Environment optimization
        env:
        - name: CUDA_VISIBLE_DEVICES
          value: "0"
        - name: PYTHONUNBUFFERED
          value: "1"
        - name: OMP_NUM_THREADS
          value: "4"
        - name: PYTORCH_CUDA_ALLOC_CONF
          value: "max_split_size_mb:512"
        
        # Health checks
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
          timeoutSeconds: 10
          failureThreshold: 3
        
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        
        # Volume mounts for optimization
        volumeMounts:
        - name: model-cache
          mountPath: /app/models
          readOnly: true
        - name: tensorrt-cache
          mountPath: /app/tensorrt_engines
        
      # Volume configuration
      volumes:
      - name: model-cache
        persistentVolumeClaim:
          claimName: model-cache-pvc
      - name: tensorrt-cache
        emptyDir:
          sizeLimit: 10Gi

---
# Horizontal Pod Autoscaler for dynamic scaling
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: its-analytics-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: its-analytics-service
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70  # Scale at 70% CPU
  - type: Pods
    pods:
      metric:
        name: gpu_utilization_percent
      target:
        type: AverageValue
        averageValue: "80"  # Scale at 80% GPU utilization
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60   # 1 minute stabilization
      policies:
      - type: Percent
        value: 50    # Scale up by 50% max
        periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300  # 5 minute stabilization
      policies:
      - type: Percent
        value: 10    # Scale down by 10% max
        periodSeconds: 60
```

This comprehensive optimization guide provides production-ready strategies for achieving maximum performance while maintaining reliability and scalability in the ITS Camera AI Analytics Service.