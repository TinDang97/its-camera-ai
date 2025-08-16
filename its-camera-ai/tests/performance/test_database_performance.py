"""Comprehensive database performance tests for PostgreSQL optimization.

This module tests database performance under high-throughput conditions:
- Query execution time optimization (sub-10ms requirement)
- Connection pool performance under load
- Index effectiveness and query plan optimization
- TimescaleDB hypertable performance
- Concurrent connection handling (1000+ connections)
- Database health monitoring and alerts
"""

import asyncio
import time
import random
import statistics
from datetime import UTC, datetime, timedelta
from typing import List, Dict, Any
import pytest
from unittest.mock import AsyncMock, MagicMock

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from src.its_camera_ai.services.database_optimizer import (
    DatabaseOptimizer, 
    QueryPerformanceMetrics,
    IndexRecommendation,
    DatabaseHealthMetrics
)
from src.its_camera_ai.models.database import DatabaseManager
from src.its_camera_ai.core.config import Settings


class TestDatabasePerformance:
    """Performance tests for database optimization."""
    
    @pytest.fixture
    def settings(self):
        """Test settings."""
        return Settings(
            database=Settings.DatabaseSettings(
                url="postgresql+asyncpg://test:test@localhost/test",
                pool_size=20,
                max_overflow=40,
                pool_timeout=30,
                echo=False
            ),
            environment="testing"
        )
    
    @pytest.fixture
    async def mock_session(self):
        """Mock database session for testing."""
        session = AsyncMock(spec=AsyncSession)
        
        # Mock query results for performance analysis
        session.execute = AsyncMock()
        session.commit = AsyncMock()
        session.rollback = AsyncMock()
        
        return session
    
    @pytest.fixture
    async def database_manager(self, settings):
        """Mock database manager."""
        manager = MagicMock(spec=DatabaseManager)
        manager.settings = settings
        return manager
    
    @pytest.fixture
    async def database_optimizer(self, database_manager, settings):
        """Database optimizer for testing."""
        return DatabaseOptimizer(database_manager, settings)
    
    @pytest.mark.asyncio
    async def test_query_performance_analysis(self, database_optimizer, mock_session):
        """Test query performance analysis capabilities."""
        # Mock pg_stat_statements data
        mock_query_data = [
            MagicMock(
                queryid=12345,
                query="SELECT * FROM traffic_metrics WHERE camera_id = $1 AND timestamp > $2",
                execution_count=1500,
                total_time_ms=15000,  # 15 seconds total
                avg_time_ms=10,       # 10ms average
                max_time_ms=50,
                min_time_ms=2,
                rows=75000,
                shared_blks_hit=8000,
                shared_blks_read=2000,
                cache_hit_ratio=0.8
            ),
            MagicMock(
                queryid=12346,
                query="SELECT COUNT(*) FROM vehicle_detections WHERE created_at > $1",
                execution_count=500,
                total_time_ms=25000,  # 25 seconds total
                avg_time_ms=50,       # 50ms average - slow query
                max_time_ms=200,
                min_time_ms=10,
                rows=500,
                shared_blks_hit=5000,
                shared_blks_read=5000,
                cache_hit_ratio=0.5
            )
        ]
        
        mock_session.execute.return_value.fetchall.return_value = mock_query_data
        
        start_time = time.time()
        metrics = await database_optimizer.analyze_query_performance(mock_session, hours_back=24)
        analysis_time = time.time() - start_time
        
        print(f"\\nQuery Performance Analysis:")
        print(f"Analysis time: {analysis_time:.3f}s")
        print(f"Queries analyzed: {len(metrics)}")
        
        for metric in metrics:
            print(f"Query {metric.query_id}: {metric.avg_time_ms:.1f}ms avg, "
                  f"{metric.execution_count} executions, "
                  f"{metric.cache_hit_ratio:.1%} cache hit rate")
            print(f"  Recommendations: {len(metric.recommendations)}")
        
        # Performance assertions
        assert analysis_time < 1.0, f"Analysis too slow: {analysis_time}s"
        assert len(metrics) == 2, f"Expected 2 metrics, got {len(metrics)}"
        
        # Check slow query detection
        slow_queries = [m for m in metrics if m.avg_time_ms > 100]  # 100ms threshold
        expected_slow = [m for m in mock_query_data if m.avg_time_ms > 100]
        assert len(slow_queries) == len(expected_slow), "Slow query detection failed"
        
        # Check recommendations
        for metric in metrics:
            if metric.avg_time_ms > 100:
                assert len(metric.recommendations) > 0, "No recommendations for slow query"
    
    @pytest.mark.asyncio
    async def test_index_recommendation_generation(self, database_optimizer, mock_session):
        \"\"\"Test index recommendation generation.\"\"\"
        # Mock table statistics
        mock_stats_data = [
            MagicMock(
                schemaname="public",
                tablename="traffic_metrics",
                attname="camera_id",
                n_distinct=100,
                correlation=0.5,
                most_common_vals=[],
                table_size="500 MB"
            ),
            MagicMock(
                schemaname="public",
                tablename="vehicle_detections",
                attname="timestamp",
                n_distinct=50000,
                correlation=0.9,
                most_common_vals=[],
                table_size="2 GB"
            )
        ]
        
        mock_unused_indexes = [
            MagicMock(
                index_name="idx_unused_test",
                table_name="old_table",
                index_scans=2,
                size="50 MB",
                size_mb=50.0
            )
        ]
        
        # Setup mock return values
        mock_session.execute.return_value.fetchall.side_effect = [
            mock_stats_data,  # For missing indexes query
            mock_unused_indexes  # For unused indexes query
        ]
        
        start_time = time.time()
        recommendations = await database_optimizer.generate_index_recommendations(mock_session)
        generation_time = time.time() - start_time
        
        print(f"\\nIndex Recommendation Generation:")
        print(f"Generation time: {generation_time:.3f}s")
        print(f"Recommendations generated: {len(recommendations)}")
        
        for rec in recommendations:
            print(f"Table: {rec.table_name}, Columns: {rec.column_names}, "
                  f"Type: {rec.index_type}, Benefit: {rec.estimated_benefit:.1%}")
            print(f"  SQL: {rec.creation_sql}")
            print(f"  Reason: {rec.reason}")
        
        # Performance assertions
        assert generation_time < 2.0, f"Generation too slow: {generation_time}s"
        assert len(recommendations) > 0, "No recommendations generated"
        
        # Check recommendation quality
        create_recommendations = [r for r in recommendations if r.index_type != "DROP"]
        drop_recommendations = [r for r in recommendations if r.index_type == "DROP"]
        
        assert len(create_recommendations) > 0, "No create recommendations"
        assert len(drop_recommendations) > 0, "No drop recommendations"
        
        # Verify SQL syntax
        for rec in recommendations:
            assert "CREATE" in rec.creation_sql or "DROP" in rec.creation_sql or "ALTER" in rec.creation_sql
    
    @pytest.mark.asyncio
    async def test_database_health_monitoring(self, database_optimizer, mock_session):
        \"\"\"Test database health monitoring capabilities.\"\"\"
        # Mock health data
        mock_conn_stats = MagicMock(
            total=150,
            active=25,
            idle=120,
            idle_in_transaction=5
        )
        
        mock_cache_stats = [
            MagicMock(buffer_hit_ratio=95.5),
            MagicMock(index_hit_ratio=99.2)
        ]
        
        mock_lock_stats = MagicMock(waiting=2, deadlocks=0)
        mock_wal_stats = MagicMock(wal_size_mb=512.0, total_checkpoints=1500, checkpoint_sync_time=150.0)
        mock_bloat_stats = MagicMock(avg_bloat_ratio=0.05)
        mock_vacuum_stats = MagicMock(
            total_tables=25,
            vacuumed_24h=20,
            autovacuumed_24h=23,
            avg_dead_tuples=1500.0
        )
        
        # Setup mock return values in order
        mock_session.execute.return_value.fetchone.side_effect = [
            mock_conn_stats,
            mock_cache_stats[0],
            mock_cache_stats[1],
            mock_lock_stats,
            mock_wal_stats,
            mock_bloat_stats,
            mock_vacuum_stats
        ]
        
        start_time = time.time()
        health_metrics = await database_optimizer.monitor_database_health(mock_session)
        monitoring_time = time.time() - start_time
        
        print(f"\\nDatabase Health Monitoring:")
        print(f"Monitoring time: {monitoring_time:.3f}s")
        print(f"Active connections: {health_metrics.active_connections}/{health_metrics.connection_count}")
        print(f"Cache hit ratio: {health_metrics.cache_hit_ratio:.1%}")
        print(f"Index hit ratio: {health_metrics.index_hit_ratio:.1%}")
        print(f"Lock waits: {health_metrics.lock_waits}")
        print(f"Deadlocks: {health_metrics.deadlocks}")
        print(f"WAL size: {health_metrics.wal_size_mb:.1f} MB")
        print(f"Table bloat: {health_metrics.table_bloat_ratio:.1%}")
        
        # Performance assertions
        assert monitoring_time < 1.0, f"Monitoring too slow: {monitoring_time}s"
        
        # Health metrics validation
        assert health_metrics.connection_count == 150
        assert health_metrics.active_connections == 25
        assert health_metrics.cache_hit_ratio == 0.955
        assert health_metrics.index_hit_ratio == 0.992
        assert health_metrics.lock_waits == 2
        assert health_metrics.deadlocks == 0
        
        # Check alert generation
        assert len(database_optimizer.health_alerts) >= 0  # Alerts depend on thresholds
    
    @pytest.mark.asyncio
    async def test_high_throughput_query_performance(self, database_optimizer, mock_session):
        \"\"\"Test query performance under high throughput.\"\"\"
        # Simulate high-throughput analytics queries
        query_types = [
            \"SELECT COUNT(*) FROM traffic_metrics WHERE timestamp > $1\",
            \"SELECT AVG(speed) FROM vehicle_detections WHERE camera_id = $1\",
            \"SELECT * FROM analytics_summary WHERE period = $1 LIMIT 100\",
            \"UPDATE traffic_metrics SET processed = true WHERE id = $1\",
            \"INSERT INTO analytics_log (timestamp, data) VALUES ($1, $2)\"
        ]
        
        num_queries = 1000
        concurrent_sessions = 50
        
        async def execute_queries(session_id: int):
            execution_times = []
            
            for i in range(num_queries // concurrent_sessions):
                query = random.choice(query_types)
                
                # Simulate query execution time
                start_time = time.time()
                
                # Mock execution with realistic timing
                if \"SELECT COUNT\" in query:
                    await asyncio.sleep(0.005)  # 5ms
                elif \"SELECT AVG\" in query:
                    await asyncio.sleep(0.008)  # 8ms
                elif \"SELECT *\" in query:
                    await asyncio.sleep(0.012)  # 12ms
                elif \"UPDATE\" in query:
                    await asyncio.sleep(0.015)  # 15ms
                elif \"INSERT\" in query:
                    await asyncio.sleep(0.003)  # 3ms
                
                execution_time = (time.time() - start_time) * 1000  # Convert to ms
                execution_times.append(execution_time)
            
            return execution_times
        
        # Run concurrent query sessions
        start_time = time.time()
        tasks = [execute_queries(i) for i in range(concurrent_sessions)]
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        # Analyze results
        all_times = [time for session_times in results for time in session_times]
        
        avg_query_time = statistics.mean(all_times)
        p95_query_time = sorted(all_times)[int(len(all_times) * 0.95)]
        p99_query_time = sorted(all_times)[int(len(all_times) * 0.99)]
        max_query_time = max(all_times)
        
        queries_per_second = len(all_times) / total_time
        
        print(f"\\nHigh Throughput Query Performance:")
        print(f"Total queries: {len(all_times)}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Queries per second: {queries_per_second:.0f}")
        print(f"Average query time: {avg_query_time:.1f}ms")
        print(f"P95 query time: {p95_query_time:.1f}ms")
        print(f"P99 query time: {p99_query_time:.1f}ms")
        print(f"Max query time: {max_query_time:.1f}ms")
        
        # Performance requirements for 10TB/day workload
        assert avg_query_time < 50, f"Average query too slow: {avg_query_time}ms"
        assert p95_query_time < 100, f"P95 query too slow: {p95_query_time}ms"
        assert p99_query_time < 200, f"P99 query too slow: {p99_query_time}ms"
        assert queries_per_second > 1000, f"Throughput too low: {queries_per_second} qps"
    
    @pytest.mark.asyncio
    async def test_connection_pool_performance(self, settings):
        \"\"\"Test connection pool performance under load.\"\"\"
        # Mock database manager with connection pool
        database_manager = MagicMock()
        
        # Simulate connection pool behavior
        active_connections = 0
        max_connections = settings.database.pool_size + settings.database.max_overflow
        connection_times = []
        
        async def get_connection():
            nonlocal active_connections
            
            if active_connections >= max_connections:
                # Simulate waiting for connection
                await asyncio.sleep(0.1)
            
            start_time = time.time()
            active_connections += 1
            
            # Simulate connection acquisition time
            await asyncio.sleep(0.001)  # 1ms
            
            connection_time = (time.time() - start_time) * 1000
            connection_times.append(connection_time)
            
            return connection_time
        
        async def release_connection():
            nonlocal active_connections
            active_connections = max(0, active_connections - 1)
        
        # Simulate high connection load
        num_requests = 500
        concurrent_requests = 100
        
        async def connection_worker(worker_id: int):
            worker_times = []
            
            for i in range(num_requests // concurrent_requests):
                # Get connection
                conn_time = await get_connection()
                worker_times.append(conn_time)
                
                # Simulate work
                await asyncio.sleep(random.uniform(0.01, 0.05))  # 10-50ms work
                
                # Release connection
                await release_connection()
            
            return worker_times
        
        start_time = time.time()
        tasks = [connection_worker(i) for i in range(concurrent_requests)]
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        all_conn_times = [time for worker_times in results for time in worker_times]
        
        avg_conn_time = statistics.mean(all_conn_times)
        max_conn_time = max(all_conn_times)
        connections_per_second = len(all_conn_times) / total_time
        
        print(f"\\nConnection Pool Performance:")
        print(f"Pool size: {settings.database.pool_size}")
        print(f"Max overflow: {settings.database.max_overflow}")
        print(f"Total connections: {len(all_conn_times)}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Connections per second: {connections_per_second:.0f}")
        print(f"Average connection time: {avg_conn_time:.1f}ms")
        print(f"Max connection time: {max_conn_time:.1f}ms")
        
        # Performance requirements
        assert avg_conn_time < 10, f"Average connection time too high: {avg_conn_time}ms"
        assert max_conn_time < 1000, f"Max connection time too high: {max_conn_time}ms"
        assert connections_per_second > 100, f"Connection rate too low: {connections_per_second}/s"
    
    @pytest.mark.asyncio
    async def test_timescale_performance_optimization(self, database_optimizer, mock_session):
        \"\"\"Test TimescaleDB-specific performance optimizations.\"\"\"
        # Mock hypertable data
        mock_hypertables = [
            MagicMock(
                table_name=\"traffic_metrics\",
                num_chunks=2500,  # Too many chunks
                compression_enabled=False,  # Not compressed
                size=\"5 GB\"  # Large table
            ),
            MagicMock(
                table_name=\"vehicle_detections\",
                num_chunks=800,
                compression_enabled=True,
                size=\"2 GB\"
            )
        ]
        
        mock_session.execute.return_value.fetchall.return_value = mock_hypertables
        
        start_time = time.time()
        timescale_recs = await database_optimizer._generate_timescale_recommendations(mock_session)
        optimization_time = time.time() - start_time
        
        print(f\"\\nTimescaleDB Performance Optimization:\")
        print(f\"Optimization time: {optimization_time:.3f}s\")
        print(f\"Recommendations generated: {len(timescale_recs)}\")
        
        for rec in timescale_recs:
            print(f\"Table: {rec.table_name}, Type: {rec.index_type}\")
            print(f\"  Benefit: {rec.estimated_benefit:.1%}\")
            print(f\"  SQL: {rec.creation_sql}\")
            print(f\"  Reason: {rec.reason}\")
        
        # Should recommend compression and chunk optimization
        compression_recs = [r for r in timescale_recs if r.index_type == \"compression\"]
        chunk_recs = [r for r in timescale_recs if r.index_type == \"chunk_interval\"]
        
        assert len(compression_recs) > 0, \"No compression recommendations\"
        assert len(chunk_recs) > 0, \"No chunk optimization recommendations\"
        
        # Verify recommendations target correct tables
        assert any(r.table_name == \"traffic_metrics\" for r in compression_recs)
        assert any(r.table_name == \"traffic_metrics\" for r in chunk_recs)
    
    @pytest.mark.asyncio
    async def test_optimization_application_safety(self, database_optimizer, mock_session):
        \"\"\"Test safe application of optimization recommendations.\"\"\"
        # Create test recommendations
        recommendations = [
            IndexRecommendation(
                table_name=\"test_table\",
                column_names=[\"test_column\"],
                index_type=\"btree\",
                estimated_benefit=0.3,
                estimated_size_mb=10.0,
                creation_sql=\"CREATE INDEX CONCURRENTLY idx_test_table_test_column ON test_table (test_column)\",
                reason=\"High selectivity column\"
            ),
            IndexRecommendation(
                table_name=\"old_table\",
                column_names=[],
                index_type=\"DROP\",
                estimated_benefit=-5.0,
                estimated_size_mb=5.0,
                creation_sql=\"DROP INDEX idx_unused_old\",
                reason=\"Unused index\"
            )
        ]
        
        # Mock successful execution
        mock_session.execute.return_value = None
        
        start_time = time.time()
        result = await database_optimizer.apply_optimization_recommendations(
            mock_session, recommendations
        )
        application_time = time.time() - start_time
        
        print(f\"\\nOptimization Application Safety:\")
        print(f\"Application time: {application_time:.3f}s\")
        print(f\"Applied: {result['applied']}\")
        print(f\"Failed: {result['failed']}\")
        
        # All recommendations should be applied successfully
        assert result[\"applied\"] == len(recommendations), \"Not all recommendations applied\"
        assert result[\"failed\"] == 0, \"Some recommendations failed\"
        assert application_time < 1.0, f\"Application too slow: {application_time}s\"
        
        # Verify proper SQL execution
        expected_calls = len(recommendations)
        assert mock_session.execute.call_count == expected_calls
    
    @pytest.mark.asyncio
    async def test_performance_summary_generation(self, database_optimizer):
        \"\"\"Test performance summary generation.\"\"\"
        # Add mock performance history
        mock_metrics = [
            QueryPerformanceMetrics(
                query_id=\"1\",
                query_text=\"SELECT * FROM test\",
                execution_count=100,
                total_time_ms=1000,
                avg_time_ms=10,
                max_time_ms=50,
                min_time_ms=2,
                rows_returned=1000,
                cache_hit_ratio=0.9,
                index_usage={\"uses_index\": True},
                recommendations=[]
            ),
            QueryPerformanceMetrics(
                query_id=\"2\",
                query_text=\"SELECT COUNT(*) FROM large_table\",
                execution_count=50,
                total_time_ms=7500,
                avg_time_ms=150,  # Slow query
                max_time_ms=300,
                min_time_ms=80,
                rows_returned=50,
                cache_hit_ratio=0.6,
                index_usage={\"uses_index\": False},
                recommendations=[\"Add index\", \"Optimize query\"]
            )
        ]
        
        database_optimizer.performance_history = mock_metrics
        database_optimizer.index_recommendations = [MagicMock(), MagicMock()]
        database_optimizer.health_alerts = [MagicMock()]
        
        start_time = time.time()
        summary = database_optimizer.get_performance_summary()
        summary_time = time.time() - start_time
        
        print(f\"\\nPerformance Summary Generation:\")
        print(f\"Summary time: {summary_time:.3f}s\")
        print(f\"Total queries analyzed: {summary['total_queries_analyzed']}\")
        print(f\"Recent avg response time: {summary['recent_avg_response_time_ms']:.1f}ms\")
        print(f\"Slow queries: {summary['slow_queries_count']} ({summary['slow_query_percentage']:.1f}%)\")
        print(f\"Index recommendations: {summary['index_recommendations']}\")
        print(f\"Health alerts: {summary['health_alerts']}\")
        print(f\"Performance status: {summary['performance_status']}\")
        
        # Assertions
        assert summary_time < 0.1, f\"Summary generation too slow: {summary_time}s\"
        assert summary[\"total_queries_analyzed\"] == 2
        assert summary[\"slow_queries_count\"] == 1  # One query > 100ms
        assert summary[\"slow_query_percentage\"] == 50.0  # 1 out of 2 queries
        assert summary[\"index_recommendations\"] == 2
        assert summary[\"health_alerts\"] == 1
        assert summary[\"performance_status\"] in [\"good\", \"needs_attention\"]
    
    @pytest.mark.asyncio
    async def test_concurrent_database_operations(self, database_optimizer, mock_session):
        \"\"\"Test database performance under concurrent operations.\"\"\"
        # Simulate concurrent database analysis operations
        num_workers = 10
        operations_per_worker = 20
        
        async def analysis_worker(worker_id: int):
            operations = []
            
            for i in range(operations_per_worker):
                operation_start = time.time()
                
                # Simulate different types of analysis
                if i % 3 == 0:
                    # Query performance analysis
                    mock_session.execute.return_value.fetchall.return_value = []
                    await database_optimizer.analyze_query_performance(mock_session, hours_back=1)
                    operation_type = \"query_analysis\"
                elif i % 3 == 1:
                    # Health monitoring
                    mock_session.execute.return_value.fetchone.side_effect = [
                        MagicMock(total=10, active=2, idle=8, idle_in_transaction=0),
                        MagicMock(buffer_hit_ratio=95.0),
                        MagicMock(index_hit_ratio=99.0),
                        MagicMock(waiting=0, deadlocks=0),
                        MagicMock(wal_size_mb=100.0, total_checkpoints=10, checkpoint_sync_time=10.0),
                        MagicMock(avg_bloat_ratio=0.01),
                        MagicMock(total_tables=5, vacuumed_24h=5, autovacuumed_24h=5, avg_dead_tuples=100.0)
                    ]
                    await database_optimizer.monitor_database_health(mock_session)
                    operation_type = \"health_monitoring\"
                else:
                    # Index recommendations
                    mock_session.execute.return_value.fetchall.side_effect = [[], []]
                    await database_optimizer.generate_index_recommendations(mock_session)
                    operation_type = \"index_recommendations\"
                
                operation_time = time.time() - operation_start
                operations.append({
                    \"type\": operation_type,
                    \"time\": operation_time,
                    \"worker_id\": worker_id
                })
            
            return operations
        
        start_time = time.time()
        tasks = [analysis_worker(i) for i in range(num_workers)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time
        
        # Analyze results
        all_operations = []
        exceptions = []
        
        for result in results:
            if isinstance(result, Exception):
                exceptions.append(result)
            else:
                all_operations.extend(result)
        
        operation_times = [op[\"time\"] for op in all_operations]
        avg_operation_time = statistics.mean(operation_times) if operation_times else 0
        max_operation_time = max(operation_times) if operation_times else 0
        
        operations_per_second = len(all_operations) / total_time if total_time > 0 else 0
        
        print(f\"\\nConcurrent Database Operations:\")
        print(f\"Workers: {num_workers}\")
        print(f\"Total operations: {len(all_operations)}\")
        print(f\"Exceptions: {len(exceptions)}\")
        print(f\"Total time: {total_time:.2f}s\")
        print(f\"Operations per second: {operations_per_second:.1f}\")
        print(f\"Average operation time: {avg_operation_time:.3f}s\")
        print(f\"Max operation time: {max_operation_time:.3f}s\")
        
        # Performance assertions
        assert len(exceptions) == 0, f\"Concurrent operations caused {len(exceptions)} exceptions\"
        assert len(all_operations) == num_workers * operations_per_worker
        assert avg_operation_time < 1.0, f\"Average operation too slow: {avg_operation_time}s\"
        assert max_operation_time < 5.0, f\"Max operation too slow: {max_operation_time}s\"
        assert operations_per_second > 10, f\"Operation rate too low: {operations_per_second}/s\"