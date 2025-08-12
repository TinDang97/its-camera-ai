"""Database manager for CLI backend integration.

Provides database connection management, query execution, and transaction
handling for CLI operations that need direct database access.
"""

import asyncio
from contextlib import asynccontextmanager
from typing import Any

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from ...core.config import Settings, get_settings
from ...core.exceptions import DatabaseError
from ...core.logging import get_logger
from ...models.database import DatabaseManager as CoreDatabaseManager

logger = get_logger(__name__)


class CLIDatabaseManager:
    """Database manager specifically designed for CLI operations.
    
    Features:
    - Connection pooling and management
    - Transaction handling with rollback support
    - Query execution with proper error handling
    - Performance monitoring for CLI operations
    - Connection health checking
    """

    def __init__(self, settings: Settings = None):
        """Initialize CLI database manager.
        
        Args:
            settings: Application settings
        """
        self.settings = settings or get_settings()
        self._db_manager: CoreDatabaseManager | None = None
        self._initialized = False

        logger.info("CLI database manager initialized")

    async def initialize(self) -> None:
        """Initialize database connection."""
        if self._initialized:
            return

        try:
            self._db_manager = CoreDatabaseManager(self.settings)
            await self._db_manager.initialize()
            self._initialized = True

            logger.info("Database manager initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize database manager: {e}")
            raise DatabaseError(f"Database initialization failed: {e}")

    async def close(self) -> None:
        """Close database connections."""
        if self._db_manager:
            await self._db_manager.close()
            self._db_manager = None
            self._initialized = False
            logger.info("Database manager closed")

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    @asynccontextmanager
    async def get_session(self):
        """Get database session context manager.
        
        Yields:
            AsyncSession: Database session
        """
        if not self._initialized:
            await self.initialize()

        if not self._db_manager:
            raise DatabaseError("Database manager not initialized")

        async with self._db_manager.get_session() as session:
            yield session

    async def execute_query(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
        fetch_all: bool = True
    ) -> list[dict[str, Any]] | dict[str, Any] | int:
        """Execute raw SQL query.
        
        Args:
            query: SQL query string
            parameters: Query parameters
            fetch_all: Whether to fetch all results or just one
            
        Returns:
            Query results
        """
        try:
            async with self.get_session() as session:
                result = await session.execute(text(query), parameters or {})

                if query.strip().upper().startswith(('SELECT', 'WITH')):
                    if fetch_all:
                        rows = result.fetchall()
                        return [dict(row._mapping) for row in rows]
                    else:
                        row = result.fetchone()
                        return dict(row._mapping) if row else None
                else:
                    # For INSERT, UPDATE, DELETE operations
                    await session.commit()
                    return result.rowcount

        except SQLAlchemyError as e:
            logger.error(f"Database query failed: {e}")
            raise DatabaseError(f"Query execution failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during query execution: {e}")
            raise DatabaseError(f"Unexpected database error: {e}")

    async def get_table_info(self, table_name: str) -> dict[str, Any]:
        """Get information about a database table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            Dictionary with table information
        """
        queries = {
            "columns": """
                SELECT 
                    column_name,
                    data_type,
                    is_nullable,
                    column_default,
                    character_maximum_length
                FROM information_schema.columns 
                WHERE table_name = :table_name
                ORDER BY ordinal_position
            """,
            "row_count": f"SELECT COUNT(*) as count FROM {table_name}",
            "table_size": """
                SELECT 
                    pg_size_pretty(pg_total_relation_size(:table_name)) as total_size,
                    pg_size_pretty(pg_relation_size(:table_name)) as table_size,
                    pg_size_pretty(pg_total_relation_size(:table_name) - pg_relation_size(:table_name)) as index_size
            """
        }

        try:
            info = {}

            # Get column information
            info["columns"] = await self.execute_query(
                queries["columns"],
                {"table_name": table_name}
            )

            # Get row count
            count_result = await self.execute_query(
                queries["row_count"],
                fetch_all=False
            )
            info["row_count"] = count_result["count"] if count_result else 0

            # Get table size information
            size_result = await self.execute_query(
                queries["table_size"],
                {"table_name": table_name},
                fetch_all=False
            )
            info["size_info"] = size_result if size_result else {}

            return info

        except Exception as e:
            logger.error(f"Failed to get table info for {table_name}: {e}")
            raise DatabaseError(f"Failed to get table information: {e}")

    async def get_database_stats(self) -> dict[str, Any]:
        """Get comprehensive database statistics.
        
        Returns:
            Dictionary with database statistics
        """
        queries = {
            "database_size": """
                SELECT pg_size_pretty(pg_database_size(current_database())) as size
            """,
            "connection_info": """
                SELECT 
                    count(*) as total_connections,
                    count(*) FILTER (WHERE state = 'active') as active_connections,
                    count(*) FILTER (WHERE state = 'idle') as idle_connections
                FROM pg_stat_activity
                WHERE datname = current_database()
            """,
            "table_stats": """
                SELECT 
                    schemaname,
                    tablename,
                    n_tup_ins as inserts,
                    n_tup_upd as updates,
                    n_tup_del as deletes,
                    n_live_tup as live_tuples,
                    n_dead_tup as dead_tuples
                FROM pg_stat_user_tables
                ORDER BY n_live_tup DESC
            """,
            "index_usage": """
                SELECT 
                    schemaname,
                    tablename,
                    indexname,
                    idx_tup_read as index_reads,
                    idx_tup_fetch as index_fetches
                FROM pg_stat_user_indexes
                WHERE idx_tup_read > 0
                ORDER BY idx_tup_read DESC
                LIMIT 10
            """
        }

        try:
            stats = {}

            # Database size
            size_result = await self.execute_query(
                queries["database_size"],
                fetch_all=False
            )
            stats["database_size"] = size_result["size"] if size_result else "Unknown"

            # Connection information
            conn_result = await self.execute_query(
                queries["connection_info"],
                fetch_all=False
            )
            stats["connections"] = conn_result if conn_result else {}

            # Table statistics
            stats["tables"] = await self.execute_query(queries["table_stats"])

            # Index usage
            stats["indexes"] = await self.execute_query(queries["index_usage"])

            return stats

        except Exception as e:
            logger.error(f"Failed to get database statistics: {e}")
            raise DatabaseError(f"Failed to get database statistics: {e}")

    async def check_connectivity(self) -> dict[str, Any]:
        """Check database connectivity and performance.
        
        Returns:
            Dictionary with connectivity information
        """
        try:
            import time
            start_time = time.time()

            # Simple connectivity test
            result = await self.execute_query(
                "SELECT 1 as test, current_timestamp as timestamp, version() as version",
                fetch_all=False
            )

            end_time = time.time()
            response_time = (end_time - start_time) * 1000  # Convert to milliseconds

            return {
                "connected": True,
                "response_time_ms": round(response_time, 2),
                "database_time": result["timestamp"] if result else None,
                "version": result["version"] if result else None,
                "status": "healthy" if response_time < 100 else "slow"
            }

        except Exception as e:
            logger.error(f"Database connectivity check failed: {e}")
            return {
                "connected": False,
                "error": str(e),
                "status": "unhealthy"
            }

    async def cleanup_old_data(
        self,
        table_name: str,
        timestamp_column: str,
        retention_days: int,
        batch_size: int = 1000
    ) -> int:
        """Clean up old data from a table.
        
        Args:
            table_name: Name of the table to clean
            timestamp_column: Column containing timestamps
            retention_days: Number of days to retain
            batch_size: Number of rows to delete per batch
            
        Returns:
            Number of rows deleted
        """
        try:
            # First, get count of rows to delete
            count_query = f"""
                SELECT COUNT(*) as count 
                FROM {table_name} 
                WHERE {timestamp_column} < NOW() - INTERVAL '{retention_days} days'
            """

            count_result = await self.execute_query(count_query, fetch_all=False)
            total_to_delete = count_result["count"] if count_result else 0

            if total_to_delete == 0:
                logger.info(f"No old data to clean in {table_name}")
                return 0

            logger.info(f"Cleaning {total_to_delete} old records from {table_name}")

            # Delete in batches to avoid locking issues
            deleted_total = 0

            while deleted_total < total_to_delete:
                delete_query = f"""
                    DELETE FROM {table_name} 
                    WHERE {timestamp_column} < NOW() - INTERVAL '{retention_days} days'
                    AND ctid IN (
                        SELECT ctid FROM {table_name} 
                        WHERE {timestamp_column} < NOW() - INTERVAL '{retention_days} days'
                        LIMIT {batch_size}
                    )
                """

                deleted_count = await self.execute_query(delete_query)

                if deleted_count == 0:
                    break  # No more rows to delete

                deleted_total += deleted_count
                logger.debug(f"Deleted {deleted_count} rows, total: {deleted_total}")

                # Small delay between batches
                await asyncio.sleep(0.1)

            logger.info(f"Cleaned {deleted_total} old records from {table_name}")
            return deleted_total

        except Exception as e:
            logger.error(f"Failed to cleanup old data from {table_name}: {e}")
            raise DatabaseError(f"Data cleanup failed: {e}")

    async def vacuum_analyze(self, table_name: str = None) -> dict[str, Any]:
        """Run VACUUM ANALYZE on database tables.
        
        Args:
            table_name: Specific table name, or None for all tables
            
        Returns:
            Operation results
        """
        try:
            import time
            start_time = time.time()

            if table_name:
                query = f"VACUUM ANALYZE {table_name}"
                operation = f"table {table_name}"
            else:
                query = "VACUUM ANALYZE"
                operation = "all tables"

            logger.info(f"Running VACUUM ANALYZE on {operation}")

            await self.execute_query(query)

            end_time = time.time()
            duration = end_time - start_time

            result = {
                "success": True,
                "operation": operation,
                "duration_seconds": round(duration, 2),
                "message": f"VACUUM ANALYZE completed for {operation}"
            }

            logger.info(f"VACUUM ANALYZE completed in {duration:.2f}s for {operation}")
            return result

        except Exception as e:
            logger.error(f"VACUUM ANALYZE failed for {operation}: {e}")
            return {
                "success": False,
                "operation": operation,
                "error": str(e),
                "message": f"VACUUM ANALYZE failed for {operation}"
            }

    async def get_slow_queries(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get information about slow queries.
        
        Args:
            limit: Maximum number of queries to return
            
        Returns:
            List of slow query information
        """
        # Note: This requires pg_stat_statements extension
        query = f"""
            SELECT 
                query,
                calls,
                total_time,
                mean_time,
                rows,
                100.0 * shared_blks_hit / nullif(shared_blks_hit + shared_blks_read, 0) AS hit_percent
            FROM pg_stat_statements 
            ORDER BY mean_time DESC 
            LIMIT {limit}
        """

        try:
            results = await self.execute_query(query)
            return results

        except Exception as e:
            # pg_stat_statements might not be enabled
            logger.debug(f"Could not get slow queries (pg_stat_statements may not be enabled): {e}")
            return []

    def is_connected(self) -> bool:
        """Check if database manager is connected.
        
        Returns:
            True if connected and initialized
        """
        return self._initialized and self._db_manager is not None
