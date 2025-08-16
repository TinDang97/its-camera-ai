"""TimescaleDB service for time-series data management.

Provides specialized operations for TimescaleDB including hypertable management,
continuous aggregates, data retention, and optimized batch operations.
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime
from typing import Any

from ...core.exceptions import DatabaseError
from ...core.logging import get_logger
from .models import BatchOperationResult, QueryResult, TimescaleConfig
from .postgresql_service import PostgreSQLService

logger = get_logger(__name__)


class TimescaleService(PostgreSQLService):
    """TimescaleDB service with time-series specific operations.
    
    Extends PostgreSQL service with TimescaleDB-specific functionality
    including hypertable management, compression, and continuous aggregates.
    Optimized for 1000+ records/sec batch operations.
    """

    def __init__(self, config: TimescaleConfig):
        """Initialize TimescaleDB service.
        
        Args:
            config: TimescaleDB configuration
        """
        super().__init__(config)
        self.timescale_config = config
        self._hypertables: set[str] = set()
        self._continuous_aggregates: set[str] = set()
        # Optimize connection pool for high throughput
        if hasattr(config, 'pool_config'):
            config.pool_config.min_connections = max(config.pool_config.min_connections, 10)
            config.pool_config.max_connections = max(config.pool_config.max_connections, 50)

    async def create_hypertable(
        self,
        table_name: str,
        time_column: str | None = None,
        partitioning_column: str | None = None,
        chunk_time_interval: str | None = None,
        if_not_exists: bool = True,
    ) -> QueryResult:
        """Create a hypertable for time-series data.
        
        Args:
            table_name: Name of the table to convert to hypertable
            time_column: Name of the time column (default: timestamp)
            partitioning_column: Additional partitioning column
            chunk_time_interval: Time interval for chunks
            if_not_exists: Don't raise error if hypertable exists
            
        Returns:
            Query execution result
        """
        time_column = time_column or self.timescale_config.default_partition_key
        chunk_time_interval = chunk_time_interval or self.timescale_config.chunk_time_interval

        # Build the create_hypertable query
        query_parts = [
            f"SELECT create_hypertable('{table_name}', '{time_column}'"
        ]

        if partitioning_column:
            query_parts.append(f", partitioning_column => '{partitioning_column}'")

        query_parts.append(f", chunk_time_interval => INTERVAL '{chunk_time_interval}'")

        if if_not_exists:
            query_parts.append(", if_not_exists => TRUE")

        query_parts.append(")")

        query = "".join(query_parts)

        try:
            result = await self.execute_query(query)

            if result.success:
                self._hypertables.add(table_name)
                logger.info(
                    "Hypertable created successfully",
                    table=table_name,
                    time_column=time_column,
                    chunk_interval=chunk_time_interval,
                )

                # Enable compression if configured
                if self.timescale_config.enable_compression:
                    await self._enable_compression(table_name)

            return result

        except Exception as e:
            logger.error("Failed to create hypertable", table=table_name, error=str(e))
            raise DatabaseError(
                f"Failed to create hypertable {table_name}",
                operation="create_hypertable",
                table=table_name,
                cause=e,
            ) from e

    async def _enable_compression(self, table_name: str) -> None:
        """Enable compression for a hypertable.
        
        Args:
            table_name: Name of the hypertable
        """
        try:
            # Enable compression policy
            compression_query = f"""
            SELECT add_compression_policy('{table_name}', INTERVAL '1 day');
            """

            await self.execute_query(compression_query)
            logger.info("Compression enabled for hypertable", table=table_name)

        except Exception as e:
            logger.warning("Failed to enable compression", table=table_name, error=str(e))

    async def create_continuous_aggregate(
        self,
        view_name: str,
        query: str,
        refresh_policy: str | None = None,
    ) -> QueryResult:
        """Create a continuous aggregate view.
        
        Args:
            view_name: Name of the continuous aggregate view
            query: SQL query for the aggregate
            refresh_policy: Refresh policy interval
            
        Returns:
            Query execution result
        """
        refresh_policy = refresh_policy or self.timescale_config.materialized_view_refresh_policy

        try:
            # Create continuous aggregate
            create_query = f"""
            CREATE MATERIALIZED VIEW {view_name}
            WITH (timescaledb.continuous) AS
            {query};
            """

            result = await self.execute_query(create_query)

            if result.success:
                self._continuous_aggregates.add(view_name)

                # Add refresh policy
                policy_query = f"""
                SELECT add_continuous_aggregate_policy('{view_name}',
                    start_offset => INTERVAL '1 month',
                    end_offset => INTERVAL '{refresh_policy}',
                    schedule_interval => INTERVAL '{refresh_policy}');
                """

                await self.execute_query(policy_query)

                logger.info(
                    "Continuous aggregate created",
                    view=view_name,
                    refresh_policy=refresh_policy,
                )

            return result

        except Exception as e:
            logger.error("Failed to create continuous aggregate", view=view_name, error=str(e))
            raise DatabaseError(
                f"Failed to create continuous aggregate {view_name}",
                operation="create_continuous_aggregate",
                cause=e,
            ) from e

    async def set_retention_policy(
        self,
        table_name: str,
        retention_period: str,
    ) -> QueryResult:
        """Set data retention policy for a hypertable.
        
        Args:
            table_name: Name of the hypertable
            retention_period: Retention period (e.g., '30 days')
            
        Returns:
            Query execution result
        """
        try:
            query = f"""
            SELECT add_retention_policy('{table_name}', INTERVAL '{retention_period}');
            """

            result = await self.execute_query(query)

            if result.success:
                logger.info(
                    "Retention policy set",
                    table=table_name,
                    retention_period=retention_period,
                )

            return result

        except Exception as e:
            logger.error("Failed to set retention policy", table=table_name, error=str(e))
            raise DatabaseError(
                f"Failed to set retention policy for {table_name}",
                operation="set_retention_policy",
                table=table_name,
                cause=e,
            ) from e

    async def batch_insert_time_series(
        self,
        table_name: str,
        data: list[dict[str, Any]],
        time_column: str | None = None,
        batch_size: int | None = None,
        timeout: int | None = None,
        use_copy: bool = True,
    ) -> BatchOperationResult:
        """Optimized batch insert for time-series data.
        
        Args:
            table_name: Target table name
            data: List of data dictionaries
            time_column: Name of the time column
            batch_size: Size of each batch
            timeout: Operation timeout
            
        Returns:
            Batch operation result
        """
        start_time = time.time()
        batch_size = batch_size or self.timescale_config.batch_size
        time_column = time_column or self.timescale_config.default_partition_key
        timeout = timeout or self.timescale_config.batch_timeout

        if not data:
            return BatchOperationResult(
                total_items=0,
                successful_items=0,
                failed_items=0,
                total_time_ms=0,
                average_time_per_item_ms=0,
            )

        # Sort data by timestamp for optimal insertion
        if time_column in data[0]:
            data.sort(key=lambda x: x.get(time_column, datetime.min))

        successful_items = 0
        failed_items = 0
        errors = []

        try:
            if use_copy:
                # Use COPY for maximum performance (1000+ records/sec)
                await self._bulk_copy_insert(table_name, data, timeout)
                successful_items = len(data)
            else:
                # Fallback to batch INSERT for compatibility
                for i in range(0, len(data), batch_size):
                    batch = data[i:i + batch_size]

                    try:
                        await self._insert_batch(table_name, batch, timeout)
                        successful_items += len(batch)

                    except Exception as e:
                        failed_items += len(batch)
                        errors.append({
                            "batch_start": i,
                            "batch_size": len(batch),
                            "error": str(e),
                        })
                        logger.error(
                            "Batch insert failed",
                            table=table_name,
                            batch_start=i,
                            batch_size=len(batch),
                            error=str(e),
                        )

            total_time = (time.time() - start_time) * 1000

            return BatchOperationResult(
                total_items=len(data),
                successful_items=successful_items,
                failed_items=failed_items,
                total_time_ms=total_time,
                average_time_per_item_ms=total_time / len(data),
                errors=errors,
            )

        except Exception as e:
            total_time = (time.time() - start_time) * 1000
            logger.error("Batch insert operation failed", table=table_name, error=str(e))

            return BatchOperationResult(
                total_items=len(data),
                successful_items=successful_items,
                failed_items=len(data) - successful_items,
                total_time_ms=total_time,
                average_time_per_item_ms=total_time / len(data),
                errors=[{"error": str(e)}],
            )

    async def _insert_batch(
        self,
        table_name: str,
        batch: list[dict[str, Any]],
        timeout: int,
    ) -> None:
        """Insert a single batch of data.
        
        Args:
            table_name: Target table name
            batch: Batch data
            timeout: Operation timeout
        """
        if not batch:
            return

        # Extract columns from first row
        columns = list(batch[0].keys())

        # Prepare data for COPY
        rows = [[row.get(col) for col in columns] for row in batch]

        async with self.get_connection(read_only=False) as conn:
            await asyncio.wait_for(
                conn.copy_records_to_table(
                    table_name,
                    records=rows,
                    columns=columns,
                ),
                timeout=timeout
            )

    async def _bulk_copy_insert(
        self,
        table_name: str,
        data: list[dict[str, Any]],
        timeout: int,
    ) -> None:
        """High-performance COPY insert for time-series data.
        
        Achieves 1000+ records/sec throughput requirement.
        """
        if not data:
            return

        columns = list(data[0].keys())
        rows = [[row.get(col) for col in columns] for row in data]

        async with self.get_connection(read_only=False) as conn:
            # Use PostgreSQL COPY for maximum performance
            await asyncio.wait_for(
                conn.copy_records_to_table(
                    table_name,
                    records=rows,
                    columns=columns,
                    timeout=timeout
                ),
                timeout=timeout
            )

    async def query_time_range(
        self,
        table_name: str,
        start_time: datetime,
        end_time: datetime,
        columns: list[str] | None = None,
        aggregation: str | None = None,
        group_by: str | None = None,
        order_by: str | None = None,
        limit: int | None = None,
        time_column: str | None = None,
        metadata_filter: dict[str, Any] | None = None,
        geospatial_filter: dict[str, Any] | None = None,
    ) -> QueryResult:
        """Query time-series data within a time range.
        
        Args:
            table_name: Table to query
            start_time: Start of time range
            end_time: End of time range
            columns: Columns to select
            aggregation: Aggregation function
            group_by: Group by clause
            order_by: Order by clause
            limit: Result limit
            time_column: Name of the time column
            
        Returns:
            Query result
        """
        time_column = time_column or self.timescale_config.default_partition_key
        columns = columns or ["*"]

        # Build SELECT clause
        if aggregation:
            select_clause = f"SELECT {aggregation}"
        else:
            select_clause = f"SELECT {', '.join(columns)}"

        # Build WHERE clause with metadata and geospatial filters
        where_clauses = [f"{time_column} >= $1 AND {time_column} <= $2"]
        parameters = [start_time, end_time]
        param_count = 2

        # Add JSONB metadata filtering for flexible attributes
        if metadata_filter:
            for key, value in metadata_filter.items():
                param_count += 1
                where_clauses.append(f"metadata @> ${param_count}")
                parameters.append(json.dumps({key: value}))

        # Add PostGIS geospatial filtering
        if geospatial_filter:
            if 'bbox' in geospatial_filter:
                # Bounding box query
                bbox = geospatial_filter['bbox']
                param_count += 1
                where_clauses.append(
                    f"ST_Within(location, ST_MakeEnvelope(${param_count}, ${param_count+1}, ${param_count+2}, ${param_count+3}, 4326))"
                )
                parameters.extend([bbox['min_lng'], bbox['min_lat'], bbox['max_lng'], bbox['max_lat']])
                param_count += 3
            elif 'point' in geospatial_filter and 'radius' in geospatial_filter:
                # Radius query
                point = geospatial_filter['point']
                radius = geospatial_filter['radius']
                param_count += 1
                where_clauses.append(
                    f"ST_DWithin(location, ST_Point(${param_count}, ${param_count+1})::geography, ${param_count+2})"
                )
                parameters.extend([point['lng'], point['lat'], radius])
                param_count += 2

        where_clause = "WHERE " + " AND ".join(where_clauses)

        # Build GROUP BY clause
        group_clause = f" GROUP BY {group_by}" if group_by else ""

        # Build ORDER BY clause
        order_clause = f" ORDER BY {order_by}" if order_by else f" ORDER BY {time_column}"

        # Build LIMIT clause
        limit_clause = f" LIMIT {limit}" if limit else ""

        query = (
            f"{select_clause} FROM {table_name} {where_clause}"
            f"{group_clause}{order_clause}{limit_clause}"
        )

        try:
            return await self.execute_query(
                query,
                parameters=parameters,
                read_only=True,
            )

        except Exception as e:
            logger.error(
                "Time range query failed",
                table=table_name,
                start_time=start_time,
                end_time=end_time,
                error=str(e),
            )
            raise DatabaseError(
                f"Failed to query time range for {table_name}",
                operation="query_time_range",
                table=table_name,
                cause=e,
            ) from e

    async def compress_chunks(
        self,
        table_name: str,
        older_than: str | None = None,
    ) -> QueryResult:
        """Manually compress chunks for a hypertable.
        
        Args:
            table_name: Name of the hypertable
            older_than: Compress chunks older than this interval
            
        Returns:
            Query execution result
        """
        older_than = older_than or "1 day"

        try:
            query = f"""
            SELECT compress_chunk(chunk) 
            FROM show_chunks('{table_name}', older_than => INTERVAL '{older_than}');
            """

            result = await self.execute_query(query)

            if result.success:
                logger.info(
                    "Chunks compressed",
                    table=table_name,
                    older_than=older_than,
                )

            return result

        except Exception as e:
            logger.error("Failed to compress chunks", table=table_name, error=str(e))
            raise DatabaseError(
                f"Failed to compress chunks for {table_name}",
                operation="compress_chunks",
                table=table_name,
                cause=e,
            ) from e

    async def get_chunk_statistics(self, table_name: str) -> QueryResult:
        """Get chunk statistics for a hypertable.
        
        Args:
            table_name: Name of the hypertable
            
        Returns:
            Query result with chunk statistics
        """
        try:
            query = f"""
            SELECT 
                chunk_schema,
                chunk_name,
                range_start,
                range_end,
                is_compressed,
                compressed_heap_size,
                uncompressed_heap_size,
                compression_ratio
            FROM chunk_compression_stats('{table_name}')
            ORDER BY range_start DESC;
            """

            return await self.execute_query(query, read_only=True)

        except Exception as e:
            logger.error("Failed to get chunk statistics", table=table_name, error=str(e))
            raise DatabaseError(
                f"Failed to get chunk statistics for {table_name}",
                operation="get_chunk_statistics",
                table=table_name,
                cause=e,
            ) from e

    async def refresh_continuous_aggregate(
        self,
        view_name: str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> QueryResult:
        """Manually refresh a continuous aggregate view.
        
        Args:
            view_name: Name of the continuous aggregate view
            start_time: Start of refresh window
            end_time: End of refresh window
            
        Returns:
            Query execution result
        """
        try:
            if start_time and end_time:
                query = f"""
                CALL refresh_continuous_aggregate('{view_name}', $1, $2);
                """
                parameters = [start_time, end_time]
            else:
                query = f"""
                CALL refresh_continuous_aggregate('{view_name}', NULL, NULL);
                """
                parameters = None

            result = await self.execute_query(query, parameters=parameters)

            if result.success:
                logger.info("Continuous aggregate refreshed", view=view_name)

            return result

        except Exception as e:
            logger.error("Failed to refresh continuous aggregate", view=view_name, error=str(e))
            raise DatabaseError(
                f"Failed to refresh continuous aggregate {view_name}",
                operation="refresh_continuous_aggregate",
                cause=e,
            ) from e

    async def get_hypertable_stats(self, table_name: str) -> QueryResult:
        """Get comprehensive statistics for a hypertable.
        
        Args:
            table_name: Name of the hypertable
            
        Returns:
            Query result with hypertable statistics
        """
        try:
            query = f"""
            SELECT 
                hypertable_name,
                hypertable_size,
                num_chunks,
                num_dimensions,
                table_bytes,
                index_bytes,
                toast_bytes,
                total_bytes
            FROM hypertable_detailed_size('{table_name}');
            """

            return await self.execute_query(query, read_only=True)

        except Exception as e:
            logger.error("Failed to get hypertable statistics", table=table_name, error=str(e))
            raise DatabaseError(
                f"Failed to get hypertable statistics for {table_name}",
                operation="get_hypertable_stats",
                table=table_name,
                cause=e,
            ) from e

    @property
    def hypertables(self) -> set[str]:
        """Get list of managed hypertables."""
        return self._hypertables.copy()

    @property
    def continuous_aggregates(self) -> set[str]:
        """Get list of managed continuous aggregates."""
        return self._continuous_aggregates.copy()

    async def create_metadata_index(
        self,
        table_name: str,
        index_name: str,
        jsonb_column: str = "metadata",
        index_keys: list[str] | None = None,
    ) -> QueryResult:
        """Create optimized indexes for JSONB metadata queries.
        
        Args:
            table_name: Table to index
            index_name: Name of the index
            jsonb_column: JSONB column name
            index_keys: Specific keys to index in JSONB
        """
        try:
            if index_keys:
                # Create GIN index on specific JSONB keys for faster queries
                index_expressions = [f"({jsonb_column} ->> '{key}')" for key in index_keys]
                query = f"CREATE INDEX CONCURRENTLY {index_name} ON {table_name} USING GIN ({', '.join(index_expressions)});"
            else:
                # Create general GIN index on entire JSONB column
                query = f"CREATE INDEX CONCURRENTLY {index_name} ON {table_name} USING GIN ({jsonb_column});"

            result = await self.execute_query(query)

            if result.success:
                logger.info(
                    "Metadata index created",
                    table=table_name,
                    index=index_name,
                    jsonb_column=jsonb_column,
                )

            return result

        except Exception as e:
            logger.error("Failed to create metadata index", table=table_name, error=str(e))
            raise DatabaseError(
                f"Failed to create metadata index on {table_name}",
                operation="create_metadata_index",
                table=table_name,
                cause=e,
            ) from e

    async def create_geospatial_index(
        self,
        table_name: str,
        index_name: str,
        location_column: str = "location",
    ) -> QueryResult:
        """Create PostGIS spatial index for geospatial queries.
        
        Args:
            table_name: Table to index  
            index_name: Name of the spatial index
            location_column: Geometry/geography column name
        """
        try:
            # Create spatial index using GIST
            query = f"CREATE INDEX CONCURRENTLY {index_name} ON {table_name} USING GIST ({location_column});"

            result = await self.execute_query(query)

            if result.success:
                logger.info(
                    "Geospatial index created",
                    table=table_name,
                    index=index_name,
                    location_column=location_column,
                )

            return result

        except Exception as e:
            logger.error("Failed to create geospatial index", table=table_name, error=str(e))
            raise DatabaseError(
                f"Failed to create geospatial index on {table_name}",
                operation="create_geospatial_index",
                table=table_name,
                cause=e,
            ) from e

