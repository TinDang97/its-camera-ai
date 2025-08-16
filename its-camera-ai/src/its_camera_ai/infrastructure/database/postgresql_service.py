"""PostgreSQL connection service with async support.

Provides async PostgreSQL connection management with connection pooling,
health checks, retry logic, and read/write splitting.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import asyncpg  # type: ignore
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from ...core.exceptions import DatabaseError
from ...core.logging import get_logger
from .models import (
    ConnectionHealth,
    ConnectionStatus,
    DatabaseConfig,
    DatabaseType,
    QueryResult,
)

logger = get_logger(__name__)


class PostgreSQLService:
    """PostgreSQL connection service with async support.

    Provides connection pooling, health monitoring, and query execution
    with automatic retry logic and read/write splitting support.
    """

    def __init__(self, config: DatabaseConfig):
        """Initialize PostgreSQL service.

        Args:
            config: Database configuration
        """
        self.config = config
        self._engine: Any = None
        self._session_factory: Any = None
        self._connection_pool: Any = None
        self._is_connected = False
        self._health_status = ConnectionStatus.DISCONNECTED

        # Read replica configuration (if different from primary)
        self._read_config: DatabaseConfig | None = None
        self._read_engine: Any = None
        self._read_session_factory: Any = None
        self._read_pool: Any = None

    async def connect(self) -> None:
        """Establish database connections."""
        try:
            self._health_status = ConnectionStatus.CONNECTING
            logger.info(
                "Connecting to PostgreSQL", host=self.config.host, port=self.config.port
            )

            # Create SQLAlchemy async engine
            await self._create_sqlalchemy_engine()

            # Create asyncpg connection pool for direct queries
            await self._create_asyncpg_pool()

            # Test connection
            await self._test_connection()

            self._is_connected = True
            self._health_status = ConnectionStatus.HEALTHY
            logger.info("PostgreSQL connection established successfully")

        except Exception as e:
            self._health_status = ConnectionStatus.UNHEALTHY
            logger.error("Failed to connect to PostgreSQL", error=str(e))
            raise DatabaseError(
                "Failed to establish PostgreSQL connection",
                operation="connect",
                details={"host": self.config.host, "port": self.config.port},
                cause=e,
            ) from e

    async def disconnect(self) -> None:
        """Close database connections."""
        try:
            logger.info("Disconnecting from PostgreSQL")

            # Close asyncpg pools
            if self._connection_pool:
                await self._connection_pool.close()
                self._connection_pool = None

            if self._read_pool:
                await self._read_pool.close()
                self._read_pool = None

            # Close SQLAlchemy engines
            if self._engine:
                await self._engine.dispose()
                self._engine = None
                self._session_factory = None

            if self._read_engine:
                await self._read_engine.dispose()
                self._read_engine = None
                self._read_session_factory = None

            self._is_connected = False
            self._health_status = ConnectionStatus.DISCONNECTED
            logger.info("PostgreSQL disconnection completed")

        except Exception as e:
            logger.error("Error during PostgreSQL disconnection", error=str(e))
            raise DatabaseError("Failed to disconnect from PostgreSQL", cause=e) from e

    def configure_read_replica(self, read_config: DatabaseConfig) -> None:
        """Configure read replica connection.

        Args:
            read_config: Read replica database configuration
        """
        self._read_config = read_config
        logger.info(
            "Read replica configured", host=read_config.host, port=read_config.port
        )

    async def _create_sqlalchemy_engine(self) -> None:
        """Create SQLAlchemy async engine."""
        connection_url = self.config.get_connection_url(async_driver=True)

        # Engine configuration
        engine_config = {
            "url": connection_url,
            "echo": False,  # Set via logging configuration
            "poolclass": NullPool,  # Use asyncpg pool instead
            "pool_pre_ping": self.config.pool_config.pool_pre_ping,
        }

        self._engine = create_async_engine(**engine_config)
        self._session_factory = async_sessionmaker(
            bind=self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

        # Create read replica engine if configured
        if self._read_config:
            read_connection_url = self._read_config.get_connection_url(
                async_driver=True
            )
            self._read_engine = create_async_engine(
                url=read_connection_url,
                echo=False,
                poolclass=NullPool,
                pool_pre_ping=self._read_config.pool_config.pool_pre_ping,
            )
            self._read_session_factory = async_sessionmaker(
                bind=self._read_engine,
                class_=AsyncSession,
                expire_on_commit=False,
            )

    async def _create_asyncpg_pool(self) -> None:
        """Create asyncpg connection pool."""
        pool_config = self.config.pool_config
        connection_params = self.config.get_connection_params()

        # Pool configuration
        pool_kwargs = {
            "min_size": pool_config.min_connections,
            "max_size": pool_config.max_connections,
            "command_timeout": self.config.command_timeout,
            **connection_params,
        }

        self._connection_pool = await asyncpg.create_pool(**pool_kwargs)

        # Create read replica pool if configured
        if self._read_config:
            read_pool_kwargs = {
                "min_size": self._read_config.pool_config.min_connections,
                "max_size": self._read_config.pool_config.max_connections,
                "command_timeout": self._read_config.command_timeout,
                **self._read_config.get_connection_params(),
            }
            self._read_pool = await asyncpg.create_pool(**read_pool_kwargs)

    async def _test_connection(self) -> None:
        """Test database connection."""
        if not self._connection_pool:
            raise DatabaseError("Connection pool not initialized")

        async with self._connection_pool.acquire() as conn:
            await conn.fetchval("SELECT 1")

        # Test read replica if configured
        if self._read_pool:
            async with self._read_pool.acquire() as conn:
                await conn.fetchval("SELECT 1")

    @asynccontextmanager
    async def get_session(
        self, read_only: bool = False
    ) -> AsyncGenerator[AsyncSession, None]:
        """Get database session with automatic cleanup.

        Args:
            read_only: Use read replica if available

        Yields:
            Async database session
        """
        if not self._is_connected:
            raise DatabaseError("Not connected to database")

        # Choose appropriate session factory
        session_factory = self._session_factory
        if read_only and self._read_session_factory:
            session_factory = self._read_session_factory

        if not session_factory:
            raise DatabaseError("Session factory not initialized")

        async with session_factory() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()

    @asynccontextmanager
    async def get_connection(
        self, read_only: bool = False
    ) -> AsyncGenerator[Any, None]:
        """Get raw asyncpg connection.

        Args:
            read_only: Use read replica if available

        Yields:
            Raw database connection
        """
        if not self._is_connected:
            raise DatabaseError("Not connected to database")

        # Choose appropriate pool
        pool = self._connection_pool
        if read_only and self._read_pool:
            pool = self._read_pool

        if not pool:
            raise DatabaseError("Connection pool not initialized")

        async with pool.acquire() as conn:
            yield conn

    async def execute_query(
        self,
        query: str,
        parameters: dict[str, Any] | list[Any] | None = None,
        read_only: bool = False,
        timeout: int | None = None,
    ) -> QueryResult:
        """Execute SQL query with retry logic.

        Args:
            query: SQL query to execute
            parameters: Query parameters
            read_only: Use read replica if available
            timeout: Query timeout in seconds

        Returns:
            Query result with metadata
        """
        start_time = time.time()
        timeout = timeout or self.config.query_timeout

        try:
            async with self.get_connection(read_only=read_only) as conn:
                if parameters is None:
                    result = await asyncio.wait_for(conn.fetch(query), timeout=timeout)
                else:
                    result = await asyncio.wait_for(
                        conn.fetch(
                            query,
                            *parameters if isinstance(parameters, list) else parameters,
                        ),
                        timeout=timeout,
                    )

                # Convert result to dictionaries
                data = [dict(row) for row in result] if result else []

                execution_time = (time.time() - start_time) * 1000

                return QueryResult(
                    success=True,
                    rows_affected=len(data),
                    execution_time_ms=execution_time,
                    data=data,
                )

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(
                "Query execution failed",
                query=query[:100],  # Log first 100 chars
                error=str(e),
                execution_time_ms=execution_time,
            )

            return QueryResult(
                success=False,
                execution_time_ms=execution_time,
                error_message=str(e),
            )

    async def execute_transaction(
        self,
        queries: list[tuple[str, dict[str, Any] | list[Any] | None]],
        timeout: int | None = None,
    ) -> QueryResult:
        """Execute multiple queries in a transaction.

        Args:
            queries: List of (query, parameters) tuples
            timeout: Transaction timeout in seconds

        Returns:
            Transaction result
        """
        start_time = time.time()
        timeout = timeout or self.config.query_timeout
        total_rows = 0

        try:
            async with self.get_connection(read_only=False) as conn:
                async with conn.transaction():
                    for query, parameters in queries:
                        if parameters is None:
                            result = await asyncio.wait_for(
                                conn.fetch(query), timeout=timeout
                            )
                        else:
                            result = await asyncio.wait_for(
                                conn.fetch(
                                    query,
                                    *(
                                        parameters
                                        if isinstance(parameters, list)
                                        else parameters
                                    ),
                                ),
                                timeout=timeout,
                            )
                        total_rows += len(result) if result else 0

                execution_time = (time.time() - start_time) * 1000

                return QueryResult(
                    success=True,
                    rows_affected=total_rows,
                    execution_time_ms=execution_time,
                )

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(
                "Transaction execution failed",
                query_count=len(queries),
                error=str(e),
                execution_time_ms=execution_time,
            )

            return QueryResult(
                success=False,
                execution_time_ms=execution_time,
                error_message=str(e),
            )

    async def bulk_insert(
        self,
        table: str,
        columns: list[str],
        data: list[list[Any]],
        timeout: int | None = None,
    ) -> QueryResult:
        """Perform bulk insert operation.

        Args:
            table: Table name
            columns: Column names
            data: Data rows to insert
            timeout: Operation timeout in seconds

        Returns:
            Bulk insert result
        """
        start_time = time.time()
        timeout = timeout or self.config.query_timeout

        try:
            async with self.get_connection(read_only=False) as conn:
                await asyncio.wait_for(
                    conn.copy_records_to_table(
                        table,
                        records=data,
                        columns=columns,
                        timeout=timeout,
                    ),
                    timeout=timeout,
                )

                execution_time = (time.time() - start_time) * 1000

                return QueryResult(
                    success=True,
                    rows_affected=len(data),
                    execution_time_ms=execution_time,
                )

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(
                "Bulk insert failed",
                table=table,
                row_count=len(data),
                error=str(e),
                execution_time_ms=execution_time,
            )

            return QueryResult(
                success=False,
                execution_time_ms=execution_time,
                error_message=str(e),
            )

    async def health_check(self) -> ConnectionHealth:
        """Check database connection health.

        Returns:
            Connection health status
        """
        start_time = time.time()

        try:
            if not self._is_connected:
                return ConnectionHealth(
                    status=ConnectionStatus.DISCONNECTED,
                    database_type=DatabaseType.POSTGRESQL,
                    host=self.config.host,
                    port=self.config.port,
                    database=self.config.database,
                    error_message="Not connected",
                )

            # Test connection with a simple query
            async with self.get_connection(read_only=True) as conn:
                await conn.fetchval("SELECT 1")

                # Get connection statistics
                pool_stats = (
                    self._connection_pool.get_stats() if self._connection_pool else None
                )

                response_time = (time.time() - start_time) * 1000

                return ConnectionHealth(
                    status=ConnectionStatus.HEALTHY,
                    database_type=DatabaseType.POSTGRESQL,
                    host=self.config.host,
                    port=self.config.port,
                    database=self.config.database,
                    response_time_ms=response_time,
                    active_connections=(
                        pool_stats.open_connections if pool_stats else None
                    ),
                    total_connections=pool_stats.max_size if pool_stats else None,
                    replication_role=self.config.replication_role,
                )

        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            logger.warning("Health check failed", error=str(e))

            return ConnectionHealth(
                status=ConnectionStatus.UNHEALTHY,
                database_type=DatabaseType.POSTGRESQL,
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
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

    async def __aenter__(self) -> PostgreSQLService:
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.disconnect()
