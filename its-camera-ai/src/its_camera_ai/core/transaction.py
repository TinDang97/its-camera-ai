"""Transaction management utilities for database operations.

This module provides transaction management patterns including
nested transaction support, savepoints, and proper error handling.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any
from uuid import uuid4

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from .exceptions import DatabaseTransactionError
from .logging import get_logger

logger = get_logger(__name__)


class TransactionManager:
    """Manages database transactions with savepoint support.
    
    Provides support for nested transactions using savepoints,
    automatic rollback on errors, and proper transaction boundaries.
    """

    def __init__(self, session: AsyncSession):
        """Initialize transaction manager with a session.
        
        Args:
            session: SQLAlchemy async session
        """
        self.session = session
        self._savepoint_stack: list[str] = []
        self._transaction_id = str(uuid4())[:8]

    @asynccontextmanager
    async def transaction(
        self,
        isolation_level: str | None = None,
        read_only: bool = False
    ) -> AsyncGenerator[AsyncSession, None]:
        """Start a new transaction or nested transaction (savepoint).
        
        Args:
            isolation_level: Optional isolation level for the transaction
            read_only: Whether this is a read-only transaction
            
        Yields:
            AsyncSession: The database session within transaction
            
        Raises:
            DatabaseTransactionError: If transaction operations fail
        """
        is_nested = self.session.in_transaction()
        savepoint_name = None

        try:
            if is_nested:
                # Create a savepoint for nested transaction
                savepoint_name = f"sp_{len(self._savepoint_stack)}_{self._transaction_id}"
                await self.session.begin_nested()
                self._savepoint_stack.append(savepoint_name)

                logger.debug(
                    "Started nested transaction",
                    savepoint=savepoint_name,
                    transaction_id=self._transaction_id,
                    depth=len(self._savepoint_stack)
                )
            else:
                # Start new transaction
                if isolation_level:
                    await self.session.connection(execution_options={
                        'isolation_level': isolation_level
                    })

                await self.session.begin()

                logger.debug(
                    "Started new transaction",
                    transaction_id=self._transaction_id,
                    isolation_level=isolation_level,
                    read_only=read_only
                )

            yield self.session

            # Commit the transaction/savepoint
            if is_nested:
                # Commit savepoint (no explicit commit needed for nested)
                if self._savepoint_stack:
                    self._savepoint_stack.pop()
                logger.debug(
                    "Committed nested transaction",
                    savepoint=savepoint_name,
                    transaction_id=self._transaction_id
                )
            else:
                await self.session.commit()
                logger.debug(
                    "Committed transaction",
                    transaction_id=self._transaction_id
                )

        except Exception as e:
            # Handle rollback
            try:
                if is_nested:
                    # Rollback to savepoint
                    await self.session.rollback()
                    if savepoint_name and savepoint_name in self._savepoint_stack:
                        self._savepoint_stack.remove(savepoint_name)

                    logger.warning(
                        "Rolled back nested transaction",
                        savepoint=savepoint_name,
                        transaction_id=self._transaction_id,
                        error=str(e)
                    )
                else:
                    # Rollback entire transaction
                    await self.session.rollback()
                    logger.warning(
                        "Rolled back transaction",
                        transaction_id=self._transaction_id,
                        error=str(e)
                    )

            except Exception as rollback_error:
                logger.error(
                    "Failed to rollback transaction",
                    transaction_id=self._transaction_id,
                    rollback_error=str(rollback_error),
                    original_error=str(e)
                )
                raise DatabaseTransactionError(
                    f"Transaction rollback failed: {str(rollback_error)}",
                    rollback_error
                ) from rollback_error

            # Re-raise original exception
            if isinstance(e, SQLAlchemyError):
                raise DatabaseTransactionError(
                    f"Database transaction failed: {str(e)}", e
                ) from e
            else:
                raise

    @asynccontextmanager
    async def read_only_transaction(self) -> AsyncGenerator[AsyncSession, None]:
        """Start a read-only transaction.
        
        This is a convenience method for read-only operations.
        
        Yields:
            AsyncSession: The database session within read-only transaction
        """
        async with self.transaction(read_only=True) as session:
            yield session

    @asynccontextmanager
    async def serializable_transaction(self) -> AsyncGenerator[AsyncSession, None]:
        """Start a serializable isolation level transaction.
        
        This is a convenience method for operations requiring
        the highest isolation level.
        
        Yields:
            AsyncSession: The database session within serializable transaction
        """
        async with self.transaction(isolation_level="SERIALIZABLE") as session:
            yield session

    async def execute_in_transaction(
        self,
        operation: callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute an operation within a transaction.
        
        Args:
            operation: Async callable to execute
            *args: Positional arguments for operation
            **kwargs: Keyword arguments for operation
            
        Returns:
            Result of the operation
            
        Raises:
            DatabaseTransactionError: If transaction fails
        """
        async with self.transaction() as session:
            if 'session' in operation.__code__.co_varnames:
                kwargs['session'] = session
            return await operation(*args, **kwargs)


@asynccontextmanager
async def managed_transaction(
    session: AsyncSession,
    isolation_level: str | None = None,
    read_only: bool = False
) -> AsyncGenerator[AsyncSession, None]:
    """Standalone transaction context manager.
    
    This is a convenience function for when you don't need
    a full TransactionManager instance.
    
    Args:
        session: SQLAlchemy async session
        isolation_level: Optional isolation level
        read_only: Whether this is a read-only transaction
        
    Yields:
        AsyncSession: The database session within transaction
    """
    tx_manager = TransactionManager(session)
    async with tx_manager.transaction(isolation_level, read_only) as tx_session:
        yield tx_session


class BatchProcessor:
    """Utility for processing operations in batches with transactions.
    
    Useful for bulk operations that need to be processed in chunks
    to avoid memory issues and lock timeouts.
    """

    def __init__(
        self,
        session: AsyncSession,
        batch_size: int = 1000,
        commit_every: int = 10
    ):
        """Initialize batch processor.
        
        Args:
            session: SQLAlchemy async session
            batch_size: Number of items to process per batch
            commit_every: Number of batches to process before committing
        """
        self.session = session
        self.batch_size = batch_size
        self.commit_every = commit_every
        self.tx_manager = TransactionManager(session)

    async def process_batches(
        self,
        items: list[Any],
        processor: callable,
        **processor_kwargs
    ) -> list[Any]:
        """Process items in batches with transaction management.
        
        Args:
            items: List of items to process
            processor: Async callable to process each batch
            **processor_kwargs: Additional kwargs for processor
            
        Returns:
            List of results from each batch
            
        Raises:
            DatabaseTransactionError: If batch processing fails
        """
        results = []
        batches_processed = 0

        async with self.tx_manager.transaction() as session:
            for i in range(0, len(items), self.batch_size):
                batch = items[i:i + self.batch_size]

                try:
                    # Process batch within a savepoint
                    async with self.tx_manager.transaction() as batch_session:
                        batch_result = await processor(
                            batch,
                            session=batch_session,
                            **processor_kwargs
                        )
                        results.append(batch_result)

                    batches_processed += 1

                    # Commit periodically to avoid long-running transactions
                    if batches_processed % self.commit_every == 0:
                        await session.commit()
                        await session.begin()

                        logger.debug(
                            "Intermediate commit in batch processing",
                            batches_processed=batches_processed,
                            total_items=len(items)
                        )

                except Exception as e:
                    logger.error(
                        "Batch processing failed",
                        batch_start=i,
                        batch_size=len(batch),
                        error=str(e)
                    )
                    raise DatabaseTransactionError(
                        f"Batch processing failed at index {i}: {str(e)}", e
                    ) from e

        logger.info(
            "Batch processing completed",
            total_items=len(items),
            total_batches=len(results),
            batch_size=self.batch_size
        )

        return results
