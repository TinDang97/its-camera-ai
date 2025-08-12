"""Alembic environment configuration for ITS Camera AI.

Provides async database migration support with proper model imports
and high-performance configuration for production deployments.
"""

import asyncio
import sys
from logging.config import fileConfig
from pathlib import Path

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context

# Add src directory to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# Import models and configuration
from its_camera_ai.core.config import get_settings
from its_camera_ai.models import BaseModel

# This is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Add your model's MetaData object here for 'autogenerate' support
target_metadata = BaseModel.metadata

# Get database URL from settings
settings = get_settings()
config.set_main_option("sqlalchemy.url", settings.get_database_url(async_driver=False))


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        # High-performance migration options
        compare_type=True,
        compare_server_default=True,
        include_schemas=True,
        # Custom naming convention for indexes and constraints
        render_as_batch=False,
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """Run migrations with database connection."""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        # Migration configuration
        compare_type=True,
        compare_server_default=True,
        include_schemas=True,
        # Performance optimizations
        render_as_batch=False,
        # Custom options for PostgreSQL
        postgresql_include_object=lambda obj, name, type_, reflected, compare_to: (
            # Exclude system tables and extensions
            not (type_ == "table" and name.startswith(("pg_", "information_schema")))
        ),
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Run migrations in async mode.
    
    Creates an async engine and runs migrations in a separate
    thread to handle async database operations.
    """
    # Create async engine with production-optimized settings
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
        # Additional async configuration
        echo=False,  # Disable SQL echo for migrations
        future=True,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.
    """
    asyncio.run(run_async_migrations())


# Determine whether to run in offline or online mode
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
