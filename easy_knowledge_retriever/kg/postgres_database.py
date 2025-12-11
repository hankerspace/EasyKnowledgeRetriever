import asyncio
import json
import os
import re
import datetime
from datetime import timezone
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, TypeVar, Union, final
import numpy as np
import configparser
import ssl
import itertools

from .types import KnowledgeGraph, KnowledgeGraphNode, KnowledgeGraphEdge

from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_fixed,
)

from .graph_storage.base import BaseGraphStorage
from .kv_storage.base import BaseKVStorage, DocProcessingStatus, DocStatus, DocStatusStorage
from .vector_storage.base import BaseVectorStorage
from .namespace import NameSpace, is_namespace
from easy_knowledge_retriever.utils.logger import logger
from easy_knowledge_retriever.kg.concurrency import get_data_init_lock
from easy_knowledge_retriever.constants import (
    DEFAULT_EMBEDDING_BATCH_NUM,
    DEFAULT_MAX_GRAPH_NODES,
)


import asyncpg  # type: ignore
from asyncpg import Pool  # type: ignore

T = TypeVar("T")




class PostgreSQLDB:
    def __init__(self, config: dict[str, Any], **kwargs: Any):
        self.host = config["host"]
        self.port = config["port"]
        self.user = config["user"]
        self.password = config["password"]
        self.database = config["database"]
        self.workspace = config["workspace"]
        self.max = int(config["max_connections"])
        self.increment = 1
        self.pool: Pool | None = None

        # SSL configuration
        self.ssl_mode = config.get("ssl_mode")
        self.ssl_cert = config.get("ssl_cert")
        self.ssl_key = config.get("ssl_key")
        self.ssl_root_cert = config.get("ssl_root_cert")
        self.ssl_crl = config.get("ssl_crl")

        # Vector configuration
        self.vector_index_type = config.get("vector_index_type")
        self.hnsw_m = config.get("hnsw_m")
        self.hnsw_ef = config.get("hnsw_ef")
        self.ivfflat_lists = config.get("ivfflat_lists")
        self.vchordrq_build_options = config.get("vchordrq_build_options")
        self.vchordrq_probes = config.get("vchordrq_probes")
        self.vchordrq_epsilon = config.get("vchordrq_epsilon")

        # Server settings
        self.server_settings = config.get("server_settings")

        # Statement LRU cache size (keep as-is, allow None for optional configuration)
        self.statement_cache_size = config.get("statement_cache_size")

        if self.user is None or self.password is None or self.database is None:
            raise ValueError("Missing database user, password, or database")

        # Guard concurrent pool resets
        self._pool_reconnect_lock = asyncio.Lock()

        self._transient_exceptions = (
            asyncio.TimeoutError,
            TimeoutError,
            ConnectionError,
            OSError,
            asyncpg.exceptions.InterfaceError,
            asyncpg.exceptions.TooManyConnectionsError,
            asyncpg.exceptions.CannotConnectNowError,
            asyncpg.exceptions.PostgresConnectionError,
            asyncpg.exceptions.ConnectionDoesNotExistError,
            asyncpg.exceptions.ConnectionFailureError,
        )

        # Connection retry configuration
        self.connection_retry_attempts = config["connection_retry_attempts"]
        self.connection_retry_backoff = config["connection_retry_backoff"]
        self.connection_retry_backoff_max = max(
            self.connection_retry_backoff,
            config["connection_retry_backoff_max"],
        )
        self.pool_close_timeout = config["pool_close_timeout"]
        logger.info(
            "PostgreSQL, Retry config: attempts=%s, backoff=%.1fs, backoff_max=%.1fs, pool_close_timeout=%.1fs",
            self.connection_retry_attempts,
            self.connection_retry_backoff,
            self.connection_retry_backoff_max,
            self.pool_close_timeout,
        )

    def _create_ssl_context(self) -> ssl.SSLContext | None:
        """Create SSL context based on configuration parameters."""
        if not self.ssl_mode:
            return None

        ssl_mode = self.ssl_mode.lower()

        # For simple modes that don't require custom context
        if ssl_mode in ["disable", "allow", "prefer", "require"]:
            if ssl_mode == "disable":
                return None
            elif ssl_mode in ["require", "prefer", "allow"]:
                # Return None for simple SSL requirement, handled in initdb
                return None

        # For modes that require certificate verification
        if ssl_mode in ["verify-ca", "verify-full"]:
            try:
                context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)

                # Configure certificate verification
                if ssl_mode == "verify-ca":
                    context.check_hostname = False
                elif ssl_mode == "verify-full":
                    context.check_hostname = True

                # Load root certificate if provided
                if self.ssl_root_cert:
                    if os.path.exists(self.ssl_root_cert):
                        context.load_verify_locations(cafile=self.ssl_root_cert)
                        logger.info(
                            f"PostgreSQL, Loaded SSL root certificate: {self.ssl_root_cert}"
                        )
                    else:
                        logger.warning(
                            f"PostgreSQL, SSL root certificate file not found: {self.ssl_root_cert}"
                        )

                # Load client certificate and key if provided
                if self.ssl_cert and self.ssl_key:
                    if os.path.exists(self.ssl_cert) and os.path.exists(self.ssl_key):
                        context.load_cert_chain(self.ssl_cert, self.ssl_key)
                        logger.info(
                            f"PostgreSQL, Loaded SSL client certificate: {self.ssl_cert}"
                        )
                    else:
                        logger.warning(
                            "PostgreSQL, SSL client certificate or key file not found"
                        )

                # Load certificate revocation list if provided
                if self.ssl_crl:
                    if os.path.exists(self.ssl_crl):
                        context.load_verify_locations(crlfile=self.ssl_crl)
                        logger.info(f"PostgreSQL, Loaded SSL CRL: {self.ssl_crl}")
                    else:
                        logger.warning(
                            f"PostgreSQL, SSL CRL file not found: {self.ssl_crl}"
                        )

                return context

            except Exception as e:
                logger.error(f"PostgreSQL, Failed to create SSL context: {e}")
                raise ValueError(f"SSL configuration error: {e}")

        # Unknown SSL mode
        logger.warning(f"PostgreSQL, Unknown SSL mode: {ssl_mode}, SSL disabled")
        return None

    async def initdb(self):
        # Prepare connection parameters
        connection_params = {
            "user": self.user,
            "password": self.password,
            "database": self.database,
            "host": self.host,
            "port": self.port,
            "min_size": 1,
            "max_size": self.max,
        }

        # Only add statement_cache_size if it's configured
        if self.statement_cache_size is not None:
            connection_params["statement_cache_size"] = int(self.statement_cache_size)
            logger.info(
                f"PostgreSQL, statement LRU cache size set as: {self.statement_cache_size}"
            )

        # Add SSL configuration if provided
        ssl_context = self._create_ssl_context()
        if ssl_context is not None:
            connection_params["ssl"] = ssl_context
            logger.info("PostgreSQL, SSL configuration applied")
        elif self.ssl_mode:
            # Handle simple SSL modes without custom context
            if self.ssl_mode.lower() in ["require", "prefer"]:
                connection_params["ssl"] = True
            elif self.ssl_mode.lower() == "disable":
                connection_params["ssl"] = False
            logger.info(f"PostgreSQL, SSL mode set to: {self.ssl_mode}")

        # Add server settings if provided
        if self.server_settings:
            try:
                settings = {}
                # The format is expected to be a query string, e.g., "key1=value1&key2=value2"
                pairs = self.server_settings.split("&")
                for pair in pairs:
                    if "=" in pair:
                        key, value = pair.split("=", 1)
                        settings[key] = value
                if settings:
                    connection_params["server_settings"] = settings
                    logger.info(f"PostgreSQL, Server settings applied: {settings}")
            except Exception as e:
                logger.warning(
                    f"PostgreSQL, Failed to parse server_settings: {self.server_settings}, error: {e}"
                )

        wait_strategy = (
            wait_exponential(
                multiplier=self.connection_retry_backoff,
                min=self.connection_retry_backoff,
                max=self.connection_retry_backoff_max,
            )
            if self.connection_retry_backoff > 0
            else wait_fixed(0)
        )

        async def _create_pool_once() -> None:
            pool = await asyncpg.create_pool(**connection_params)  # type: ignore
            try:
                async with pool.acquire() as connection:
                    await self.configure_vector_extension(connection)
            except Exception:
                await pool.close()
                raise
            self.pool = pool

        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self.connection_retry_attempts),
                retry=retry_if_exception_type(self._transient_exceptions),
                wait=wait_strategy,
                before_sleep=self._before_sleep,
                reraise=True,
            ):
                with attempt:
                    await _create_pool_once()

            ssl_status = "with SSL" if connection_params.get("ssl") else "without SSL"
            logger.info(
                f"PostgreSQL, Connected to database at {self.host}:{self.port}/{self.database} {ssl_status}"
            )
        except Exception as e:
            logger.error(
                f"PostgreSQL, Failed to connect database at {self.host}:{self.port}/{self.database}, Got:{e}"
            )
            raise

    async def _ensure_pool(self) -> None:
        """Ensure the connection pool is initialised."""
        if self.pool is None:
            async with self._pool_reconnect_lock:
                if self.pool is None:
                    await self.initdb()

    async def _reset_pool(self) -> None:
        async with self._pool_reconnect_lock:
            if self.pool is not None:
                try:
                    await asyncio.wait_for(
                        self.pool.close(), timeout=self.pool_close_timeout
                    )
                except asyncio.TimeoutError:
                    logger.error(
                        "PostgreSQL, Timed out closing connection pool after %.2fs",
                        self.pool_close_timeout,
                    )
                except Exception as close_error:  # pragma: no cover - defensive logging
                    logger.warning(
                        f"PostgreSQL, Failed to close existing connection pool cleanly: {close_error!r}"
                    )
            self.pool = None

    async def _before_sleep(self, retry_state: RetryCallState) -> None:
        """Hook invoked by tenacity before sleeping between retries."""
        exc = retry_state.outcome.exception() if retry_state.outcome else None
        logger.warning(
            "PostgreSQL transient connection issue on attempt %s/%s: %r",
            retry_state.attempt_number,
            self.connection_retry_attempts,
            exc,
        )
        await self._reset_pool()

    async def _run_with_retry(
        self,
        operation: Callable[[asyncpg.Connection], Awaitable[T]],
        *,
        with_age: bool = False,
        graph_name: str | None = None,
    ) -> T:
        """
        Execute a database operation with automatic retry for transient failures.

        Args:
            operation: Async callable that receives an active connection.
            with_age: Whether to configure Apache AGE on the connection.
            graph_name: AGE graph name; required when with_age is True.

        Returns:
            The result returned by the operation.

        Raises:
            Exception: Propagates the last error if all retry attempts fail or a non-transient error occurs.
        """
        wait_strategy = (
            wait_exponential(
                multiplier=self.connection_retry_backoff,
                min=self.connection_retry_backoff,
                max=self.connection_retry_backoff_max,
            )
            if self.connection_retry_backoff > 0
            else wait_fixed(0)
        )

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self.connection_retry_attempts),
            retry=retry_if_exception_type(self._transient_exceptions),
            wait=wait_strategy,
            before_sleep=self._before_sleep,
            reraise=True,
        ):
            with attempt:
                await self._ensure_pool()
                assert self.pool is not None
                async with self.pool.acquire() as connection:  # type: ignore[arg-type]
                    if with_age and graph_name:
                        await self.configure_age(connection, graph_name)
                    elif with_age and not graph_name:
                        raise ValueError("Graph name is required when with_age is True")
                    if self.vector_index_type == "VCHORDRQ":
                        await self.configure_vchordrq(connection)
                    return await operation(connection)

    @staticmethod
    async def configure_vector_extension(connection: asyncpg.Connection) -> None:
        """Create VECTOR extension if it doesn't exist for vector similarity operations."""
        try:
            await connection.execute("CREATE EXTENSION IF NOT EXISTS vector")  # type: ignore
            logger.info("PostgreSQL, VECTOR extension enabled")
        except Exception as e:
            logger.warning(f"Could not create VECTOR extension: {e}")
            # Don't raise - let the system continue without vector extension

    @staticmethod
    async def configure_age_extension(connection: asyncpg.Connection) -> None:
        """Create AGE extension if it doesn't exist for graph operations."""
        try:
            await connection.execute("CREATE EXTENSION IF NOT EXISTS AGE CASCADE")  # type: ignore
            logger.info("PostgreSQL, AGE extension enabled")
        except Exception as e:
            logger.warning(f"Could not create AGE extension: {e}")
            # Don't raise - let the system continue without AGE extension

    @staticmethod
    async def configure_age(connection: asyncpg.Connection, graph_name: str) -> None:
        """Set the Apache AGE environment and creates a graph if it does not exist.

        This method:
        - Sets the PostgreSQL `search_path` to include `ag_catalog`, ensuring that Apache AGE functions can be used without specifying the schema.
        - Attempts to create a new graph with the provided `graph_name` if it does not already exist.
        - Silently ignores errors related to the graph already existing.

        """
        try:
            await connection.execute(  # type: ignore
                'SET search_path = ag_catalog, "$user", public'
            )
            await connection.execute(  # type: ignore
                f"select create_graph('{graph_name}')"
            )
        except (
            asyncpg.exceptions.InvalidSchemaNameError,
            asyncpg.exceptions.UniqueViolationError,
        ):
            pass

    async def configure_vchordrq(self, connection: asyncpg.Connection) -> None:
        """Configure VCHORDRQ extension for vector similarity search.

        Raises:
            asyncpg.exceptions.UndefinedObjectError: If VCHORDRQ extension is not installed
            asyncpg.exceptions.InvalidParameterValueError: If parameter value is invalid

        Note:
            This method does not catch exceptions. Configuration errors will fail-fast,
            while transient connection errors will be retried by _run_with_retry.
        """
        # Handle probes parameter - only set if non-empty value is provided
        if self.vchordrq_probes and str(self.vchordrq_probes).strip():
            await connection.execute(f"SET vchordrq.probes TO '{self.vchordrq_probes}'")
            logger.debug(f"PostgreSQL, VCHORDRQ probes set to: {self.vchordrq_probes}")

        # Handle epsilon parameter independently - check for None to allow 0.0 as valid value
        if self.vchordrq_epsilon is not None:
            await connection.execute(f"SET vchordrq.epsilon TO {self.vchordrq_epsilon}")
            logger.debug(
                f"PostgreSQL, VCHORDRQ epsilon set to: {self.vchordrq_epsilon}"
            )

    async def _migrate_llm_cache_schema(self):
        """Migrate LLM cache schema: add new columns and remove deprecated mode field"""
        try:
            # Check if all columns exist
            check_columns_sql = """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'easy_knowledge_retriever_llm_cache'
            AND column_name IN ('chunk_id', 'cache_type', 'queryparam', 'mode')
            """

            existing_columns = await self.query(check_columns_sql, multirows=True)
            existing_column_names = (
                {col["column_name"] for col in existing_columns}
                if existing_columns
                else set()
            )

            # Add missing chunk_id column
            if "chunk_id" not in existing_column_names:
                logger.info("Adding chunk_id column to EKR_LLM_CACHE table")
                add_chunk_id_sql = """
                ALTER TABLE EKR_LLM_CACHE
                ADD COLUMN chunk_id VARCHAR(255) NULL
                """
                await self.execute(add_chunk_id_sql)
                logger.info(
                    "Successfully added chunk_id column to EKR_LLM_CACHE table"
                )
            else:
                logger.info(
                    "chunk_id column already exists in EKR_LLM_CACHE table"
                )

            # Add missing cache_type column
            if "cache_type" not in existing_column_names:
                logger.info("Adding cache_type column to EKR_LLM_CACHE table")
                add_cache_type_sql = """
                ALTER TABLE EKR_LLM_CACHE
                ADD COLUMN cache_type VARCHAR(32) NULL
                """
                await self.execute(add_cache_type_sql)
                logger.info(
                    "Successfully added cache_type column to EKR_LLM_CACHE table"
                )

                # Migrate existing data using optimized regex pattern
                logger.info(
                    "Migrating existing LLM cache data to populate cache_type field (optimized)"
                )
                optimized_update_sql = """
                UPDATE EKR_LLM_CACHE
                SET cache_type = CASE
                    WHEN id ~ '^[^:]+:[^:]+:' THEN split_part(id, ':', 2)
                    ELSE 'extract'
                END
                WHERE cache_type IS NULL
                """
                await self.execute(optimized_update_sql)
                logger.info("Successfully migrated existing LLM cache data")
            else:
                logger.info(
                    "cache_type column already exists in EKR_LLM_CACHE table"
                )

            # Add missing queryparam column
            if "queryparam" not in existing_column_names:
                logger.info("Adding queryparam column to EKR_LLM_CACHE table")
                add_queryparam_sql = """
                ALTER TABLE EKR_LLM_CACHE
                ADD COLUMN queryparam JSONB NULL
                """
                await self.execute(add_queryparam_sql)
                logger.info(
                    "Successfully added queryparam column to EKR_LLM_CACHE table"
                )
            else:
                logger.info(
                    "queryparam column already exists in EKR_LLM_CACHE table"
                )

            # Remove deprecated mode field if it exists
            if "mode" in existing_column_names:
                logger.info(
                    "Removing deprecated mode column from EKR_LLM_CACHE table"
                )

                # First, drop the primary key constraint that includes mode
                drop_pk_sql = """
                ALTER TABLE EKR_LLM_CACHE
                DROP CONSTRAINT IF EXISTS EKR_LLM_CACHE_PK
                """
                await self.execute(drop_pk_sql)
                logger.info("Dropped old primary key constraint")

                # Drop the mode column
                drop_mode_sql = """
                ALTER TABLE EKR_LLM_CACHE
                DROP COLUMN mode
                """
                await self.execute(drop_mode_sql)
                logger.info(
                    "Successfully removed mode column from EKR_LLM_CACHE table"
                )

                # Create new primary key constraint without mode
                add_pk_sql = """
                ALTER TABLE EKR_LLM_CACHE
                ADD CONSTRAINT EKR_LLM_CACHE_PK PRIMARY KEY (workspace, id)
                """
                await self.execute(add_pk_sql)
                logger.info("Created new primary key constraint (workspace, id)")
            else:
                logger.info("mode column does not exist in EKR_LLM_CACHE table")

        except Exception as e:
            logger.warning(f"Failed to migrate LLM cache schema: {e}")

    async def _migrate_timestamp_columns(self):
        """Migrate timestamp columns in tables to witimezone-free types, assuming original data is in UTC time"""
        # Tables and columns that need migration
        tables_to_migrate = {
            "EKR_VDB_ENTITY": ["create_time", "update_time"],
            "EKR_VDB_RELATION": ["create_time", "update_time"],
            "EKR_DOC_CHUNKS": ["create_time", "update_time"],
            "EKR_DOC_STATUS": ["created_at", "updated_at"],
        }

        try:
            # Optimization: Batch check all columns in one query instead of 8 separate queries
            table_names_lower = [t.lower() for t in tables_to_migrate.keys()]
            all_column_names = list(
                set(col for cols in tables_to_migrate.values() for col in cols)
            )

            check_all_columns_sql = """
            SELECT table_name, column_name, data_type
            FROM information_schema.columns
            WHERE table_name = ANY($1)
            AND column_name = ANY($2)
            """

            all_columns_result = await self.query(
                check_all_columns_sql,
                [table_names_lower, all_column_names],
                multirows=True,
            )

            # Build lookup dict: (table_name, column_name) -> data_type
            column_types = {}
            if all_columns_result:
                column_types = {
                    (row["table_name"].upper(), row["column_name"]): row["data_type"]
                    for row in all_columns_result
                }

            # Now iterate and migrate only what's needed
            for table_name, columns in tables_to_migrate.items():
                for column_name in columns:
                    try:
                        data_type = column_types.get((table_name, column_name))

                        if not data_type:
                            logger.warning(
                                f"Column {table_name}.{column_name} does not exist, skipping migration"
                            )
                            continue

                        # Check column type
                        if data_type == "timestamp without time zone":
                            logger.debug(
                                f"Column {table_name}.{column_name} is already witimezone-free, no migration needed"
                            )
                            continue

                        # Execute migration, explicitly specifying UTC timezone for interpreting original data
                        logger.info(
                            f"Migrating {table_name}.{column_name} from {data_type} to TIMESTAMP(0) type"
                        )
                        migration_sql = f"""
                        ALTER TABLE {table_name}
                        ALTER COLUMN {column_name} TYPE TIMESTAMP(0),
                        ALTER COLUMN {column_name} SET DEFAULT CURRENT_TIMESTAMP
                        """

                        await self.execute(migration_sql)
                        logger.info(
                            f"Successfully migrated {table_name}.{column_name} to timezone-free type"
                        )
                    except Exception as e:
                        # Log error but don't interrupt the process
                        logger.warning(
                            f"Failed to migrate {table_name}.{column_name}: {e}"
                        )
        except Exception as e:
            logger.error(f"Failed to batch check timestamp columns: {e}")

    async def _migrate_doc_chunks_to_vdb_chunks(self):
        """
        Migrate data from EKR_DOC_CHUNKS to EKR_VDB_CHUNKS if specific conditions are met.
        This migration is intended for users who are upgrading and have an older table structure
        where EKR_DOC_CHUNKS contained a `content_vector` column.

        """
        try:
            # 1. Check if the new table EKR_VDB_CHUNKS is empty
            vdb_chunks_count_sql = "SELECT COUNT(1) as count FROM EKR_VDB_CHUNKS"
            vdb_chunks_count_result = await self.query(vdb_chunks_count_sql)
            if vdb_chunks_count_result and vdb_chunks_count_result["count"] > 0:
                logger.info(
                    "Skipping migration: EKR_VDB_CHUNKS already contains data."
                )
                return

            # 2. Check if `content_vector` column exists in the old table
            check_column_sql = """
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'easy_knowledge_retriever_doc_chunks' AND column_name = 'content_vector'
            """
            column_exists = await self.query(check_column_sql)
            if not column_exists:
                logger.info(
                    "Skipping migration: `content_vector` not found in EKR_DOC_CHUNKS"
                )
                return

            # 3. Check if the old table EKR_DOC_CHUNKS has data
            doc_chunks_count_sql = "SELECT COUNT(1) as count FROM EKR_DOC_CHUNKS"
            doc_chunks_count_result = await self.query(doc_chunks_count_sql)
            if not doc_chunks_count_result or doc_chunks_count_result["count"] == 0:
                logger.info("Skipping migration: EKR_DOC_CHUNKS is empty.")
                return

            # 4. Perform the migration
            logger.info(
                "Starting data migration from EKR_DOC_CHUNKS to EKR_VDB_CHUNKS..."
            )
            migration_sql = """
            INSERT INTO EKR_VDB_CHUNKS (
                id, workspace, full_doc_id, chunk_order_index, tokens, content,
                content_vector, file_path, create_time, update_time
            )
            SELECT
                id, workspace, full_doc_id, chunk_order_index, tokens, content,
                content_vector, file_path, create_time, update_time
            FROM EKR_DOC_CHUNKS
            ON CONFLICT (workspace, id) DO NOTHING;
            """
            await self.execute(migration_sql)
            logger.info("Data migration to EKR_VDB_CHUNKS completed successfully.")

        except Exception as e:
            logger.error(f"Failed during data migration to EKR_VDB_CHUNKS: {e}")
            # Do not re-raise, to allow the application to start

    async def _check_llm_cache_needs_migration(self):
        """Check if LLM cache data needs migration by examining any record with old format"""
        try:
            # Optimized query: directly check for old format records without sorting
            check_sql = """
            SELECT 1 FROM EKR_LLM_CACHE
            WHERE id NOT LIKE '%:%'
            LIMIT 1
            """
            result = await self.query(check_sql)

            # If any old format record exists, migration is needed
            return result is not None

        except Exception as e:
            logger.warning(f"Failed to check LLM cache migration status: {e}")
            return False

    async def _migrate_llm_cache_to_flattened_keys(self):
        """Optimized version: directly execute single UPDATE migration to migrate old format cache keys to flattened format"""
        try:
            # Check if migration is needed
            check_sql = """
            SELECT COUNT(*) as count FROM EKR_LLM_CACHE
            WHERE id NOT LIKE '%:%'
            """
            result = await self.query(check_sql)

            if not result or result["count"] == 0:
                logger.info("No old format LLM cache data found, skipping migration")
                return

            old_count = result["count"]
            logger.info(f"Found {old_count} old format cache records")

            # Check potential primary key conflicts (optional but recommended)
            conflict_check_sql = """
            WITH new_ids AS (
                SELECT
                    workspace,
                    mode,
                    id as old_id,
                    mode || ':' ||
                    CASE WHEN mode = 'default' THEN 'extract' ELSE 'unknown' END || ':' ||
                    md5(original_prompt) as new_id
                FROM EKR_LLM_CACHE
                WHERE id NOT LIKE '%:%'
            )
            SELECT COUNT(*) as conflicts
            FROM new_ids n1
            JOIN EKR_LLM_CACHE existing
            ON existing.workspace = n1.workspace
            AND existing.mode = n1.mode
            AND existing.id = n1.new_id
            WHERE existing.id LIKE '%:%'  -- Only check conflicts with existing new format records
            """

            conflict_result = await self.query(conflict_check_sql)
            if conflict_result and conflict_result["conflicts"] > 0:
                logger.warning(
                    f"Found {conflict_result['conflicts']} potential ID conflicts with existing records"
                )
                # Can choose to continue or abort, here we choose to continue and log warning

            # Execute single UPDATE migration
            logger.info("Starting optimized LLM cache migration...")
            migration_sql = """
            UPDATE EKR_LLM_CACHE
            SET
                id = mode || ':' ||
                     CASE WHEN mode = 'default' THEN 'extract' ELSE 'unknown' END || ':' ||
                     md5(original_prompt),
                cache_type = CASE WHEN mode = 'default' THEN 'extract' ELSE 'unknown' END,
                update_time = CURRENT_TIMESTAMP
            WHERE id NOT LIKE '%:%'
            """

            # Execute migration
            await self.execute(migration_sql)

            # Verify migration results
            verify_sql = """
            SELECT COUNT(*) as remaining_old FROM EKR_LLM_CACHE
            WHERE id NOT LIKE '%:%'
            """
            verify_result = await self.query(verify_sql)
            remaining = verify_result["remaining_old"] if verify_result else -1

            if remaining == 0:
                logger.info(
                    f"✅ Successfully migrated {old_count} LLM cache records to flattened format"
                )
            else:
                logger.warning(
                    f"⚠️ Migration completed but {remaining} old format records remain"
                )

        except Exception as e:
            logger.error(f"Optimized LLM cache migration failed: {e}")
            raise

    async def _migrate_doc_status_add_chunks_list(self):
        """Add chunks_list column to EKR_DOC_STATUS table if it doesn't exist"""
        try:
            # Check if chunks_list column exists
            check_column_sql = """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'easy_knowledge_retriever_doc_status'
            AND column_name = 'chunks_list'
            """

            column_info = await self.query(check_column_sql)
            if not column_info:
                logger.info("Adding chunks_list column to EKR_DOC_STATUS table")
                add_column_sql = """
                ALTER TABLE EKR_DOC_STATUS
                ADD COLUMN chunks_list JSONB NULL DEFAULT '[]'::jsonb
                """
                await self.execute(add_column_sql)
                logger.info(
                    "Successfully added chunks_list column to EKR_DOC_STATUS table"
                )
            else:
                logger.info(
                    "chunks_list column already exists in EKR_DOC_STATUS table"
                )
        except Exception as e:
            logger.warning(
                f"Failed to add chunks_list column to EKR_DOC_STATUS: {e}"
            )

    async def _migrate_text_chunks_add_llm_cache_list(self):
        """Add llm_cache_list column to EKR_DOC_CHUNKS table if it doesn't exist"""
        try:
            # Check if llm_cache_list column exists
            check_column_sql = """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'easy_knowledge_retriever_doc_chunks'
            AND column_name = 'llm_cache_list'
            """

            column_info = await self.query(check_column_sql)
            if not column_info:
                logger.info("Adding llm_cache_list column to EKR_DOC_CHUNKS table")
                add_column_sql = """
                ALTER TABLE EKR_DOC_CHUNKS
                ADD COLUMN llm_cache_list JSONB NULL DEFAULT '[]'::jsonb
                """
                await self.execute(add_column_sql)
                logger.info(
                    "Successfully added llm_cache_list column to EKR_DOC_CHUNKS table"
                )
            else:
                logger.info(
                    "llm_cache_list column already exists in EKR_DOC_CHUNKS table"
                )
        except Exception as e:
            logger.warning(
                f"Failed to add llm_cache_list column to EKR_DOC_CHUNKS: {e}"
            )

    async def _migrate_doc_status_add_track_id(self):
        """Add track_id column to EKR_DOC_STATUS table if it doesn't exist and create index"""
        try:
            # Check if track_id column exists
            check_column_sql = """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'easy_knowledge_retriever_doc_status'
            AND column_name = 'track_id'
            """

            column_info = await self.query(check_column_sql)
            if not column_info:
                logger.info("Adding track_id column to EKR_DOC_STATUS table")
                add_column_sql = """
                ALTER TABLE EKR_DOC_STATUS
                ADD COLUMN track_id VARCHAR(255) NULL
                """
                await self.execute(add_column_sql)
                logger.info(
                    "Successfully added track_id column to EKR_DOC_STATUS table"
                )
            else:
                logger.info(
                    "track_id column already exists in EKR_DOC_STATUS table"
                )

            # Check if track_id index exists
            check_index_sql = """
            SELECT indexname
            FROM pg_indexes
            WHERE tablename = 'easy_knowledge_retriever_doc_status'
            AND indexname = 'idx_easy_knowledge_retriever_doc_status_track_id'
            """

            index_info = await self.query(check_index_sql)
            if not index_info:
                logger.info(
                    "Creating index on track_id column for EKR_DOC_STATUS table"
                )
                create_index_sql = """
                CREATE INDEX idx_easy_knowledge_retriever_doc_status_track_id ON EKR_DOC_STATUS (track_id)
                """
                await self.execute(create_index_sql)
                logger.info(
                    "Successfully created index on track_id column for EKR_DOC_STATUS table"
                )
            else:
                logger.info(
                    "Index on track_id column already exists for EKR_DOC_STATUS table"
                )

        except Exception as e:
            logger.warning(
                f"Failed to add track_id column or index to EKR_DOC_STATUS: {e}"
            )

    async def _migrate_doc_status_add_metadata_error_msg(self):
        """Add metadata and error_msg columns to EKR_DOC_STATUS table if they don't exist"""
        try:
            # Check if metadata column exists
            check_metadata_sql = """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'easy_knowledge_retriever_doc_status'
            AND column_name = 'metadata'
            """

            metadata_info = await self.query(check_metadata_sql)
            if not metadata_info:
                logger.info("Adding metadata column to EKR_DOC_STATUS table")
                add_metadata_sql = """
                ALTER TABLE EKR_DOC_STATUS
                ADD COLUMN metadata JSONB NULL DEFAULT '{}'::jsonb
                """
                await self.execute(add_metadata_sql)
                logger.info(
                    "Successfully added metadata column to EKR_DOC_STATUS table"
                )
            else:
                logger.info(
                    "metadata column already exists in EKR_DOC_STATUS table"
                )

            # Check if error_msg column exists
            check_error_msg_sql = """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'easy_knowledge_retriever_doc_status'
            AND column_name = 'error_msg'
            """

            error_msg_info = await self.query(check_error_msg_sql)
            if not error_msg_info:
                logger.info("Adding error_msg column to EKR_DOC_STATUS table")
                add_error_msg_sql = """
                ALTER TABLE EKR_DOC_STATUS
                ADD COLUMN error_msg TEXT NULL
                """
                await self.execute(add_error_msg_sql)
                logger.info(
                    "Successfully added error_msg column to EKR_DOC_STATUS table"
                )
            else:
                logger.info(
                    "error_msg column already exists in EKR_DOC_STATUS table"
                )

        except Exception as e:
            logger.warning(
                f"Failed to add metadata/error_msg columns to EKR_DOC_STATUS: {e}"
            )

    async def _migrate_field_lengths(self):
        """Migrate database field lengths: entity_name, source_id, target_id, and file_path"""
        # Define the field changes needed
        field_migrations = [
            {
                "table": "EKR_VDB_ENTITY",
                "column": "entity_name",
                "old_type": "character varying(255)",
                "new_type": "VARCHAR(512)",
                "description": "entity_name from 255 to 512",
            },
            {
                "table": "EKR_VDB_RELATION",
                "column": "source_id",
                "old_type": "character varying(256)",
                "new_type": "VARCHAR(512)",
                "description": "source_id from 256 to 512",
            },
            {
                "table": "EKR_VDB_RELATION",
                "column": "target_id",
                "old_type": "character varying(256)",
                "new_type": "VARCHAR(512)",
                "description": "target_id from 256 to 512",
            },
            {
                "table": "EKR_DOC_CHUNKS",
                "column": "file_path",
                "old_type": "character varying(256)",
                "new_type": "TEXT",
                "description": "file_path to TEXT NULL",
            },
            {
                "table": "EKR_VDB_CHUNKS",
                "column": "file_path",
                "old_type": "character varying(256)",
                "new_type": "TEXT",
                "description": "file_path to TEXT NULL",
            },
        ]

        try:
            # Optimization: Batch check all columns in one query instead of 5 separate queries
            unique_tables = list(set(m["table"].lower() for m in field_migrations))
            unique_columns = list(set(m["column"] for m in field_migrations))

            check_all_columns_sql = """
            SELECT table_name, column_name, data_type, character_maximum_length, is_nullable
            FROM information_schema.columns
            WHERE table_name = ANY($1)
            AND column_name = ANY($2)
            """

            all_columns_result = await self.query(
                check_all_columns_sql, [unique_tables, unique_columns], multirows=True
            )

            # Build lookup dict: (table_name, column_name) -> column_info
            column_info_map = {}
            if all_columns_result:
                column_info_map = {
                    (row["table_name"].upper(), row["column_name"]): row
                    for row in all_columns_result
                }

            # Now iterate and migrate only what's needed
            for migration in field_migrations:
                try:
                    column_info = column_info_map.get(
                        (migration["table"], migration["column"])
                    )

                    if not column_info:
                        logger.warning(
                            f"Column {migration['table']}.{migration['column']} does not exist, skipping migration"
                        )
                        continue

                    current_type = column_info.get("data_type", "").lower()
                    current_length = column_info.get("character_maximum_length")

                    # Check if migration is needed
                    needs_migration = False

                    if migration["column"] == "entity_name" and current_length == 255:
                        needs_migration = True
                    elif (
                        migration["column"] in ["source_id", "target_id"]
                        and current_length == 256
                    ):
                        needs_migration = True
                    elif (
                        migration["column"] == "file_path"
                        and current_type == "character varying"
                    ):
                        needs_migration = True

                    if needs_migration:
                        logger.info(
                            f"Migrating {migration['table']}.{migration['column']}: {migration['description']}"
                        )

                        # Execute the migration
                        alter_sql = f"""
                        ALTER TABLE {migration["table"]}
                        ALTER COLUMN {migration["column"]} TYPE {migration["new_type"]}
                        """

                        await self.execute(alter_sql)
                        logger.info(
                            f"Successfully migrated {migration['table']}.{migration['column']}"
                        )
                    else:
                        logger.debug(
                            f"Column {migration['table']}.{migration['column']} already has correct type, no migration needed"
                        )

                except Exception as e:
                    # Log error but don't interrupt the process
                    logger.warning(
                        f"Failed to migrate {migration['table']}.{migration['column']}: {e}"
                    )
        except Exception as e:
            logger.error(f"Failed to batch check field lengths: {e}")

    async def check_tables(self):
        # First create all tables
        for k, v in TABLES.items():
            try:
                await self.query(f"SELECT 1 FROM {k} LIMIT 1")
            except Exception:
                try:
                    logger.info(f"PostgreSQL, Try Creating table {k} in database")
                    await self.execute(v["ddl"])
                    logger.info(
                        f"PostgreSQL, Creation success table {k} in PostgreSQL database"
                    )
                except Exception as e:
                    logger.error(
                        f"PostgreSQL, Failed to create table {k} in database, Please verify the connection with PostgreSQL database, Got: {e}"
                    )
                    raise e

        # Batch check all indexes at once (optimization: single query instead of N queries)
        try:
            table_names = list(TABLES.keys())
            table_names_lower = [t.lower() for t in table_names]

            # Get all existing indexes for our tables in one query
            check_all_indexes_sql = """
            SELECT indexname, tablename
            FROM pg_indexes
            WHERE tablename = ANY($1)
            """
            existing_indexes_result = await self.query(
                check_all_indexes_sql, [table_names_lower], multirows=True
            )

            # Build a set of existing index names for fast lookup
            existing_indexes = set()
            if existing_indexes_result:
                existing_indexes = {row["indexname"] for row in existing_indexes_result}

            # Create missing indexes
            for k in table_names:
                # Create index for id column if missing
                index_name = f"idx_{k.lower()}_id"
                if index_name not in existing_indexes:
                    try:
                        create_index_sql = f"CREATE INDEX {index_name} ON {k}(id)"
                        logger.info(
                            f"PostgreSQL, Creating index {index_name} on table {k}"
                        )
                        await self.execute(create_index_sql)
                    except Exception as e:
                        logger.error(
                            f"PostgreSQL, Failed to create index {index_name}, Got: {e}"
                        )

                # Create composite index for (workspace, id) if missing
                composite_index_name = f"idx_{k.lower()}_workspace_id"
                if composite_index_name not in existing_indexes:
                    try:
                        create_composite_index_sql = (
                            f"CREATE INDEX {composite_index_name} ON {k}(workspace, id)"
                        )
                        logger.info(
                            f"PostgreSQL, Creating composite index {composite_index_name} on table {k}"
                        )
                        await self.execute(create_composite_index_sql)
                    except Exception as e:
                        logger.error(
                            f"PostgreSQL, Failed to create composite index {composite_index_name}, Got: {e}"
                        )
        except Exception as e:
            logger.error(f"PostgreSQL, Failed to batch check/create indexes: {e}")

        # Create vector indexs
        if self.vector_index_type:
            logger.info(
                f"PostgreSQL, Create vector indexs, type: {self.vector_index_type}"
            )
            try:
                if self.vector_index_type in ["HNSW", "IVFFLAT", "VCHORDRQ"]:
                    await self._create_vector_indexes()
                else:
                    logger.warning(
                        "Doesn't support this vector index type: {self.vector_index_type}. "
                        "Supported types: HNSW, IVFFLAT, VCHORDRQ"
                    )
            except Exception as e:
                logger.error(
                    f"PostgreSQL, Failed to create vector index, type: {self.vector_index_type}, Got: {e}"
                )
        # After all tables are created, attempt to migrate timestamp fields
        try:
            await self._migrate_timestamp_columns()
        except Exception as e:
            logger.error(f"PostgreSQL, Failed to migrate timestamp columns: {e}")
            # Don't throw an exception, allow the initialization process to continue

        # Migrate LLM cache schema: add new columns and remove deprecated mode field
        try:
            await self._migrate_llm_cache_schema()
        except Exception as e:
            logger.error(f"PostgreSQL, Failed to migrate LLM cache schema: {e}")
            # Don't throw an exception, allow the initialization process to continue

        # Finally, attempt to migrate old doc chunks data if needed
        try:
            await self._migrate_doc_chunks_to_vdb_chunks()
        except Exception as e:
            logger.error(f"PostgreSQL, Failed to migrate doc_chunks to vdb_chunks: {e}")

        # Check and migrate LLM cache to flattened keys if needed
        try:
            if await self._check_llm_cache_needs_migration():
                await self._migrate_llm_cache_to_flattened_keys()
        except Exception as e:
            logger.error(f"PostgreSQL, LLM cache migration failed: {e}")

        # Migrate doc status to add chunks_list field if needed
        try:
            await self._migrate_doc_status_add_chunks_list()
        except Exception as e:
            logger.error(
                f"PostgreSQL, Failed to migrate doc status chunks_list field: {e}"
            )

        # Migrate text chunks to add llm_cache_list field if needed
        try:
            await self._migrate_text_chunks_add_llm_cache_list()
        except Exception as e:
            logger.error(
                f"PostgreSQL, Failed to migrate text chunks llm_cache_list field: {e}"
            )

        # Migrate field lengths for entity_name, source_id, target_id, and file_path
        try:
            await self._migrate_field_lengths()
        except Exception as e:
            logger.error(f"PostgreSQL, Failed to migrate field lengths: {e}")

        # Migrate doc status to add track_id field if needed
        try:
            await self._migrate_doc_status_add_track_id()
        except Exception as e:
            logger.error(
                f"PostgreSQL, Failed to migrate doc status track_id field: {e}"
            )

        # Migrate doc status to add metadata and error_msg fields if needed
        try:
            await self._migrate_doc_status_add_metadata_error_msg()
        except Exception as e:
            logger.error(
                f"PostgreSQL, Failed to migrate doc status metadata/error_msg fields: {e}"
            )

        # Create pagination optimization indexes for EKR_DOC_STATUS
        try:
            await self._create_pagination_indexes()
        except Exception as e:
            logger.error(f"PostgreSQL, Failed to create pagination indexes: {e}")

        # Migrate to ensure new tables EKR_FULL_ENTITIES and EKR_FULL_RELATIONS exist
        try:
            await self._migrate_create_full_entities_relations_tables()
        except Exception as e:
            logger.error(
                f"PostgreSQL, Failed to create full entities/relations tables: {e}"
            )

    async def _migrate_create_full_entities_relations_tables(self):
        """Create EKR_FULL_ENTITIES and EKR_FULL_RELATIONS tables if they don't exist"""
        tables_to_check = [
            {
                "name": "EKR_FULL_ENTITIES",
                "ddl": TABLES["EKR_FULL_ENTITIES"]["ddl"],
                "description": "Full entities storage table",
            },
            {
                "name": "EKR_FULL_RELATIONS",
                "ddl": TABLES["EKR_FULL_RELATIONS"]["ddl"],
                "description": "Full relations storage table",
            },
        ]

        for table_info in tables_to_check:
            table_name = table_info["name"]
            try:
                # Check if table exists
                check_table_sql = """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_name = $1
                AND table_schema = 'public'
                """
                params = {"table_name": table_name.lower()}
                table_exists = await self.query(check_table_sql, list(params.values()))

                if not table_exists:
                    logger.info(f"Creating table {table_name}")
                    await self.execute(table_info["ddl"])
                    logger.info(
                        f"Successfully created {table_info['description']}: {table_name}"
                    )

                    # Create basic indexes for the new table
                    try:
                        # Create index for id column
                        index_name = f"idx_{table_name.lower()}_id"
                        create_index_sql = (
                            f"CREATE INDEX {index_name} ON {table_name}(id)"
                        )
                        await self.execute(create_index_sql)
                        logger.info(f"Created index {index_name} on table {table_name}")

                        # Create composite index for (workspace, id) columns
                        composite_index_name = f"idx_{table_name.lower()}_workspace_id"
                        create_composite_index_sql = f"CREATE INDEX {composite_index_name} ON {table_name}(workspace, id)"
                        await self.execute(create_composite_index_sql)
                        logger.info(
                            f"Created composite index {composite_index_name} on table {table_name}"
                        )

                    except Exception as e:
                        logger.warning(
                            f"Failed to create indexes for table {table_name}: {e}"
                        )

                else:
                    logger.debug(f"Table {table_name} already exists")

            except Exception as e:
                logger.error(f"Failed to create table {table_name}: {e}")

    async def _create_pagination_indexes(self):
        """Create indexes to optimize pagination queries for EKR_DOC_STATUS"""
        indexes = [
            {
                "name": "idx_easy_knowledge_retriever_doc_status_workspace_status_updated_at",
                "sql": "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_easy_knowledge_retriever_doc_status_workspace_status_updated_at ON EKR_DOC_STATUS (workspace, status, updated_at DESC)",
                "description": "Composite index for workspace + status + updated_at pagination",
            },
            {
                "name": "idx_easy_knowledge_retriever_doc_status_workspace_status_created_at",
                "sql": "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_easy_knowledge_retriever_doc_status_workspace_status_created_at ON EKR_DOC_STATUS (workspace, status, created_at DESC)",
                "description": "Composite index for workspace + status + created_at pagination",
            },
            {
                "name": "idx_easy_knowledge_retriever_doc_status_workspace_updated_at",
                "sql": "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_easy_knowledge_retriever_doc_status_workspace_updated_at ON EKR_DOC_STATUS (workspace, updated_at DESC)",
                "description": "Index for workspace + updated_at pagination (all statuses)",
            },
            {
                "name": "idx_easy_knowledge_retriever_doc_status_workspace_created_at",
                "sql": "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_easy_knowledge_retriever_doc_status_workspace_created_at ON EKR_DOC_STATUS (workspace, created_at DESC)",
                "description": "Index for workspace + created_at pagination (all statuses)",
            },
            {
                "name": "idx_easy_knowledge_retriever_doc_status_workspace_id",
                "sql": "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_easy_knowledge_retriever_doc_status_workspace_id ON EKR_DOC_STATUS (workspace, id)",
                "description": "Index for workspace + id sorting",
            },
            {
                "name": "idx_easy_knowledge_retriever_doc_status_workspace_file_path",
                "sql": "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_easy_knowledge_retriever_doc_status_workspace_file_path ON EKR_DOC_STATUS (workspace, file_path)",
                "description": "Index for workspace + file_path sorting",
            },
        ]

        for index in indexes:
            try:
                # Check if index already exists
                check_sql = """
                SELECT indexname
                FROM pg_indexes
                WHERE tablename = 'easy_knowledge_retriever_doc_status'
                AND indexname = $1
                """

                params = {"indexname": index["name"]}
                existing = await self.query(check_sql, list(params.values()))

                if not existing:
                    logger.info(f"Creating pagination index: {index['description']}")
                    await self.execute(index["sql"])
                    logger.info(f"Successfully created index: {index['name']}")
                else:
                    logger.debug(f"Index already exists: {index['name']}")

            except Exception as e:
                logger.warning(f"Failed to create index {index['name']}: {e}")

    async def _create_vector_indexes(self):
        vdb_tables = [
            "EKR_VDB_CHUNKS",
            "EKR_VDB_ENTITY",
            "EKR_VDB_RELATION",
        ]

        create_sql = {
            "HNSW": f"""
                CREATE INDEX {{vector_index_name}}
                ON {{k}} USING hnsw (content_vector vector_cosine_ops)
                WITH (m = {self.hnsw_m}, ef_construction = {self.hnsw_ef})
            """,
            "IVFFLAT": f"""
                CREATE INDEX {{vector_index_name}}
                ON {{k}} USING ivfflat (content_vector vector_cosine_ops)
                WITH (lists = {self.ivfflat_lists})
            """,
            "VCHORDRQ": f"""
                CREATE INDEX {{vector_index_name}}
                ON {{k}} USING vchordrq (content_vector vector_cosine_ops)
                {f'WITH (options = $${self.vchordrq_build_options}$$)' if self.vchordrq_build_options else ''}
            """,
        }

        embedding_dim = int(os.environ.get("EMBEDDING_DIM", 1024))
        for k in vdb_tables:
            vector_index_name = (
                f"idx_{k.lower()}_{self.vector_index_type.lower()}_cosine"
            )
            check_vector_index_sql = f"""
                    SELECT 1 FROM pg_indexes
                    WHERE indexname = '{vector_index_name}' AND tablename = '{k.lower()}'
                """
            try:
                vector_index_exists = await self.query(check_vector_index_sql)
                if not vector_index_exists:
                    # Only set vector dimension when index doesn't exist
                    alter_sql = f"ALTER TABLE {k} ALTER COLUMN content_vector TYPE VECTOR({embedding_dim})"
                    await self.execute(alter_sql)
                    logger.debug(f"Ensured vector dimension for {k}")
                    logger.info(
                        f"Creating {self.vector_index_type} index {vector_index_name} on table {k}"
                    )
                    await self.execute(
                        create_sql[self.vector_index_type].format(
                            vector_index_name=vector_index_name, k=k
                        )
                    )
                    logger.info(
                        f"Successfully created vector index {vector_index_name} on table {k}"
                    )
                else:
                    logger.info(
                        f"{self.vector_index_type} vector index {vector_index_name} already exists on table {k}"
                    )
            except Exception as e:
                logger.error(f"Failed to create vector index on table {k}, Got: {e}")

    async def query(
        self,
        sql: str,
        params: list[Any] | None = None,
        multirows: bool = False,
        with_age: bool = False,
        graph_name: str | None = None,
    ) -> dict[str, Any] | None | list[dict[str, Any]]:
        async def _operation(connection: asyncpg.Connection) -> Any:
            prepared_params = tuple(params) if params else ()
            if prepared_params:
                rows = await connection.fetch(sql, *prepared_params)
            else:
                rows = await connection.fetch(sql)

            if multirows:
                if rows:
                    columns = [col for col in rows[0].keys()]
                    return [dict(zip(columns, row)) for row in rows]
                return []

            if rows:
                columns = rows[0].keys()
                return dict(zip(columns, rows[0]))
            return None

        try:
            return await self._run_with_retry(
                _operation, with_age=with_age, graph_name=graph_name
            )
        except Exception as e:
            logger.error(f"PostgreSQL database, error:{e}")
            raise

    async def execute(
        self,
        sql: str,
        data: dict[str, Any] | None = None,
        upsert: bool = False,
        ignore_if_exists: bool = False,
        with_age: bool = False,
        graph_name: str | None = None,
    ):
        async def _operation(connection: asyncpg.Connection) -> Any:
            prepared_values = tuple(data.values()) if data else ()
            try:
                if not data:
                    return await connection.execute(sql)
                return await connection.execute(sql, *prepared_values)
            except (
                asyncpg.exceptions.UniqueViolationError,
                asyncpg.exceptions.DuplicateTableError,
                asyncpg.exceptions.DuplicateObjectError,
                asyncpg.exceptions.InvalidSchemaNameError,
            ) as e:
                if ignore_if_exists:
                    logger.debug("PostgreSQL, ignoring duplicate during execute: %r", e)
                    return None
                if upsert:
                    logger.info(
                        "PostgreSQL, duplicate detected but treated as upsert success: %r",
                        e,
                    )
                    return None
                raise

        try:
            await self._run_with_retry(
                _operation, with_age=with_age, graph_name=graph_name
            )
        except Exception as e:
            logger.error(f"PostgreSQL database,\nsql:{sql},\ndata:{data},\nerror:{e}")
            raise


class ClientManager:
    _instances: dict[str, Any] = {"db": None, "ref_count": 0}
    _lock = asyncio.Lock()

    @staticmethod
    def get_config() -> dict[str, Any]:
        config = configparser.ConfigParser()
        config.read("config.ini", "utf-8")

        return {
            "host": os.environ.get(
                "POSTGRES_HOST",
                config.get("postgres", "host", fallback="localhost"),
            ),
            "port": os.environ.get(
                "POSTGRES_PORT", config.get("postgres", "port", fallback=5432)
            ),
            "user": os.environ.get(
                "POSTGRES_USER", config.get("postgres", "user", fallback="postgres")
            ),
            "password": os.environ.get(
                "POSTGRES_PASSWORD",
                config.get("postgres", "password", fallback=None),
            ),
            "database": os.environ.get(
                "POSTGRES_DATABASE",
                config.get("postgres", "database", fallback="postgres"),
            ),
            "workspace": os.environ.get(
                "POSTGRES_WORKSPACE",
                config.get("postgres", "workspace", fallback=None),
            ),
            "max_connections": os.environ.get(
                "POSTGRES_MAX_CONNECTIONS",
                config.get("postgres", "max_connections", fallback=50),
            ),
            # SSL configuration
            "ssl_mode": os.environ.get(
                "POSTGRES_SSL_MODE",
                config.get("postgres", "ssl_mode", fallback=None),
            ),
            "ssl_cert": os.environ.get(
                "POSTGRES_SSL_CERT",
                config.get("postgres", "ssl_cert", fallback=None),
            ),
            "ssl_key": os.environ.get(
                "POSTGRES_SSL_KEY",
                config.get("postgres", "ssl_key", fallback=None),
            ),
            "ssl_root_cert": os.environ.get(
                "POSTGRES_SSL_ROOT_CERT",
                config.get("postgres", "ssl_root_cert", fallback=None),
            ),
            "ssl_crl": os.environ.get(
                "POSTGRES_SSL_CRL",
                config.get("postgres", "ssl_crl", fallback=None),
            ),
            "vector_index_type": os.environ.get(
                "POSTGRES_VECTOR_INDEX_TYPE",
                config.get("postgres", "vector_index_type", fallback="HNSW"),
            ),
            "hnsw_m": int(
                os.environ.get(
                    "POSTGRES_HNSW_M",
                    config.get("postgres", "hnsw_m", fallback="16"),
                )
            ),
            "hnsw_ef": int(
                os.environ.get(
                    "POSTGRES_HNSW_EF",
                    config.get("postgres", "hnsw_ef", fallback="64"),
                )
            ),
            "ivfflat_lists": int(
                os.environ.get(
                    "POSTGRES_IVFFLAT_LISTS",
                    config.get("postgres", "ivfflat_lists", fallback="100"),
                )
            ),
            "vchordrq_build_options": os.environ.get(
                "POSTGRES_VCHORDRQ_BUILD_OPTIONS",
                config.get("postgres", "vchordrq_build_options", fallback=""),
            ),
            "vchordrq_probes": os.environ.get(
                "POSTGRES_VCHORDRQ_PROBES",
                config.get("postgres", "vchordrq_probes", fallback=""),
            ),
            "vchordrq_epsilon": float(
                os.environ.get(
                    "POSTGRES_VCHORDRQ_EPSILON",
                    config.get("postgres", "vchordrq_epsilon", fallback="1.9"),
                )
            ),
            # Server settings for Supabase
            "server_settings": os.environ.get(
                "POSTGRES_SERVER_SETTINGS",
                config.get("postgres", "server_options", fallback=None),
            ),
            "statement_cache_size": os.environ.get(
                "POSTGRES_STATEMENT_CACHE_SIZE",
                config.get("postgres", "statement_cache_size", fallback=None),
            ),
            # Connection retry configuration
            "connection_retry_attempts": min(
                10,
                int(
                    os.environ.get(
                        "POSTGRES_CONNECTION_RETRIES",
                        config.get("postgres", "connection_retries", fallback=3),
                    )
                ),
            ),
            "connection_retry_backoff": min(
                5.0,
                float(
                    os.environ.get(
                        "POSTGRES_CONNECTION_RETRY_BACKOFF",
                        config.get(
                            "postgres", "connection_retry_backoff", fallback=0.5
                        ),
                    )
                ),
            ),
            "connection_retry_backoff_max": min(
                60.0,
                float(
                    os.environ.get(
                        "POSTGRES_CONNECTION_RETRY_BACKOFF_MAX",
                        config.get(
                            "postgres",
                            "connection_retry_backoff_max",
                            fallback=5.0,
                        ),
                    )
                ),
            ),
            "pool_close_timeout": min(
                30.0,
                float(
                    os.environ.get(
                        "POSTGRES_POOL_CLOSE_TIMEOUT",
                        config.get("postgres", "pool_close_timeout", fallback=5.0),
                    )
                ),
            ),
        }

    @classmethod
    async def get_client(cls) -> PostgreSQLDB:
        async with cls._lock:
            if cls._instances["db"] is None:
                config = ClientManager.get_config()
                db = PostgreSQLDB(config)
                await db.initdb()
                await db.check_tables()
                cls._instances["db"] = db
                cls._instances["ref_count"] = 0
            cls._instances["ref_count"] += 1
            return cls._instances["db"]

    @classmethod
    async def release_client(cls, db: PostgreSQLDB):
        async with cls._lock:
            if db is not None:
                if db is cls._instances["db"]:
                    cls._instances["ref_count"] -= 1
                    if cls._instances["ref_count"] == 0:
                        await db.pool.close()
                        logger.info("Closed PostgreSQL database connection pool")
                        cls._instances["db"] = None
                else:
                    await db.pool.close()




NAMESPACE_TABLE_MAP = {
    NameSpace.KV_STORE_FULL_DOCS: "EKR_DOC_FULL",
    NameSpace.KV_STORE_TEXT_CHUNKS: "EKR_DOC_CHUNKS",
    NameSpace.KV_STORE_FULL_ENTITIES: "EKR_FULL_ENTITIES",
    NameSpace.KV_STORE_FULL_RELATIONS: "EKR_FULL_RELATIONS",
    NameSpace.KV_STORE_ENTITY_CHUNKS: "EKR_ENTITY_CHUNKS",
    NameSpace.KV_STORE_RELATION_CHUNKS: "EKR_RELATION_CHUNKS",
    NameSpace.KV_STORE_LLM_RESPONSE_CACHE: "EKR_LLM_CACHE",
    NameSpace.VECTOR_STORE_CHUNKS: "EKR_VDB_CHUNKS",
    NameSpace.VECTOR_STORE_ENTITIES: "EKR_VDB_ENTITY",
    NameSpace.VECTOR_STORE_RELATIONSHIPS: "EKR_VDB_RELATION",
    NameSpace.DOC_STATUS: "EKR_DOC_STATUS",
}


def namespace_to_table_name(namespace: str) -> str:
    for k, v in NAMESPACE_TABLE_MAP.items():
        if is_namespace(namespace, k):
            return v


TABLES = {
    "EKR_DOC_FULL": {
        "ddl": """CREATE TABLE EKR_DOC_FULL (
                    id VARCHAR(255),
                    workspace VARCHAR(255),
                    doc_name VARCHAR(1024),
                    content TEXT,
                    meta JSONB,
                    create_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                    update_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
	                CONSTRAINT EKR_DOC_FULL_PK PRIMARY KEY (workspace, id)
                    )"""
    },
    "EKR_DOC_CHUNKS": {
        "ddl": """CREATE TABLE EKR_DOC_CHUNKS (
                    id VARCHAR(255),
                    workspace VARCHAR(255),
                    full_doc_id VARCHAR(256),
                    chunk_order_index INTEGER,
                    tokens INTEGER,
                    content TEXT,
                    file_path TEXT NULL,
                    llm_cache_list JSONB NULL DEFAULT '[]'::jsonb,
                    create_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                    update_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
	                CONSTRAINT EKR_DOC_CHUNKS_PK PRIMARY KEY (workspace, id)
                    )"""
    },
    "EKR_VDB_CHUNKS": {
        "ddl": f"""CREATE TABLE EKR_VDB_CHUNKS (
                    id VARCHAR(255),
                    workspace VARCHAR(255),
                    full_doc_id VARCHAR(256),
                    chunk_order_index INTEGER,
                    tokens INTEGER,
                    content TEXT,
                    content_vector VECTOR({os.environ.get("EMBEDDING_DIM", 1024)}),
                    file_path TEXT NULL,
                    create_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                    update_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
	                CONSTRAINT EKR_VDB_CHUNKS_PK PRIMARY KEY (workspace, id)
                    )"""
    },
    "EKR_VDB_ENTITY": {
        "ddl": f"""CREATE TABLE EKR_VDB_ENTITY (
                    id VARCHAR(255),
                    workspace VARCHAR(255),
                    entity_name VARCHAR(512),
                    content TEXT,
                    content_vector VECTOR({os.environ.get("EMBEDDING_DIM", 1024)}),
                    create_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                    update_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                    chunk_ids VARCHAR(255)[] NULL,
                    file_path TEXT NULL,
	                CONSTRAINT EKR_VDB_ENTITY_PK PRIMARY KEY (workspace, id)
                    )"""
    },
    "EKR_VDB_RELATION": {
        "ddl": f"""CREATE TABLE EKR_VDB_RELATION (
                    id VARCHAR(255),
                    workspace VARCHAR(255),
                    source_id VARCHAR(512),
                    target_id VARCHAR(512),
                    content TEXT,
                    content_vector VECTOR({os.environ.get("EMBEDDING_DIM", 1024)}),
                    create_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                    update_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                    chunk_ids VARCHAR(255)[] NULL,
                    file_path TEXT NULL,
	                CONSTRAINT EKR_VDB_RELATION_PK PRIMARY KEY (workspace, id)
                    )"""
    },
    "EKR_LLM_CACHE": {
        "ddl": """CREATE TABLE EKR_LLM_CACHE (
	                workspace varchar(255) NOT NULL,
	                id varchar(255) NOT NULL,
                    original_prompt TEXT,
                    return_value TEXT,
                    chunk_id VARCHAR(255) NULL,
                    cache_type VARCHAR(32),
                    queryparam JSONB NULL,
                    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
	                CONSTRAINT EKR_LLM_CACHE_PK PRIMARY KEY (workspace, id)
                    )"""
    },
    "EKR_DOC_STATUS": {
        "ddl": """CREATE TABLE EKR_DOC_STATUS (
	               workspace varchar(255) NOT NULL,
	               id varchar(255) NOT NULL,
	               content_summary varchar(255) NULL,
	               content_length int4 NULL,
	               chunks_count int4 NULL,
	               status varchar(64) NULL,
	               file_path TEXT NULL,
	               chunks_list JSONB NULL DEFAULT '[]'::jsonb,
	               track_id varchar(255) NULL,
	               metadata JSONB NULL DEFAULT '{}'::jsonb,
	               error_msg TEXT NULL,
	               created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
	               updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
	               CONSTRAINT EKR_DOC_STATUS_PK PRIMARY KEY (workspace, id)
	              )"""
    },
    "EKR_FULL_ENTITIES": {
        "ddl": """CREATE TABLE EKR_FULL_ENTITIES (
                    id VARCHAR(255),
                    workspace VARCHAR(255),
                    entity_names JSONB,
                    count INTEGER,
                    create_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                    update_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                    CONSTRAINT EKR_FULL_ENTITIES_PK PRIMARY KEY (workspace, id)
                    )"""
    },
    "EKR_FULL_RELATIONS": {
        "ddl": """CREATE TABLE EKR_FULL_RELATIONS (
                    id VARCHAR(255),
                    workspace VARCHAR(255),
                    relation_pairs JSONB,
                    count INTEGER,
                    create_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                    update_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                    CONSTRAINT EKR_FULL_RELATIONS_PK PRIMARY KEY (workspace, id)
                    )"""
    },
    "EKR_ENTITY_CHUNKS": {
        "ddl": """CREATE TABLE EKR_ENTITY_CHUNKS (
                    id VARCHAR(512),
                    workspace VARCHAR(255),
                    chunk_ids JSONB,
                    count INTEGER,
                    create_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                    update_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                    CONSTRAINT EKR_ENTITY_CHUNKS_PK PRIMARY KEY (workspace, id)
                    )"""
    },
    "EKR_RELATION_CHUNKS": {
        "ddl": """CREATE TABLE EKR_RELATION_CHUNKS (
                    id VARCHAR(512),
                    workspace VARCHAR(255),
                    chunk_ids JSONB,
                    count INTEGER,
                    create_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                    update_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                    CONSTRAINT EKR_RELATION_CHUNKS_PK PRIMARY KEY (workspace, id)
                    )"""
    },
}


SQL_TEMPLATES = {
    # SQL for KVStorage
    "get_by_id_full_docs": """SELECT id, COALESCE(content, '') as content,
                                COALESCE(doc_name, '') as file_path
                                FROM EKR_DOC_FULL WHERE workspace=$1 AND id=$2
                            """,
    "get_by_id_text_chunks": """SELECT id, tokens, COALESCE(content, '') as content,
                                chunk_order_index, full_doc_id, file_path,
                                COALESCE(llm_cache_list, '[]'::jsonb) as llm_cache_list,
                                EXTRACT(EPOCH FROM create_time)::BIGINT as create_time,
                                EXTRACT(EPOCH FROM update_time)::BIGINT as update_time
                                FROM EKR_DOC_CHUNKS WHERE workspace=$1 AND id=$2
                            """,
    "get_by_id_llm_response_cache": """SELECT id, original_prompt, return_value, chunk_id, cache_type, queryparam,
                                EXTRACT(EPOCH FROM create_time)::BIGINT as create_time,
                                EXTRACT(EPOCH FROM update_time)::BIGINT as update_time
                                FROM EKR_LLM_CACHE WHERE workspace=$1 AND id=$2
                               """,
    "get_by_ids_full_docs": """SELECT id, COALESCE(content, '') as content,
                                 COALESCE(doc_name, '') as file_path
                                 FROM EKR_DOC_FULL WHERE workspace=$1 AND id = ANY($2)
                            """,
    "get_by_ids_text_chunks": """SELECT id, tokens, COALESCE(content, '') as content,
                                  chunk_order_index, full_doc_id, file_path,
                                  COALESCE(llm_cache_list, '[]'::jsonb) as llm_cache_list,
                                  EXTRACT(EPOCH FROM create_time)::BIGINT as create_time,
                                  EXTRACT(EPOCH FROM update_time)::BIGINT as update_time
                                   FROM EKR_DOC_CHUNKS WHERE workspace=$1 AND id = ANY($2)
                                """,
    "get_by_ids_llm_response_cache": """SELECT id, original_prompt, return_value, chunk_id, cache_type, queryparam,
                                 EXTRACT(EPOCH FROM create_time)::BIGINT as create_time,
                                 EXTRACT(EPOCH FROM update_time)::BIGINT as update_time
                                 FROM EKR_LLM_CACHE WHERE workspace=$1 AND id = ANY($2)
                                """,
    "get_by_id_full_entities": """SELECT id, entity_names, count,
                                EXTRACT(EPOCH FROM create_time)::BIGINT as create_time,
                                EXTRACT(EPOCH FROM update_time)::BIGINT as update_time
                                FROM EKR_FULL_ENTITIES WHERE workspace=$1 AND id=$2
                               """,
    "get_by_id_full_relations": """SELECT id, relation_pairs, count,
                                EXTRACT(EPOCH FROM create_time)::BIGINT as create_time,
                                EXTRACT(EPOCH FROM update_time)::BIGINT as update_time
                                FROM EKR_FULL_RELATIONS WHERE workspace=$1 AND id=$2
                               """,
    "get_by_ids_full_entities": """SELECT id, entity_names, count,
                                 EXTRACT(EPOCH FROM create_time)::BIGINT as create_time,
                                 EXTRACT(EPOCH FROM update_time)::BIGINT as update_time
                                 FROM EKR_FULL_ENTITIES WHERE workspace=$1 AND id = ANY($2)
                                """,
    "get_by_ids_full_relations": """SELECT id, relation_pairs, count,
                                 EXTRACT(EPOCH FROM create_time)::BIGINT as create_time,
                                 EXTRACT(EPOCH FROM update_time)::BIGINT as update_time
                                 FROM EKR_FULL_RELATIONS WHERE workspace=$1 AND id = ANY($2)
                                """,
    "get_by_id_entity_chunks": """SELECT id, chunk_ids, count,
                                EXTRACT(EPOCH FROM create_time)::BIGINT as create_time,
                                EXTRACT(EPOCH FROM update_time)::BIGINT as update_time
                                FROM EKR_ENTITY_CHUNKS WHERE workspace=$1 AND id=$2
                               """,
    "get_by_id_relation_chunks": """SELECT id, chunk_ids, count,
                                EXTRACT(EPOCH FROM create_time)::BIGINT as create_time,
                                EXTRACT(EPOCH FROM update_time)::BIGINT as update_time
                                FROM EKR_RELATION_CHUNKS WHERE workspace=$1 AND id=$2
                               """,
    "get_by_ids_entity_chunks": """SELECT id, chunk_ids, count,
                                 EXTRACT(EPOCH FROM create_time)::BIGINT as create_time,
                                 EXTRACT(EPOCH FROM update_time)::BIGINT as update_time
                                 FROM EKR_ENTITY_CHUNKS WHERE workspace=$1 AND id = ANY($2)
                                """,
    "get_by_ids_relation_chunks": """SELECT id, chunk_ids, count,
                                 EXTRACT(EPOCH FROM create_time)::BIGINT as create_time,
                                 EXTRACT(EPOCH FROM update_time)::BIGINT as update_time
                                 FROM EKR_RELATION_CHUNKS WHERE workspace=$1 AND id = ANY($2)
                                """,
    "filter_keys": "SELECT id FROM {table_name} WHERE workspace=$1 AND id IN ({ids})",
    "upsert_doc_full": """INSERT INTO EKR_DOC_FULL (id, content, doc_name, workspace)
                        VALUES ($1, $2, $3, $4)
                        ON CONFLICT (workspace,id) DO UPDATE
                           SET content = $2,
                               doc_name = $3,
                               update_time = CURRENT_TIMESTAMP
                       """,
    "upsert_llm_response_cache": """INSERT INTO EKR_LLM_CACHE(workspace,id,original_prompt,return_value,chunk_id,cache_type,queryparam)
                                      VALUES ($1, $2, $3, $4, $5, $6, $7)
                                      ON CONFLICT (workspace,id) DO UPDATE
                                      SET original_prompt = EXCLUDED.original_prompt,
                                      return_value=EXCLUDED.return_value,
                                      chunk_id=EXCLUDED.chunk_id,
                                      cache_type=EXCLUDED.cache_type,
                                      queryparam=EXCLUDED.queryparam,
                                      update_time = CURRENT_TIMESTAMP
                                     """,
    "upsert_text_chunk": """INSERT INTO EKR_DOC_CHUNKS (workspace, id, tokens,
                      chunk_order_index, full_doc_id, content, file_path, llm_cache_list,
                      create_time, update_time)
                      VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                      ON CONFLICT (workspace,id) DO UPDATE
                      SET tokens=EXCLUDED.tokens,
                      chunk_order_index=EXCLUDED.chunk_order_index,
                      full_doc_id=EXCLUDED.full_doc_id,
                      content = EXCLUDED.content,
                      file_path=EXCLUDED.file_path,
                      llm_cache_list=EXCLUDED.llm_cache_list,
                      update_time = EXCLUDED.update_time
                     """,
    "upsert_full_entities": """INSERT INTO EKR_FULL_ENTITIES (workspace, id, entity_names, count,
                      create_time, update_time)
                      VALUES ($1, $2, $3, $4, $5, $6)
                      ON CONFLICT (workspace,id) DO UPDATE
                      SET entity_names=EXCLUDED.entity_names,
                      count=EXCLUDED.count,
                      update_time = EXCLUDED.update_time
                     """,
    "upsert_full_relations": """INSERT INTO EKR_FULL_RELATIONS (workspace, id, relation_pairs, count,
                      create_time, update_time)
                      VALUES ($1, $2, $3, $4, $5, $6)
                      ON CONFLICT (workspace,id) DO UPDATE
                      SET relation_pairs=EXCLUDED.relation_pairs,
                      count=EXCLUDED.count,
                      update_time = EXCLUDED.update_time
                     """,
    "upsert_entity_chunks": """INSERT INTO EKR_ENTITY_CHUNKS (workspace, id, chunk_ids, count,
                      create_time, update_time)
                      VALUES ($1, $2, $3, $4, $5, $6)
                      ON CONFLICT (workspace,id) DO UPDATE
                      SET chunk_ids=EXCLUDED.chunk_ids,
                      count=EXCLUDED.count,
                      update_time = EXCLUDED.update_time
                     """,
    "upsert_relation_chunks": """INSERT INTO EKR_RELATION_CHUNKS (workspace, id, chunk_ids, count,
                      create_time, update_time)
                      VALUES ($1, $2, $3, $4, $5, $6)
                      ON CONFLICT (workspace,id) DO UPDATE
                      SET chunk_ids=EXCLUDED.chunk_ids,
                      count=EXCLUDED.count,
                      update_time = EXCLUDED.update_time
                     """,
    # SQL for VectorStorage
    "upsert_chunk": """INSERT INTO EKR_VDB_CHUNKS (workspace, id, tokens,
                      chunk_order_index, full_doc_id, content, content_vector, file_path,
                      create_time, update_time)
                      VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                      ON CONFLICT (workspace,id) DO UPDATE
                      SET tokens=EXCLUDED.tokens,
                      chunk_order_index=EXCLUDED.chunk_order_index,
                      full_doc_id=EXCLUDED.full_doc_id,
                      content = EXCLUDED.content,
                      content_vector=EXCLUDED.content_vector,
                      file_path=EXCLUDED.file_path,
                      update_time = EXCLUDED.update_time
                     """,
    "upsert_entity": """INSERT INTO EKR_VDB_ENTITY (workspace, id, entity_name, content,
                      content_vector, chunk_ids, file_path, create_time, update_time)
                      VALUES ($1, $2, $3, $4, $5, $6::varchar[], $7, $8, $9)
                      ON CONFLICT (workspace,id) DO UPDATE
                      SET entity_name=EXCLUDED.entity_name,
                      content=EXCLUDED.content,
                      content_vector=EXCLUDED.content_vector,
                      chunk_ids=EXCLUDED.chunk_ids,
                      file_path=EXCLUDED.file_path,
                      update_time=EXCLUDED.update_time
                     """,
    "upsert_relationship": """INSERT INTO EKR_VDB_RELATION (workspace, id, source_id,
                      target_id, content, content_vector, chunk_ids, file_path, create_time, update_time)
                      VALUES ($1, $2, $3, $4, $5, $6, $7::varchar[], $8, $9, $10)
                      ON CONFLICT (workspace,id) DO UPDATE
                      SET source_id=EXCLUDED.source_id,
                      target_id=EXCLUDED.target_id,
                      content=EXCLUDED.content,
                      content_vector=EXCLUDED.content_vector,
                      chunk_ids=EXCLUDED.chunk_ids,
                      file_path=EXCLUDED.file_path,
                      update_time = EXCLUDED.update_time
                     """,
    "relationships": """
                     SELECT r.source_id AS src_id,
                            r.target_id AS tgt_id,
                            EXTRACT(EPOCH FROM r.create_time)::BIGINT AS created_at
                     FROM EKR_VDB_RELATION r
                     WHERE r.workspace = $1
                       AND r.content_vector <=> '[{embedding_string}]'::vector < $2
                     ORDER BY r.content_vector <=> '[{embedding_string}]'::vector
                     LIMIT $3;
                     """,
    "entities": """
                SELECT e.entity_name,
                       EXTRACT(EPOCH FROM e.create_time)::BIGINT AS created_at
                FROM EKR_VDB_ENTITY e
                WHERE e.workspace = $1
                  AND e.content_vector <=> '[{embedding_string}]'::vector < $2
                ORDER BY e.content_vector <=> '[{embedding_string}]'::vector
                LIMIT $3;
                """,
    "chunks": """
              SELECT c.id,
                     c.content,
                     c.file_path,
                     EXTRACT(EPOCH FROM c.create_time)::BIGINT AS created_at
              FROM EKR_VDB_CHUNKS c
              WHERE c.workspace = $1
                AND c.content_vector <=> '[{embedding_string}]'::vector < $2
              ORDER BY c.content_vector <=> '[{embedding_string}]'::vector
              LIMIT $3;
              """,
    # DROP tables
    "drop_specifiy_table_workspace": """
        DELETE FROM {table_name} WHERE workspace=$1
       """,
}
