import os
import asyncio
import multiprocessing as mp
from multiprocessing import Manager
import time
from typing import Any, Dict, List, Optional

from .exceptions import PipelineNotInitializedError
from easy_knowledge_retriever.kg import state
from easy_knowledge_retriever.kg.utils import direct_log, get_final_namespace
from easy_knowledge_retriever.kg.concurrency import (
    KeyedUnifiedLock,
    get_internal_lock,
    get_namespace_lock,
)


def initialize_share_data(workers: int = 1):
    """
    Initialize shared storage data for single or multi-process mode.
    
    Args:
        workers: Number of workers. If > 1, initializes multiprocessing shared data.
    """
    # Check if already initialized
    if state._initialized:
        direct_log(
            f"Process {os.getpid()} Shared-Data already initialized (multiprocess={state._is_multiprocess})"
        )
        return

    state._workers = workers

    if workers > 1:
        state._is_multiprocess = True
        state._manager = Manager()
        state._lock_registry = state._manager.dict()
        state._lock_registry_count = state._manager.dict()
        state._lock_cleanup_data = state._manager.dict()
        state._registry_guard = state._manager.RLock()
        state._internal_lock = state._manager.Lock()
        state._data_init_lock = state._manager.Lock()
        state._shared_dicts = state._manager.dict()
        state._init_flags = state._manager.dict()
        state._update_flags = state._manager.dict()

        state._storage_keyed_lock = KeyedUnifiedLock()

        # Initialize async locks for multiprocess mode
        state._async_locks = {
            "internal_lock": asyncio.Lock(),
            "graph_db_lock": asyncio.Lock(),
            "data_init_lock": asyncio.Lock(),
        }

        direct_log(
            f"Process {os.getpid()} Shared-Data created for Multiple Process (workers={workers})"
        )
    else:
        state._is_multiprocess = False
        state._internal_lock = asyncio.Lock()
        state._data_init_lock = asyncio.Lock()
        state._shared_dicts = {}
        state._init_flags = {}
        state._update_flags = {}
        state._async_locks = None  # No need for async locks in single process mode

        state._storage_keyed_lock = KeyedUnifiedLock()
        direct_log(f"Process {os.getpid()} Shared-Data created for Single Process")

    # Initialize multiprocess cleanup times
    state._earliest_mp_cleanup_time = None
    state._last_mp_cleanup_time = None

    # Mark as initialized
    state._initialized = True


def finalize_share_data():
    """
    Release shared resources and clean up.
    """
    # Check if already initialized
    if not state._initialized:
        direct_log(
            f"Process {os.getpid()} storage data not initialized, nothing to finalize"
        )
        return

    direct_log(
        f"Process {os.getpid()} finalizing storage data (multiprocess={state._is_multiprocess})"
    )

    # In multi-process mode, shut down the Manager
    if state._is_multiprocess and state._manager is not None:
        try:
            # Clear shared resources before shutting down Manager
            if state._shared_dicts is not None:
                # Clear pipeline status history messages first if exists
                try:
                    pipeline_status = state._shared_dicts.get("pipeline_status", {})
                    if "history_messages" in pipeline_status:
                        pipeline_status["history_messages"].clear()
                except Exception:
                    pass  # Ignore any errors during history messages cleanup
                state._shared_dicts.clear()
            if state._init_flags is not None:
                state._init_flags.clear()
            if state._update_flags is not None:
                # Clear each namespace's update flags list and Value objects
                try:
                    for namespace in state._update_flags:
                        flags_list = state._update_flags[namespace]
                        if isinstance(flags_list, list):
                            # Clear Value objects in the list
                            for flag in flags_list:
                                if hasattr(
                                    flag, "value"
                                ):  # Check if it's a Value object
                                    flag.value = False
                            flags_list.clear()
                except Exception:
                    pass  # Ignore any errors during update flags cleanup
                state._update_flags.clear()

            # Shut down the Manager - this will automatically clean up all shared resources
            state._manager.shutdown()
            direct_log(f"Process {os.getpid()} Manager shutdown complete")
        except Exception as e:
            direct_log(
                f"Process {os.getpid()} Error shutting down Manager: {e}", level="ERROR"
            )

    # Reset global variables
    state._manager = None
    state._initialized = None
    state._is_multiprocess = None
    state._shared_dicts = None
    state._init_flags = None
    state._internal_lock = None
    state._data_init_lock = None
    state._update_flags = None
    state._async_locks = None
    state._default_workspace = None

    direct_log(f"Process {os.getpid()} storage data finalization complete")


def set_default_workspace(workspace: str | None = None):
    """
    Set default workspace for namespace operations for backward compatibility.
    """
    if workspace is None:
        workspace = ""
    state._default_workspace = workspace
    direct_log(
        f"Default workspace set to: '{state._default_workspace}' (empty means global)",
        level="DEBUG",
    )


def get_default_workspace() -> str:
    """
    Get default workspace for backward compatibility.
    """
    return state._default_workspace


async def get_namespace_data(
    namespace: str, first_init: bool = False, workspace: str | None = None
) -> Dict[str, Any]:
    """get the shared data reference for specific namespace"""
    if state._shared_dicts is None:
        direct_log(
            f"Error: Try to getnanmespace before it is initialized, pid={os.getpid()}",
            level="ERROR",
        )
        raise ValueError("Shared dictionaries not initialized")

    final_namespace = get_final_namespace(namespace, workspace)

    async with get_internal_lock():
        if final_namespace not in state._shared_dicts:
            # Special handling for pipeline_status namespace
            if (
                final_namespace.endswith(":pipeline_status")
                or final_namespace == "pipeline_status"
            ) and not first_init:
                # Check if pipeline_status should have been initialized but wasn't
                raise PipelineNotInitializedError(final_namespace)

            # For other namespaces or when allow_create=True, create them dynamically
            if state._is_multiprocess and state._manager is not None:
                state._shared_dicts[final_namespace] = state._manager.dict()
            else:
                state._shared_dicts[final_namespace] = {}

    return state._shared_dicts[final_namespace]


async def try_initialize_namespace(
    namespace: str, workspace: str | None = None
) -> bool:
    """
    Returns True if the current worker(process) gets initialization permission for loading data later.
    """
    if state._init_flags is None:
        raise ValueError("Try to create nanmespace before Shared-Data is initialized")

    final_namespace = get_final_namespace(namespace, workspace)

    async with get_internal_lock():
        if final_namespace not in state._init_flags:
            state._init_flags[final_namespace] = True
            direct_log(
                f"Process {os.getpid()} ready to initialize storage namespace: [{final_namespace}]"
            )
            return True
        direct_log(
            f"Process {os.getpid()} storage namespace already initialized: [{final_namespace}]"
        )

    return False


async def initialize_pipeline_status(workspace: str | None = None):
    """
    Initialize pipeline_status share data with default values.
    """
    pipeline_namespace = await get_namespace_data(
        "pipeline_status", first_init=True, workspace=workspace
    )

    async with get_internal_lock():
        # Check if already initialized by checking for required fields
        if "busy" in pipeline_namespace:
            return

        # Create a shared list object for history_messages
        history_messages = state._manager.list() if state._is_multiprocess else []
        pipeline_namespace.update(
            {
                "autoscanned": False,  # Auto-scan started
                "busy": False,  # Control concurrent processes
                "job_name": "-",  # Current job name (indexing files/indexing texts)
                "job_start": None,  # Job start time
                "docs": 0,  # Total number of documents to be indexed
                "batchs": 0,  # Number of batches for processing documents
                "cur_batch": 0,  # Current processing batch
                "request_pending": False,  # Flag for pending request for processing
                "latest_message": "",  # Latest message from pipeline processing
                "history_messages": history_messages,  # 使用共享列表对象
            }
        )

        final_namespace = get_final_namespace("pipeline_status", workspace)
        direct_log(
            f"Process {os.getpid()} Pipeline namespace '{final_namespace}' initialized"
        )


async def get_update_flag(namespace: str, workspace: str | None = None):
    """
    Create a namespace's update flag for a workers.
    """
    if state._update_flags is None:
        raise ValueError("Try to create namespace before Shared-Data is initialized")

    final_namespace = get_final_namespace(namespace, workspace)

    async with get_internal_lock():
        if final_namespace not in state._update_flags:
            if state._is_multiprocess and state._manager is not None:
                state._update_flags[final_namespace] = state._manager.list()
            else:
                state._update_flags[final_namespace] = []
            direct_log(
                f"Process {os.getpid()} initialized updated flags for namespace: [{final_namespace}]"
            )

        if state._is_multiprocess and state._manager is not None:
            new_update_flag = state._manager.Value("b", False)
        else:
            # Create a simple mutable object to store boolean value for compatibility with mutiprocess
            class MutableBoolean:
                def __init__(self, initial_value=False):
                    self.value = initial_value

            new_update_flag = MutableBoolean(False)

        state._update_flags[final_namespace].append(new_update_flag)
        return new_update_flag


async def set_all_update_flags(namespace: str, workspace: str | None = None):
    """Set all update flag of namespace indicating all workers need to reload data from files"""
    if state._update_flags is None:
        raise ValueError("Try to create namespace before Shared-Data is initialized")

    final_namespace = get_final_namespace(namespace, workspace)

    async with get_internal_lock():
        if final_namespace not in state._update_flags:
            raise ValueError(f"Namespace {final_namespace} not found in update flags")
        # Update flags for both modes
        for i in range(len(state._update_flags[final_namespace])):
            state._update_flags[final_namespace][i].value = True


async def clear_all_update_flags(namespace: str, workspace: str | None = None):
    """Clear all update flag of namespace indicating all workers need to reload data from files"""
    if state._update_flags is None:
        raise ValueError("Try to create namespace before Shared-Data is initialized")

    final_namespace = get_final_namespace(namespace, workspace)

    async with get_internal_lock():
        if final_namespace not in state._update_flags:
            raise ValueError(f"Namespace {final_namespace} not found in update flags")
        # Update flags for both modes
        for i in range(len(state._update_flags[final_namespace])):
            state._update_flags[final_namespace][i].value = False


async def get_all_update_flags_status(workspace: str | None = None) -> Dict[str, list]:
    """
    Get update flags status for all namespaces.
    """
    if state._update_flags is None:
        return {}

    if workspace is None:
        workspace = get_default_workspace()

    result = {}
    async with get_internal_lock():
        for namespace, flags in state._update_flags.items():
            # Check if namespace has a workspace prefix (contains ':')
            if ":" in namespace:
                # Namespace has workspace prefix like "space1:pipeline_status"
                # Only include if workspace matches the prefix
                # Use rsplit to split from the right since workspace can contain colons
                namespace_split = namespace.rsplit(":", 1)
                if not workspace or namespace_split[0] != workspace:
                    continue
            else:
                # Namespace has no workspace prefix like "pipeline_status"
                # Only include if we're querying the default (empty) workspace
                if workspace:
                    continue

            worker_statuses = []
            for flag in flags:
                if state._is_multiprocess:
                    worker_statuses.append(flag.value)
                else:
                    worker_statuses.append(flag)
            result[namespace] = worker_statuses

    return result
