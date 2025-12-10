from typing import Any, Dict, Optional, Union
import multiprocessing as mp
from multiprocessing.synchronize import Lock as ProcessLock
import asyncio
import time

# Type definitions for locks
LockType = Union[ProcessLock, asyncio.Lock]

# Global state variables
_manager: Optional[mp.Manager] = None
_workers: Optional[int] = None
_is_multiprocess: Optional[bool] = None

# Registry for locks and shared data
_lock_registry: Optional[Dict[str, ProcessLock]] = None
_lock_registry_count: Optional[Dict[str, int]] = None
_lock_cleanup_data: Optional[Dict[str, time.time]] = None
_registry_guard = None

# Specific locks
_internal_lock: Optional[LockType] = None
_data_init_lock: Optional[LockType] = None
_storage_keyed_lock = None  # KeyedUnifiedLock instance

# Shared dictionaries
_shared_dicts: Optional[Dict[str, Any]] = None
_init_flags: Optional[Dict[str, bool]] = None
_update_flags: Optional[Dict[str, bool]] = None

# Async locks for multiprocess mode
_async_locks: Optional[Dict[str, asyncio.Lock]] = None

# Cleanup time tracking
_earliest_mp_cleanup_time: Optional[float] = None
_last_mp_cleanup_time: Optional[float] = None

# Initialization flag
_initialized: bool = False

# Default workspace
_default_workspace: Optional[str] = None

# Debug counters
_debug_n_locks_acquired: int = 0
