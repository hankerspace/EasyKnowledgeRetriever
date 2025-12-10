import asyncio
import os
import time
from typing import Any, Dict, List, Optional, TypeVar, Generic, Union
from multiprocessing.synchronize import Lock as ProcessLock
from contextvars import ContextVar

from easy_knowledge_retriever.kg import state
from easy_knowledge_retriever.kg.utils import direct_log, get_combined_key, get_final_namespace, DEBUG_LOCKS

T = TypeVar("T")

# Constants
CLEANUP_KEYED_LOCKS_AFTER_SECONDS = 300
CLEANUP_THRESHOLD = 500
MIN_CLEANUP_INTERVAL_SECONDS = 30


def inc_debug_n_locks_acquired():
    if DEBUG_LOCKS:
        state._debug_n_locks_acquired += 1
        print(f"DEBUG: Keyed Lock acquired, total: {state._debug_n_locks_acquired:>5}")


def dec_debug_n_locks_acquired():
    if DEBUG_LOCKS:
        if state._debug_n_locks_acquired > 0:
            state._debug_n_locks_acquired -= 1
            print(f"DEBUG: Keyed Lock released, total: {state._debug_n_locks_acquired:>5}")
        else:
            raise RuntimeError("Attempting to release lock when no locks are acquired")


def get_debug_n_locks_acquired():
    return state._debug_n_locks_acquired


class UnifiedLock(Generic[T]):
    """Provide a unified lock interface type for asyncio.Lock and multiprocessing.Lock"""

    def __init__(
        self,
        lock: Union[ProcessLock, asyncio.Lock],
        is_async: bool,
        name: str = "unnamed",
        enable_logging: bool = True,
        async_lock: Optional[asyncio.Lock] = None,
    ):
        self._lock = lock
        self._is_async = is_async
        self._pid = os.getpid()  # for debug only
        self._name = name  # for debug only
        self._enable_logging = enable_logging  # for debug only
        self._async_lock = async_lock  # auxiliary lock for coroutine synchronization

    async def __aenter__(self) -> "UnifiedLock[T]":
        try:
            # If in multiprocess mode and async lock exists, acquire it first
            if not self._is_async and self._async_lock is not None:
                await self._async_lock.acquire()
                direct_log(
                    f"== Lock == Process {self._pid}: Acquired async lock '{self._name}",
                    level="DEBUG",
                    enable_output=self._enable_logging,
                )

            # Then acquire the main lock
            if self._is_async:
                await self._lock.acquire()
            else:
                self._lock.acquire()

            direct_log(
                f"== Lock == Process {self._pid}: Acquired lock {self._name} (async={self._is_async})",
                level="INFO",
                enable_output=self._enable_logging,
            )
            return self
        except Exception as e:
            # If main lock acquisition fails, release the async lock if it was acquired
            if (
                not self._is_async
                and self._async_lock is not None
                and self._async_lock.locked()
            ):
                self._async_lock.release()

            direct_log(
                f"== Lock == Process {self._pid}: Failed to acquire lock '{self._name}': {e}",
                level="ERROR",
                enable_output=True,
            )
            raise

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        main_lock_released = False
        try:
            # Release main lock first
            if self._is_async:
                self._lock.release()
            else:
                self._lock.release()
            main_lock_released = True

            direct_log(
                f"== Lock == Process {self._pid}: Released lock {self._name} (async={self._is_async})",
                level="INFO",
                enable_output=self._enable_logging,
            )

            # Then release async lock if in multiprocess mode
            if not self._is_async and self._async_lock is not None:
                self._async_lock.release()
                direct_log(
                    f"== Lock == Process {self._pid}: Released async lock {self._name}",
                    level="DEBUG",
                    enable_output=self._enable_logging,
                )

        except Exception as e:
            direct_log(
                f"== Lock == Process {self._pid}: Failed to release lock '{self._name}': {e}",
                level="ERROR",
                enable_output=True,
            )

            # If main lock release failed but async lock hasn't been released, try to release it
            if (
                not main_lock_released
                and not self._is_async
                and self._async_lock is not None
            ):
                try:
                    direct_log(
                        f"== Lock == Process {self._pid}: Attempting to release async lock after main lock failure",
                        level="DEBUG",
                        enable_output=self._enable_logging,
                    )
                    self._async_lock.release()
                    direct_log(
                        f"== Lock == Process {self._pid}: Successfully released async lock after main lock failure",
                        level="INFO",
                        enable_output=self._enable_logging,
                    )
                except Exception as inner_e:
                    direct_log(
                        f"== Lock == Process {self._pid}: Failed to release async lock after main lock failure: {inner_e}",
                        level="ERROR",
                        enable_output=True,
                    )

            raise

    def __enter__(self) -> "UnifiedLock[T]":
        """For backward compatibility"""
        try:
            if self._is_async:
                raise RuntimeError("Use 'async with' for lock")
            direct_log(
                f"== Lock == Process {self._pid}: Acquiring lock {self._name} (sync)",
                level="DEBUG",
                enable_output=self._enable_logging,
            )
            self._lock.acquire()
            direct_log(
                f"== Lock == Process {self._pid}: Acquired lock {self._name} (sync)",
                level="INFO",
                enable_output=self._enable_logging,
            )
            return self
        except Exception as e:
            direct_log(
                f"== Lock == Process {self._pid}: Failed to acquire lock '{self._name}' (sync): {e}",
                level="ERROR",
                enable_output=True,
            )
            raise

    def __exit__(self, exc_type, exc_val, exc_tb):
        """For backward compatibility"""
        try:
            if self._is_async:
                raise RuntimeError("Use 'async with' for lock")
            direct_log(
                f"== Lock == Process {self._pid}: Releasing lock '{self._name}' (sync)",
                level="DEBUG",
                enable_output=self._enable_logging,
            )
            self._lock.release()
            direct_log(
                f"== Lock == Process {self._pid}: Released lock {self._name} (sync)",
                level="INFO",
                enable_output=self._enable_logging,
            )
        except Exception as e:
            direct_log(
                f"== Lock == Process {self._pid}: Failed to release lock '{self._name}' (sync): {e}",
                level="ERROR",
                enable_output=True,
            )
            raise

    def locked(self) -> bool:
        if self._is_async:
            return self._lock.locked()
        else:
            # Multiprocessing Lock doesn't always have locked(), but RLock does.
            # Assuming standard Lock/RLock interface
            if hasattr(self._lock, "locked"):
                return self._lock.locked()
            return False  # Fallback if not supported


def _perform_lock_cleanup(
    lock_type: str,
    cleanup_data: Dict[str, float],
    lock_registry: Optional[Dict[str, Any]],
    lock_count: Optional[Dict[str, int]],
    earliest_cleanup_time: Optional[float],
    last_cleanup_time: Optional[float],
    current_time: float,
    threshold_check: bool = True,
) -> tuple[int, Optional[float], Optional[float]]:
    """
    Generic lock cleanup function to unify cleanup logic for both multiprocess and async locks.
    """
    if len(cleanup_data) == 0:
        return 0, earliest_cleanup_time, last_cleanup_time

    # If threshold check is needed and threshold not reached, return directly
    if threshold_check and len(cleanup_data) < CLEANUP_THRESHOLD:
        return 0, earliest_cleanup_time, last_cleanup_time

    # Time rollback detection
    if last_cleanup_time is not None and current_time < last_cleanup_time:
        direct_log(
            f"== {lock_type} Lock == Time rollback detected, resetting cleanup time",
            level="WARNING",
            enable_output=False,
        )
        last_cleanup_time = None

    # Check cleanup conditions
    has_expired_locks = (
        earliest_cleanup_time is not None
        and current_time - earliest_cleanup_time > CLEANUP_KEYED_LOCKS_AFTER_SECONDS
    )

    interval_satisfied = (
        last_cleanup_time is None
        or current_time - last_cleanup_time > MIN_CLEANUP_INTERVAL_SECONDS
    )

    if not (has_expired_locks and interval_satisfied):
        return 0, earliest_cleanup_time, last_cleanup_time

    try:
        cleaned_count = 0
        new_earliest_time = None

        # Calculate total count before cleanup
        total_cleanup_len = len(cleanup_data)

        # Perform cleanup operation
        for cleanup_key, cleanup_time in list(cleanup_data.items()):
            if current_time - cleanup_time > CLEANUP_KEYED_LOCKS_AFTER_SECONDS:
                # Remove from cleanup data
                cleanup_data.pop(cleanup_key, None)

                # Remove from lock registry if exists
                if lock_registry is not None:
                    lock_registry.pop(cleanup_key, None)
                if lock_count is not None:
                    lock_count.pop(cleanup_key, None)

                cleaned_count += 1
            else:
                # Track the earliest time among remaining locks
                if new_earliest_time is None or cleanup_time < new_earliest_time:
                    new_earliest_time = cleanup_time

        # Update state only after successful cleanup
        if cleaned_count > 0:
            new_last_cleanup_time = current_time

            # Log cleanup results
            next_cleanup_in = max(
                (new_earliest_time + CLEANUP_KEYED_LOCKS_AFTER_SECONDS - current_time)
                if new_earliest_time
                else float("inf"),
                MIN_CLEANUP_INTERVAL_SECONDS,
            )

            direct_log(
                f"== {lock_type} Lock == Cleaned up {cleaned_count}/{total_cleanup_len} expired locks, "
                f"next cleanup in {next_cleanup_in:.1f}s",
                enable_output=False,
                level="INFO",
            )

            return cleaned_count, new_earliest_time, new_last_cleanup_time
        else:
            return 0, earliest_cleanup_time, last_cleanup_time

    except Exception as e:
        direct_log(
            f"== {lock_type} Lock == Cleanup failed: {e}",
            level="ERROR",
            enable_output=True,
        )
        return 0, earliest_cleanup_time, last_cleanup_time


def _get_or_create_shared_raw_mp_lock(
    factory_name: str, key: str
) -> Optional[Any]:
    """Return the *singleton* manager.Lock() proxy for keyed lock, creating if needed."""
    if not state._is_multiprocess:
        return None

    with state._registry_guard:
        combined_key = get_combined_key(factory_name, key)
        raw = state._lock_registry.get(combined_key)
        count = state._lock_registry_count.get(combined_key)
        if raw is None:
            raw = state._manager.Lock()
            state._lock_registry[combined_key] = raw
            count = 0
        else:
            if count is None:
                raise RuntimeError(
                    f"Shared-Data lock registry for {factory_name} is corrupted for key {key}"
                )
            if (
                count == 0 and combined_key in state._lock_cleanup_data
            ):  # Reusing an key waiting for cleanup, remove it from cleanup list
                state._lock_cleanup_data.pop(combined_key)
        count += 1
        state._lock_registry_count[combined_key] = count
        return raw


def _release_shared_raw_mp_lock(factory_name: str, key: str):
    """Release the *singleton* manager.Lock() proxy for *key*."""
    if not state._is_multiprocess:
        return

    with state._registry_guard:
        combined_key = get_combined_key(factory_name, key)
        raw = state._lock_registry.get(combined_key)
        count = state._lock_registry_count.get(combined_key)
        if raw is None and count is None:
            return
        elif raw is None or count is None:
            raise RuntimeError(
                f"Shared-Data lock registry for {factory_name} is corrupted for key {key}"
            )

        count -= 1
        if count < 0:
            raise RuntimeError(
                f"Attempting to release lock for {key} more times than it was acquired"
            )

        state._lock_registry_count[combined_key] = count

        current_time = time.time()
        if count == 0:
            state._lock_cleanup_data[combined_key] = current_time

            # Update earliest multiprocess cleanup time (only when earlier)
            if (
                state._earliest_mp_cleanup_time is None
                or current_time < state._earliest_mp_cleanup_time
            ):
                state._earliest_mp_cleanup_time = current_time

        # Use generic cleanup function
        cleaned_count, new_earliest_time, new_last_cleanup_time = _perform_lock_cleanup(
            lock_type="mp",
            cleanup_data=state._lock_cleanup_data,
            lock_registry=state._lock_registry,
            lock_count=state._lock_registry_count,
            earliest_cleanup_time=state._earliest_mp_cleanup_time,
            last_cleanup_time=state._last_mp_cleanup_time,
            current_time=current_time,
            threshold_check=True,
        )

        # Update global state if cleanup was performed
        if cleaned_count > 0:
            state._earliest_mp_cleanup_time = new_earliest_time
            state._last_mp_cleanup_time = new_last_cleanup_time


class KeyedUnifiedLock:
    """
    Manager for unified keyed locks, supporting both single and multi-process
    """

    def __init__(self, *, default_enable_logging: bool = True) -> None:
        self._default_enable_logging = default_enable_logging
        self._async_lock: Dict[str, asyncio.Lock] = {}  # local keyed locks
        self._async_lock_count: Dict[
            str, int
        ] = {}  # local keyed locks referenced count
        self._async_lock_cleanup_data: Dict[
            str, time.time
        ] = {}  # local keyed locks timeout
        # self._mp_locks is not needed since we access state._lock_registry directly via helpers
        
        self._earliest_async_cleanup_time: Optional[float] = (
            None  # track earliest async cleanup time
        )
        self._last_async_cleanup_time: Optional[float] = (
            None  # track last async cleanup time for minimum interval
        )

    def __call__(
        self, namespace: str, keys: list[str], *, enable_logging: Optional[bool] = None
    ):
        """
        Ergonomic helper so you can write:

            async with storage_keyed_lock("namespace", ["key1", "key2"]):
                ...
        """
        if enable_logging is None:
            enable_logging = self._default_enable_logging
        return _KeyedLockContext(
            self,
            namespace=namespace,
            keys=keys,
            enable_logging=enable_logging,
        )

    def _get_or_create_async_lock(self, combined_key: str) -> asyncio.Lock:
        async_lock = self._async_lock.get(combined_key)
        count = self._async_lock_count.get(combined_key, 0)
        if async_lock is None:
            async_lock = asyncio.Lock()
            self._async_lock[combined_key] = async_lock
        elif count == 0 and combined_key in self._async_lock_cleanup_data:
            self._async_lock_cleanup_data.pop(combined_key)
        count += 1
        self._async_lock_count[combined_key] = count
        return async_lock

    def _release_async_lock(self, combined_key: str):
        count = self._async_lock_count.get(combined_key, 0)
        count -= 1

        current_time = time.time()
        if count == 0:
            self._async_lock_cleanup_data[combined_key] = current_time

            # Update earliest async cleanup time (only when earlier)
            if (
                self._earliest_async_cleanup_time is None
                or current_time < self._earliest_async_cleanup_time
            ):
                self._earliest_async_cleanup_time = current_time
        self._async_lock_count[combined_key] = count

        # Use generic cleanup function
        cleaned_count, new_earliest_time, new_last_cleanup_time = _perform_lock_cleanup(
            lock_type="async",
            cleanup_data=self._async_lock_cleanup_data,
            lock_registry=self._async_lock,
            lock_count=self._async_lock_count,
            earliest_cleanup_time=self._earliest_async_cleanup_time,
            last_cleanup_time=self._last_async_cleanup_time,
            current_time=current_time,
            threshold_check=True,
        )

        # Update instance state if cleanup was performed
        if cleaned_count > 0:
            self._earliest_async_cleanup_time = new_earliest_time
            self._last_async_cleanup_time = new_last_cleanup_time

    def _get_lock_for_key(
        self, namespace: str, key: str, enable_logging: bool = False
    ) -> UnifiedLock:
        # 1. Create combined key for this namespace:key combination
        combined_key = get_combined_key(namespace, key)

        # 2. get (or create) the per‑process async gate for this combined key
        # Is synchronous, so no need to acquire a lock
        async_lock = self._get_or_create_async_lock(combined_key)

        # 3. fetch the shared raw lock
        raw_lock = _get_or_create_shared_raw_mp_lock(namespace, key)
        is_multiprocess = raw_lock is not None
        if not is_multiprocess:
            raw_lock = async_lock

        # 4. build a *fresh* UnifiedLock with the chosen logging flag
        if is_multiprocess:
            return UnifiedLock(
                lock=raw_lock,
                is_async=False,  # manager.Lock is synchronous
                name=combined_key,
                enable_logging=enable_logging,
                async_lock=async_lock,  # prevents event‑loop blocking
            )
        else:
            return UnifiedLock(
                lock=raw_lock,
                is_async=True,
                name=combined_key,
                enable_logging=enable_logging,
                async_lock=None,  # No need for async lock in single process mode
            )

    def _release_lock_for_key(self, namespace: str, key: str):
        combined_key = get_combined_key(namespace, key)
        self._release_async_lock(combined_key)
        _release_shared_raw_mp_lock(namespace, key)

    def cleanup_expired_locks(self) -> Dict[str, Any]:
        """
        Cleanup expired locks for both async and multiprocess locks.
        """
        cleanup_stats = {"mp_cleaned": 0, "async_cleaned": 0}

        current_time = time.time()

        # 1. Cleanup multiprocess locks using generic function
        if (
            state._is_multiprocess
            and state._lock_registry is not None
            and state._registry_guard is not None
        ):
            try:
                with state._registry_guard:
                    if state._lock_cleanup_data is not None:
                        # Use generic cleanup function without threshold check
                        cleaned_count, new_earliest_time, new_last_cleanup_time = (
                            _perform_lock_cleanup(
                                lock_type="mp",
                                cleanup_data=state._lock_cleanup_data,
                                lock_registry=state._lock_registry,
                                lock_count=state._lock_registry_count,
                                earliest_cleanup_time=state._earliest_mp_cleanup_time,
                                last_cleanup_time=state._last_mp_cleanup_time,
                                current_time=current_time,
                                threshold_check=False,  # Force cleanup in cleanup_expired_locks
                            )
                        )

                        # Update global state if cleanup was performed
                        if cleaned_count > 0:
                            state._earliest_mp_cleanup_time = new_earliest_time
                            state._last_mp_cleanup_time = new_last_cleanup_time
                            cleanup_stats["mp_cleaned"] = cleaned_count

            except Exception as e:
                direct_log(
                    f"Error during multiprocess lock cleanup: {e}",
                    level="ERROR",
                    enable_output=True,
                )

        # 2. Cleanup async locks using generic function
        try:
            # Use generic cleanup function without threshold check
            cleaned_count, new_earliest_time, new_last_cleanup_time = (
                _perform_lock_cleanup(
                    lock_type="async",
                    cleanup_data=self._async_lock_cleanup_data,
                    lock_registry=self._async_lock,
                    lock_count=self._async_lock_count,
                    earliest_cleanup_time=self._earliest_async_cleanup_time,
                    last_cleanup_time=self._last_async_cleanup_time,
                    current_time=current_time,
                    threshold_check=False,  # Force cleanup in cleanup_expired_locks
                )
            )

            # Update instance state if cleanup was performed
            if cleaned_count > 0:
                self._earliest_async_cleanup_time = new_earliest_time
                self._last_async_cleanup_time = new_last_cleanup_time
                cleanup_stats["async_cleaned"] = cleaned_count

        except Exception as e:
            direct_log(
                f"Error during async lock cleanup: {e}",
                level="ERROR",
                enable_output=True,
            )

        # 3. Get current status after cleanup
        current_status = self.get_lock_status()

        return {
            "process_id": os.getpid(),
            "cleanup_performed": cleanup_stats,
            "current_status": current_status,
        }

    def get_lock_status(self) -> Dict[str, int]:
        """
        Get current status of both async and multiprocess locks.
        """
        status = {
            "total_mp_locks": 0,
            "pending_mp_cleanup": 0,
            "total_async_locks": 0,
            "pending_async_cleanup": 0,
        }

        try:
            # Count multiprocess locks
            if state._is_multiprocess and state._lock_registry_count is not None:
                if state._registry_guard is not None:
                    with state._registry_guard:
                        status["total_mp_locks"] = len(state._lock_registry_count)
                        if state._lock_cleanup_data is not None:
                            status["pending_mp_cleanup"] = len(state._lock_cleanup_data)

            # Count async locks
            status["total_async_locks"] = len(self._async_lock_count)
            status["pending_async_cleanup"] = len(self._async_lock_cleanup_data)

        except Exception as e:
            direct_log(
                f"Error getting keyed lock status: {e}",
                level="ERROR",
                enable_output=True,
            )

        return status


class _KeyedLockContext:
    def __init__(
        self,
        parent: KeyedUnifiedLock,
        namespace: str,
        keys: list[str],
        enable_logging: bool,
    ) -> None:
        self._parent = parent
        self._namespace = namespace

        # The sorting is critical to ensure proper lock and release order
        # to avoid deadlocks
        self._keys = sorted(keys)
        self._enable_logging = (
            enable_logging
            if enable_logging is not None
            else parent._default_enable_logging
        )
        self._ul: Optional[List[Dict[str, Any]]] = None  # set in __aenter__

    # ----- enter -----
    async def __aenter__(self):
        if self._ul is not None:
            raise RuntimeError("KeyedUnifiedLock already acquired in current context")

        self._ul = []

        try:
            # Acquire locks for all keys in the namespace
            for key in self._keys:
                lock = None
                entry = None

                try:
                    # 1. Get lock object (reference count is incremented here)
                    lock = self._parent._get_lock_for_key(
                        self._namespace, key, enable_logging=self._enable_logging
                    )

                    # 2. Immediately create and add entry to list (critical for rollback to work)
                    entry = {
                        "key": key,
                        "lock": lock,
                        "entered": False,
                        "debug_inc": False,
                        "ref_incremented": True,  # Mark that reference count has been incremented
                    }
                    self._ul.append(
                        entry
                    )  # Add immediately after _get_lock_for_key for rollback to work

                    # 3. Try to acquire the lock
                    # Use try-finally to ensure state is updated atomically
                    lock_acquired = False
                    try:
                        await lock.__aenter__()
                        lock_acquired = True  # Lock successfully acquired
                    finally:
                        if lock_acquired:
                            entry["entered"] = True
                            inc_debug_n_locks_acquired()
                            entry["debug_inc"] = True

                except asyncio.CancelledError:
                    # Lock acquisition was cancelled
                    # The finally block above ensures entry["entered"] is correct
                    direct_log(
                        f"Lock acquisition cancelled for key {key}",
                        level="WARNING",
                        enable_output=self._enable_logging,
                    )
                    raise
                except Exception as e:
                    # Other exceptions, log and re-raise
                    direct_log(
                        f"Lock acquisition failed for key {key}: {e}",
                        level="ERROR",
                        enable_output=True,
                    )
                    raise

            return self

        except BaseException:
            # Critical: if any exception occurs (including CancelledError) during lock acquisition,
            # we must rollback all already acquired locks to prevent lock leaks
            # Use shield to ensure rollback completes
            await asyncio.shield(self._rollback_acquired_locks())
            raise

    async def _rollback_acquired_locks(self):
        """Rollback all acquired locks in case of exception during __aenter__"""
        if not self._ul:
            return

        async def rollback_single_entry(entry):
            """Rollback a single lock acquisition"""
            key = entry["key"]
            lock = entry["lock"]
            debug_inc = entry["debug_inc"]
            entered = entry["entered"]
            ref_incremented = entry.get(
                "ref_incremented", True
            )  # Default to True for safety

            errors = []

            # 1. If lock was acquired, release it
            if entered:
                try:
                    await lock.__aexit__(None, None, None)
                except Exception as e:
                    errors.append(("lock_exit", e))
                    direct_log(
                        f"Lock rollback error for key {key}: {e}",
                        level="ERROR",
                        enable_output=True,
                    )

            # 2. Release reference count (if it was incremented)
            if ref_incremented:
                try:
                    self._parent._release_lock_for_key(self._namespace, key)
                except Exception as e:
                    errors.append(("ref_release", e))
                    direct_log(
                        f"Lock rollback reference release error for key {key}: {e}",
                        level="ERROR",
                        enable_output=True,
                    )

            # 3. Decrement debug counter
            if debug_inc:
                try:
                    dec_debug_n_locks_acquired()
                except Exception as e:
                    errors.append(("debug_dec", e))
                    direct_log(
                        f"Lock rollback counter decrementing error for key {key}: {e}",
                        level="ERROR",
                        enable_output=True,
                    )

            return errors

        # Release already acquired locks in reverse order
        for entry in reversed(self._ul):
            # Use shield to protect each lock's rollback
            try:
                await asyncio.shield(rollback_single_entry(entry))
            except Exception as e:
                # Log but continue rolling back other locks
                direct_log(
                    f"Lock rollback unexpected error for {entry['key']}: {e}",
                    level="ERROR",
                    enable_output=True,
                )

        self._ul = None

    # ----- exit -----
    async def __aexit__(self, exc_type, exc, tb):
        if self._ul is None:
            return

        async def release_all_locks():
            """Release all locks with comprehensive error handling, protected from cancellation"""

            async def release_single_entry(entry, exc_type, exc, tb):
                """Release a single lock with full protection"""
                key = entry["key"]
                lock = entry["lock"]
                debug_inc = entry["debug_inc"]
                entered = entry["entered"]

                errors = []

                # 1. Release the lock
                if entered:
                    try:
                        await lock.__aexit__(exc_type, exc, tb)
                    except Exception as e:
                        errors.append(("lock_exit", e))
                        direct_log(
                            f"Lock release error for key {key}: {e}",
                            level="ERROR",
                            enable_output=True,
                            )

                # 2. Release reference count
                try:
                    self._parent._release_lock_for_key(self._namespace, key)
                except Exception as e:
                    errors.append(("ref_release", e))
                    direct_log(
                        f"Lock release reference error for key {key}: {e}",
                        level="ERROR",
                        enable_output=True,
                    )

                # 3. Decrement debug counter
                if debug_inc:
                    try:
                        dec_debug_n_locks_acquired()
                    except Exception as e:
                        errors.append(("debug_dec", e))
                        direct_log(
                            f"Lock release counter decrementing error for key {key}: {e}",
                            level="ERROR",
                            enable_output=True,
                        )

                return errors

            all_errors = []

            # Release locks in reverse order
            # This entire loop is protected by the outer shield
            for entry in reversed(self._ul):
                try:
                    errors = await release_single_entry(entry, exc_type, exc, tb)
                    for error_type, error in errors:
                        all_errors.append((entry["key"], error_type, error))
                except Exception as e:
                    all_errors.append((entry["key"], "unexpected", e))
                    direct_log(
                        f"Lock release unexpected error for {entry['key']}: {e}",
                        level="ERROR",
                        enable_output=True,
                    )

            return all_errors

        # CRITICAL: Protect the entire release process with shield
        # This ensures that even if cancellation occurs, all locks are released
        try:
            all_errors = await asyncio.shield(release_all_locks())
        except Exception as e:
            direct_log(
                f"Critical error during __aexit__ cleanup: {e}",
                level="ERROR",
                enable_output=True,
            )
            all_errors = []
        finally:
            # Always clear the lock list, even if shield was cancelled
            self._ul = None

        # If there were release errors and no other exception, raise the first release error
        if all_errors and exc_type is None:
            raise all_errors[0][2]  # (key, error_type, error)


class NamespaceLock:
    """
    Reusable namespace lock wrapper that creates a fresh context on each use.
    """

    def __init__(
        self, namespace: str, workspace: str | None = None, enable_logging: bool = False
    ):
        self._namespace = namespace
        self._workspace = workspace
        self._enable_logging = enable_logging
        # Use ContextVar to provide per-coroutine storage for lock context
        # This ensures each coroutine has its own independent context
        self._ctx_var: ContextVar[Optional[_KeyedLockContext]] = ContextVar(
            "lock_ctx", default=None
        )

    async def __aenter__(self):
        """Create a fresh context each time we enter"""
        # Check if this coroutine already has an active lock context
        if self._ctx_var.get() is not None:
            raise RuntimeError(
                "NamespaceLock already acquired in current coroutine context"
            )

        final_namespace = get_final_namespace(self._namespace, self._workspace)
        ctx = get_storage_keyed_lock(
            ["default_key"],
            namespace=final_namespace,
            enable_logging=self._enable_logging,
        )

        # Acquire the lock first, then store context only after successful acquisition
        # This prevents the ContextVar from being set if acquisition fails (e.g., due to cancellation),
        # which would permanently brick the lock
        result = await ctx.__aenter__()
        self._ctx_var.set(ctx)
        return result

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the current context and clean up"""
        # Retrieve this coroutine's context
        ctx = self._ctx_var.get()
        if ctx is None:
            raise RuntimeError("NamespaceLock exited without being entered")

        result = await ctx.__aexit__(exc_type, exc_val, exc_tb)
        # Clear this coroutine's context
        self._ctx_var.set(None)
        return result


def get_internal_lock(enable_logging: bool = False) -> UnifiedLock:
    """return unified storage lock for data consistency"""
    async_lock = state._async_locks.get("internal_lock") if state._is_multiprocess else None
    return UnifiedLock(
        lock=state._internal_lock,
        is_async=not state._is_multiprocess,
        name="internal_lock",
        enable_logging=enable_logging,
        async_lock=async_lock,
    )


def get_storage_keyed_lock(
    keys: str | list[str], namespace: str = "default", enable_logging: bool = False
) -> _KeyedLockContext:
    """Return unified storage keyed lock for ensuring atomic operations across different namespaces"""
    if state._storage_keyed_lock is None:
        raise RuntimeError("Shared-Data is not initialized")
    if isinstance(keys, str):
        keys = [keys]
    return state._storage_keyed_lock(namespace, keys, enable_logging=enable_logging)


def get_data_init_lock(enable_logging: bool = False) -> UnifiedLock:
    """return unified data initialization lock for ensuring atomic data initialization"""
    async_lock = state._async_locks.get("data_init_lock") if state._is_multiprocess else None
    return UnifiedLock(
        lock=state._data_init_lock,
        is_async=not state._is_multiprocess,
        name="data_init_lock",
        enable_logging=enable_logging,
        async_lock=async_lock,
    )


def cleanup_keyed_lock() -> Dict[str, Any]:
    """Force cleanup of expired keyed locks"""
    if not state._initialized or state._storage_keyed_lock is None:
        return {
            "process_id": os.getpid(),
            "cleanup_performed": {"mp_cleaned": 0, "async_cleaned": 0},
            "current_status": {
                "total_mp_locks": 0,
                "pending_mp_cleanup": 0,
                "total_async_locks": 0,
                "pending_async_cleanup": 0,
            },
        }

    return state._storage_keyed_lock.cleanup_expired_locks()


def get_keyed_lock_status() -> Dict[str, Any]:
    """Get current status of keyed locks without performing cleanup."""
    if not state._initialized or state._storage_keyed_lock is None:
        return {
            "process_id": os.getpid(),
            "total_mp_locks": 0,
            "pending_mp_cleanup": 0,
            "total_async_locks": 0,
            "pending_async_cleanup": 0,
        }

    status = state._storage_keyed_lock.get_lock_status()
    status["process_id"] = os.getpid()
    return status


def get_namespace_lock(
    namespace: str, workspace: str | None = None, enable_logging: bool = False
) -> NamespaceLock:
    """Get a reusable namespace lock wrapper."""
    return NamespaceLock(namespace, workspace, enable_logging)


def get_pipeline_status_lock(
    enable_logging: bool = False, workspace: str = None
) -> NamespaceLock:
    """Return unified storage lock for pipeline status data consistency."""
    actual_workspace = workspace if workspace else state._default_workspace
    return get_namespace_lock(
        "pipeline_status", workspace=actual_workspace, enable_logging=enable_logging
    )
