from __future__ import annotations
import asyncio
import weakref
import logging
import time

from functools import wraps
from dataclasses import dataclass
from typing import Any
from .logger import logger

class UnlimitedSemaphore:
    """A context manager that allows unlimited access."""

    async def __aenter__(self):
        pass

    async def __aexit__(self, exc_type, exc, tb):
        pass


@dataclass
class TaskState:
    """Task state tracking for priority queue management"""

    future: asyncio.Future
    start_time: float
    execution_start_time: float = None
    worker_started: bool = False
    cancellation_requested: bool = False
    cleanup_done: bool = False


# Custom exception classes
class QueueFullError(Exception):
    """Raised when the queue is full and the wait times out"""

    pass


class WorkerTimeoutError(Exception):
    """Worker-level timeout exception with specific timeout information"""

    def __init__(self, timeout_value: float, timeout_type: str = "execution"):
        self.timeout_value = timeout_value
        self.timeout_type = timeout_type
        super().__init__(f"Worker {timeout_type} timeout after {timeout_value}s")


class HealthCheckTimeoutError(Exception):
    """Health Check-level timeout exception"""

    def __init__(self, timeout_value: float, execution_duration: float):
        self.timeout_value = timeout_value
        self.execution_duration = execution_duration
        super().__init__(
            f"Task forcefully terminated due to execution timeout (>{timeout_value}s, actual: {execution_duration:.1f}s)"
        )


def priority_limit_async_func_call(
    max_size: int,
    llm_timeout: float = None,
    max_execution_timeout: float = None,
    max_task_duration: float = None,
    max_queue_size: int = 1000,
    cleanup_timeout: float = 2.0,
    queue_name: str = "limit_async",
):
    """
    Enhanced priority-limited asynchronous function call decorator with robust timeout handling

    This decorator provides a comprehensive solution for managing concurrent LLM requests with:
    - Multi-layer timeout protection (LLM -> Worker -> Health Check -> User)
    - Task state tracking to prevent race conditions
    - Enhanced health check system with stuck task detection
    - Proper resource cleanup and error recovery

    Args:
        max_size: Maximum number of concurrent calls
        max_queue_size: Maximum queue capacity to prevent memory overflow
        llm_timeout: LLM provider timeout (from global config), used to calculate other timeouts
        max_execution_timeout: Maximum time for worker to execute function (defaults to llm_timeout + 30s)
        max_task_duration: Maximum time before health check intervenes (defaults to llm_timeout + 60s)
        cleanup_timeout: Maximum time to wait for cleanup operations (defaults to 2.0s)
        queue_name: Optional queue name for logging identification (defaults to "limit_async")

    Returns:
        Decorator function
    """

    def final_decro(func):
        # Ensure func is callable
        if not callable(func):
            raise TypeError(f"Expected a callable object, got {type(func)}")

        # Calculate timeout hierarchy if llm_timeout is provided (Dynamic Timeout Calculation)
        if llm_timeout is not None:
            nonlocal max_execution_timeout, max_task_duration
            if max_execution_timeout is None:
                max_execution_timeout = (
                    llm_timeout * 2
                )  # Reserved timeout buffer for low-level retry
            if max_task_duration is None:
                max_task_duration = (
                    llm_timeout * 2 + 15
                )  # Reserved timeout buffer for health check phase

        queue = asyncio.PriorityQueue(maxsize=max_queue_size)
        tasks = set()
        initialization_lock = asyncio.Lock()
        counter = 0
        shutdown_event = asyncio.Event()
        initialized = False
        worker_health_check_task = None

        # Enhanced task state management
        task_states = {}  # task_id -> TaskState
        task_states_lock = asyncio.Lock()
        active_futures = weakref.WeakSet()
        reinit_count = 0

        async def worker():
            """Enhanced worker that processes tasks with proper timeout and state management"""
            try:
                while not shutdown_event.is_set():
                    try:
                        # Get task from queue with timeout for shutdown checking
                        try:
                            (
                                priority,
                                count,
                                task_id,
                                args,
                                kwargs,
                            ) = await asyncio.wait_for(queue.get(), timeout=1.0)
                        except asyncio.TimeoutError:
                            continue

                        # Get task state and mark worker as started
                        async with task_states_lock:
                            if task_id not in task_states:
                                queue.task_done()
                                continue
                            task_state = task_states[task_id]
                            task_state.worker_started = True
                            # Record execution start time when worker actually begins processing
                            task_state.execution_start_time = (
                                asyncio.get_event_loop().time()
                            )

                        # Check if task was cancelled before worker started
                        if (
                            task_state.cancellation_requested
                            or task_state.future.cancelled()
                        ):
                            async with task_states_lock:
                                task_states.pop(task_id, None)
                            queue.task_done()
                            continue

                        try:
                            # Execute function with timeout protection
                            if max_execution_timeout is not None:
                                result = await asyncio.wait_for(
                                    func(*args, **kwargs), timeout=max_execution_timeout
                                )
                            else:
                                result = await func(*args, **kwargs)

                            # Set result if future is still valid
                            if not task_state.future.done():
                                task_state.future.set_result(result)

                        except asyncio.TimeoutError:
                            # Worker-level timeout (max_execution_timeout exceeded)
                            logger.warning(
                                f"{queue_name}: Worker timeout for task {task_id} after {max_execution_timeout}s"
                            )
                            if not task_state.future.done():
                                task_state.future.set_exception(
                                    WorkerTimeoutError(
                                        max_execution_timeout, "execution"
                                    )
                                )
                        except asyncio.CancelledError:
                            # Task was cancelled during execution
                            if not task_state.future.done():
                                task_state.future.cancel()
                            logger.debug(
                                f"{queue_name}: Task {task_id} cancelled during execution"
                            )
                        except Exception as e:
                            # Function execution error
                            logger.error(
                                f"{queue_name}: Error in decorated function for task {task_id}: {str(e)}"
                            )
                            if not task_state.future.done():
                                task_state.future.set_exception(e)
                        finally:
                            # Clean up task state
                            async with task_states_lock:
                                task_states.pop(task_id, None)
                            queue.task_done()

                    except Exception as e:
                        # Critical error in worker loop
                        logger.error(
                            f"{queue_name}: Critical error in worker: {str(e)}"
                        )
                        await asyncio.sleep(0.1)
            finally:
                logger.debug(f"{queue_name}: Worker exiting")

        async def enhanced_health_check():
            """Enhanced health check with stuck task detection and recovery"""
            nonlocal initialized
            try:
                while not shutdown_event.is_set():
                    await asyncio.sleep(5)  # Check every 5 seconds

                    current_time = asyncio.get_event_loop().time()

                    # Detect and handle stuck tasks based on execution start time
                    if max_task_duration is not None:
                        stuck_tasks = []
                        async with task_states_lock:
                            for task_id, task_state in list(task_states.items()):
                                # Only check tasks that have started execution
                                if (
                                    task_state.worker_started
                                    and task_state.execution_start_time is not None
                                    and current_time - task_state.execution_start_time
                                    > max_task_duration
                                ):
                                    stuck_tasks.append(
                                        (
                                            task_id,
                                            current_time
                                            - task_state.execution_start_time,
                                        )
                                    )

                        # Force cleanup of stuck tasks
                        for task_id, execution_duration in stuck_tasks:
                            logger.warning(
                                f"{queue_name}: Detected stuck task {task_id} (execution time: {execution_duration:.1f}s), forcing cleanup"
                            )
                            async with task_states_lock:
                                if task_id in task_states:
                                    task_state = task_states[task_id]
                                    if not task_state.future.done():
                                        task_state.future.set_exception(
                                            HealthCheckTimeoutError(
                                                max_task_duration, execution_duration
                                            )
                                        )
                                    task_states.pop(task_id, None)

                    # Worker recovery logic
                    current_tasks = set(tasks)
                    done_tasks = {t for t in current_tasks if t.done()}
                    tasks.difference_update(done_tasks)

                    active_tasks_count = len(tasks)
                    workers_needed = max_size - active_tasks_count

                    if workers_needed > 0:
                        logger.info(
                            f"{queue_name}: Creating {workers_needed} new workers"
                        )
                        new_tasks = set()
                        for _ in range(workers_needed):
                            task = asyncio.create_task(worker())
                            new_tasks.add(task)
                            task.add_done_callback(tasks.discard)
                        tasks.update(new_tasks)

            except Exception as e:
                logger.error(f"{queue_name}: Error in enhanced health check: {str(e)}")
            finally:
                logger.debug(f"{queue_name}: Enhanced health check task exiting")
                initialized = False

        async def ensure_workers():
            """Ensure worker system is initialized with enhanced error handling"""
            nonlocal initialized, worker_health_check_task, tasks, reinit_count

            if initialized:
                return

            async with initialization_lock:
                if initialized:
                    return

                if reinit_count > 0:
                    reinit_count += 1
                    logger.warning(
                        f"{queue_name}: Reinitializing system (count: {reinit_count})"
                    )
                else:
                    reinit_count = 1

                # Clean up completed tasks
                current_tasks = set(tasks)
                done_tasks = {t for t in current_tasks if t.done()}
                tasks.difference_update(done_tasks)

                active_tasks_count = len(tasks)
                if active_tasks_count > 0 and reinit_count > 1:
                    logger.warning(
                        f"{queue_name}: {active_tasks_count} tasks still running during reinitialization"
                    )

                # Create worker tasks
                workers_needed = max_size - active_tasks_count
                for _ in range(workers_needed):
                    task = asyncio.create_task(worker())
                    tasks.add(task)
                    task.add_done_callback(tasks.discard)

                # Start enhanced health check
                worker_health_check_task = asyncio.create_task(enhanced_health_check())

                initialized = True
                # Log dynamic timeout configuration
                timeout_info = []
                if llm_timeout is not None:
                    timeout_info.append(f"Func: {llm_timeout}s")
                if max_execution_timeout is not None:
                    timeout_info.append(f"Worker: {max_execution_timeout}s")
                if max_task_duration is not None:
                    timeout_info.append(f"Health Check: {max_task_duration}s")

                timeout_str = (
                    f"(Timeouts: {', '.join(timeout_info)})" if timeout_info else ""
                )
                logger.info(
                    f"{queue_name}: {workers_needed} new workers initialized {timeout_str}"
                )

        async def shutdown():
            """Gracefully shut down all workers and cleanup resources"""
            logger.info(f"{queue_name}: Shutting down priority queue workers")

            shutdown_event.set()

            # Cancel all active futures
            for future in list(active_futures):
                if not future.done():
                    future.cancel()

            # Cancel all pending tasks
            async with task_states_lock:
                for task_id, task_state in list(task_states.items()):
                    if not task_state.future.done():
                        task_state.future.cancel()
                task_states.clear()

            # Wait for queue to empty with timeout
            try:
                await asyncio.wait_for(queue.join(), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning(
                    f"{queue_name}: Timeout waiting for queue to empty during shutdown"
                )

            # Cancel worker tasks
            for task in list(tasks):
                if not task.done():
                    task.cancel()

            # Wait for all tasks to complete
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

            # Cancel health check task
            if worker_health_check_task and not worker_health_check_task.done():
                worker_health_check_task.cancel()
                try:
                    await worker_health_check_task
                except asyncio.CancelledError:
                    pass

            logger.info(f"{queue_name}: Priority queue workers shutdown complete")

        @wraps(func)
        async def wait_func(
            *args, _priority=10, _timeout=None, _queue_timeout=None, **kwargs
        ):
            """
            Execute function with enhanced priority-based concurrency control and timeout handling

            Args:
                *args: Positional arguments passed to the function
                _priority: Call priority (lower values have higher priority)
                _timeout: Maximum time to wait for completion (in seconds, none means determinded by max_execution_timeout of the queue)
                _queue_timeout: Maximum time to wait for entering the queue (in seconds)
                **kwargs: Keyword arguments passed to the function

            Returns:
                The result of the function call

            Raises:
                TimeoutError: If the function call times out at any level
                QueueFullError: If the queue is full and waiting times out
                Any exception raised by the decorated function
            """
            await ensure_workers()

            # Generate unique task ID
            task_id = f"{id(asyncio.current_task())}_{asyncio.get_event_loop().time()}"
            future = asyncio.Future()

            # Create task state
            task_state = TaskState(
                future=future, start_time=asyncio.get_event_loop().time()
            )

            try:
                # Register task state
                async with task_states_lock:
                    task_states[task_id] = task_state

                active_futures.add(future)

                # Get counter for FIFO ordering
                nonlocal counter
                async with initialization_lock:
                    current_count = counter
                    counter += 1

                # Queue the task with timeout handling
                try:
                    if _queue_timeout is not None:
                        await asyncio.wait_for(
                            queue.put(
                                (_priority, current_count, task_id, args, kwargs)
                            ),
                            timeout=_queue_timeout,
                        )
                    else:
                        await queue.put(
                            (_priority, current_count, task_id, args, kwargs)
                        )
                except asyncio.TimeoutError:
                    raise QueueFullError(
                        f"{queue_name}: Queue full, timeout after {_queue_timeout} seconds"
                    )
                except Exception as e:
                    # Clean up on queue error
                    if not future.done():
                        future.set_exception(e)
                    raise

                # Wait for result with timeout handling
                try:
                    if _timeout is not None:
                        return await asyncio.wait_for(future, _timeout)
                    else:
                        return await future
                except asyncio.TimeoutError:
                    # This is user-level timeout (asyncio.wait_for caused)
                    # Mark cancellation request
                    async with task_states_lock:
                        if task_id in task_states:
                            task_states[task_id].cancellation_requested = True

                    # Cancel future
                    if not future.done():
                        future.cancel()

                    # Wait for worker cleanup with timeout
                    cleanup_start = asyncio.get_event_loop().time()
                    while (
                        task_id in task_states
                        and asyncio.get_event_loop().time() - cleanup_start
                        < cleanup_timeout
                    ):
                        await asyncio.sleep(0.1)

                    raise TimeoutError(
                        f"{queue_name}: User timeout after {_timeout} seconds"
                    )
                except WorkerTimeoutError as e:
                    # This is Worker-level timeout, directly propagate exception information
                    raise TimeoutError(f"{queue_name}: {str(e)}")
                except HealthCheckTimeoutError as e:
                    # This is Health Check-level timeout, directly propagate exception information
                    raise TimeoutError(f"{queue_name}: {str(e)}")

            finally:
                # Ensure cleanup
                active_futures.discard(future)
                async with task_states_lock:
                    task_states.pop(task_id, None)

        # Add shutdown method to decorated function
        wait_func.shutdown = shutdown

        return wait_func

    return final_decro


def always_get_an_event_loop() -> asyncio.AbstractEventLoop:
    """
    Ensure that there is always an event loop available.

    This function tries to get the current event loop. If the current event loop is closed or does not exist,
    it creates a new event loop and sets it as the current event loop.

    Returns:
        asyncio.AbstractEventLoop: The current or newly created event loop.
    """
    try:
        # Try to get the current event loop
        current_loop = asyncio.get_event_loop()
        if current_loop.is_closed():
            raise RuntimeError("Event loop is closed.")
        return current_loop

    except RuntimeError:
        # If no event loop exists or it is closed, create a new one
        logger.info("Creating a new event loop in main thread.")
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        return new_loop
