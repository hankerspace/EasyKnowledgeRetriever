import sys
import os
import logging
from typing import Optional
from easy_knowledge_retriever.kg import state

# Constants
DEBUG_LOCKS = False


def direct_log(message: str, enable_output: bool = True, level: str = "DEBUG"):
    """
    Log a message directly to stderr to ensure visibility in all processes,
    including the Gunicorn master process.

    Args:
        message: The message to log
        level: Log level for message (control the visibility of the message by comparing with the current logger level)
        enable_output: Enable or disable log message (Force to turn off the message,)
    """
    if not enable_output:
        return

    # Get the current logger level from the easy_knowledge_retriever logger
    try:
        from easy_knowledge_retriever.utils.logger import logger

        current_level = logger.getEffectiveLevel()
    except ImportError:
        # Fallback if easy_knowledge_retriever.utils is not available
        current_level = 20  # INFO

    # Convert string level to numeric level for comparison
    level_mapping = {
        "DEBUG": 10,  # DEBUG
        "INFO": 20,  # INFO
        "WARNING": 30,  # WARNING
        "ERROR": 40,  # ERROR
        "CRITICAL": 50,  # CRITICAL
    }
    message_level = level_mapping.get(level.upper(), logging.DEBUG)

    if message_level >= current_level:
        print(f"{level}: {message}", file=sys.stderr, flush=True)


def get_final_namespace(namespace: str, workspace: str | None = None) -> str:
    """
    Get the final namespace string, incorporating the workspace if present.
    Uses the default workspace from state if workspace is None.
    """
    if workspace is None:
        workspace = state._default_workspace

    if workspace is None:
        direct_log(
            f"Error: Invoke namespace operation without workspace, pid={os.getpid()}",
            level="ERROR",
        )
        raise ValueError("Invoke namespace operation without workspace")

    final_namespace = f"{workspace}:{namespace}" if workspace else f"{namespace}"
    return final_namespace


def get_combined_key(factory_name: str, key: str) -> str:
    """Return the combined key for the factory and key."""
    return f"{factory_name}:{key}"
