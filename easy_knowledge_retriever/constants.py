"""
Centralized configuration constants for EasyKnowledgeRetriever.

This module defines default values for configuration constants used across
different parts of the EasyKnowledgeRetriever system. Centralizing these values ensures
consistency and makes maintenance easier.
"""

import os


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    """Read a positive int from the environment, falling back to `default`.

    Invalid or out-of-range values fall back rather than raising: a typo in a
    deployment env var should not prevent the library from importing.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return value if value >= minimum else default


# Default values for server settings
DEFAULT_MAX_GRAPH_NODES = 1000

# Default values for extraction settings

DEFAULT_MAX_GLEANING = 1
DEFAULT_ENTITY_NAME_MAX_LENGTH = 256

# Number of description fragments to trigger LLM summary
DEFAULT_FORCE_LLM_SUMMARY_ON_MERGE = 8
# Max description token size to trigger LLM summary
DEFAULT_SUMMARY_MAX_TOKENS = 1200
# Recommended LLM summary output length in tokens
DEFAULT_SUMMARY_LENGTH_RECOMMENDED = 600
# Maximum token size sent to LLM for summary
DEFAULT_SUMMARY_CONTEXT_SIZE = 12000
# Default entities to extract if ENTITY_TYPES is not specified in .env
DEFAULT_ENTITY_TYPES = [
    "Person",
    "Creature",
    "Organization",
    "Location",
    "Event",
    "Concept",
    "Action",
    "Method",
    "Content",
    "Data",
    "Artifact",
    "NaturalObject",
]

# Separator for: description, source_id and relation-key fields(Can not be changed after data inserted)
GRAPH_FIELD_SEP = "<SEP>"

# Query and retrieval configuration defaults


# Rerank configuration defaults
# Default source ids limit in meta data for entity and relation
DEFAULT_MAX_SOURCE_IDS_PER_ENTITY = 300
DEFAULT_MAX_SOURCE_IDS_PER_RELATION = 300
### control chunk_ids limitation method: FIFO, FIFO
###    FIFO: First in first out
###    KEEP: Keep oldest (less merge action and faster)
SOURCE_IDS_LIMIT_METHOD_KEEP = "KEEP"
SOURCE_IDS_LIMIT_METHOD_FIFO = "FIFO"
DEFAULT_SOURCE_IDS_LIMIT_METHOD = SOURCE_IDS_LIMIT_METHOD_FIFO
VALID_SOURCE_IDS_LIMIT_METHODS = {
    SOURCE_IDS_LIMIT_METHOD_KEEP,
    SOURCE_IDS_LIMIT_METHOD_FIFO,
}
# Maximum number of file paths stored in entity/relation file_path field (For displayed only, does not affect query performance)
DEFAULT_MAX_FILE_PATHS = 100

# Field length of file_path in Milvus Schema for entity and relation (Should not be changed)
# file_path must store all file paths up to the DEFAULT_MAX_FILE_PATHS limit within the metadata.
DEFAULT_MAX_FILE_PATH_LENGTH = 32768
# Placeholder for more file paths in meta data for entity and relation (Should not be changed)
DEFAULT_FILE_PATH_MORE_PLACEHOLDER = "truncated"


# Default maximum parallel insert operations
DEFAULT_MAX_PARALLEL_INSERT = 1


# Logging configuration defaults
DEFAULT_LOG_MAX_BYTES = 10485760  # Default 10MB
DEFAULT_LOG_BACKUP_COUNT = 5  # Default 5 backups
DEFAULT_LOG_FILENAME = "easy_knowledge_retriever.log"  # Default log filename



# LLM execution defaults
#
# Concurrency is env-tunable because the right value depends entirely on the
# provider's rate limits, not on the code. The defaults stay conservative (1 =
# fully serialized) so an unconfigured deployment cannot burn through a quota,
# but ingestion of a large document is unusably slow at 1 -- raise
# EKR_MAX_ASYNC / EKR_EMBEDDING_MAX_ASYNC once you know your limits.
DEFAULT_TEMPERATURE = 1.0
DEFAULT_MAX_ASYNC = _env_int("EKR_MAX_ASYNC", 1, minimum=1)
DEFAULT_LLM_TIMEOUT = _env_int("EKR_LLM_TIMEOUT", 60, minimum=1)

# Embedding execution defaults
DEFAULT_EMBEDDING_BATCH_NUM = _env_int("EKR_EMBEDDING_BATCH_NUM", 16, minimum=1)
DEFAULT_EMBEDDING_FUNC_MAX_ASYNC = _env_int("EKR_EMBEDDING_MAX_ASYNC", 1, minimum=1)
DEFAULT_EMBEDDING_TIMEOUT = _env_int("EKR_EMBEDDING_TIMEOUT", 60, minimum=1)

# MinerU PDF parsing: hard ceiling on the parser subprocess, so a hung parse
# fails the ingestion instead of hanging it forever.
DEFAULT_MINERU_TIMEOUT = _env_int("EKR_MINERU_TIMEOUT", 3600, minimum=1)

# Default values for retrieval settings
DEFAULT_RELATED_CHUNK_NUMBER = 5
DEFAULT_KG_CHUNK_PICK_METHOD = "VECTOR"
DEFAULT_MAX_TOTAL_TOKENS = 30000
