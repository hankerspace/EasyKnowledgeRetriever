from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from .base import BaseConfig
from llm.utils import EmbeddingFunc
# Default values moved from constants.py
DEFAULT_EMBEDDING_BATCH_NUM = 1
DEFAULT_EMBEDDING_FUNC_MAX_ASYNC = 1
DEFAULT_EMBEDDING_TIMEOUT = 1000000

@dataclass
class EmbeddingConfig(BaseConfig):
    embedding_func: Optional[EmbeddingFunc] = None
    batch_num: int = DEFAULT_EMBEDDING_BATCH_NUM
    max_async: int = DEFAULT_EMBEDDING_FUNC_MAX_ASYNC
    timeout: int = DEFAULT_EMBEDDING_TIMEOUT
    cache_config: Dict[str, Any] = field(
        default_factory=lambda: {
            "enabled": False,
            "similarity_threshold": 0.95,
            "use_llm_check": False,
        }
    )
    token_limit: Optional[int] = None # Will be set from embedding_func if available
