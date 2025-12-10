from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Callable
from .base import BaseConfig
# Default values moved from constants.py
DEFAULT_SUMMARY_MAX_TOKENS = 1200
DEFAULT_SUMMARY_CONTEXT_SIZE = 12000
DEFAULT_SUMMARY_LENGTH_RECOMMENDED = 600
DEFAULT_MAX_ASYNC = 1
DEFAULT_LLM_TIMEOUT = 1000000

DEFAULT_TEMPERATURE = 1.0

@dataclass
class LLMConfig(BaseConfig):
    model_func: Optional[Callable[..., object]] = None
    model_name: str = "gpt-4o-mini"
    summary_max_tokens: int = DEFAULT_SUMMARY_MAX_TOKENS
    summary_context_size: int = DEFAULT_SUMMARY_CONTEXT_SIZE
    summary_length_recommended: int = DEFAULT_SUMMARY_LENGTH_RECOMMENDED
    max_async: int = DEFAULT_MAX_ASYNC
    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    api_config: Dict[str, Any] = field(default_factory=dict) # e.g. api_key, base_url
    timeout: int = DEFAULT_LLM_TIMEOUT
    temperature: float = DEFAULT_TEMPERATURE
