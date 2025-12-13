from abc import ABC, abstractmethod
from typing import Any, List
import numpy as np
from .openai import openai_complete_if_cache, openai_embed
from easy_knowledge_retriever.constants import (
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_ASYNC,
    DEFAULT_LLM_TIMEOUT,
    DEFAULT_SUMMARY_MAX_TOKENS,
    DEFAULT_SUMMARY_CONTEXT_SIZE,
    DEFAULT_SUMMARY_LENGTH_RECOMMENDED,
    DEFAULT_EMBEDDING_BATCH_NUM,
    DEFAULT_EMBEDDING_FUNC_MAX_ASYNC,
    DEFAULT_EMBEDDING_TIMEOUT,
)

class BaseLLMService(ABC):
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        max_async: int = DEFAULT_MAX_ASYNC,
        timeout: int = DEFAULT_LLM_TIMEOUT,
        temperature: float = DEFAULT_TEMPERATURE,
        summary_max_tokens: int = DEFAULT_SUMMARY_MAX_TOKENS,
        summary_context_size: int = DEFAULT_SUMMARY_CONTEXT_SIZE,
        summary_length_recommended: int = DEFAULT_SUMMARY_LENGTH_RECOMMENDED,
    ):
        self.model_name = model_name
        self.max_async = max_async
        self.timeout = timeout
        self.temperature = temperature
        self.summary_max_tokens = summary_max_tokens
        self.summary_context_size = summary_context_size
        self.summary_length_recommended = summary_length_recommended

    @abstractmethod
    async def __call__(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt."""
        pass

class OpenAILLMService(BaseLLMService):
    def __init__(
        self, 
        model: str, 
        base_url: str | None = None, 
        api_key: str | None = None, 
        temperature: float = DEFAULT_TEMPERATURE,
        max_async: int = DEFAULT_MAX_ASYNC,
        timeout: int = DEFAULT_LLM_TIMEOUT,
        summary_max_tokens: int = DEFAULT_SUMMARY_MAX_TOKENS,
        summary_context_size: int = DEFAULT_SUMMARY_CONTEXT_SIZE,
        summary_length_recommended: int = DEFAULT_SUMMARY_LENGTH_RECOMMENDED,
        **kwargs
    ):
        super().__init__(
            model_name=model,
            max_async=max_async,
            timeout=timeout,
            temperature=temperature,
            summary_max_tokens=summary_max_tokens,
            summary_context_size=summary_context_size,
            summary_length_recommended=summary_length_recommended,
        )
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.kwargs = kwargs

    async def __call__(self, prompt: str, **kwargs) -> str:
        # Merge kwargs: call-time kwargs override init kwargs
        call_kwargs = {**self.kwargs, **kwargs}
        if "temperature" not in call_kwargs:
            call_kwargs["temperature"] = self.temperature
        
        # Avoid duplicate keyword arguments
        api_key = call_kwargs.pop("api_key", self.api_key)
        base_url = call_kwargs.pop("base_url", self.base_url)
        model = call_kwargs.pop("model", self.model)

        return await openai_complete_if_cache(
            model=model,
            prompt=prompt,
            base_url=base_url,
            api_key=api_key,
            **call_kwargs
        )

class BaseEmbeddingService(ABC):
    def __init__(
        self,
        batch_num: int = DEFAULT_EMBEDDING_BATCH_NUM,
        max_async: int = DEFAULT_EMBEDDING_FUNC_MAX_ASYNC,
        timeout: int = DEFAULT_EMBEDDING_TIMEOUT,
        cache_config: dict | None = None,
        embedding_dim: int = 1536,
    ):
        self.batch_num = batch_num
        self.max_async = max_async
        self.timeout = timeout
        self.cache_config = cache_config
        self.max_token_size = None # To be set by subclass or dynamically
        self.embedding_dim = embedding_dim

    @abstractmethod
    async def __call__(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for texts."""
        pass

class OpenAIEmbeddingService(BaseEmbeddingService):
    def __init__(
        self, 
        model: str, 
        base_url: str | None = None, 
        api_key: str | None = None,
        batch_num: int = DEFAULT_EMBEDDING_BATCH_NUM,
        max_async: int = DEFAULT_EMBEDDING_FUNC_MAX_ASYNC,
        timeout: int = DEFAULT_EMBEDDING_TIMEOUT,
        embedding_dim: int = 1536,
        cache_config: dict[str, Any] | None = None,
    ):
        super().__init__(
            batch_num=batch_num,
            max_async=max_async,
            timeout=timeout,
            cache_config=cache_config,
            embedding_dim=embedding_dim,
        )
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.max_token_size = 8191 # OpenAI limit

    async def __call__(self, texts: List[str], **kwargs) -> np.ndarray:
        return await openai_embed(
            texts=texts,
            model=self.model,
            base_url=self.base_url,
            api_key=self.api_key,
            batch_size=self.batch_num
        )
