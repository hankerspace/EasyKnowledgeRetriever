from abc import ABC, abstractmethod
from typing import Any, List
import numpy as np
from .openai import openai_complete_if_cache, openai_embed
from config.llm_config import DEFAULT_TEMPERATURE

class BaseLLMService(ABC):
    @abstractmethod
    async def __call__(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt."""
        pass

class OpenAILLMService(BaseLLMService):
    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None, temperature: float = DEFAULT_TEMPERATURE, **kwargs):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.temperature = temperature
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
    @abstractmethod
    async def __call__(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for texts."""
        pass

class OpenAIEmbeddingService(BaseEmbeddingService):
    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key

    async def __call__(self, texts: List[str], **kwargs) -> np.ndarray:
        return await openai_embed(
            texts=texts,
            model=self.model,
            base_url=self.base_url,
            api_key=self.api_key
        )
