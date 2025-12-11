from __future__ import annotations

from typing import Any, List, Dict, Optional
from .generic import generic_rerank_api
from .base import BaseRerankerService

class OpenAIRerankerService(BaseRerankerService):
    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: Optional[str],
        extra_body: Optional[Dict[str, Any]] = None,
        enable_chunking: bool = False,
        max_tokens_per_doc: int = 480,
    ):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.extra_body = extra_body
        self.enable_chunking = enable_chunking
        self.max_tokens_per_doc = max_tokens_per_doc

    async def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        return await openai_compatible_rerank(
            query=query,
            documents=documents,
            model=self.model,
            base_url=self.base_url,
            api_key=self.api_key,
            top_n=top_n,
            extra_body=self.extra_body,
            enable_chunking=self.enable_chunking,
            max_tokens_per_doc=self.max_tokens_per_doc
        )

async def openai_compatible_rerank(
    query: str,
    documents: List[str],
    model: str,
    base_url: str,
    api_key: Optional[str],
    top_n: Optional[int] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    enable_chunking: bool = False,
    max_tokens_per_doc: int = 480,
) -> List[Dict[str, Any]]:
    """
    Rerank documents using an OpenAI API compatible endpoint.
    This usually expects a POST request to {base_url} (e.g. /v1/rerank)
    with JSON body: {"model": "...", "query": "...", "documents": [...], "top_n": ...}

    Args:
        query: The search query
        documents: List of strings to rerank
        model: Model name to use
        base_url: API endpoint URL (full URL including path, e.g. http://localhost:8000/v1/rerank)
        api_key: API key for authentication
        top_n: Number of top results to return
        extra_body: Additional body parameters
        enable_chunking: Whether to chunk documents exceeding token limit
        max_tokens_per_doc: Maximum tokens per document for chunking

    Returns:
        List of dictionary of ["index": int, "relevance_score": float]
    """
    # Just reuse the generic API as it already implements the standard format
    # used by most OpenAI-compatible rerankers (like Infinity, TEI, etc.)
    return await generic_rerank_api(
        query=query,
        documents=documents,
        model=model,
        base_url=base_url,
        api_key=api_key,
        top_n=top_n,
        return_documents=False,
        extra_body=extra_body,
        response_format="standard",
        request_format="standard",
        enable_chunking=enable_chunking,
        max_tokens_per_doc=max_tokens_per_doc
    )
