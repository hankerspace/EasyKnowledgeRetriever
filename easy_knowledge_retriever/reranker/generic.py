from __future__ import annotations

import aiohttp
from typing import Any, List, Dict, Optional
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from easy_knowledge_retriever.utils.logger import logger
from .utils import chunk_documents_for_rerank, aggregate_chunk_scores


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=(
        retry_if_exception_type(aiohttp.ClientError)
        | retry_if_exception_type(aiohttp.ClientResponseError)
    ),
)
async def generic_rerank_api(
    query: str,
    documents: List[str],
    model: str,
    base_url: str,
    api_key: Optional[str],
    top_n: Optional[int] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    response_format: str = "standard",
    enable_chunking: bool = False,
    max_tokens_per_doc: int = 480,
) -> List[Dict[str, Any]]:
    """
    Generic rerank API call for Jina/Cohere/Aliyun models.

    Args:
        query: The search query
        documents: List of strings to rerank
        model: Model name to use
        base_url: API endpoint URL
        api_key: API key for authentication
        top_n: Number of top results to return
        return_documents: Whether to return document text (Jina only)
        extra_body: Additional body parameters
        response_format: Response format type ("standard" for Jina/Cohere, "aliyun" for Aliyun)
        request_format: Request format type
        enable_chunking: Whether to chunk documents exceeding token limit
        max_tokens_per_doc: Maximum tokens per document for chunking

    Returns:
        List of dictionary of ["index": int, "relevance_score": float]
    """
    if not base_url:
        raise ValueError("Base URL is required")

    headers = {"Content-Type": "application/json"}
    if api_key is not None:
        headers["Authorization"] = f"Bearer {api_key}"

    # Handle document chunking if enabled
    original_documents = documents
    doc_indices = None
    original_top_n = top_n  # Save original top_n for post-aggregation limiting

    if enable_chunking:
        documents, doc_indices = chunk_documents_for_rerank(
            documents, max_tokens=max_tokens_per_doc
        )
        logger.debug(
            f"Chunked {len(original_documents)} documents into {len(documents)} chunks"
        )
        # When chunking is enabled, disable top_n at API level to get all chunk scores
        # This ensures proper document-level coverage after aggregation
        # We'll apply top_n to aggregated document results instead
        if top_n is not None:
            logger.debug(
                f"Chunking enabled: disabled API-level top_n={top_n} to ensure complete document coverage"
            )
            top_n = None

    # Build request payload based on request format
    # Standard format
    payload = {
        "model": model,
        "query": query,
        "documents": documents,
    }

    # Add optional parameters
    if top_n is not None:
        payload["top_n"] = top_n

    # Add extra parameters
    if extra_body:
        payload.update(extra_body)

    logger.debug(
        f"Rerank request: {len(documents)} documents, model: {model}, format: {response_format}"
    )

    async with aiohttp.ClientSession() as session:
        async with session.post(base_url, headers=headers, json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                content_type = response.headers.get("content-type", "").lower()
                is_html_error = (
                    error_text.strip().startswith("<!DOCTYPE html>")
                    or "text/html" in content_type
                )
                if is_html_error:
                    if response.status == 502:
                        clean_error = "Bad Gateway (502) - Rerank service temporarily unavailable. Please try again in a few minutes."
                    elif response.status == 503:
                        clean_error = "Service Unavailable (503) - Rerank service is temporarily overloaded. Please try again later."
                    elif response.status == 504:
                        clean_error = "Gateway Timeout (504) - Rerank service request timed out. Please try again."
                    else:
                        clean_error = f"HTTP {response.status} - Rerank service error. Please try again later."
                else:
                    clean_error = error_text
                logger.error(f"Rerank API error {response.status}: {clean_error}")
                raise aiohttp.ClientResponseError(
                    request_info=response.request_info,
                    history=response.history,
                    status=response.status,
                    message=f"Rerank API error: {clean_error}",
                )

            response_json = await response.json()

            if response_format == "aliyun":
                 raise ValueError("Aliyun format not supported")
            elif response_format == "standard":
                # Standard format: {"results": [...]}
                results = response_json.get("results", [])
                if not isinstance(results, list):
                    logger.warning(
                        f"Expected 'results' to be list, got {type(results)}: {results}"
                    )
                    results = []
            else:
                raise ValueError(f"Unsupported response format: {response_format}")

            if not results:
                logger.warning("Rerank API returned empty results")
                return []

            # Standardize return format
            standardized_results = [
                {"index": result["index"], "relevance_score": result["relevance_score"]}
                for result in results
            ]

            # Aggregate chunk scores back to original documents if chunking was enabled
            if enable_chunking and doc_indices:
                standardized_results = aggregate_chunk_scores(
                    standardized_results,
                    doc_indices,
                    len(original_documents),
                    aggregation="max",
                )
                # Apply original top_n limit at document level (post-aggregation)
                # This preserves document-level semantics: top_n limits documents, not chunks
                if (
                    original_top_n is not None
                    and len(standardized_results) > original_top_n
                ):
                    standardized_results = standardized_results[:original_top_n]

            return standardized_results
