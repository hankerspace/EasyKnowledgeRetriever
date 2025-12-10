from __future__ import annotations
import json
import logging
import time
import re
import numpy as np
from dataclasses import dataclass
from typing import Any, Callable

# Import generic utils from the main utils package
# We assume utils.py will still contain these generic functions
from easy_knowledge_retriever.utils.hashing import compute_args_hash, generate_cache_key
from easy_knowledge_retriever.utils.logger import logger
from easy_knowledge_retriever.utils.text_utils import sanitize_text_for_encoding
from easy_knowledge_retriever.utils.common_utils import statistic_data

@dataclass
class EmbeddingFunc:
    """Embedding function wrapper with dimension validation
    This class wraps an embedding function to ensure that the output embeddings have the correct dimension.
    This class should not be wrapped multiple times.

    Args:
        embedding_dim: Expected dimension of the embeddings
        func: The actual embedding function to wrap
        max_token_size: Optional token limit for the embedding model
        send_dimensions: Whether to inject embedding_dim as a keyword argument
    """

    embedding_dim: int
    func: callable
    max_token_size: int | None = None  # Token limit for the embedding model
    send_dimensions: bool = (
        False  # Control whether to send embedding_dim to the function
    )

    async def __call__(self, *args, **kwargs) -> np.ndarray:
        # Only inject embedding_dim when send_dimensions is True
        if self.send_dimensions:
            # Check if user provided embedding_dim parameter
            if "embedding_dim" in kwargs:
                user_provided_dim = kwargs["embedding_dim"]
                # If user's value differs from class attribute, output warning
                if (
                    user_provided_dim is not None
                    and user_provided_dim != self.embedding_dim
                ):
                    logger.warning(
                        f"Ignoring user-provided embedding_dim={user_provided_dim}, "
                        f"using declared embedding_dim={self.embedding_dim} from decorator"
                    )

            # Inject embedding_dim from decorator
            kwargs["embedding_dim"] = self.embedding_dim

        # Call the actual embedding function
        result = await self.func(*args, **kwargs)

        # Validate embedding dimensions using total element count
        total_elements = result.size  # Total number of elements in the numpy array
        expected_dim = self.embedding_dim

        # Check if total elements can be evenly divided by embedding_dim
        if total_elements % expected_dim != 0:
            raise ValueError(
                f"Embedding dimension mismatch detected: "
                f"total elements ({total_elements}) cannot be evenly divided by "
                f"expected dimension ({expected_dim}). "
            )

        # Optional: Verify vector count matches input text count
        actual_vectors = total_elements // expected_dim
        if args and isinstance(args[0], (list, tuple)):
            expected_vectors = len(args[0])
            if actual_vectors != expected_vectors:
                raise ValueError(
                    f"Vector count mismatch: "
                    f"expected {expected_vectors} vectors but got {actual_vectors} vectors (from embedding result)."
                )

        return result


def pack_user_ass_to_openai_messages(*args: str):
    roles = ["user", "assistant"]
    return [
        {"role": roles[i % 2], "content": content} for i, content in enumerate(args)
    ]


@dataclass
class CacheData:
    args_hash: str
    content: str
    prompt: str
    mode: str = "default"
    cache_type: str = "query"
    chunk_id: str | None = None
    queryparam: dict | None = None


async def handle_cache(
    hashing_kv,
    args_hash,
    prompt,
    mode="default",
    cache_type="unknown",
    enable_cache=True,
) -> tuple[str, int] | None:
    """Generic cache handling function with flattened cache keys

    Returns:
        tuple[str, int] | None: (content, create_time) if cache hit, None if cache miss
    """
    if hashing_kv is None:
        return None

    if not enable_cache:
        return None

    # Use flattened cache key format: {mode}:{cache_type}:{hash}
    flattened_key = generate_cache_key(mode, cache_type, args_hash)
    cache_entry = await hashing_kv.get_by_id(flattened_key)
    if cache_entry:
        logger.debug(f"Flattened cache hit(key:{flattened_key})")
        content = cache_entry["return"]
        timestamp = cache_entry.get("create_time", 0)
        return content, timestamp

    logger.debug(f"Cache missed(mode:{mode} type:{cache_type})")
    return None


async def save_to_cache(hashing_kv, cache_data: CacheData):
    """Save data to cache using flattened key structure.

    Args:
        hashing_kv: The key-value storage for caching
        cache_data: The cache data to save
    """
    # Skip if storage is None or content is a streaming response
    if hashing_kv is None or not cache_data.content:
        return

    # If content is a streaming response, don't cache it
    if hasattr(cache_data.content, "__aiter__"):
        logger.debug("Streaming response detected, skipping cache")
        return

    # Use flattened cache key format: {mode}:{cache_type}:{hash}
    flattened_key = generate_cache_key(
        cache_data.mode, cache_data.cache_type, cache_data.args_hash
    )

    # Check if we already have identical content cached
    existing_cache = await hashing_kv.get_by_id(flattened_key)
    if existing_cache:
        existing_content = existing_cache.get("return")
        if existing_content == cache_data.content:
            logger.warning(
                f"Cache duplication detected for {flattened_key}, skipping update"
            )
            return

    # Create cache entry with flattened structure
    cache_entry = {
        "return": cache_data.content,
        "cache_type": cache_data.cache_type,
        "chunk_id": cache_data.chunk_id if cache_data.chunk_id is not None else None,
        "original_prompt": cache_data.prompt,
        "queryparam": cache_data.queryparam
        if cache_data.queryparam is not None
        else None,
    }

    logger.info(f" == LLM cache == saving: {flattened_key}")

    # Save using flattened key
    await hashing_kv.upsert({flattened_key: cache_entry})


def remove_think_tags(text: str) -> str:
    """Remove <think>...</think> tags from the text
    Remove  orphon ...</think> tags from the text also"""
    return re.sub(
        r"^(<think>.*?</think>|.*</think>)", "", text, flags=re.DOTALL
    ).strip()


async def use_llm_func_with_cache(
    user_prompt: str,
    use_llm_func: callable,
    llm_response_cache: "BaseKVStorage | None" = None,
    system_prompt: str | None = None,
    max_tokens: int = None,
    history_messages: list[dict[str, str]] = None,
    cache_type: str = "extract",
    chunk_id: str | None = None,
    cache_keys_collector: list = None,
    enable_cache: bool = True,
) -> tuple[str, int]:
    """Call LLM function with cache support and text sanitization

    If cache is available and enabled (determined by handle_cache based on mode),
    retrieve result from cache; otherwise call LLM function and save result to cache.

    This function applies text sanitization to prevent UTF-8 encoding errors for all LLM providers.

    Args:
        input_text: Input text to send to LLM
        use_llm_func: LLM function with higher priority
        llm_response_cache: Cache storage instance
        max_tokens: Maximum tokens for generation
        history_messages: History messages list
        cache_type: Type of cache
        chunk_id: Chunk identifier to store in cache
        text_chunks_storage: Text chunks storage to update llm_cache_list
        cache_keys_collector: Optional list to collect cache keys for batch processing

    Returns:
        tuple[str, int]: (LLM response text, timestamp)
            - For cache hits: (content, cache_create_time)
            - For cache misses: (content, current_timestamp)
    """
    # Sanitize input text to prevent UTF-8 encoding errors for all LLM providers
    safe_user_prompt = sanitize_text_for_encoding(user_prompt)
    safe_system_prompt = (
        sanitize_text_for_encoding(system_prompt) if system_prompt else None
    )

    # Sanitize history messages if provided
    safe_history_messages = None
    if history_messages:
        safe_history_messages = []
        for i, msg in enumerate(history_messages):
            safe_msg = msg.copy()
            if "content" in safe_msg:
                safe_msg["content"] = sanitize_text_for_encoding(safe_msg["content"])
            safe_history_messages.append(safe_msg)
        history = json.dumps(safe_history_messages, ensure_ascii=False)
    else:
        history = None

    if llm_response_cache:
        prompt_parts = []
        if safe_user_prompt:
            prompt_parts.append(safe_user_prompt)
        if safe_system_prompt:
            prompt_parts.append(safe_system_prompt)
        if history:
            prompt_parts.append(history)
        _prompt = "\n".join(prompt_parts)

        arg_hash = compute_args_hash(_prompt)
        # Generate cache key for this LLM call
        cache_key = generate_cache_key("default", cache_type, arg_hash)

        cached_result = await handle_cache(
            llm_response_cache,
            arg_hash,
            _prompt,
            "default",
            cache_type=cache_type,
            enable_cache=enable_cache,
        )
        if cached_result:
            content, timestamp = cached_result
            logger.debug(f"Found cache for {arg_hash}")
            statistic_data["llm_cache"] += 1

            # Add cache key to collector if provided
            if cache_keys_collector is not None:
                cache_keys_collector.append(cache_key)

            return content, timestamp
        statistic_data["llm_call"] += 1

        # Call LLM with sanitized input
        kwargs = {}
        if safe_history_messages:
            kwargs["history_messages"] = safe_history_messages
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        res: str = await use_llm_func(
            safe_user_prompt, system_prompt=safe_system_prompt, **kwargs
        )

        res = remove_think_tags(res)

        # Generate timestamp for cache miss (LLM call completion time)
        current_timestamp = int(time.time())

        if enable_cache:
            await save_to_cache(
                llm_response_cache,
                CacheData(
                    args_hash=arg_hash,
                    content=res,
                    prompt=_prompt,
                    cache_type=cache_type,
                    chunk_id=chunk_id,
                ),
            )

            # Add cache key to collector if provided
            if cache_keys_collector is not None:
                cache_keys_collector.append(cache_key)

        return res, current_timestamp

    # When cache is disabled, directly call LLM with sanitized input
    kwargs = {}
    if safe_history_messages:
        kwargs["history_messages"] = safe_history_messages
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    try:
        res = await use_llm_func(
            safe_user_prompt, system_prompt=safe_system_prompt, **kwargs
        )
    except Exception as e:
        # Add [LLM func] prefix to error message
        error_msg = f"[LLM func] {str(e)}"
        # Re-raise with the same exception type but modified message
        raise type(e)(error_msg) from e

    # Generate timestamp for non-cached LLM call
    current_timestamp = int(time.time())
    return remove_think_tags(res), current_timestamp
