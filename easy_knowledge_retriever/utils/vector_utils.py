import asyncio
import logging
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Sequence, Union, Set, Iterable, Collection
from .logger import logger
from .hashing import compute_mdhash_id
from easy_knowledge_retriever.constants import (
    DEFAULT_SOURCE_IDS_LIMIT_METHOD,
    VALID_SOURCE_IDS_LIMIT_METHODS,
    SOURCE_IDS_LIMIT_METHOD_FIFO,
    GRAPH_FIELD_SEP,
)

async def safe_vdb_operation_with_exception(
    func: Callable | None = None,
    *args,
    operation: Callable | None = None,
    operation_name: str = "unknown_operation",
    entity_name: str | None = None,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    **kwargs,
) -> Optional[Any]:
    """Execute a VectorDB operation safely, logging errors and returning None on failure.

    This wrapper function ensures that VectorDB operations do not crash the application
    if expected errors occur (like timeout or connection issues). It catches Exception,
    logs the error with context, and returns None.

    Args:
        func: The async function to execute (positional or keyword)
        operation: Alternative keyword for func
        operation_name: Name of operation for logging
        entity_name: Related entity name for logging
        max_retries: Number of retries
        retry_delay: Delay between retries in seconds
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function

    Returns:
        The result of the function call, or None if an exception occurred.
    """
    target_func = operation if operation is not None else func
    if target_func is None:
        logger.error("safe_vdb_operation_with_exception called without a function/operation.")
        return None

    op_name = operation_name or getattr(target_func, "__name__", str(target_func))

    for attempt in range(max_retries):
        try:
            return await target_func(*args, **kwargs)
        except Exception as e:
            is_last_attempt = attempt == max_retries - 1
            error_prefix = "Failed" if is_last_attempt else f"Error (attempt {attempt + 1}/{max_retries})"

            msg = f"{error_prefix} in VectorDB operation '{op_name}': {e}"
            if entity_name:
                msg += f" for entity '{entity_name}'"

            logger.error(msg)

            if not is_last_attempt:
                await asyncio.sleep(retry_delay)

    return None


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    return dot_product / (norm_v1 * norm_v2)


def pick_by_weighted_polling(
    entity_info: list[dict],
    max_related_chunks: int,
) -> list[str]:
    """Pick chunks by weighted polling from entity info.
    
    Args:
        entity_info: List of dictionaries containing entity and chunk info
        max_related_chunks: Maximum number of chunks to select
        
    Returns:
        list[str]: Selected chunk IDs
    """
    if not entity_info:
        return []

    # Flatten and deduplicate chunks from all entities, preserving order
    # Entities should already be sorted by relevance in entity_info
    # Chunks within entities are already sorted by occurrence
    selected_ids = []
    seen = set()
    
    for item in entity_info:
        for cid in item.get("sorted_chunks", []):
            if cid not in seen:
                selected_ids.append(cid)
                seen.add(cid)
                
    return selected_ids[:max_related_chunks]


async def pick_by_vector_similarity(
    query: str,
    text_chunks_storage: Any, # BaseKVStorage
    chunks_vdb: Any, # BaseVectorStorage
    num_of_chunks: int,
    entity_info: list[dict],
    embedding_func: Callable,
    query_embedding: np.ndarray | None = None,
) -> list[str]:
    """Pick top-k chunks by vector similarity.
    
    Args:
        query: Query string
        text_chunks_storage: Storage for text chunks
        chunks_vdb: Vector database for chunks
        num_of_chunks: Number of chunks to pick
        entity_info: List of entity info containing related chunks
        embedding_func: Function to generate embeddings
        query_embedding: Pre-computed query embedding
        
    Returns:
        list[str]: Selected chunk IDs
    """
    # 1. Collect all unique chunk IDs
    unique_chunk_ids = set()
    for item in entity_info:
        unique_chunk_ids.update(item.get("chunks", []))
    
    if not unique_chunk_ids:
        return []

    chunk_ids_list = list(unique_chunk_ids)
    
    # 2. Get chunk contents
    # We fetch content from KV storage because we need to re-embed or match
    chunks_data = await text_chunks_storage.get_by_ids(chunk_ids_list)
    
    # Pair valid chunks with their IDs
    valid_pairs = [] # List of (id, data)
    for cid, data in zip(chunk_ids_list, chunks_data):
        if data is not None:
            valid_pairs.append((cid, data))
    
    if not valid_pairs:
        return []

    # 3. Get embeddings
    # Optimally we should get from VDB, but filtering by ID list in VDB might be inefficient or unsupported
    # generic interface wise. Safest is to embed content.
    contents = [data.get("content", "") for _, data in valid_pairs]
    
    if asyncio.iscoroutinefunction(embedding_func):
        embeddings = await embedding_func(contents)
    else:
        embeddings = embedding_func(contents)
        
    # 4. Handle Query Embedding
    if query_embedding is None:
         if asyncio.iscoroutinefunction(embedding_func):
             query_batch = await embedding_func([query])
         else:
             query_batch = embedding_func([query])
         query_embedding = query_batch[0 if len(query_batch) > 0 else 0]
             
    # 5. Compute Similarity and Sort
    similarities = []
    for i, emb in enumerate(embeddings):
        # Ensure dimensions match
        if len(emb) != len(query_embedding):
            # logger.warning(f"Dimension mismatch in pick_by_vector_similarity: {len(emb)} vs {len(query_embedding)}")
            continue
            
        sim = cosine_similarity(query_embedding, emb)
        # Use the ID preserved in valid_pairs
        similarities.append((valid_pairs[i][0], sim))
        
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    return [cid for cid, score in similarities[:num_of_chunks]]


def apply_rerank_if_enabled(
    rerank_model, final_results: str, query: str, top_k: int
):
    """Apply reranking to results if a rerank model is provided."""
    if rerank_model:
        # Check if it is the new standard Reranker (StandardReranker)
        # Using duck typing to check for 'rerank' method
        if hasattr(rerank_model, "rerank") and callable(rerank_model.rerank):
             # For standard Reranker
            logger.info("Applying standard reranking...")
            # We need to parse final_results back to list/tuples if it's a string
            # But wait, the previous code passed a string?
            # Looking at original code: "results = rerank_model.rerank(query, results)"
            # It seems results was expected to be a list of strings or tuples.
            # If final_results is a string (concatenated context), we can't rerank it easily.
            # Usually rerank is applied BEFORE concatenation.
            # This function signature might be slightly off compared to usage. 
            # Let's assume final_results passed here is what the caller has. 
            # If it is a string, we might just return it. 
            # BUT, let's look at how it was used in `_query_done` in `lightrag.py` (not visible here).
            # The original utils.py had this logic.
            return rerank_model.rerank(query, final_results, top_k=top_k)
        
        # Fallback for old rerankers (like JinaReranker which might be callable directly or have predict)
        # Assuming usage like: rerank_model.predict(query, results)
        logger.warning("Using legacy reranker interface. Please migrate to StandardReranker.")
        try:
             return rerank_model.predict(query, final_results) # Hypothetical legacy method
        except AttributeError:
             return final_results # Fail safe

    return final_results


def process_chunks_unified(
    chunks: list[dict],
    limit_ids_method: str = "FIFO",
    source_ids_limit: int = 100,
    source_id: str | None = None,
) -> list[dict]:
    """Process text chunks by generating IDs and handling source ID limits.

    This unified function performs:
    1. Content hashing for deduplication
    2. Generation of unique chunk IDs
    3. Management of source ID lists with limits (FIFO/Ignore new)
    4. Integration of reference IDs

    Args:
        chunks: List of chunk dictionaries containing 'content'
        limit_ids_method: Strategy for limiting source IDs ('FIFO' or 'IGNORE_NEW')
        source_ids_limit: Maximum number of source IDs allowed per chunk
        source_id: Optional source ID to add to chunks (e.g., filename)

    Returns:
        Processed chunks with 'chunk_id', 'source_id', etc.
    """
    if not chunks:
        return []

    unique_chunks = []
    seen_hashes = set()

    for chunk in chunks:
        content = chunk.get("content", "")
        if not content:
            continue

        # 1. Generate content hash for deduplication
        content_hash = compute_mdhash_id(content, prefix="")
        
        if content_hash in seen_hashes:
            continue
        
        seen_hashes.add(content_hash)
        
        # Create processed chunk
        processed_chunk = chunk.copy()
        
        # 2. Add chunk ID
        processed_chunk["chunk_id"] = content_hash
        
        # 3. Handle Source IDs
        # Get existing source IDs (if any)
        existing_source_ids = processed_chunk.get("source_id", [])
        if isinstance(existing_source_ids, str):
            # Handle legacy string format (should act like a list of 1)
             if existing_source_ids:
                 existing_source_ids = [existing_source_ids]
             else:
                 existing_source_ids = []
        
        # Add new source_id if provided
        new_source_ids = list(existing_source_ids)
        if source_id and source_id not in new_source_ids:
            new_source_ids.append(source_id)
            
        # Apply limit strategy
        processed_chunk["source_id"] = apply_source_ids_limit(
            new_source_ids, 
            limit=source_ids_limit, 
            method=limit_ids_method,
            identifier=f"Chunk {content_hash[:8]}"
        )
        
        unique_chunks.append(processed_chunk)

    # 4. Add concise ID (optional, for readability/ordering)
    # The original code added "DC{i+1}" format.
    final_chunks = []
    for i, chunk in enumerate(unique_chunks):
        chunk_with_id = chunk.copy()
        # We preserve the detailed hash ID but also add/overwrite 'id' if needed?
        # Original code: chunk_with_id["id"] = f"DC{i + 1}"
        # This seems to be a 'local' ID for the current batch?
        # Let's keep it to be safe with existing logic
        chunk_with_id["id"] = f"DC{i + 1}"
        final_chunks.append(chunk_with_id)

    return final_chunks


def normalize_source_ids_limit_method(method: str | None) -> str:
    """Normalize the source ID limiting strategy and fall back to default when invalid."""

    if not method:
        return DEFAULT_SOURCE_IDS_LIMIT_METHOD

    normalized = method.upper()
    if normalized not in VALID_SOURCE_IDS_LIMIT_METHODS:
        logger.warning(
            "Unknown SOURCE_IDS_LIMIT_METHOD '%s', falling back to %s",
            method,
            DEFAULT_SOURCE_IDS_LIMIT_METHOD,
        )
        return DEFAULT_SOURCE_IDS_LIMIT_METHOD

    return normalized


def merge_source_ids(
    existing_ids: Iterable[str] | None, new_ids: Iterable[str] | None
) -> list[str]:
    """Merge two iterables of source IDs while preserving order and removing duplicates."""

    merged: list[str] = []
    seen: set[str] = set()

    for sequence in (existing_ids, new_ids):
        if not sequence:
            continue
        for source_id in sequence:
            if not source_id:
                continue
            if source_id not in seen:
                seen.add(source_id)
                merged.append(source_id)

    return merged


def apply_source_ids_limit(
    source_ids: Sequence[str],
    limit: int,
    method: str,
    *,
    identifier: str | None = None,
) -> list[str]:
    """Apply a limit strategy to a sequence of source IDs."""

    if limit <= 0:
        return []

    source_ids_list = list(source_ids)
    if len(source_ids_list) <= limit:
        return source_ids_list

    normalized_method = normalize_source_ids_limit_method(method)

    if normalized_method == SOURCE_IDS_LIMIT_METHOD_FIFO:
        truncated = source_ids_list[-limit:]
    else:  # IGNORE_NEW
        truncated = source_ids_list[:limit]

    if identifier and len(truncated) < len(source_ids_list):
        logger.debug(
            "Source_id truncated: %s | %s keeping %s of %s entries",
            identifier,
            normalized_method,
            len(truncated),
            len(source_ids_list),
        )

    return truncated


def compute_incremental_chunk_ids(
    existing_full_chunk_ids: list[str],
    old_chunk_ids: list[str],
    new_chunk_ids: list[str],
) -> list[str]:
    """
    Compute incrementally updated chunk IDs based on changes.

    This function applies delta changes (additions and removals) to an existing
    list of chunk IDs while maintaining order and ensuring deduplication.
    Delta additions from new_chunk_ids are placed at the end.

    Args:
        existing_full_chunk_ids: Complete list of existing chunk IDs from storage
        old_chunk_ids: Previous chunk IDs from source_id (chunks being replaced)
        new_chunk_ids: New chunk IDs from updated source_id (chunks being added)

    Returns:
        Updated list of chunk IDs with deduplication
    """
    # Calculate changes
    chunks_to_remove = set(old_chunk_ids) - set(new_chunk_ids)
    chunks_to_add = set(new_chunk_ids) - set(old_chunk_ids)

    # Apply changes to full chunk_ids
    # Step 1: Remove chunks that are no longer needed
    updated_chunk_ids = [
        cid for cid in existing_full_chunk_ids if cid not in chunks_to_remove
    ]

    # Step 2: Add new chunks (preserving order from new_chunk_ids)
    # Note: 'cid not in updated_chunk_ids' check ensures deduplication
    for cid in new_chunk_ids:
        if cid in chunks_to_add and cid not in updated_chunk_ids:
            updated_chunk_ids.append(cid)

    return updated_chunk_ids


def subtract_source_ids(
    source_ids: Iterable[str],
    ids_to_remove: Collection[str],
) -> list[str]:
    """Remove a collection of IDs from an ordered iterable while preserving order."""

    removal_set = set(ids_to_remove)
    if not removal_set:
        return [source_id for source_id in source_ids if source_id]

    return [
        source_id
        for source_id in source_ids
        if source_id and source_id not in removal_set
    ]


def make_relation_chunk_key(src: str, tgt: str) -> str:
    """Create a deterministic storage key for relation chunk tracking."""

    return GRAPH_FIELD_SEP.join(sorted((src, tgt)))


def parse_relation_chunk_key(key: str) -> tuple[str, str]:
    """Parse a relation chunk storage key back into its entity pair."""

    parts = key.split(GRAPH_FIELD_SEP)
    if len(parts) != 2:
        raise ValueError(f"Invalid relation chunk key: {key}")
    return parts[0], parts[1]


async def aexport_data(
    chunk_entity_relation_graph,
    entities_vdb,
    relationships_vdb,
    output_path: str,
    file_format: str = "csv",
    include_vector_data: bool = False,
) -> None:
    """
    Asynchronously exports all entities, relations, and relationships to various formats.

    Args:
        chunk_entity_relation_graph: Graph storage instance for entities and relations
        entities_vdb: Vector database storage for entities
        relationships_vdb: Vector database storage for relationships
        output_path: The path to the output file (including extension).
        file_format: Output format - "csv", "excel", "md", "txt".
            - csv: Comma-separated values file
            - excel: Microsoft Excel file with multiple sheets
            - md: Markdown tables
            - txt: Plain text formatted output
        include_vector_data: Whether to include data from the vector database.
    """
    import csv
    import json
    
    # Collect data
    entities_data = []
    relations_data = []
    relationships_data = []

    # --- Entities ---
    all_entities = await chunk_entity_relation_graph.get_all_labels()
    for entity_name in all_entities:
        # Get entity information from graph
        node_data = await chunk_entity_relation_graph.get_node(entity_name)
        source_id = node_data.get("source_id") if node_data else None

        entity_info = {
            "graph_data": node_data,
            "source_id": source_id,
        }

        # Optional: Get vector database information
        if include_vector_data:
            entity_id = compute_mdhash_id(entity_name, prefix="ent-")
            vector_data = await entities_vdb.get_by_id(entity_id)
            entity_info["vector_data"] = vector_data

        entity_row = {
            "entity_name": entity_name,
            "source_id": source_id,
            "graph_data": str(
                entity_info["graph_data"]
            ),  # Convert to string to ensure compatibility
        }
        if include_vector_data and "vector_data" in entity_info:
            entity_row["vector_data"] = str(entity_info["vector_data"])
        entities_data.append(entity_row)

    # --- Relations ---
    for src_entity in all_entities:
        for tgt_entity in all_entities:
            if src_entity == tgt_entity:
                continue

            edge_exists = await chunk_entity_relation_graph.has_edge(
                src_entity, tgt_entity
            )
            if edge_exists:
                # Get edge information from graph
                edge_data = await chunk_entity_relation_graph.get_edge(
                    src_entity, tgt_entity
                )
                source_id = edge_data.get("source_id") if edge_data else None

                relation_info = {
                    "graph_data": edge_data,
                    "source_id": source_id,
                }

                # Optional: Get vector database information
                if include_vector_data:
                    rel_id = compute_mdhash_id(src_entity + tgt_entity, prefix="rel-")
                    vector_data = await relationships_vdb.get_by_id(rel_id)
                    relation_info["vector_data"] = vector_data

                relation_row = {
                    "src_entity": src_entity,
                    "tgt_entity": tgt_entity,
                    "source_id": relation_info["source_id"],
                    "graph_data": str(relation_info["graph_data"]),  # Convert to string
                }
                if include_vector_data and "vector_data" in relation_info:
                    relation_row["vector_data"] = str(relation_info["vector_data"])
                relations_data.append(relation_row)

    # --- Relationships (from VectorDB) ---
    all_relationships = await relationships_vdb.client_storage
    for rel in all_relationships["data"]:
        relationships_data.append(
            {
                "relationship_id": rel["__id__"],
                "data": str(rel),  # Convert to string for compatibility
            }
        )

    # Export based on format
    if file_format == "csv":
        # CSV export
        with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
            # Entities
            if entities_data:
                csvfile.write("# ENTITIES\n")
                writer = csv.DictWriter(csvfile, fieldnames=entities_data[0].keys())
                writer.writeheader()
                writer.writerows(entities_data)
                csvfile.write("\n\n")

            # Relations
            if relations_data:
                csvfile.write("# RELATIONS\n")
                writer = csv.DictWriter(csvfile, fieldnames=relations_data[0].keys())
                writer.writeheader()
                writer.writerows(relations_data)
                csvfile.write("\n\n")

            # Relationships
            if relationships_data:
                csvfile.write("# RELATIONSHIPS\n")
                writer = csv.DictWriter(
                    csvfile, fieldnames=relationships_data[0].keys()
                )
                writer.writeheader()
                writer.writerows(relationships_data)

    elif file_format == "excel":
        # Excel export
        import pandas as pd

        entities_df = pd.DataFrame(entities_data) if entities_data else pd.DataFrame()
        relations_df = (
            pd.DataFrame(relations_data) if relations_data else pd.DataFrame()
        )
        relationships_df = (
            pd.DataFrame(relationships_data) if relationships_data else pd.DataFrame()
        )

        with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
            if not entities_df.empty:
                entities_df.to_excel(writer, sheet_name="Entities", index=False)
            if not relations_df.empty:
                relations_df.to_excel(writer, sheet_name="Relations", index=False)
            if not relationships_df.empty:
                relationships_df.to_excel(
                    writer, sheet_name="Relationships", index=False
                )

    elif file_format == "md":
        # Markdown export
        with open(output_path, "w", encoding="utf-8") as mdfile:
            mdfile.write("# EasyKnowledgeRetriever Data Export\n\n")

            # Entities
            mdfile.write("## Entities\n\n")
            if entities_data:
                # Write header
                mdfile.write("| " + " | ".join(entities_data[0].keys()) + " |\n")
                mdfile.write(
                    "| " + " | ".join(["---"] * len(entities_data[0].keys())) + " |\n"
                )

                # Write rows
                for entity in entities_data:
                    mdfile.write(
                        "| " + " | ".join(str(v) for v in entity.values()) + " |\n"
                    )
                mdfile.write("\n\n")
            else:
                mdfile.write("*No entity data available*\n\n")

            # Relations
            mdfile.write("## Relations\n\n")
            if relations_data:
                # Write header
                mdfile.write("| " + " | ".join(relations_data[0].keys()) + " |\n")
                mdfile.write(
                    "| " + " | ".join(["---"] * len(relations_data[0].keys())) + " |\n"
                )

                # Write rows
                for relation in relations_data:
                    mdfile.write(
                        "| " + " | ".join(str(v) for v in relation.values()) + " |\n"
                    )
                mdfile.write("\n\n")
            else:
                mdfile.write("*No relation data available*\n\n")

            # Relationships
            mdfile.write("## Relationships (VectorDB)\n\n")
            if relationships_data:
                # Write header
                mdfile.write("| " + " | ".join(relationships_data[0].keys()) + " |\n")
                mdfile.write(
                    "| " + " | ".join(["---"] * len(relationships_data[0].keys())) + " |\n"
                )

                # Write rows
                for relationship in relationships_data:
                    mdfile.write(
                        "| " + " | ".join(str(v) for v in relationship.values()) + " |\n"
                    )
                mdfile.write("\n\n")
            else:
                mdfile.write("*No relationship data available*\n\n")

    elif file_format == "txt":
        # Text export (simple dump)
        with open(output_path, "w", encoding="utf-8") as txtfile:
            txtfile.write("EASYKNOWLEDGERETRIEVER DATA EXPORT\n")
            txtfile.write("==============================\n\n")

            txtfile.write("ENTITIES\n")
            txtfile.write("--------\n")
            for entity in entities_data:
                txtfile.write(str(entity) + "\n")
            txtfile.write("\n")

            txtfile.write("RELATIONS\n")
            txtfile.write("---------\n")
            for relation in relations_data:
                txtfile.write(str(relation) + "\n")
            txtfile.write("\n")

            txtfile.write("RELATIONSHIPS\n")
            txtfile.write("-------------\n")
            for relationship in relationships_data:
                txtfile.write(str(relationship) + "\n")
            txtfile.write("\n")
            txtfile.write("\n")

from easy_knowledge_retriever.reranker.base import BaseRerankerService

async def process_retrieved_chunks(
    query: str,
    unique_chunks: list[dict],
    query_param: Any,
    chunk_token_limit: int,
    reranker_service: Optional[BaseRerankerService] = None,
) -> list[dict]:
    """Process and filter chunks for retrieval context.
    
    Args:
        query: Query string
        unique_chunks: List of chunk dictionaries
        query_param: QueryParam object
        chunk_token_limit: Maximum tokens allowed for chunks
        reranker_service: Optional reranker service to use
        
    Returns:
        Filtered and sorted list of chunk dictionaries
    """
    if not unique_chunks:
        return []

    final_chunks = unique_chunks
    
    if reranker_service:
        try:
            # Extract contents
            contents = [c.get("content", "") for c in final_chunks]
            # Rerank
            rerank_results = await reranker_service.rerank(query, contents)
            # Reorder unique_chunks based on results
            # results is list of {index, relevance_score}
            # Sort rerank results by score just in case, though rerank usually returns sorted
            rerank_results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
            
            reordered_chunks = []
            for r in rerank_results:
                idx = r.get("index")
                if idx is not None and 0 <= idx < len(final_chunks):
                    reordered_chunks.append(final_chunks[idx])
            
            # If we lost some chunks (shouldn't happen if reranker behaves), append missing ones or just use reordered
            if len(reordered_chunks) < len(final_chunks):
                seen_indices = {r.get("index") for r in rerank_results}
                for i, chunk in enumerate(final_chunks):
                    if i not in seen_indices:
                        reordered_chunks.append(chunk)
            
            final_chunks = reordered_chunks
            logger.info(f"Reranked {len(final_chunks)} chunks using {reranker_service.__class__.__name__}")
        except Exception as e:
            logger.error(f"Reranking failed: {e}. Falling back to original order.")

    # 2. Token Truncation
    from easy_knowledge_retriever.utils.tokenizer import truncate_list_by_token_size
    
    truncated_chunks = truncate_list_by_token_size(
        final_chunks,
        key=lambda x: x.get("content", ""),
        max_token_size=chunk_token_limit
    )
    
    return truncated_chunks
