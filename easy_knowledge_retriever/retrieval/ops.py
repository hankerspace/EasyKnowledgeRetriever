from __future__ import annotations

import asyncio
from typing import Any, List, Tuple

from easy_knowledge_retriever.kg.graph_storage.base import BaseGraphStorage
from easy_knowledge_retriever.kg.kv_storage.base import BaseKVStorage
from easy_knowledge_retriever.kg.vector_storage.base import BaseVectorStorage
from easy_knowledge_retriever.utils.logger import logger

async def get_vector_context(
    query: str,
    chunks_vdb: BaseVectorStorage,
    top_k: int,
    query_embedding: list[float] = None,
) -> list[dict]:
    """
    Retrieve text chunks from the vector database without reranking or truncation.

    Args:
        query: The query string to search for
        chunks_vdb: Vector database containing document chunks
        top_k: Number of chunks to retrieve
        query_embedding: Optional pre-computed query embedding

    Returns:
        List of text chunks with metadata
    """
    try:
        cosine_threshold = chunks_vdb.cosine_better_than_threshold

        results = await chunks_vdb.query(
            query, top_k=top_k, query_embedding=query_embedding
        )
        if not results:
            logger.info(
                f"Naive query: 0 chunks (chunk_top_k:{top_k} cosine:{cosine_threshold})"
            )
            return []

        valid_chunks = []
        for result in results:
            if "content" in result:
                chunk_with_metadata = {
                    "content": result["content"],
                    "created_at": result.get("created_at", None),
                    "file_path": result.get("file_path", "unknown_source"),
                    "source_type": "vector",  # Mark the source type
                    "chunk_id": result.get("id"),  # Add chunk_id for deduplication
                }
                valid_chunks.append(chunk_with_metadata)

        logger.info(
            f"Naive query: {len(valid_chunks)} chunks (chunk_top_k:{top_k} cosine:{cosine_threshold})"
        )
        return valid_chunks

    except Exception as e:
        logger.error(f"Error in get_vector_context: {e}")
        return []


async def get_node_data(
    query: str,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    top_k: int,
) -> Tuple[List[dict], List[dict]]:
    """
    Search for entities in vector DB and retrieve their details and related edges from KG.
    """
    # get similar entities
    logger.info(
        f"Query nodes: {query} (top_k:{top_k}, cosine:{entities_vdb.cosine_better_than_threshold})"
    )

    results = await entities_vdb.query(query, top_k=top_k)

    if not len(results):
        return [], []

    # Extract all entity IDs from your results list
    node_ids = [r["entity_name"] for r in results]

    # Call the batch node retrieval and degree functions concurrently.
    nodes_dict, degrees_dict = await asyncio.gather(
        knowledge_graph_inst.get_nodes_batch(node_ids),
        knowledge_graph_inst.node_degrees_batch(node_ids),
    )

    # Now, if you need the node data and degree in order:
    node_datas = [nodes_dict.get(nid) for nid in node_ids]
    node_degrees = [degrees_dict.get(nid, 0) for nid in node_ids]

    if not all([n is not None for n in node_datas]):
        logger.warning("Some nodes are missing, maybe the storage is damaged")

    node_datas = [
        {
            **n,
            "entity_name": k["entity_name"],
            "rank": d,
            "created_at": k.get("created_at"),
        }
        for k, n, d in zip(results, node_datas, node_degrees)
        if n is not None
    ]

    use_relations = await find_most_related_edges_from_entities(
        node_datas,
        knowledge_graph_inst,
    )

    logger.info(
        f"Local query: {len(node_datas)} entites, {len(use_relations)} relations"
    )

    # Entities are sorted by cosine similarity
    # Relations are sorted by rank + weight
    return node_datas, use_relations


async def find_most_related_edges_from_entities(
    node_datas: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
) -> List[dict]:
    node_names = [dp["entity_name"] for dp in node_datas]
    batch_edges_dict = await knowledge_graph_inst.get_nodes_edges_batch(node_names)

    all_edges = []
    seen = set()

    for node_name in node_names:
        this_edges = batch_edges_dict.get(node_name, [])
        for e in this_edges:
            sorted_edge = tuple(sorted(e))
            if sorted_edge not in seen:
                seen.add(sorted_edge)
                all_edges.append(sorted_edge)

    # Prepare edge pairs in two forms:
    # For the batch edge properties function, use dicts.
    edge_pairs_dicts = [{"src": e[0], "tgt": e[1]} for e in all_edges]
    # For edge degrees, use tuples.
    edge_pairs_tuples = list(all_edges)  # all_edges is already a list of tuples

    # Call the batched functions concurrently.
    edge_data_dict, edge_degrees_dict = await asyncio.gather(
        knowledge_graph_inst.get_edges_batch(edge_pairs_dicts),
        knowledge_graph_inst.edge_degrees_batch(edge_pairs_tuples),
    )

    # Reconstruct edge_datas list in the same order as the deduplicated results.
    all_edges_data = []
    for pair in all_edges:
        edge_props = edge_data_dict.get(pair)
        if edge_props is not None:
            if "weight" not in edge_props:
                logger.warning(
                    f"Edge {pair} missing 'weight' attribute, using default value 1.0"
                )
                edge_props["weight"] = 1.0

            combined = {
                "src_tgt": pair,
                "rank": edge_degrees_dict.get(pair, 0),
                **edge_props,
            }
            all_edges_data.append(combined)

    all_edges_data = sorted(
        all_edges_data, key=lambda x: (x["rank"], x["weight"]), reverse=True
    )

    return all_edges_data


async def get_edge_data(
    keywords: str | list[str],
    knowledge_graph_inst: BaseGraphStorage,
    relationships_vdb: BaseVectorStorage,
    top_k: int,
) -> Tuple[List[dict], List[dict]]:
    """
    Search for edges in vector DB and retrieve their details and related entities from KG.
    """
    logger.info(
        f"Query edges: {keywords} (top_k:{top_k}, cosine:{relationships_vdb.cosine_better_than_threshold})"
    )

    results = await relationships_vdb.query(keywords, top_k=top_k)

    if not len(results):
        return [], []

    # Prepare edge pairs in two forms:
    # For the batch edge properties function, use dicts.
    edge_pairs_dicts = [{"src": r["src_id"], "tgt": r["tgt_id"]} for r in results]
    edge_data_dict = await knowledge_graph_inst.get_edges_batch(edge_pairs_dicts)

    # Reconstruct edge_datas list in the same order as results.
    edge_datas = []
    for k in results:
        pair = (k["src_id"], k["tgt_id"])
        edge_props = edge_data_dict.get(pair)
        if edge_props is not None:
            if "weight" not in edge_props:
                logger.warning(
                    f"Edge {pair} missing 'weight' attribute, using default value 1.0"
                )
                edge_props["weight"] = 1.0

            # Keep edge data without rank, maintain vector search order
            combined = {
                "src_id": k["src_id"],
                "tgt_id": k["tgt_id"],
                "created_at": k.get("created_at", None),
                **edge_props,
            }
            edge_datas.append(combined)

    # Relations maintain vector search order (sorted by similarity)

    use_entities = await find_most_related_entities_from_relationships(
        edge_datas,
        knowledge_graph_inst,
    )

    logger.info(
        f"Global query: {len(use_entities)} entites, {len(edge_datas)} relations"
    )

    return edge_datas, use_entities


async def find_most_related_entities_from_relationships(
    edge_datas: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
) -> List[dict]:
    entity_names = []
    seen = set()

    for e in edge_datas:
        if e["src_id"] not in seen:
            entity_names.append(e["src_id"])
            seen.add(e["src_id"])
        if e["tgt_id"] not in seen:
            entity_names.append(e["tgt_id"])
            seen.add(e["tgt_id"])

    # Only get nodes data, no need for node degrees
    nodes_dict = await knowledge_graph_inst.get_nodes_batch(entity_names)

    # Rebuild the list in the same order as entity_names
    node_datas = []
    for entity_name in entity_names:
        node = nodes_dict.get(entity_name)
        if node is None:
            logger.warning(f"Node '{entity_name}' not found in batch retrieval.")
            continue
        # Combine the node data with the entity name, no rank needed
        combined = {**node, "entity_name": entity_name}
        node_datas.append(combined)

    return node_datas
