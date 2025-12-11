from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from ...llm.utils import EmbeddingFunc
from ..base import StorageNameSpace
from ..types import KnowledgeGraph


@dataclass
class BaseGraphStorage(StorageNameSpace, ABC):
    """All operations related to edges in graph should be undirected."""

    embedding_func: EmbeddingFunc | None = None

    @abstractmethod
    async def has_node(self, node_id: str) -> bool:
        """Check if a node exists in the graph.

        Args:
            node_id: The ID of the node to check

        Returns:
            True if the node exists, False otherwise
        """

    @abstractmethod
    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        """Check if an edge exists between two nodes.

        Args:
            source_node_id: The ID of the source node
            target_node_id: The ID of the target node

        Returns:
            True if the edge exists, False otherwise
        """

    @abstractmethod
    async def node_degree(self, node_id: str) -> int:
        """Get the degree (number of connected edges) of a node.

        Args:
            node_id: The ID of the node

        Returns:
            The number of edges connected to the node
        """

    @abstractmethod
    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        """Get the total degree of an edge (sum of degrees of its source and target nodes).

        Args:
            src_id: The ID of the source node
            tgt_id: The ID of the target node

        Returns:
            The sum of the degrees of the source and target nodes
        """

    @abstractmethod
    async def get_node(self, node_id: str) -> dict[str, str] | None:
        """Get node by its ID, returning only node properties.

        Args:
            node_id: The ID of the node to retrieve

        Returns:
            A dictionary of node properties if found, None otherwise
        """

    @abstractmethod
    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> dict[str, str] | None:
        """Get edge properties between two nodes.

        Args:
            source_node_id: The ID of the source node
            target_node_id: The ID of the target node

        Returns:
            A dictionary of edge properties if found, None otherwise
        """

    @abstractmethod
    async def get_node_edges(self, source_node_id: str) -> list[tuple[str, str]] | None:
        """Get all edges connected to a node.

        Args:
            source_node_id: The ID of the node to get edges for

        Returns:
            A list of (source_id, target_id) tuples representing edges,
            or None if the node doesn't exist
        """

    async def get_nodes_batch(self, node_ids: list[str]) -> dict[str, dict]:
        """Get nodes as a batch using UNWIND

        Default implementation fetches nodes one by one.
        Override this method for better performance in storage backends
        that support batch operations.
        """
        result = {}
        for node_id in node_ids:
            node = await self.get_node(node_id)
            if node is not None:
                result[node_id] = node
        return result

    async def node_degrees_batch(self, node_ids: list[str]) -> dict[str, int]:
        """Node degrees as a batch using UNWIND

        Default implementation fetches node degrees one by one.
        Override this method for better performance in storage backends
        that support batch operations.
        """
        result = {}
        for node_id in node_ids:
            degree = await self.node_degree(node_id)
            result[node_id] = degree
        return result

    async def edge_degrees_batch(
        self, edge_pairs: list[tuple[str, str]]
    ) -> dict[tuple[str, str], int]:
        """Edge degrees as a batch using UNWIND also uses node_degrees_batch

        Default implementation calculates edge degrees one by one.
        Override this method for better performance in storage backends
        that support batch operations.
        """
        result = {}
        for src_id, tgt_id in edge_pairs:
            degree = await self.edge_degree(src_id, tgt_id)
            result[(src_id, tgt_id)] = degree
        return result

    async def get_edges_batch(
        self, pairs: list[dict[str, str]]
    ) -> dict[tuple[str, str], dict]:
        """Get edges as a batch using UNWIND

        Default implementation fetches edges one by one.
        Override this method for better performance in storage backends
        that support batch operations.
        """
        result = {}
        for pair in pairs:
            src_id = pair["src"]
            tgt_id = pair["tgt"]
            edge = await self.get_edge(src_id, tgt_id)
            if edge is not None:
                result[(src_id, tgt_id)] = edge
        return result

    async def get_nodes_edges_batch(
        self, node_ids: list[str]
    ) -> dict[str, list[tuple[str, str]]]:
        """Get nodes edges as a batch using UNWIND

        Default implementation fetches node edges one by one.
        Override this method for better performance in storage backends
        that support batch operations.
        """
        result = {}
        for node_id in node_ids:
            edges = await self.get_node_edges(node_id)
            result[node_id] = edges if edges is not None else []
        return result

    @abstractmethod
    async def upsert_node(self, node_id: str, node_data: dict[str, str]) -> None:
        """Insert a new node or update an existing node in the graph.

        Importance notes for in-memory storage:
        1. Changes will be persisted to disk during the next index_done_callback
        2. Only one process should updating the storage at a time before index_done_callback,
           KG-storage-log should be used to avoid data corruption

        Args:
            node_id: The ID of the node to insert or update
            node_data: A dictionary of node properties
        """

    @abstractmethod
    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ) -> None:
        """Insert a new edge or update an existing edge in the graph.

        Importance notes for in-memory storage:
        1. Changes will be persisted to disk during the next index_done_callback
        2. Only one process should updating the storage at a time before index_done_callback,
           KG-storage-log should be used to avoid data corruption

        Args:
            source_node_id: The ID of the source node
            target_node_id: The ID of the target node
            edge_data: A dictionary of edge properties
        """

    @abstractmethod
    async def delete_node(self, node_id: str) -> None:
        """Delete a node from the graph.

        Importance notes for in-memory storage:
        1. Changes will be persisted to disk during the next index_done_callback
        2. Only one process should updating the storage at a time before index_done_callback,
           KG-storage-log should be used to avoid data corruption

        Args:
            node_id: The ID of the node to delete
        """

    @abstractmethod
    async def remove_nodes(self, nodes: list[str]):
        """Delete multiple nodes

        Importance notes:
        1. Changes will be persisted to disk during the next index_done_callback
        2. Only one process should updating the storage at a time before index_done_callback,
           KG-storage-log should be used to avoid data corruption

        Args:
            nodes: List of node IDs to be deleted
        """

    @abstractmethod
    async def remove_edges(self, edges: list[tuple[str, str]]):
        """Delete multiple edges

        Importance notes:
        1. Changes will be persisted to disk during the next index_done_callback
        2. Only one process should updating the storage at a time before index_done_callback,
           KG-storage-log should be used to avoid data corruption

        Args:
            edges: List of edges to be deleted, each edge is a (source, target) tuple
        """

    # TODO: deprecated
    @abstractmethod
    async def get_all_labels(self) -> list[str]:
        """Get all labels in the graph.

        Returns:
            A list of all node labels in the graph, sorted alphabetically
        """

    @abstractmethod
    async def get_knowledge_graph(
        self, node_label: str, max_depth: int = 3, max_nodes: int = 1000
    ) -> KnowledgeGraph:
        """
        Retrieve a connected subgraph of nodes where the label includes the specified `node_label`.

        Args:
            node_label: Label of the starting node，* means all nodes
            max_depth: Maximum depth of the subgraph, Defaults to 3
            max_nodes: Maxiumu nodes to return, Defaults to 1000（BFS if possible)

        Returns:
            KnowledgeGraph object containing nodes and edges, with an is_truncated flag
            indicating whether the graph was truncated due to max_nodes limit
        """

    @abstractmethod
    async def get_all_nodes(self) -> list[dict]:
        """Get all nodes in the graph.

        Returns:
            A list of all nodes, where each node is a dictionary of its properties
            (Edge is bidirectional for some storage implementation; deduplication must be handled by the caller)
        """

    @abstractmethod
    async def get_all_edges(self) -> list[dict]:
        """Get all edges in the graph.

        Returns:
            A list of all edges, where each edge is a dictionary of its properties
        """

    @abstractmethod
    async def get_popular_labels(self, limit: int = 300) -> list[str]:
        """Get popular labels by node degree (most connected entities)

        Args:
            limit: Maximum number of labels to return

        Returns:
            List of labels sorted by degree (highest first)
        """

    @abstractmethod
    async def search_labels(self, query: str, limit: int = 50) -> list[str]:
        """Search labels with fuzzy matching

        Args:
            query: Search query string
            limit: Maximum number of results to return

        Returns:
            List of matching labels sorted by relevance
        """
