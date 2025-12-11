from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ...llm.utils import EmbeddingFunc
from ..base import StorageNameSpace


@dataclass
class BaseVectorStorage(StorageNameSpace, ABC):
    embedding_func: EmbeddingFunc | None = None
    cosine_better_than_threshold: float = field(default=0.2)
    meta_fields: set[str] = field(default_factory=set)
    embedding_dim: int = field(default=1536)

    @abstractmethod
    async def query(
        self, query: str, top_k: int, query_embedding: list[float] = None
    ) -> list[dict[str, Any]]:
        """Query the vector storage and retrieve top_k results.

        Args:
            query: The query string to search for
            top_k: Number of top results to return
            query_embedding: Optional pre-computed embedding for the query.
                           If provided, skips embedding computation for better performance.
        """

    @abstractmethod
    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        """Insert or update vectors in the storage.

        Importance notes for in-memory storage:
        1. Changes will be persisted to disk during the next index_done_callback
        2. Only one process should updating the storage at a time before index_done_callback,
           KG-storage-log should be used to avoid data corruption
        """

    @abstractmethod
    async def delete_entity(self, entity_name: str) -> None:
        """Delete a single entity by its name.

        Importance notes for in-memory storage:
        1. Changes will be persisted to disk during the next index_done_callback
        2. Only one process should updating the storage at a time before index_done_callback,
           KG-storage-log should be used to avoid data corruption
        """

    @abstractmethod
    async def delete_entity_relation(self, entity_name: str) -> None:
        """Delete relations for a given entity.

        Importance notes for in-memory storage:
        1. Changes will be persisted to disk during the next index_done_callback
        2. Only one process should updating the storage at a time before index_done_callback,
           KG-storage-log should be used to avoid data corruption
        """

    @abstractmethod
    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        """Get vector data by its ID

        Args:
            id: The unique identifier of the vector

        Returns:
            The vector data if found, or None if not found
        """
        pass

    @abstractmethod
    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        """Get multiple vector data by their IDs

        Args:
            ids: List of unique identifiers

        Returns:
            List of vector data objects that were found
        """
        pass

    @abstractmethod
    async def delete(self, ids: list[str]):
        """Delete vectors with specified IDs

        Importance notes for in-memory storage:
        1. Changes will be persisted to disk during the next index_done_callback
        2. Only one process should updating the storage at a time before index_done_callback,
           KG-storage-log should be used to avoid data corruption

        Args:
            ids: List of vector IDs to be deleted
        """

    @abstractmethod
    async def get_vectors_by_ids(self, ids: list[str]) -> dict[str, list[float]]:
        """Get vectors by their IDs, returning only ID and vector data for efficiency

        Args:
            ids: List of unique identifiers

        Returns:
            Dictionary mapping IDs to their vector embeddings
            Format: {id: [vector_values], ...}
        """
        pass
