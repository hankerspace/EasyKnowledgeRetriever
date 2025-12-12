from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ...llm.utils import EmbeddingFunc
from ..base import StorageNameSpace


@dataclass
class BaseKVStorage(StorageNameSpace, ABC):
    embedding_func: EmbeddingFunc | None = None

    @abstractmethod
    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        """Get value by id"""

    @abstractmethod
    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        """Get values by ids"""

    @abstractmethod
    async def filter_keys(self, keys: set[str]) -> set[str]:
        """Return un-exist keys"""

    @abstractmethod
    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        """Upsert data

        Importance notes for in-memory storage:
        1. Changes will be persisted to disk during the next index_done_callback
        2. update flags to notify other processes that data persistence is needed
        """

    @abstractmethod
    async def delete(self, ids: list[str]) -> None:
        """Delete specific records from storage by their IDs

        Importance notes for in-memory storage:
        1. Changes will be persisted to disk during the next index_done_callback
        2. update flags to notify other processes that data persistence is needed

        Args:
            ids (list[str]): List of document IDs to be deleted from storage

        Returns:
            None
        """

    @abstractmethod
    async def is_empty(self) -> bool:
        """Check if the storage is empty

        Returns:
            bool: True if storage contains no data, False otherwise
        """


class DocStatus(str, Enum):
    """Document processing status"""

    PENDING = "pending"
    PROCESSING = "processing"
    PREPROCESSED = "preprocessed"
    PROCESSED = "processed"
    FAILED = "failed"


@dataclass
class DocProcessingStatus:
    """Document processing status data structure"""

    content_summary: str
    """First 100 chars of document content, used for preview"""
    content_length: int
    """Total length of document"""
    file_path: str
    """File path of the document"""
    status: DocStatus
    """Current processing status"""
    created_at: str
    """ISO format timestamp when document was created"""
    updated_at: str
    """ISO format timestamp when document was last updated"""
    track_id: str | None = None
    """Tracking ID for monitoring progress"""
    chunks_count: int | None = None
    """Number of chunks after splitting, used for processing"""
    chunks_list: list[str] | None = field(default_factory=list)
    """List of chunk IDs associated with this document, used for deletion"""
    error_msg: str | None = None
    """Error message if failed"""
    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata"""
    multimodal_processed: bool | None = field(default=None, repr=False)
    """Internal field: indicates if multimodal processing is complete. Not shown in repr() but accessible for debugging."""

    def __post_init__(self):
        """
        Handle status conversion based on multimodal_processed field.

        Business rules:
        - If multimodal_processed is False and status is PROCESSED,
          then change status to PREPROCESSED
        - The multimodal_processed field is kept (with repr=False) for internal use and debugging
        """
        # Apply status conversion logic
        if self.multimodal_processed is not None:
            if (
                self.multimodal_processed is False
                and self.status == DocStatus.PROCESSED
            ):
                self.status = DocStatus.PREPROCESSED


@dataclass
class DocStatusStorage(BaseKVStorage, ABC):
    """Base class for document status storage"""

    @abstractmethod
    async def get_status_counts(self) -> dict[str, int]:
        """Get counts of documents in each status"""

    @abstractmethod
    async def get_docs_by_status(
        self, status: DocStatus
    ) -> dict[str, DocProcessingStatus]:
        """Get all documents with a specific status"""

    @abstractmethod
    async def get_docs_by_track_id(
        self, track_id: str
    ) -> dict[str, DocProcessingStatus]:
        """Get all documents with a specific track_id"""

    @abstractmethod
    async def get_docs_paginated(
        self,
        status_filter: DocStatus | None = None,
        page: int = 1,
        page_size: int = 50,
        sort_field: str = "updated_at",
        sort_direction: str = "desc",
    ) -> tuple[list[tuple[str, DocProcessingStatus]], int]:
        """Get documents with pagination support

        Args:
            status_filter: Filter by document status, None for all statuses
            page: Page number (1-based)
            page_size: Number of documents per page (10-200)
            sort_field: Field to sort by ('created_at', 'updated_at', 'id')
            sort_direction: Sort direction ('asc' or 'desc')

        Returns:
            Tuple of (list of (doc_id, DocProcessingStatus) tuples, total_count)
        """

    @abstractmethod
    async def get_all_status_counts(self) -> dict[str, int]:
        """Get counts of documents in each status for all documents

        Returns:
            Dictionary mapping status names to counts
        """

    @abstractmethod
    async def get_doc_by_file_path(self, file_path: str) -> tuple[str, dict[str, Any]] | None:
        """Get document by file path

        Args:
            file_path: The file path to search for

        Returns:
            tuple[str, dict[str, Any]] | None: Tuple of (doc_id, doc_data) if found, None otherwise
            doc_data follows the same format as get_by_ids method
        """
