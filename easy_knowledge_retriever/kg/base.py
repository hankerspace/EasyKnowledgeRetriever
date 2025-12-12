from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Literal,
    TypedDict,
    TypeVar,
    Callable,
    Optional,
    Dict,
    List,
    AsyncIterator,
)


class TextChunkSchema(TypedDict):
    tokens: int
    content: str
    full_doc_id: str
    chunk_order_index: int
    page_start: Optional[int]


T = TypeVar("T")


@dataclass
class QueryParam:
    """Configuration parameters for query execution in EasyKnowledgeRetriever."""

    mode: Literal["local", "global", "hybrid", "naive", "mix", "bypass"] = "mix"
    """Specifies the retrieval mode:
    - "local": Focuses on context-dependent information.
    - "global": Utilizes global knowledge.
    - "hybrid": Combines local and global retrieval methods.
    - "naive": Performs a basic search without advanced techniques.
    - "mix": Integrates knowledge graph and vector retrieval.
    """

    only_need_context: bool = False
    """If True, only returns the retrieved context without generating a response."""

    only_need_prompt: bool = False
    """If True, only returns the generated prompt without producing a response."""

    response_type: str = "Multiple Paragraphs"
    """Defines the response format. Examples: 'Multiple Paragraphs', 'Single Paragraph', 'Bullet Points'."""

    stream: bool = False
    """If True, enables streaming output for real-time responses."""

    top_k: int = 40
    """Number of top items to retrieve. Represents entities in 'local' mode and relationships in 'global' mode."""

    chunk_top_k: int = None
    """Number of text chunks to retrieve initially from vector search and keep after reranking.
    If None, defaults to top_k value.
    """

    max_entity_tokens: int = None
    """Maximum number of tokens allocated for entity context in unified token control system."""

    max_relation_tokens: int = None
    """Maximum number of tokens allocated for relationship context in unified token control system."""

    max_total_tokens: int = None
    """Maximum total tokens budget for the entire query context (entities + relations + chunks + system prompt)."""

    hl_keywords: list[str] = field(default_factory=list)
    """List of high-level keywords to prioritize in retrieval."""

    ll_keywords: list[str] = field(default_factory=list)
    """List of low-level keywords to refine retrieval focus."""

    # History mesages is only send to LLM for context, not used for retrieval
    conversation_history: list[dict[str, str]] = field(default_factory=list)
    """Stores past conversation history to maintain context.
    Format: [{"role": "user/assistant", "content": "message"}].
    """

    # TODO: deprecated. No longer used in the codebase, all conversation_history messages is send to LLM
    history_turns: int = 0
    """Number of complete conversation turns (user-assistant pairs) to consider in the response context."""

    model_func: Callable[..., object] | None = None
    """Optional override for the LLM model function to use for this specific query.
    If provided, this will be used instead of the global model function.
    This allows using different models for different query modes.
    """

    user_prompt: str | None = None
    """User-provided prompt for the query.
    Addition instructions for LLM. If provided, this will be inject into the prompt template.
    It's purpose is the let user customize the way LLM generate the response.
    """

    include_references: bool = False
    """If True, includes reference list in the response for supported endpoints.
    This parameter controls whether the API response includes a references field
    containing citation information for the retrieved content.
    """

    query_decomposition: bool = False
    """If True, enables query decomposition for complex queries.
    The query will be split into sub-queries, retrieved separately, and then results merged.
    """


@dataclass
class StorageNameSpace(ABC):
    namespace: str | None = None
    workspace: str = ""
    working_dir: str = ""

    async def initialize(self):
        """Initialize the storage"""
        pass

    async def finalize(self):
        """Finalize the storage"""
        pass

    def create(self, namespace: str, **kwargs) -> StorageNameSpace:
        """Create a new instance with a specific namespace"""
        from dataclasses import replace
        return replace(self, namespace=namespace, **kwargs)

    @abstractmethod
    async def index_done_callback(self) -> None:
        """Commit the storage operations after indexing"""

    @abstractmethod
    async def drop(self) -> dict[str, str]:
        """Drop all data from storage and clean up resources

        This abstract method defines the contract for dropping all data from a storage implementation.
        Each storage type must implement this method to:
        1. Clear all data from memory and/or external storage
        2. Remove any associated storage files if applicable
        3. Reset the storage to its initial state
        4. Handle cleanup of any resources
        5. Notify other processes if necessary
        6. This action should persistent the data to disk immediately.

        Returns:
            dict[str, str]: Operation status and message with the following format:
                {
                    "status": str,  # "success" or "error"
                    "message": str  # "data dropped" on success, error details on failure
                }

        Implementation specific:
        - On success: return {"status": "success", "message": "data dropped"}
        - On failure: return {"status": "error", "message": "<error details>"}
        - If not supported: return {"status": "error", "message": "unsupported"}
        """


class StoragesStatus(str, Enum):
    """Storages status"""

    NOT_CREATED = "not_created"
    CREATED = "created"
    INITIALIZED = "initialized"
    FINALIZED = "finalized"


@dataclass
class DeletionResult:
    """Represents the result of a deletion operation."""

    status: Literal["success", "not_found", "fail"]
    doc_id: str
    message: str
    status_code: int = 200
    file_path: str | None = None


# Unified Query Result Data Structures for Reference List Support


@dataclass
class Entity:
    """Represents an entity in the knowledge graph."""
    entity_name: str
    entity_type: str
    description: str
    source_id: str
    file_path: str = ""
    created_at: str = ""
    reference_id: str = ""

@dataclass
class Relationship:
    """Represents a relationship between entities."""
    src_id: str
    tgt_id: str
    description: str
    keywords: str
    weight: float
    source_id: str
    file_path: str = ""
    created_at: str = ""
    reference_id: str = ""

@dataclass
class Chunk:
    """Represents a text chunk."""
    content: str
    file_path: str
    chunk_id: str
    reference_id: str = ""
    page_start: Optional[int] = None
    page_end: Optional[int] = None

@dataclass
class Reference:
    """Represents a file reference."""
    reference_id: str
    file_path: str

@dataclass
class QueryResult:
    """
    Unified query result data structure for all query modes.

    Attributes:
        content: Text content for non-streaming responses (LLM response)
        response_iterator: Streaming response iterator for streaming responses
        raw_data: Complete structured data including references and metadata (legacy dict)
        is_streaming: Whether this is a streaming result
        context: The context text used for the query
        entities: List of entities used
        relationships: List of relationships used
        chunks: List of text chunks used
        references: List of references
        metadata: Query metadata
        status: Operation status
        message: Operation message
    """

    content: Optional[str] = None
    response_iterator: Optional[AsyncIterator[str]] = None
    raw_data: Optional[Dict[str, Any]] = None
    is_streaming: bool = False
    
    # New structured fields
    query: str = ""
    system_prompt: str = ""
    user_prompt: str = ""
    context: Optional[str] = None
    entities: List[Entity] = field(default_factory=list)
    relationships: List[Relationship] = field(default_factory=list)
    chunks: List[Chunk] = field(default_factory=list)
    references: List[Reference] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    status: Literal["success", "failure"] = "success"
    message: str = ""

    @property
    def reference_list(self) -> List[Dict[str, str]]:
        """
        Convenient property to extract reference list from raw_data.

        Returns:
            List[Dict[str, str]]: Reference list in format:
            [{"reference_id": "1", "file_path": "/path/to/file.pdf"}, ...]
        """
        if self.references:
            return [{"reference_id": r.reference_id, "file_path": r.file_path} for r in self.references]
        if self.raw_data:
            return self.raw_data.get("data", {}).get("references", [])
        return []

    @property
    def files(self) -> List[str]:
        """Returns list of unique file paths."""
        return list(set(r.file_path for r in self.references))


@dataclass
class QueryContextResult:
    """
    Unified query context result data structure.

    Attributes:
        context: LLM context string
        raw_data: Complete structured data including reference_list
    """

    context: str
    raw_data: Dict[str, Any]

    @property
    def reference_list(self) -> List[Dict[str, str]]:
        """Convenient property to extract reference list from raw_data."""
        return self.raw_data.get("data", {}).get("references", [])
