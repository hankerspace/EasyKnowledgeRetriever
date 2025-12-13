from __future__ import annotations

import traceback
import asyncio
import configparser
import inspect
import os
import time
import warnings
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from functools import partial
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Iterator,
    cast,
    final,
    Literal,
    Optional,
    List,
    Dict,
    Union,
)
from easy_knowledge_retriever.utils.logger import logger
from easy_knowledge_retriever.utils.hashing import compute_mdhash_id, generate_track_id
from easy_knowledge_retriever.utils.file_utils import load_json, write_json
from easy_knowledge_retriever.llm.utils import EmbeddingFunc
from easy_knowledge_retriever.llm.service import BaseLLMService, BaseEmbeddingService
from easy_knowledge_retriever.llm.prompts import PROMPTS
from easy_knowledge_retriever.kg.exceptions import PipelineCancelledException
from easy_knowledge_retriever.constants import (

    DEFAULT_MAX_GLEANING,
    DEFAULT_FORCE_LLM_SUMMARY_ON_MERGE,
    DEFAULT_MAX_GRAPH_NODES,
    DEFAULT_MAX_SOURCE_IDS_PER_ENTITY,
    DEFAULT_MAX_SOURCE_IDS_PER_RELATION,
    DEFAULT_ENTITY_TYPES,

    DEFAULT_SOURCE_IDS_LIMIT_METHOD,
    DEFAULT_MAX_FILE_PATHS,
    DEFAULT_FILE_PATH_MORE_PLACEHOLDER,
    DEFAULT_MAX_PARALLEL_INSERT,
)



from easy_knowledge_retriever.kg.registry import (
    STORAGES,
    verify_storage_implementation,
)




from easy_knowledge_retriever.retrieval.base import BaseRetrieval
from easy_knowledge_retriever.retrieval.retrieval_factory import RetrievalFactory
from easy_knowledge_retriever.reranker.base import BaseRerankerService




from easy_knowledge_retriever.kg.shared_memory import (
    get_namespace_data,
    get_default_workspace,
    set_default_workspace,
)
from easy_knowledge_retriever.kg.concurrency import (
    get_data_init_lock,
    get_namespace_lock,
)

from easy_knowledge_retriever.kg.base import (
    QueryParam,
    StorageNameSpace,
    StoragesStatus,
    DeletionResult,
    QueryResult,
    QueryContextResult,
    Entity,
    Relationship,
    Chunk,
    Reference,
)
from easy_knowledge_retriever.kg.kv_storage.base import (
    BaseKVStorage,
    DocStatusStorage,
    DocStatus,
    DocProcessingStatus,
)
from easy_knowledge_retriever.kg.graph_storage.base import BaseGraphStorage
from easy_knowledge_retriever.kg.vector_storage.base import BaseVectorStorage
from easy_knowledge_retriever.kg.services import (
    BaseKVStorageService,
    BaseVectorStorageService,
    BaseGraphStorageService,
    BaseDocStatusStorageService,
    KVStorageService,
    VectorStorageService,
    GraphStorageService,
    DocStatusStorageService
)
from easy_knowledge_retriever.kg.namespace import NameSpace
from easy_knowledge_retriever.operations.chunking import chunking_by_token_size
from easy_knowledge_retriever.operations.extraction import extract_entities
from easy_knowledge_retriever.operations.graph_ops import (
    merge_nodes_and_edges,
    rebuild_knowledge_from_chunks,
)
from easy_knowledge_retriever.retrieval.query_processing import (
    kg_query,
    naive_query,
    decompose_query,
    merge_query_results,
)
from easy_knowledge_retriever.constants import GRAPH_FIELD_SEP
from easy_knowledge_retriever.utils.tokenizer import Tokenizer, TiktokenTokenizer
from easy_knowledge_retriever.utils.async_utils import (
    always_get_an_event_loop,
    priority_limit_async_func_call,
)
from easy_knowledge_retriever.utils.common_utils import (
    lazy_external_import,
    convert_to_user_format,
)
from easy_knowledge_retriever.utils.vector_utils import (
    subtract_source_ids,
    make_relation_chunk_key,
    normalize_source_ids_limit_method,
)
from easy_knowledge_retriever.utils.text_utils import sanitize_text_for_encoding, get_content_summary
from easy_knowledge_retriever.kg.types import KnowledgeGraph



@final
@dataclass
class EasyKnowledgeRetriever:
    """Easy Knowledge Retreiver: Simple and Fast Retrieval-Augmented Generation."""

    # Directory
    # ---

    working_dir: str = field(default="./rag_storage")
    """Directory where cache and temporary files are stored."""

    # Storage
    # ---

    kv_storage: Union[str, BaseKVStorageService, BaseKVStorage] = field(default="JsonKVStorage")
    """Storage backend for key-value data."""

    vector_storage: Union[str, BaseVectorStorageService, BaseVectorStorage] = field(default="NanoVectorDBStorage")
    """Storage backend for vector embeddings."""

    graph_storage: Union[str, BaseGraphStorageService, BaseGraphStorage] = field(default="NetworkXStorage")
    """Storage backend for knowledge graphs."""

    doc_status_storage: Union[str, BaseDocStatusStorageService, DocStatusStorage] = field(default="JsonDocStatusStorage")
    """Storage type for tracking document processing statuses."""

    # Workspace
    # ---

    workspace: str = field(default="")
    """Workspace for data isolation. Defaults to empty string if WORKSPACE environment variable is not set."""

    # Logging (Deprecated, use setup_logger in utils.py instead)
    # ---
    log_level: int | None = field(default=None)
    log_file_path: str | None = field(default=None)

    # Query parameters
    # ---

    top_k: int = field(default=40)
    """Number of entities/relations to retrieve for each query."""

    chunk_top_k: int = field(
        default=20
    )
    """Maximum number of chunks in context."""

    max_entity_tokens: int = field(
        default=6000
    )
    """Maximum number of tokens for entity in context."""

    max_relation_tokens: int = field(
        default=8000
    )
    """Maximum number of tokens for relation in context."""

    max_total_tokens: int = field(
        default=30000
    )
    """Maximum total tokens in context (including system prompt, entities, relations and chunks)."""

    cosine_threshold: int = field(
        default=0.2
    )
    """Cosine threshold of vector DB retrieval for entities, relations and chunks."""

    related_chunk_number: int = field(
        default=5
    )
    """Number of related chunks to grab from single entity or relation."""

    kg_chunk_pick_method: str = field(
        default="VECTOR"
    )
    """Method for selecting text chunks: 'WEIGHT' for weight-based selection, 'VECTOR' for embedding similarity-based selection."""

    # Entity extraction
    # ---

    entity_extract_max_gleaning: int = field(
        default=DEFAULT_MAX_GLEANING
    )
    """Maximum number of entity extraction attempts for ambiguous content."""

    force_llm_summary_on_merge: int = field(
        default=DEFAULT_FORCE_LLM_SUMMARY_ON_MERGE
    )

    # Text chunking
    # ---

    chunk_token_size: int = field(default=1200)
    """Maximum number of tokens per text chunk when splitting documents."""

    chunk_overlap_token_size: int = field(
        default=100
    )
    """Number of overlapping tokens between consecutive text chunks to preserve context."""

    tokenizer: Optional[Tokenizer] = field(default=None)
    """
    A function that returns a Tokenizer instance.
    If None, and a `tiktoken_model_name` is provided, a TiktokenTokenizer will be created.
    If both are None, the default TiktokenTokenizer is used.
    """

    tiktoken_model_name: str = field(default="gpt-4o-mini")
    """Model name used for tokenization when chunking text with tiktoken. Defaults to `gpt-4o-mini`."""

    chunking_func: Callable[
        [
            Tokenizer,
            str,
            Optional[str],
            bool,
            int,
            int,
        ],
        Union[List[Dict[str, Any]], Awaitable[List[Dict[str, Any]]]],
    ] = field(default_factory=lambda: chunking_by_token_size)
    """
    Custom chunking function for splitting text into chunks before processing.

    The function can be either synchronous or asynchronous.

    The function should take the following parameters:

        - `tokenizer`: A Tokenizer instance to use for tokenization.
        - `content`: The text to be split into chunks.
        - `split_by_character`: The character to split the text on. If None, the text is split into chunks of `chunk_token_size` tokens.
        - `split_by_character_only`: If True, the text is split only on the specified character.
        - `chunk_overlap_token_size`: The number of overlapping tokens between consecutive chunks.
        - `chunk_token_size`: The maximum number of tokens per chunk.


    The function should return a list of dictionaries (or an awaitable that resolves to a list),
    where each dictionary contains the following keys:
        - `tokens` (int): The number of tokens in the chunk.
        - `content` (str): The text content of the chunk.
        - `chunk_order_index` (int): Zero-based index indicating the chunk's order in the document.

    Defaults to `chunking_by_token_size` if not specified.
    """


    # Embedding
    # ---

    embedding_service: BaseEmbeddingService | None = field(default=None)
    """Service for computing text embeddings. Must be set before use."""

    embedding_func: Callable | None = field(default=None, init=False)
    """Function derived from embedding_service. Init=False as it's set in post_init."""



    # LLM Configuration
    # ---

    llm_service: BaseLLMService | None = field(default=None)
    """Service for interacting with the large language model (LLM). Must be set before use."""

    llm_model_func: Callable | None = field(default=None, init=False)
    """Function derived from easy_knowledge_retriever.llm_service. Init=False as it's set in post_init."""



    # Rerank Configuration
    # ---

    reranker_service: Optional[BaseRerankerService] = field(default=None)
    """Service for reranking retrieved documents. Optional."""


    """Minimum rerank score threshold for filtering chunks after reranking."""

    # Storage
    # ---


    enable_llm_cache: bool = field(default=True)
    """Enables caching for LLM responses to avoid redundant computations."""

    enable_llm_cache_for_entity_extract: bool = field(default=True)
    """If True, enables caching for entity extraction steps to reduce LLM costs."""

    # Extensions
    # ---

    max_parallel_insert: int = field(
        default=DEFAULT_MAX_PARALLEL_INSERT
    )
    """Maximum number of parallel insert operations."""

    max_graph_nodes: int = field(
        default=DEFAULT_MAX_GRAPH_NODES
    )
    """Maximum number of graph nodes to return in knowledge graph queries."""

    max_source_ids_per_entity: int = field(
        default=DEFAULT_MAX_SOURCE_IDS_PER_ENTITY
    )
    """Maximum number of source (chunk) ids in entity Grpah + VDB."""

    max_source_ids_per_relation: int = field(
        default=DEFAULT_MAX_SOURCE_IDS_PER_RELATION
    )
    """Maximum number of source (chunk) ids in relation Graph + VDB."""

    source_ids_limit_method: str = field(
        default_factory=lambda: normalize_source_ids_limit_method(
            DEFAULT_SOURCE_IDS_LIMIT_METHOD
        )
    )
    """Strategy for enforcing source_id limits: IGNORE_NEW or FIFO."""

    max_file_paths: int = field(
        default=DEFAULT_MAX_FILE_PATHS
    )
    """Maximum number of file paths to store in entity/relation file_path field."""

    file_path_more_placeholder: str = field(default=DEFAULT_FILE_PATH_MORE_PLACEHOLDER)
    """Placeholder text when file paths exceed max_file_paths limit."""

    language: str = "English"
    entity_types: List[str] = field(default_factory=lambda: DEFAULT_ENTITY_TYPES)

    cosine_better_than_threshold: float = field(
        default=0.2
    )



    _storages_status: StoragesStatus = field(default=StoragesStatus.NOT_CREATED)



    def __post_init__(self):
        # -- New Config Pattern Logic --
        # We try to initialize configs from kwargs if not explicit passed (backward compatibility)
        # For this refactor, we will rely on values being present in the dataclass fields either from defaults or init.
        
        # Initialize storage factory
        # We need a way to pass this in or create it. 
        # Since we are keeping this class as a dataclass for now (hard to change completely without breaking everything)
        # We will create internal config objects based on self's fields if they weren't injected.
        
        # Sync Global Config from self
        # self is effectively a superset of GlobalConfig + other things
        # We create a dictionary of current values to use for factory
        
        # Ensure working directory exists (Moved from below)
        if not os.path.exists(self.working_dir):
            logger.info(f"Creating working directory {self.working_dir}")
            os.makedirs(self.working_dir)



        from easy_knowledge_retriever.kg.shared_memory import (
            initialize_share_data,
        )

        # Merge llm_config into llm_model_kwargs


        # Handle deprecated parameters
        if self.log_level is not None:
            warnings.warn(
                "WARNING: log_level parameter is deprecated, use setup_logger in utils.py instead",
                UserWarning,
                stacklevel=2,
            )
        if self.log_file_path is not None:
            warnings.warn(
                "WARNING: log_file_path parameter is deprecated, use setup_logger in utils.py instead",
                UserWarning,
                stacklevel=2,
            )

        # Remove these attributes to prevent their use
        if hasattr(self, "log_level"):
            delattr(self, "log_level")
        if hasattr(self, "log_file_path"):
            delattr(self, "log_file_path")

        initialize_share_data()

        # Verify storage implementation compatibility and environment variables
        # Using factory verification implicitly or explicitly
        # We keep this check for now as it provides fast feedback
        storage_configs = [
            ("KV_STORAGE", self.kv_storage),
            ("VECTOR_STORAGE", self.vector_storage),
            ("GRAPH_STORAGE", self.graph_storage),
            ("DOC_STATUS_STORAGE", self.doc_status_storage),
        ]

        for storage_type, storage_name in storage_configs:
            # Verify storage implementation compatibility
            if isinstance(storage_name, str):
                verify_storage_implementation(storage_type, storage_name)


        # Init Tokenizer
        # Post-initialization hook to handle backward compatabile tokenizer initialization based on provided parameters
        if self.tokenizer is None:
            if self.tiktoken_model_name:
                self.tokenizer = TiktokenTokenizer(self.tiktoken_model_name)
            else:
                self.tokenizer = TiktokenTokenizer()



        # Validate config
        if self.force_llm_summary_on_merge < 3:
            logger.warning(
                f"force_llm_summary_on_merge should be at least 3, got {self.force_llm_summary_on_merge}"
            )
        
        # Validations involving LLM configs are deferred or skipped as those configs are now in llm_service

        # Log configuration
        config_dict = asdict(self)
        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in config_dict.items()])
        logger.debug(f"EasyKnowledgeRetriever init with param:\n  {_print_config}\n")

        # Init Embedding
        if self.embedding_service:
            # Set embedding_func from service
            self.embedding_func = self.embedding_service
            
            # Apply priority wrapper decorator
            self.embedding_func = priority_limit_async_func_call(
                self.embedding_service.max_async,
                llm_timeout=self.embedding_service.timeout,
                queue_name="Embedding func",
            )(self.embedding_func)
            
            logger.debug(f"Initialized embedding_func from service with max_async={self.embedding_service.max_async}")


        # Initialize all storages using Factory (Service pattern)

        # 1. Initialize Services from strings if necessary (Legacy/Config support)
        if isinstance(self.kv_storage, str):
            self.kv_storage = KVStorageService(
                storage_name=self.kv_storage, 
                workspace=self.workspace,
                working_dir=self.working_dir,
            )
        else:
             # If passed as object, ensure working_dir is set if possible (duck typing check)
             pass
                 
        if isinstance(self.vector_storage, str):
            cosine_threshold = self.cosine_better_than_threshold
            self.vector_storage = VectorStorageService(
                storage_name=self.vector_storage, 
                workspace=self.workspace,
                working_dir=self.working_dir,
                cosine_better_than_threshold=cosine_threshold,
            )
        else:
             pass
        
        if isinstance(self.graph_storage, str):
            self.graph_storage = GraphStorageService(
                storage_name=self.graph_storage, 
                workspace=self.workspace,
                working_dir=self.working_dir,
            )
        else:
             pass

        if isinstance(self.doc_status_storage, str):
            self.doc_status_storage = DocStatusStorageService(
                storage_name=self.doc_status_storage, 
                workspace=self.workspace,
                working_dir=self.working_dir,
            )
        else:
             pass

        # 2. Create actual storage instances using services
        
        # KV Storages
        self.llm_response_cache: BaseKVStorage = self.kv_storage.create(
            namespace=NameSpace.KV_STORE_LLM_RESPONSE_CACHE,
            embedding_func=self.embedding_func,
        )

        self.text_chunks: BaseKVStorage = self.kv_storage.create(
            namespace=NameSpace.KV_STORE_TEXT_CHUNKS,
            embedding_func=self.embedding_func,
        )

        self.full_docs: BaseKVStorage = self.kv_storage.create(
            namespace=NameSpace.KV_STORE_FULL_DOCS,
            embedding_func=self.embedding_func,
        )

        self.full_entities: BaseKVStorage = self.kv_storage.create(
            namespace=NameSpace.KV_STORE_FULL_ENTITIES,
            embedding_func=self.embedding_func,
        )

        self.full_relations: BaseKVStorage = self.kv_storage.create(
            namespace=NameSpace.KV_STORE_FULL_RELATIONS,
            embedding_func=self.embedding_func,
        )

        self.entity_chunks: BaseKVStorage = self.kv_storage.create(
            namespace=NameSpace.KV_STORE_ENTITY_CHUNKS,
            embedding_func=self.embedding_func,
        )

        self.relation_chunks: BaseKVStorage = self.kv_storage.create(
            namespace=NameSpace.KV_STORE_RELATION_CHUNKS,
            embedding_func=self.embedding_func,
        )

        # Graph Storage
        self.chunk_entity_relation_graph: BaseGraphStorage = self.graph_storage.create(
            namespace=NameSpace.GRAPH_STORE_CHUNK_ENTITY_RELATION,
            embedding_func=self.embedding_func,
        )

        # Vector Storages
        # Prepare vector storage kwargs including cosine threshold
        cosine_threshold = self.cosine_better_than_threshold
        embedding_dim = getattr(self.embedding_service, "embedding_dim", 1536) if self.embedding_service else 1536
        
        self.entities_vdb: BaseVectorStorage = self.vector_storage.create(
            namespace=NameSpace.VECTOR_STORE_ENTITIES,
            embedding_func=self.embedding_func,
            meta_fields={"entity_name", "source_id", "content", "file_path"},
            cosine_better_than_threshold=cosine_threshold,
            embedding_dim=embedding_dim,
        )
        self.relationships_vdb: BaseVectorStorage = self.vector_storage.create(
            namespace=NameSpace.VECTOR_STORE_RELATIONSHIPS,
            embedding_func=self.embedding_func,
            meta_fields={"src_id", "tgt_id", "description", "keywords", "weight", "source_id", "file_path"},
            cosine_better_than_threshold=cosine_threshold,
            embedding_dim=embedding_dim,
        )
        self.chunks_vdb: BaseVectorStorage = self.vector_storage.create(
            namespace=NameSpace.VECTOR_STORE_CHUNKS,
            embedding_func=self.embedding_func,
            meta_fields={"content", "full_doc_id", "source_id", "file_path"},
            cosine_better_than_threshold=cosine_threshold,
            embedding_dim=embedding_dim,
        )

        # Doc Status Storage
        self.doc_status: DocStatusStorage = self.doc_status_storage.create(
            namespace=NameSpace.DOC_STATUS,
            embedding_func=self.embedding_func,
        )

        # Directly use llm_response_cache, don't create a new object
        hashing_kv = self.llm_response_cache

        # Get timeout from LLM model kwargs for dynamic timeout calculation
        # Get timeout from LLM model kwargs for dynamic timeout calculation
        if self.llm_service:
            # Set llm_model_func from service
            self.llm_model_func = self.llm_service
            
            self.llm_model_func = priority_limit_async_func_call(
                self.llm_service.max_async,
                llm_timeout=self.llm_service.timeout,
                queue_name="LLM func",
            )(
                partial(
                    self.llm_service,  # type: ignore
                    hashing_kv=hashing_kv,
                )
            )
            # Attach the wrapped function to the service instance so it's accessible downstream
            self.llm_service.llm_model_func = self.llm_model_func
            logger.debug(f"Initialized llm_model_func from service with max_async={self.llm_service.max_async}")

        self._storages_status = StoragesStatus.CREATED

    async def initialize_storages(self):
        """Storage initialization must be called one by one to prevent deadlock"""
        if self._storages_status == StoragesStatus.CREATED:
            # Set the first initialized workspace will set the default workspace
            # Allows namespace operation without specifying workspace for backward compatibility
            default_workspace = get_default_workspace()
            if default_workspace is None:
                set_default_workspace(self.workspace)
            elif default_workspace != self.workspace:
                logger.info(
                    f"Creating EasyKnowledgeRetriever instance with workspace='{self.workspace}' "
                    f"while default workspace is set to '{default_workspace}'"
                )

            # Auto-initialize pipeline_status for this workspace
            from easy_knowledge_retriever.kg.shared_memory import initialize_pipeline_status

            await initialize_pipeline_status(workspace=self.workspace)

            for storage in (
                self.full_docs,
                self.text_chunks,
                self.full_entities,
                self.full_relations,
                self.entity_chunks,
                self.relation_chunks,
                self.entities_vdb,
                self.relationships_vdb,
                self.chunks_vdb,
                self.chunk_entity_relation_graph,
                self.llm_response_cache,
                self.doc_status,
            ):
                if storage:
                    # logger.debug(f"Initializing storage: {storage}")
                    await storage.initialize()

            self._storages_status = StoragesStatus.INITIALIZED
            logger.debug("All storage types initialized")

    async def finalize_storages(self):
        """Asynchronously finalize the storages with improved error handling"""
        if self._storages_status == StoragesStatus.INITIALIZED:
            storages = [
                ("full_docs", self.full_docs),
                ("text_chunks", self.text_chunks),
                ("full_entities", self.full_entities),
                ("full_relations", self.full_relations),
                ("entity_chunks", self.entity_chunks),
                ("relation_chunks", self.relation_chunks),
                ("entities_vdb", self.entities_vdb),
                ("relationships_vdb", self.relationships_vdb),
                ("chunks_vdb", self.chunks_vdb),
                ("chunk_entity_relation_graph", self.chunk_entity_relation_graph),
                ("llm_response_cache", self.llm_response_cache),
                ("doc_status", self.doc_status),
            ]

            # Finalize each storage individually to ensure one failure doesn't prevent others from closing
            successful_finalizations = []
            failed_finalizations = []

            for storage_name, storage in storages:
                if storage:
                    try:
                        await storage.finalize()
                        successful_finalizations.append(storage_name)
                        logger.debug(f"Successfully finalized {storage_name}")
                    except Exception as e:
                        error_msg = f"Failed to finalize {storage_name}: {e}"
                        logger.error(error_msg)
                        failed_finalizations.append(storage_name)

            # Log summary of finalization results
            if successful_finalizations:
                logger.info(
                    f"Successfully finalized {len(successful_finalizations)} storages"
                )

            if failed_finalizations:
                logger.error(
                    f"Failed to finalize {len(failed_finalizations)} storages: {', '.join(failed_finalizations)}"
                )
            else:
                logger.debug("All storages finalized successfully")

            self._storages_status = StoragesStatus.FINALIZED

    async def check_and_migrate_data(self):
        """Check if data migration is needed and perform migration if necessary"""
        async with get_data_init_lock():
            try:
                # Check if migration is needed:
                # 1. chunk_entity_relation_graph has entities and relations (count > 0)
                # 2. full_entities and full_relations are empty

                # Get all entity labels from graph
                all_entity_labels = (
                    await self.chunk_entity_relation_graph.get_all_labels()
                )

                if not all_entity_labels:
                    logger.debug("No entities found in graph, skipping migration check")
                    return

                try:
                    # Initialize chunk tracking storage after migration
                    await self._migrate_chunk_tracking_storage()
                except Exception as e:
                    logger.error(f"Error during chunk_tracking migration: {e}")
                    raise e

                # Check if full_entities and full_relations are empty
                # Get all processed documents to check their entity/relation data
                try:
                    processed_docs = await self.doc_status.get_docs_by_status(
                        DocStatus.PROCESSED
                    )

                    if not processed_docs:
                        logger.debug("No processed documents found, skipping migration")
                        return

                    # Check first few documents to see if they have full_entities/full_relations data
                    migration_needed = True
                    checked_count = 0
                    max_check = min(5, len(processed_docs))  # Check up to 5 documents

                    for doc_id in list(processed_docs.keys())[:max_check]:
                        checked_count += 1
                        entity_data = await self.full_entities.get_by_id(doc_id)
                        relation_data = await self.full_relations.get_by_id(doc_id)

                        if entity_data or relation_data:
                            migration_needed = False
                            break

                    if not migration_needed:
                        logger.debug(
                            "Full entities/relations data already exists, no migration needed"
                        )
                        return

                    logger.info(
                        f"Data migration needed: found {len(all_entity_labels)} entities in graph but no full_entities/full_relations data"
                    )

                    # Perform migration
                    await self._migrate_entity_relation_data(processed_docs)

                except Exception as e:
                    logger.error(f"Error during migration check: {e}")
                    raise e

            except Exception as e:
                logger.error(f"Error in data migration check: {e}")
                raise e

    async def _migrate_entity_relation_data(self, processed_docs: dict):
        """Migrate existing entity and relation data to full_entities and full_relations storage"""
        logger.info(f"Starting data migration for {len(processed_docs)} documents")

        # Create mapping from chunk_id to doc_id
        chunk_to_doc = {}
        for doc_id, doc_status in processed_docs.items():
            chunk_ids = (
                doc_status.chunks_list
                if hasattr(doc_status, "chunks_list") and doc_status.chunks_list
                else []
            )
            for chunk_id in chunk_ids:
                chunk_to_doc[chunk_id] = doc_id

        # Initialize document entity and relation mappings
        doc_entities = {}  # doc_id -> set of entity_names
        doc_relations = {}  # doc_id -> set of relation_pairs (as tuples)

        # Get all nodes and edges from graph
        all_nodes = await self.chunk_entity_relation_graph.get_all_nodes()
        all_edges = await self.chunk_entity_relation_graph.get_all_edges()

        # Process all nodes once
        for node in all_nodes:
            if "source_id" in node:
                entity_id = node.get("entity_id") or node.get("id")
                if not entity_id:
                    continue

                # Get chunk IDs from source_id
                source_ids = node["source_id"].split(GRAPH_FIELD_SEP)

                # Find which documents this entity belongs to
                for chunk_id in source_ids:
                    doc_id = chunk_to_doc.get(chunk_id)
                    if doc_id:
                        if doc_id not in doc_entities:
                            doc_entities[doc_id] = set()
                        doc_entities[doc_id].add(entity_id)

        # Process all edges once
        for edge in all_edges:
            if "source_id" in edge:
                src = edge.get("source")
                tgt = edge.get("target")
                if not src or not tgt:
                    continue

                # Get chunk IDs from source_id
                source_ids = edge["source_id"].split(GRAPH_FIELD_SEP)

                # Find which documents this relation belongs to
                for chunk_id in source_ids:
                    doc_id = chunk_to_doc.get(chunk_id)
                    if doc_id:
                        if doc_id not in doc_relations:
                            doc_relations[doc_id] = set()
                        # Use tuple for set operations, convert to list later
                        doc_relations[doc_id].add(tuple(sorted((src, tgt))))

        # Store the results in full_entities and full_relations
        migration_count = 0

        # Store entities
        if doc_entities:
            entities_data = {}
            for doc_id, entity_set in doc_entities.items():
                entities_data[doc_id] = {
                    "entity_names": list(entity_set),
                    "count": len(entity_set),
                }
            await self.full_entities.upsert(entities_data)

        # Store relations
        if doc_relations:
            relations_data = {}
            for doc_id, relation_set in doc_relations.items():
                # Convert tuples back to lists
                relations_data[doc_id] = {
                    "relation_pairs": [list(pair) for pair in relation_set],
                    "count": len(relation_set),
                }
            await self.full_relations.upsert(relations_data)

        migration_count = len(
            set(list(doc_entities.keys()) + list(doc_relations.keys()))
        )

        # Persist the migrated data
        await self.full_entities.index_done_callback()
        await self.full_relations.index_done_callback()

        logger.info(
            f"Data migration completed: migrated {migration_count} documents with entities/relations"
        )

    async def _migrate_chunk_tracking_storage(self) -> None:
        """Ensure entity/relation chunk tracking KV stores exist and are seeded."""

        if not self.entity_chunks or not self.relation_chunks:
            return

        need_entity_migration = False
        need_relation_migration = False

        try:
            need_entity_migration = await self.entity_chunks.is_empty()
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error(f"Failed to check entity chunks storage: {exc}")
            raise exc

        try:
            need_relation_migration = await self.relation_chunks.is_empty()
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error(f"Failed to check relation chunks storage: {exc}")
            raise exc

        if not need_entity_migration and not need_relation_migration:
            return

        BATCH_SIZE = 500  # Process 500 records per batch

        if need_entity_migration:
            try:
                nodes = await self.chunk_entity_relation_graph.get_all_nodes()
            except Exception as exc:
                logger.error(f"Failed to fetch nodes for chunk migration: {exc}")
                nodes = []

            logger.info(f"Starting chunk_tracking data migration: {len(nodes)} nodes")

            # Process nodes in batches
            total_nodes = len(nodes)
            total_batches = (total_nodes + BATCH_SIZE - 1) // BATCH_SIZE
            total_migrated = 0

            for batch_idx in range(total_batches):
                start_idx = batch_idx * BATCH_SIZE
                end_idx = min((batch_idx + 1) * BATCH_SIZE, total_nodes)
                batch_nodes = nodes[start_idx:end_idx]

                upsert_payload: dict[str, dict[str, object]] = {}
                for node in batch_nodes:
                    entity_id = node.get("entity_id") or node.get("id")
                    if not entity_id:
                        continue

                    raw_source = node.get("source_id") or ""
                    chunk_ids = [
                        chunk_id
                        for chunk_id in raw_source.split(GRAPH_FIELD_SEP)
                        if chunk_id
                    ]
                    if not chunk_ids:
                        continue

                    upsert_payload[entity_id] = {
                        "chunk_ids": chunk_ids,
                        "count": len(chunk_ids),
                    }

                if upsert_payload:
                    await self.entity_chunks.upsert(upsert_payload)
                    total_migrated += len(upsert_payload)
                    logger.info(
                        f"Processed entity batch {batch_idx + 1}/{total_batches}: {len(upsert_payload)} records (total: {total_migrated}/{total_nodes})"
                    )

            if total_migrated > 0:
                # Persist entity_chunks data to disk
                await self.entity_chunks.index_done_callback()
                logger.info(
                    f"Entity chunk_tracking migration completed: {total_migrated} records persisted"
                )

        if need_relation_migration:
            try:
                edges = await self.chunk_entity_relation_graph.get_all_edges()
            except Exception as exc:
                logger.error(f"Failed to fetch edges for chunk migration: {exc}")
                edges = []

            logger.info(f"Starting chunk_tracking data migration: {len(edges)} edges")

            # Process edges in batches
            total_edges = len(edges)
            total_batches = (total_edges + BATCH_SIZE - 1) // BATCH_SIZE
            total_migrated = 0

            for batch_idx in range(total_batches):
                start_idx = batch_idx * BATCH_SIZE
                end_idx = min((batch_idx + 1) * BATCH_SIZE, total_edges)
                batch_edges = edges[start_idx:end_idx]

                upsert_payload: dict[str, dict[str, object]] = {}
                for edge in batch_edges:
                    src = edge.get("source") or edge.get("src_id") or edge.get("src")
                    tgt = edge.get("target") or edge.get("tgt_id") or edge.get("tgt")
                    if not src or not tgt:
                        continue

                    raw_source = edge.get("source_id") or ""
                    chunk_ids = [
                        chunk_id
                        for chunk_id in raw_source.split(GRAPH_FIELD_SEP)
                        if chunk_id
                    ]
                    if not chunk_ids:
                        continue

                    storage_key = make_relation_chunk_key(src, tgt)
                    upsert_payload[storage_key] = {
                        "chunk_ids": chunk_ids,
                        "count": len(chunk_ids),
                    }

                if upsert_payload:
                    await self.relation_chunks.upsert(upsert_payload)
                    total_migrated += len(upsert_payload)
                    logger.info(
                        f"Processed relation batch {batch_idx + 1}/{total_batches}: {len(upsert_payload)} records (total: {total_migrated}/{total_edges})"
                    )

            if total_migrated > 0:
                # Persist relation_chunks data to disk
                await self.relation_chunks.index_done_callback()
                logger.info(
                    f"Relation chunk_tracking migration completed: {total_migrated} records persisted"
                )

    async def get_graph_labels(self):
        text = await self.chunk_entity_relation_graph.get_all_labels()
        return text

    async def get_knowledge_graph(
        self,
        node_label: str,
        max_depth: int = 3,
        max_nodes: int = None,
    ) -> KnowledgeGraph:
        """Get knowledge graph for a given label

        Args:
            node_label (str): Label to get knowledge graph for
            max_depth (int): Maximum depth of graph
            max_nodes (int, optional): Maximum number of nodes to return. Defaults to self.max_graph_nodes.

        Returns:
            KnowledgeGraph: Knowledge graph containing nodes and edges
        """
        # Use self.max_graph_nodes as default if max_nodes is None
        if max_nodes is None:
            max_nodes = self.max_graph_nodes
        else:
            # Limit max_nodes to not exceed self.max_graph_nodes
            max_nodes = min(max_nodes, self.max_graph_nodes)

        return await self.chunk_entity_relation_graph.get_knowledge_graph(
            node_label, max_depth, max_nodes
        )

    def _get_storage_class(self, storage_name: str) -> Callable[..., Any]:
        # Direct imports for default storage implementations
        if storage_name == "JsonKVStorage":
            from easy_knowledge_retriever.kg.kv_storage.json_kv_impl import JsonKVStorage

            return JsonKVStorage
        elif storage_name == "NanoVectorDBStorage":
            from easy_knowledge_retriever.kg.vector_storage.nano_vector_db_impl import NanoVectorDBStorage

            return NanoVectorDBStorage
        elif storage_name == "NetworkXStorage":
            from easy_knowledge_retriever.kg.graph_storage.networkx_impl import NetworkXStorage

            return NetworkXStorage
        elif storage_name == "JsonDocStatusStorage":
            from easy_knowledge_retriever.kg.kv_storage.json_doc_status_impl import JsonDocStatusStorage

            return JsonDocStatusStorage
        elif storage_name == "MilvusVectorDBStorage":
            from easy_knowledge_retriever.kg.vector_storage.milvus_impl import MilvusVectorDBStorage

            return MilvusVectorDBStorage
        else:
            # Fallback to dynamic import for other storage implementations
            import_path = STORAGES[storage_name]
            storage_class = lazy_external_import(import_path, storage_name)
            return storage_class

    def insert(
        self,
        input: str | list[str] | list[dict[str, Any]],
        split_by_character: str | None = None,
        split_by_character_only: bool = False,
        ids: str | list[str] | None = None,
        file_paths: str | list[str] | None = None,
        track_id: str | None = None,
    ) -> str:
        """Sync Insert documents with checkpoint support

        Args:
            input: Single document string, list of document strings, or list of dicts ({"content": "...", "pages": [...]})
            split_by_character: if split_by_character is not None, split the string by character, if chunk longer than
            chunk_token_size, it will be split again by token size.
            split_by_character_only: if split_by_character_only is True, split the string by character only, when
            split_by_character is None, this parameter is ignored.
            ids: single string of the document ID or list of unique document IDs, if not provided, MD5 hash IDs will be generated
            file_paths: single string of the file path or list of file paths, used for citation
            track_id: tracking ID for monitoring processing status, if not provided, will be generated

        Returns:
            str: tracking ID for monitoring processing status
        """
        loop = always_get_an_event_loop()
        return loop.run_until_complete(
            self.ainsert(
                input,
                split_by_character,
                split_by_character_only,
                ids,
                file_paths,
                track_id,
            )
        )
    async def ingest(self, file_path: str, start_page: Optional[int] = None, end_page: Optional[int] = None) -> Dict[str, Any]:
        """
        Ingest a document using Mineru parser.

        Args:
            file_path: Path to the document file (PDF).
            start_page: Optional start page index (0-based).
            end_page: Optional end page index (0-based) or count.
            
        Returns:
            The parsed data structure.
        """
        import os
        from easy_knowledge_retriever.operations.mineru_parser import MineruParser
        from easy_knowledge_retriever.operations.image_processing import ImageSummarizer

        # Use a subdirectory in working_dir for parsed docs
        parsed_docs_dir = os.path.join(self.working_dir, "parsed_docs")
        if not os.path.exists(parsed_docs_dir):
            os.makedirs(parsed_docs_dir)

        parser = MineruParser(output_dir=parsed_docs_dir)
        try:
            # Check if document has already been parsed.
            # Mineru creates a directory structure: working_dir / file_stem / 'auto' / file_stem_content_list.json
            import pathlib
            file_path_obj = pathlib.Path(file_path)
            file_stem = file_path_obj.stem
            expected_output_json = pathlib.Path(parsed_docs_dir) / file_stem / "auto" / f"{file_stem}_content_list.json"
            
            parsed_data = None
            if expected_output_json.exists():
                print(f"Document already parsed: {file_path}. Loading existing data...")
                # We need to reuse the parser's logic to process the content list, 
                # but parser.parse runs the command. 
                # Let's extract the loading logic or just use a helper method.
                # Accessing private method _process_content_list is acceptable here as they are in same package/ownership.
                try:
                    import json
                    with open(expected_output_json, "r", encoding="utf-8") as f:
                        content_list = json.load(f)
                    parsed_data = parser._process_content_list(content_list, expected_output_json.parent)
                    print("Loaded existing parsed data.")
                except Exception as e:
                    print(f"Failed to load existing parsed data, re-parsing: {e}")
            
            if parsed_data is None:
                print(f"Parsing document: {file_path}...")
                parsed_data = parser.parse(file_path, start_page=start_page, end_page=end_page)

            # Calculate a stable ID based on the original text content (before image summarization)
            # This ensures that even if image summaries change (LLM non-determinism),
            # the document ID remains the same, preventing duplicates.
            doc_content_raw = parsed_data.get("content", "")
            cleaned_content = sanitize_text_for_encoding(doc_content_raw)
            doc_id = compute_mdhash_id(cleaned_content, prefix="doc-")

            # --- Duplicate Detection and Cleanup ---
            # Check if this file was previously processed with a different ID (e.g., if content processing logic changed)
            # If so, remove the old record to prevent duplicates (issue where same file is processed twice)
            try:
                existing_doc_result = await self.doc_status.get_doc_by_file_path(file_path)
                if existing_doc_result:
                    existing_doc_id, existing_doc_data = existing_doc_result

                    if existing_doc_id != doc_id:
                        print(f"Duplicate document detected with different ID. Removing old record: {existing_doc_id}")
                        logger.warning(f"Duplicate document detected for {file_path}. New ID: {doc_id}, Old ID: {existing_doc_id}. Removing old record.")

                        # Acquire deletion lock and delete old document
                        # We use adelete_by_doc_id to ensure comprehensive cleanup (chunks, graph, etc.)
                        delete_result = await self.adelete_by_doc_id(existing_doc_id)
                        if delete_result.status != "success":
                             logger.error(f"Failed to delete old duplicate document {existing_doc_id}: {delete_result.message}")
                    else:
                        # Same ID, existing logic in ainsert will handle idempotent check
                        pass
            except Exception as e:
                logger.error(f"Error checking for duplicate document: {e}")
                # Don't block ingestion if check fails
                pass
            
            # --- Multimodal Processing (Image Summarization) ---
            if self.llm_model_func:
                # We need the underlying LLM service method to pass 'messages'
                # self.llm_model_func is wrapped by priority_limit_async_func_call
                # and partial. We assume the underlying func or the partial
                # can accept 'messages' as we implemented in OpenAILLMService.
                
                # Check if we can unwrap or use the wrapped function directly.
                # The priority_wrapper calls `await func(*args, **kwargs)`.
                # So if we pass messages kwarg, it should propagate.
                
                # However, ImageSummarizer expects a specific signature protocol.
                # Let's verify if `self.llm_model_func` is sufficient or if we need access 
                # to the original Service object.
                # The typical pattern in this codebase is to use `llm_model_func` for generic generation.
                # But ImageSummarizer needs high control over message structure.
                
                # If llm_model_func is a partial wrapping OpenAI service, it should work.
                
                # Create a wrapper or adapter if needed.
                # Here we pass self.llm_model_func which is async and accepts kwargs.
                
                print("Processing images with VLM...")
                summarizer = ImageSummarizer(self.llm_model_func, llm_response_cache=self.llm_response_cache)
                
                # Iterate over pages to associate images with page numbers
                if "pages" in parsed_data:
                    for page in parsed_data["pages"]:
                        page_num = page.get("page_number", 0)
                        images = page.get("images", [])
                        
                        for img_path in images:
                            print(f"Summarizing image on page {page_num}: {img_path}")
                            summary = await summarizer.summarize(img_path)
                            
                            if summary and not summary.startswith("[Error"):
                                # Create a text chunk for the image summary
                                # We append it to the page's content so it gets chunked naturally?
                                # OR we add it as a separate modality block?
                                
                                # Option A: Append to page content (Simplest for retrieval)
                                # " [IMAGE SUMMARY: ... ] "
                                image_context = f"\\n\\n[IMAGE on Page {page_num}]\\nSummary: {summary}\\nImage Path: {img_path}\\n"
                                page["content"] += image_context

                                
                                # Also update main content if it matters (usually main content is concatenation of pages)
                                # But ainsert likely uses pages if provided?
                                # Let's check ainsert logic. It uses chunking_func which likely uses full_text or pages.
                                # Actually chunking_by_token_size uses 'content' key of each page if 'pages' is provided.
                                
                                # So appending to page["content"] is correct.
                                pass
                
                # Re-construct full text content from updated pages to keep consistency
                parsed_data["content"] = "\n".join([p.get("content", "") for p in parsed_data.get("pages", [])])


            print(f"Indexing document content...")
            # ingest into RAG
            # We wrap input in list as ainsert expects a list of documents
            await self.ainsert(input=[parsed_data], ids=[doc_id], file_paths=[file_path])
            
            return parsed_data
        except Exception as e:
            print(f"Error ingesting {file_path}: {e}")
            raise


    async def ainsert(
        self,
        input: str | list[str] | list[dict[str, Any]],
        split_by_character: str | None = None,
        split_by_character_only: bool = False,
        ids: str | list[str] | None = None,
        file_paths: str | list[str] | None = None,
        track_id: str | None = None,
    ) -> str:
        """Async Insert documents with checkpoint support

        Args:
            input: Single document string, list of document strings, or list of dicts ({"content": "...", "pages": [...]})
            split_by_character: if split_by_character is not None, split the string by character, if chunk longer than
            chunk_token_size, it will be split again by token size.
            split_by_character_only: if split_by_character_only is True, split the string by character only, when
            split_by_character is None, this parameter is ignored.
            ids: list of unique document IDs, if not provided, MD5 hash IDs will be generated
            file_paths: list of file paths corresponding to each document, used for citation
            track_id: tracking ID for monitoring processing status, if not provided, will be generated

        Returns:
            str: tracking ID for monitoring processing status
        """
        # Generate track_id if not provided
        if track_id is None:
            track_id = generate_track_id("insert")

        await self.apipeline_enqueue_documents(input, ids, file_paths, track_id)
        await self.apipeline_process_enqueue_documents(
            split_by_character, split_by_character_only
        )

        return track_id

    async def apipeline_enqueue_documents(
        self,
        input: str | list[str] | list[dict[str, Any]],
        ids: list[str] | None = None,
        file_paths: str | list[str] | None = None,
        track_id: str | None = None,
    ) -> str:
        """
        Pipeline for Processing Documents

        1. Validate ids if provided or generate MD5 hash IDs and remove duplicate contents
        2. Generate document initial status
        3. Filter out already processed documents
        4. Enqueue document in status

        Args:
            input: Single document string, list of document strings, or list of dicts ({"content": "...", "pages": [...]})
            ids: list of unique document IDs, if not provided, MD5 hash IDs will be generated
            file_paths: list of file paths corresponding to each document, used for citation
            track_id: tracking ID for monitoring processing status, if not provided, will be generated with "enqueue" prefix

        Returns:
            str: tracking ID for monitoring processing status
        """
        # Generate track_id if not provided
        if track_id is None or track_id.strip() == "":
            track_id = generate_track_id("enqueue")
        if isinstance(input, str):
            input = [input]
        if isinstance(ids, str):
            ids = [ids]
        if isinstance(file_paths, str):
            file_paths = [file_paths]
            
        # Normalize input to list of dicts/strings
        normalized_input = []
        for item in input:
            if isinstance(item, str):
                normalized_input.append({"content": item})
            elif isinstance(item, dict):
                 normalized_input.append(item)
            else:
                 raise ValueError("Input must be string or dict")

        # If file_paths is provided, ensure it matches the number of documents
        if file_paths is not None:
            if isinstance(file_paths, str):
                file_paths = [file_paths]
            if len(file_paths) != len(input):
                raise ValueError(
                    "Number of file paths must match the number of documents"
                )
        else:
            # If no file paths provided, use placeholder
            file_paths = ["unknown_source"] * len(input)

        # 1. Validate ids if provided or generate MD5 hash IDs and remove duplicate contents
        if ids is not None:
            # Check if the number of IDs matches the number of documents
            if len(ids) != len(input):
                raise ValueError("Number of IDs must match the number of documents")

            # Check if IDs are unique
            if len(ids) != len(set(ids)):
                raise ValueError("IDs must be unique")

            # Generate contents dict and remove duplicates in one pass
            unique_contents = {}
            for id_, doc_data, path in zip(ids, normalized_input, file_paths):
                # doc_data is now a dict
                doc_content = doc_data.get("content", "")
                doc_pages = doc_data.get("pages", None)
                cleaned_content = sanitize_text_for_encoding(doc_content)
                
                if cleaned_content not in unique_contents:
                    unique_contents[cleaned_content] = (id_, path, doc_pages)

            # Reconstruct contents with unique content
            contents = {
                id_: {"content": content, "file_path": file_path, "pages": pages}
                for content, (id_, file_path, pages) in unique_contents.items()
            }
        else:
            # Clean input text and remove duplicates in one pass
            unique_content_with_paths = {}
            for doc_data, path in zip(normalized_input, file_paths):
                doc_content = doc_data.get("content", "")
                doc_pages = doc_data.get("pages", None)
                cleaned_content = sanitize_text_for_encoding(doc_content)
                
                if cleaned_content not in unique_content_with_paths:
                    unique_content_with_paths[cleaned_content] = (path, doc_pages)

            # Generate contents dict of MD5 hash IDs and documents with paths
            contents = {
                compute_mdhash_id(content, prefix="doc-"): {
                    "content": content,
                    "file_path": path,
                    "pages": pages,
                }
                for content, (path, pages) in unique_content_with_paths.items()
            }

        # 2. Generate document initial status (without content)
        new_docs: dict[str, Any] = {
            id_: {
                "status": DocStatus.PENDING,
                "content_summary": get_content_summary(content_data["content"]),
                "content_length": len(content_data["content"]),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "file_path": content_data[
                    "file_path"
                ],  # Store file path in document status
                "track_id": track_id,  # Store track_id in document status
            }
            for id_, content_data in contents.items()
        }

        # 3. Filter out already processed documents
        # Get docs ids
        all_new_doc_ids = set(new_docs.keys())
        # Exclude IDs of documents that are already enqueued
        unique_new_doc_ids = await self.doc_status.filter_keys(all_new_doc_ids)

        # Log ignored document IDs (documents that were filtered out because they already exist)
        ignored_ids = list(all_new_doc_ids - unique_new_doc_ids)
        if ignored_ids:
            for doc_id in ignored_ids:
                file_path = new_docs.get(doc_id, {}).get("file_path", "unknown_source")
                logger.warning(
                    f"Ignoring document ID (already exists): {doc_id} ({file_path})"
                )
            if len(ignored_ids) > 3:
                logger.warning(
                    f"Total Ignoring {len(ignored_ids)} document IDs that already exist in storage"
                )

        # Filter new_docs to only include documents with unique IDs
        new_docs = {
            doc_id: new_docs[doc_id]
            for doc_id in unique_new_doc_ids
            if doc_id in new_docs
        }

        if not new_docs:
            logger.warning("No new unique documents were found.")
            return

        # 4. Store document content in full_docs and status in doc_status
        #    Store full document content separately
        full_docs_data = {
            doc_id: {
                "content": contents[doc_id]["content"],
                "file_path": contents[doc_id]["file_path"],
                "pages": contents[doc_id].get("pages"),
            }
            for doc_id in new_docs.keys()
        }
        await self.full_docs.upsert(full_docs_data)
        # Persist data to disk immediately
        await self.full_docs.index_done_callback()

        # Store document status (without content)
        await self.doc_status.upsert(new_docs)
        logger.debug(f"Stored {len(new_docs)} new unique documents")

        return track_id

    async def apipeline_enqueue_error_documents(
        self,
        error_files: list[dict[str, Any]],
        track_id: str | None = None,
    ) -> None:
        """
        Record file extraction errors in doc_status storage.

        This function creates error document entries in the doc_status storage for files
        that failed during the extraction process. Each error entry contains information
        about the failure to help with debugging and monitoring.

        Args:
            error_files: List of dictionaries containing error information for each failed file.
                Each dictionary should contain:
                - file_path: Original file name/path
                - error_description: Brief error description (for content_summary)
                - original_error: Full error message (for error_msg)
                - file_size: File size in bytes (for content_length, 0 if unknown)
            track_id: Optional tracking ID for grouping related operations

        Returns:
            None
        """
        if not error_files:
            logger.debug("No error files to record")
            return

        # Generate track_id if not provided
        if track_id is None or track_id.strip() == "":
            track_id = generate_track_id("error")

        error_docs: dict[str, Any] = {}
        current_time = datetime.now(timezone.utc).isoformat()

        for error_file in error_files:
            file_path = error_file.get("file_path", "unknown_file")
            error_description = error_file.get(
                "error_description", "File extraction failed"
            )
            original_error = error_file.get("original_error", "Unknown error")
            file_size = error_file.get("file_size", 0)

            # Generate unique doc_id with "error-" prefix
            doc_id_content = f"{file_path}-{error_description}"
            doc_id = compute_mdhash_id(doc_id_content, prefix="error-")

            error_docs[doc_id] = {
                "status": DocStatus.FAILED,
                "content_summary": error_description,
                "content_length": file_size,
                "error_msg": original_error,
                "chunks_count": 0,  # No chunks for failed files
                "created_at": current_time,
                "updated_at": current_time,
                "file_path": file_path,
                "track_id": track_id,
                "metadata": {
                    "error_type": "file_extraction_error",
                },
            }

        # Store error documents in doc_status
        if error_docs:
            await self.doc_status.upsert(error_docs)
            # Log each error for debugging
            for doc_id, error_doc in error_docs.items():
                logger.error(
                    f"File processing error: - ID: {doc_id} {error_doc['file_path']}"
                )

    async def _validate_and_fix_document_consistency(
        self,
        to_process_docs: dict[str, DocProcessingStatus],
        pipeline_status: dict,
        pipeline_status_lock: asyncio.Lock,
    ) -> dict[str, DocProcessingStatus]:
        """Validate and fix document data consistency by deleting inconsistent entries, but preserve failed documents"""
        inconsistent_docs = []
        failed_docs_to_preserve = []
        successful_deletions = 0

        # Check each document's data consistency
        for doc_id, status_doc in to_process_docs.items():
            # Check if corresponding content exists in full_docs
            content_data = await self.full_docs.get_by_id(doc_id)
            if not content_data:
                # Check if this is a failed document that should be preserved
                if (
                    hasattr(status_doc, "status")
                    and status_doc.status == DocStatus.FAILED
                ):
                    failed_docs_to_preserve.append(doc_id)
                else:
                    inconsistent_docs.append(doc_id)

        # Log information about failed documents that will be preserved
        if failed_docs_to_preserve:
            async with pipeline_status_lock:
                preserve_message = f"Preserving {len(failed_docs_to_preserve)} failed document entries for manual review"
                logger.info(preserve_message)
                pipeline_status["latest_message"] = preserve_message
                pipeline_status["history_messages"].append(preserve_message)

            # Remove failed documents from processing list but keep them in doc_status
            for doc_id in failed_docs_to_preserve:
                to_process_docs.pop(doc_id, None)

        # Delete inconsistent document entries(excluding failed documents)
        if inconsistent_docs:
            async with pipeline_status_lock:
                summary_message = (
                    f"Inconsistent document entries found: {len(inconsistent_docs)}"
                )
                logger.info(summary_message)
                pipeline_status["latest_message"] = summary_message
                pipeline_status["history_messages"].append(summary_message)

            successful_deletions = 0
            for doc_id in inconsistent_docs:
                try:
                    status_doc = to_process_docs[doc_id]
                    file_path = getattr(status_doc, "file_path", "unknown_source")

                    # Delete doc_status entry
                    await self.doc_status.delete([doc_id])
                    successful_deletions += 1

                    # Log successful deletion
                    async with pipeline_status_lock:
                        log_message = (
                            f"Deleted inconsistent entry: {doc_id} ({file_path})"
                        )
                        logger.info(log_message)
                        pipeline_status["latest_message"] = log_message
                        pipeline_status["history_messages"].append(log_message)

                    # Remove from processing list
                    to_process_docs.pop(doc_id, None)

                except Exception as e:
                    # Log deletion failure
                    async with pipeline_status_lock:
                        error_message = f"Failed to delete entry: {doc_id} - {str(e)}"
                        logger.error(error_message)
                        pipeline_status["latest_message"] = error_message
                        pipeline_status["history_messages"].append(error_message)

        # Final summary log
        # async with pipeline_status_lock:
        #     final_message = f"Successfully deleted {successful_deletions} inconsistent entries, preserved {len(failed_docs_to_preserve)} failed documents"
        #     logger.info(final_message)
        #     pipeline_status["latest_message"] = final_message
        #     pipeline_status["history_messages"].append(final_message)

        # Reset PROCESSING and FAILED documents that pass consistency checks to PENDING status
        docs_to_reset = {}
        reset_count = 0

        for doc_id, status_doc in to_process_docs.items():
            # Check if document has corresponding content in full_docs (consistency check)
            content_data = await self.full_docs.get_by_id(doc_id)
            if content_data:  # Document passes consistency check
                # Check if document is in PROCESSING or FAILED status
                if hasattr(status_doc, "status") and status_doc.status in [
                    DocStatus.PROCESSING,
                    DocStatus.FAILED,
                ]:
                    # Prepare document for status reset to PENDING
                    docs_to_reset[doc_id] = {
                        "status": DocStatus.PENDING,
                        "content_summary": status_doc.content_summary,
                        "content_length": status_doc.content_length,
                        "created_at": status_doc.created_at,
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                        "file_path": getattr(status_doc, "file_path", "unknown_source"),
                        "track_id": getattr(status_doc, "track_id", ""),
                        # Clear any error messages and processing metadata
                        "error_msg": "",
                        "metadata": {},
                    }

                    # Update the status in to_process_docs as well
                    status_doc.status = DocStatus.PENDING
                    reset_count += 1

        # Update doc_status storage if there are documents to reset
        if docs_to_reset:
            await self.doc_status.upsert(docs_to_reset)

            async with pipeline_status_lock:
                reset_message = f"Reset {reset_count} documents from PROCESSING/FAILED to PENDING status"
                logger.info(reset_message)
                pipeline_status["latest_message"] = reset_message
                pipeline_status["history_messages"].append(reset_message)

        return to_process_docs

    async def apipeline_process_enqueue_documents(
        self,
        split_by_character: str | None = None,
        split_by_character_only: bool = False,
    ) -> None:
        """
        Process pending documents by splitting them into chunks, processing
        each chunk for entity and relation extraction, and updating the
        document status.

        1. Get all pending, failed, and abnormally terminated processing documents.
        2. Validate document data consistency and fix any issues
        3. Split document content into chunks
        4. Process each chunk for entity and relation extraction
        5. Update the document status
        """

        # Get pipeline status shared data and lock
        pipeline_status = await get_namespace_data(
            "pipeline_status", workspace=self.workspace
        )
        pipeline_status_lock = get_namespace_lock(
            "pipeline_status", workspace=self.workspace
        )

        # Check if another process is already processing the queue
        async with pipeline_status_lock:
            # Ensure only one worker is processing documents
            if not pipeline_status.get("busy", False):
                processing_docs, failed_docs, pending_docs = await asyncio.gather(
                    self.doc_status.get_docs_by_status(DocStatus.PROCESSING),
                    self.doc_status.get_docs_by_status(DocStatus.FAILED),
                    self.doc_status.get_docs_by_status(DocStatus.PENDING),
                )

                to_process_docs: dict[str, DocProcessingStatus] = {}
                to_process_docs.update(processing_docs)
                to_process_docs.update(failed_docs)
                to_process_docs.update(pending_docs)

                if not to_process_docs:
                    logger.info("No documents to process")
                    return

                pipeline_status.update(
                    {
                        "busy": True,
                        "job_name": "Default Job",
                        "job_start": datetime.now(timezone.utc).isoformat(),
                        "docs": 0,
                        "batchs": 0,  # Total number of files to be processed
                        "cur_batch": 0,  # Number of files already processed
                        "request_pending": False,  # Clear any previous request
                        "cancellation_requested": False,  # Initialize cancellation flag
                        "latest_message": "",
                    }
                )
                # Cleaning history_messages without breaking it as a shared list object
                del pipeline_status["history_messages"][:]
            else:
                # Another process is busy, just set request flag and return
                pipeline_status["request_pending"] = True
                logger.info(
                    "Another process is already processing the document queue. Request queued."
                )
                return

        try:
            # Process documents until no more documents or requests
            while True:
                # Check for cancellation request at the start of main loop
                async with pipeline_status_lock:
                    if pipeline_status.get("cancellation_requested", False):
                        # Clear pending request
                        pipeline_status["request_pending"] = False
                        # Celar cancellation flag
                        pipeline_status["cancellation_requested"] = False

                        log_message = "Pipeline cancelled by user"
                        logger.info(log_message)
                        pipeline_status["latest_message"] = log_message
                        pipeline_status["history_messages"].append(log_message)

                        # Exit directly, skipping request_pending check
                        return

                if not to_process_docs:
                    log_message = "All enqueued documents have been processed"
                    logger.info(log_message)
                    pipeline_status["latest_message"] = log_message
                    pipeline_status["history_messages"].append(log_message)
                    break

                # Validate document data consistency and fix any issues as part of the pipeline
                to_process_docs = await self._validate_and_fix_document_consistency(
                    to_process_docs, pipeline_status, pipeline_status_lock
                )

                if not to_process_docs:
                    log_message = (
                        "No valid documents to process after consistency check"
                    )
                    logger.info(log_message)
                    pipeline_status["latest_message"] = log_message
                    pipeline_status["history_messages"].append(log_message)
                    break

                log_message = f"Processing {len(to_process_docs)} document(s)"
                logger.info(log_message)

                # Update pipeline_status, batchs now represents the total number of files to be processed
                pipeline_status["docs"] = len(to_process_docs)
                pipeline_status["batchs"] = len(to_process_docs)
                pipeline_status["cur_batch"] = 0
                pipeline_status["latest_message"] = log_message
                pipeline_status["history_messages"].append(log_message)

                # Get first document's file path and total count for job name
                first_doc_id, first_doc = next(iter(to_process_docs.items()))
                first_doc_path = first_doc.file_path

                # Handle cases where first_doc_path is None
                if first_doc_path:
                    path_prefix = first_doc_path[:20] + (
                        "..." if len(first_doc_path) > 20 else ""
                    )
                else:
                    path_prefix = "unknown_source"

                total_files = len(to_process_docs)
                job_name = f"{path_prefix}[{total_files} files]"
                pipeline_status["job_name"] = job_name

                # Create a counter to track the number of processed files
                processed_count = 0
                # Create a semaphore to limit the number of concurrent file processing
                semaphore = asyncio.Semaphore(self.max_parallel_insert)

                async def process_document(
                    doc_id: str,
                    status_doc: DocProcessingStatus,
                    split_by_character: str | None,
                    split_by_character_only: bool,
                    pipeline_status: dict,
                    pipeline_status_lock: asyncio.Lock,
                    semaphore: asyncio.Semaphore,
                ) -> None:
                    """Process single document"""
                    # Initialize variables at the start to prevent UnboundLocalError in error handling
                    file_path = "unknown_source"
                    current_file_number = 0
                    file_extraction_stage_ok = False
                    processing_start_time = int(time.time())
                    first_stage_tasks = []
                    entity_relation_task = None

                    async with semaphore:
                        nonlocal processed_count
                        # Initialize to prevent UnboundLocalError in error handling
                        first_stage_tasks = []
                        entity_relation_task = None
                        try:
                            # Check for cancellation before starting document processing
                            async with pipeline_status_lock:
                                if pipeline_status.get("cancellation_requested", False):
                                    raise PipelineCancelledException("User cancelled")

                            # Get file path from status document
                            file_path = getattr(
                                status_doc, "file_path", "unknown_source"
                            )

                            async with pipeline_status_lock:
                                # Update processed file count and save current file number
                                processed_count += 1
                                current_file_number = (
                                    processed_count  # Save the current file number
                                )
                                pipeline_status["cur_batch"] = processed_count

                                log_message = f"Extracting stage {current_file_number}/{total_files}: {file_path}"
                                logger.info(log_message)
                                pipeline_status["history_messages"].append(log_message)
                                log_message = f"Processing d-id: {doc_id}"
                                logger.info(log_message)
                                pipeline_status["latest_message"] = log_message
                                pipeline_status["history_messages"].append(log_message)

                                # Prevent memory growth: keep only latest 5000 messages when exceeding 10000
                                if len(pipeline_status["history_messages"]) > 10000:
                                    logger.info(
                                        f"Trimming pipeline history from {len(pipeline_status['history_messages'])} to 5000 messages"
                                    )
                                    pipeline_status["history_messages"] = (
                                        pipeline_status["history_messages"][-5000:]
                                    )

                            # Get document content from full_docs
                            content_data = await self.full_docs.get_by_id(doc_id)
                            if not content_data:
                                raise Exception(
                                    f"Document content not found in full_docs for doc_id: {doc_id}"
                                )
                            content = content_data["content"]

                            # Call chunking function, supporting both sync and async implementations
                            pages = content_data.get("pages")
                            chunking_result = self.chunking_func(
                                self.tokenizer,
                                content,
                                split_by_character,
                                split_by_character_only,
                                self.chunk_overlap_token_size,
                                self.chunk_token_size,
                                pages=pages,
                            )

                            # If result is awaitable, await to get actual result
                            if inspect.isawaitable(chunking_result):
                                chunking_result = await chunking_result

                            # Validate return type
                            if not isinstance(chunking_result, (list, tuple)):
                                raise TypeError(
                                    f"chunking_func must return a list or tuple of dicts, "
                                    f"got {type(chunking_result)}"
                                )

                            # Build chunks dictionary
                            chunks: dict[str, Any] = {
                                compute_mdhash_id(dp["content"], prefix="chunk-"): {
                                    **dp,
                                    "full_doc_id": doc_id,
                                    "file_path": file_path,  # Add file path to each chunk
                                    "llm_cache_list": [],  # Initialize empty LLM cache list for each chunk
                                }
                                for dp in chunking_result
                            }

                            if not chunks:
                                logger.warning("No document chunks to process")

                            # Record processing start time
                            processing_start_time = int(time.time())

                            # Check for cancellation before entity extraction
                            async with pipeline_status_lock:
                                if pipeline_status.get("cancellation_requested", False):
                                    raise PipelineCancelledException("User cancelled")

                            # Process document in two stages
                            # Stage 1: Process text chunks and docs (parallel execution)
                            doc_status_task = asyncio.create_task(
                                self.doc_status.upsert(
                                    {
                                        doc_id: {
                                            "status": DocStatus.PROCESSING,
                                            "chunks_count": len(chunks),
                                            "chunks_list": list(
                                                chunks.keys()
                                            ),  # Save chunks list
                                            "content_summary": status_doc.content_summary,
                                            "content_length": status_doc.content_length,
                                            "created_at": status_doc.created_at,
                                            "updated_at": datetime.now(
                                                timezone.utc
                                            ).isoformat(),
                                            "file_path": file_path,
                                            "track_id": status_doc.track_id,  # Preserve existing track_id
                                            "metadata": {
                                                "processing_start_time": processing_start_time
                                            },
                                        }
                                    }
                                )
                            )
                            chunks_vdb_task = asyncio.create_task(
                                self.chunks_vdb.upsert(chunks)
                            )
                            text_chunks_task = asyncio.create_task(
                                self.text_chunks.upsert(chunks)
                            )

                            # First stage tasks (parallel execution)
                            first_stage_tasks = [
                                doc_status_task,
                                chunks_vdb_task,
                                text_chunks_task,
                            ]
                            entity_relation_task = None

                            # Execute first stage tasks
                            await asyncio.gather(*first_stage_tasks)

                            # Stage 2: Process entity relation graph (after text_chunks are saved)
                            entity_relation_task = asyncio.create_task(
                                self._process_extract_entities(
                                    chunks, pipeline_status, pipeline_status_lock
                                )
                            )
                            chunk_results = await entity_relation_task
                            file_extraction_stage_ok = True

                        except Exception as e:
                            # Check if this is a user cancellation
                            if isinstance(e, PipelineCancelledException):
                                # User cancellation - log brief message only, no traceback
                                error_msg = f"User cancelled {current_file_number}/{total_files}: {file_path}"
                                logger.warning(error_msg)
                                async with pipeline_status_lock:
                                    pipeline_status["latest_message"] = error_msg
                                    pipeline_status["history_messages"].append(
                                        error_msg
                                    )
                            else:
                                # Other exceptions - log with traceback
                                logger.error(traceback.format_exc())
                                error_msg = f"Failed to extract document {current_file_number}/{total_files}: {file_path}"
                                logger.error(error_msg)
                                async with pipeline_status_lock:
                                    pipeline_status["latest_message"] = error_msg
                                    pipeline_status["history_messages"].append(
                                        traceback.format_exc()
                                    )
                                    pipeline_status["history_messages"].append(
                                        error_msg
                                    )

                            # Cancel tasks that are not yet completed
                            all_tasks = first_stage_tasks + (
                                [entity_relation_task] if entity_relation_task else []
                            )
                            for task in all_tasks:
                                if task and not task.done():
                                    task.cancel()

                            # Persistent llm cache with error handling
                            if self.llm_response_cache:
                                try:
                                    await self.llm_response_cache.index_done_callback()
                                except Exception as persist_error:
                                    logger.error(
                                        f"Failed to persist LLM cache: {persist_error}"
                                    )

                            # Record processing end time for failed case
                            processing_end_time = int(time.time())

                            # Update document status to failed
                            await self.doc_status.upsert(
                                {
                                    doc_id: {
                                        "status": DocStatus.FAILED,
                                        "error_msg": str(e),
                                        "content_summary": status_doc.content_summary,
                                        "content_length": status_doc.content_length,
                                        "created_at": status_doc.created_at,
                                        "updated_at": datetime.now(
                                            timezone.utc
                                        ).isoformat(),
                                        "file_path": file_path,
                                        "track_id": status_doc.track_id,  # Preserve existing track_id
                                        "metadata": {
                                            "processing_start_time": processing_start_time,
                                            "processing_end_time": processing_end_time,
                                        },
                                    }
                                }
                            )

                        # Concurrency is controlled by keyed lock for individual entities and relationships
                        if file_extraction_stage_ok:
                            try:
                                # Check for cancellation before merge
                                async with pipeline_status_lock:
                                    if pipeline_status.get(
                                        "cancellation_requested", False
                                    ):
                                        raise PipelineCancelledException(
                                            "User cancelled"
                                        )

                                # Use chunk_results from entity_relation_task
                                await merge_nodes_and_edges(
                                    chunk_results=chunk_results,  # result collected from entity_relation_task
                                    knowledge_graph_inst=self.chunk_entity_relation_graph,
                                    entity_vdb=self.entities_vdb,
                                    relationships_vdb=self.relationships_vdb,
                                    llm_service=self.llm_service,
                                    source_ids_limit_method=self.source_ids_limit_method,
                                    max_source_ids_per_entity=self.max_source_ids_per_entity,
                                    max_source_ids_per_relation=self.max_source_ids_per_relation,
                                    max_file_paths=self.max_file_paths,
                                    file_path_more_placeholder=self.file_path_more_placeholder,
                                    workspace=self.workspace,
                                    full_entities_storage=self.full_entities,
                                    full_relations_storage=self.full_relations,
                                    doc_id=doc_id,
                                    pipeline_status=pipeline_status,
                                    pipeline_status_lock=pipeline_status_lock,
                                    llm_response_cache=self.llm_response_cache,
                                    entity_chunks_storage=self.entity_chunks,
                                    relation_chunks_storage=self.relation_chunks,
                                    current_file_number=current_file_number,
                                    total_files=total_files,
                                    file_path=file_path,
                                )

                                # Record processing end time
                                processing_end_time = int(time.time())

                                await self.doc_status.upsert(
                                    {
                                        doc_id: {
                                            "status": DocStatus.PROCESSED,
                                            "chunks_count": len(chunks),
                                            "chunks_list": list(chunks.keys()),
                                            "content_summary": status_doc.content_summary,
                                            "content_length": status_doc.content_length,
                                            "created_at": status_doc.created_at,
                                            "updated_at": datetime.now(
                                                timezone.utc
                                            ).isoformat(),
                                            "file_path": file_path,
                                            "track_id": status_doc.track_id,  # Preserve existing track_id
                                            "metadata": {
                                                "processing_start_time": processing_start_time,
                                                "processing_end_time": processing_end_time,
                                            },
                                        }
                                    }
                                )

                                # Call _insert_done after processing each file
                                await self._insert_done()

                                async with pipeline_status_lock:
                                    log_message = f"Completed processing file {current_file_number}/{total_files}: {file_path}"
                                    logger.info(log_message)
                                    pipeline_status["latest_message"] = log_message
                                    pipeline_status["history_messages"].append(
                                        log_message
                                    )

                            except Exception as e:
                                # Check if this is a user cancellation
                                if isinstance(e, PipelineCancelledException):
                                    # User cancellation - log brief message only, no traceback
                                    error_msg = f"User cancelled during merge {current_file_number}/{total_files}: {file_path}"
                                    logger.warning(error_msg)
                                    async with pipeline_status_lock:
                                        pipeline_status["latest_message"] = error_msg
                                        pipeline_status["history_messages"].append(
                                            error_msg
                                        )
                                else:
                                    # Other exceptions - log with traceback
                                    logger.error(traceback.format_exc())
                                    error_msg = f"Merging stage failed in document {current_file_number}/{total_files}: {file_path}"
                                    logger.error(error_msg)
                                    async with pipeline_status_lock:
                                        pipeline_status["latest_message"] = error_msg
                                        pipeline_status["history_messages"].append(
                                            traceback.format_exc()
                                        )
                                        pipeline_status["history_messages"].append(
                                            error_msg
                                        )

                                # Persistent llm cache with error handling
                                if self.llm_response_cache:
                                    try:
                                        await self.llm_response_cache.index_done_callback()
                                    except Exception as persist_error:
                                        logger.error(
                                            f"Failed to persist LLM cache: {persist_error}"
                                        )

                                # Record processing end time for failed case
                                processing_end_time = int(time.time())

                                # Update document status to failed
                                await self.doc_status.upsert(
                                    {
                                        doc_id: {
                                            "status": DocStatus.FAILED,
                                            "error_msg": str(e),
                                            "content_summary": status_doc.content_summary,
                                            "content_length": status_doc.content_length,
                                            "created_at": status_doc.created_at,
                                            "updated_at": datetime.now().isoformat(),
                                            "file_path": file_path,
                                            "track_id": status_doc.track_id,  # Preserve existing track_id
                                            "metadata": {
                                                "processing_start_time": processing_start_time,
                                                "processing_end_time": processing_end_time,
                                            },
                                        }
                                    }
                                )

                # Create processing tasks for all documents
                doc_tasks = []
                for doc_id, status_doc in to_process_docs.items():
                    doc_tasks.append(
                        process_document(
                            doc_id,
                            status_doc,
                            split_by_character,
                            split_by_character_only,
                            pipeline_status,
                            pipeline_status_lock,
                            semaphore,
                        )
                    )

                # Wait for all document processing to complete
                try:
                    await asyncio.gather(*doc_tasks)
                except PipelineCancelledException:
                    # Cancel all remaining tasks
                    for task in doc_tasks:
                        if not task.done():
                            task.cancel()

                    # Wait for all tasks to complete cancellation
                    await asyncio.wait(doc_tasks, return_when=asyncio.ALL_COMPLETED)

                    # Exit directly (document statuses already updated in process_document)
                    return

                # Check if there's a pending request to process more documents (with lock)
                has_pending_request = False
                async with pipeline_status_lock:
                    has_pending_request = pipeline_status.get("request_pending", False)
                    if has_pending_request:
                        # Clear the request flag before checking for more documents
                        pipeline_status["request_pending"] = False

                if not has_pending_request:
                    break

                log_message = "Processing additional documents due to pending request"
                logger.info(log_message)
                pipeline_status["latest_message"] = log_message
                pipeline_status["history_messages"].append(log_message)

                # Check for pending documents again
                processing_docs, failed_docs, pending_docs = await asyncio.gather(
                    self.doc_status.get_docs_by_status(DocStatus.PROCESSING),
                    self.doc_status.get_docs_by_status(DocStatus.FAILED),
                    self.doc_status.get_docs_by_status(DocStatus.PENDING),
                )

                to_process_docs = {}
                to_process_docs.update(processing_docs)
                to_process_docs.update(failed_docs)
                to_process_docs.update(pending_docs)

        finally:
            log_message = "Enqueued document processing pipeline stopped"
            logger.info(log_message)
            # Always reset busy status and cancellation flag when done or if an exception occurs (with lock)
            async with pipeline_status_lock:
                pipeline_status["busy"] = False
                pipeline_status["cancellation_requested"] = (
                    False  # Always reset cancellation flag
                )
                pipeline_status["latest_message"] = log_message
                pipeline_status["history_messages"].append(log_message)

    async def _process_extract_entities(
        self, chunk: dict[str, Any], pipeline_status=None, pipeline_status_lock=None
    ) -> list:
        try:
            chunk_results = await extract_entities(
                chunk,
                llm_model_func=self.llm_model_func,
                entity_extract_max_gleaning=self.entity_extract_max_gleaning,
                llm_service=self.llm_service,
                language=self.language,
                entity_types=self.entity_types,
                enable_llm_cache_for_entity_extract=self.enable_llm_cache_for_entity_extract,
                pipeline_status=pipeline_status,
                pipeline_status_lock=pipeline_status_lock,
                llm_response_cache=self.llm_response_cache,
                text_chunks_storage=self.text_chunks,
            )
            return chunk_results
        except Exception as e:
            error_msg = f"Failed to extract entities and relationships: {str(e)}"
            logger.error(error_msg)
            async with pipeline_status_lock:
                pipeline_status["latest_message"] = error_msg
                pipeline_status["history_messages"].append(error_msg)
            raise e

    async def _insert_done(
        self, pipeline_status=None, pipeline_status_lock=None
    ) -> None:
        tasks = [
            cast(StorageNameSpace, storage_inst).index_done_callback()
            for storage_inst in [  # type: ignore
                self.full_docs,
                self.doc_status,
                self.text_chunks,
                self.full_entities,
                self.full_relations,
                self.entity_chunks,
                self.relation_chunks,
                self.llm_response_cache,
                self.entities_vdb,
                self.relationships_vdb,
                self.chunks_vdb,
                self.chunk_entity_relation_graph,
            ]
            if storage_inst is not None
        ]
        await asyncio.gather(*tasks)

        log_message = "In memory DB persist to disk"
        logger.info(log_message)

        if pipeline_status is not None and pipeline_status_lock is not None:
            async with pipeline_status_lock:
                pipeline_status["latest_message"] = log_message
                pipeline_status["history_messages"].append(log_message)

    def insert_custom_kg(
        self, custom_kg: dict[str, Any], full_doc_id: str = None
    ) -> None:
        loop = always_get_an_event_loop()
        loop.run_until_complete(self.ainsert_custom_kg(custom_kg, full_doc_id))

    async def ainsert_custom_kg(
        self,
        custom_kg: dict[str, Any],
        full_doc_id: str = None,
    ) -> None:
        update_storage = False
        try:
            # Insert chunks into vector storage
            all_chunks_data: dict[str, dict[str, str]] = {}
            chunk_to_source_map: dict[str, str] = {}
            for chunk_data in custom_kg.get("chunks", []):
                chunk_content = sanitize_text_for_encoding(chunk_data["content"])
                source_id = chunk_data["source_id"]
                file_path = chunk_data.get("file_path", "custom_kg")
                tokens = len(self.tokenizer.encode(chunk_content))
                chunk_order_index = (
                    0
                    if "chunk_order_index" not in chunk_data.keys()
                    else chunk_data["chunk_order_index"]
                )
                chunk_id = compute_mdhash_id(chunk_content, prefix="chunk-")

                chunk_entry = {
                    "content": chunk_content,
                    "source_id": source_id,
                    "tokens": tokens,
                    "chunk_order_index": chunk_order_index,
                    "full_doc_id": full_doc_id
                    if full_doc_id is not None
                    else source_id,
                    "file_path": file_path,
                    "status": DocStatus.PROCESSED,
                }
                all_chunks_data[chunk_id] = chunk_entry
                chunk_to_source_map[source_id] = chunk_id
                update_storage = True

            if all_chunks_data:
                await asyncio.gather(
                    self.chunks_vdb.upsert(all_chunks_data),
                    self.text_chunks.upsert(all_chunks_data),
                )

            # Insert entities into knowledge graph
            all_entities_data: list[dict[str, str]] = []
            for entity_data in custom_kg.get("entities", []):
                entity_name = entity_data["entity_name"]
                entity_type = entity_data.get("entity_type", "UNKNOWN")
                description = entity_data.get("description", "No description provided")
                source_chunk_id = entity_data.get("source_id", "UNKNOWN")
                source_id = chunk_to_source_map.get(source_chunk_id, "UNKNOWN")
                file_path = entity_data.get("file_path", "custom_kg")

                # Log if source_id is UNKNOWN
                if source_id == "UNKNOWN":
                    logger.warning(
                        f"Entity '{entity_name}' has an UNKNOWN source_id. Please check the source mapping."
                    )

                # Prepare node data
                node_data: dict[str, str] = {
                    "entity_id": entity_name,
                    "entity_type": entity_type,
                    "description": description,
                    "source_id": source_id,
                    "file_path": file_path,
                    "created_at": int(time.time()),
                }
                # Insert node data into the knowledge graph
                await self.chunk_entity_relation_graph.upsert_node(
                    entity_name, node_data=node_data
                )
                node_data["entity_name"] = entity_name
                all_entities_data.append(node_data)
                update_storage = True

            # Insert relationships into knowledge graph
            all_relationships_data: list[dict[str, str]] = []
            for relationship_data in custom_kg.get("relationships", []):
                src_id = relationship_data["src_id"]
                tgt_id = relationship_data["tgt_id"]
                description = relationship_data["description"]
                keywords = relationship_data["keywords"]
                weight = relationship_data.get("weight", 1.0)
                source_chunk_id = relationship_data.get("source_id", "UNKNOWN")
                source_id = chunk_to_source_map.get(source_chunk_id, "UNKNOWN")
                file_path = relationship_data.get("file_path", "custom_kg")

                # Log if source_id is UNKNOWN
                if source_id == "UNKNOWN":
                    logger.warning(
                        f"Relationship from '{src_id}' to '{tgt_id}' has an UNKNOWN source_id. Please check the source mapping."
                    )

                # Check if nodes exist in the knowledge graph
                for need_insert_id in [src_id, tgt_id]:
                    if not (
                        await self.chunk_entity_relation_graph.has_node(need_insert_id)
                    ):
                        await self.chunk_entity_relation_graph.upsert_node(
                            need_insert_id,
                            node_data={
                                "entity_id": need_insert_id,
                                "source_id": source_id,
                                "description": "UNKNOWN",
                                "entity_type": "UNKNOWN",
                                "file_path": file_path,
                                "created_at": int(time.time()),
                            },
                        )

                # Insert edge into the knowledge graph
                await self.chunk_entity_relation_graph.upsert_edge(
                    src_id,
                    tgt_id,
                    edge_data={
                        "weight": weight,
                        "description": description,
                        "keywords": keywords,
                        "source_id": source_id,
                        "file_path": file_path,
                        "created_at": int(time.time()),
                    },
                )

                edge_data: dict[str, str] = {
                    "src_id": src_id,
                    "tgt_id": tgt_id,
                    "description": description,
                    "keywords": keywords,
                    "source_id": source_id,
                    "weight": weight,
                    "file_path": file_path,
                    "created_at": int(time.time()),
                }
                all_relationships_data.append(edge_data)
                update_storage = True

            # Insert entities into vector storage with consistent format
            data_for_vdb = {
                compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                    "content": dp["entity_name"] + "\n" + dp["description"],
                    "entity_name": dp["entity_name"],
                    "source_id": dp["source_id"],
                    "description": dp["description"],
                    "entity_type": dp["entity_type"],
                    "file_path": dp.get("file_path", "custom_kg"),
                }
                for dp in all_entities_data
            }
            await self.entities_vdb.upsert(data_for_vdb)

            # Insert relationships into vector storage with consistent format
            data_for_vdb = {
                compute_mdhash_id(dp["src_id"] + dp["tgt_id"], prefix="rel-"): {
                    "src_id": dp["src_id"],
                    "tgt_id": dp["tgt_id"],
                    "source_id": dp["source_id"],
                    "content": f"{dp['keywords']}\t{dp['src_id']}\n{dp['tgt_id']}\n{dp['description']}",
                    "keywords": dp["keywords"],
                    "description": dp["description"],
                    "weight": dp["weight"],
                    "file_path": dp.get("file_path", "custom_kg"),
                }
                for dp in all_relationships_data
            }
            await self.relationships_vdb.upsert(data_for_vdb)

        except Exception as e:
            logger.error(f"Error in ainsert_custom_kg: {e}")
            raise
        finally:
            if update_storage:
                await self._insert_done()

    async def retrieve(self, query: str, retrieval: BaseRetrieval) -> QueryContextResult:
        """
        Execute the retrieval strategy.

        Args:
            query: Query text.
            retrieval: Retrieval strategy object.

        Returns:
            QueryContextResult: Context data.
        """
        if getattr(retrieval, "query_decomposition", False):
            try:
                sub_queries = await decompose_query(query, self.llm_model_func)
                if len(sub_queries) > 1:
                    logger.info(f"Query decomposition active. Sub-queries: {sub_queries}")
                    results = []
                    for sub_q in sub_queries:
                        # Recursively call retrieve with the sub-query
                        # We temporarily disable decomposition to avoid infinite loop if logic was different,
                        # but here we call retrieval.retrieve directly so it's fine unless retrieval.retrieve calls back self.retrieve.
                        # retrieval.retrieve uses self (rag) but usually calls self.query or internal methods.
                        # Standard implementations call _common_kg_retrieve or search directly.
                        results.append(await retrieval.retrieve(sub_q, self))
                    
                    return merge_query_results(results)
            except Exception as e:
                logger.error(f"Error during query decomposition: {e}. Falling back to standard retrieval.")
        
        return await retrieval.retrieve(query, self)

    def query(
        self,
        query: str,
        param: QueryParam = QueryParam(),
        retrieval: BaseRetrieval = None,
        system_prompt: str | None = None,
    ) -> QueryResult:
        """
        Perform a sync query.

        Args:
            query (str): The query to be executed.
            param (QueryParam): Configuration parameters for query execution.
            retrieval (BaseRetrieval): Retrieval strategy object.
            prompt (Optional[str]): Custom prompts for fine-tuned control over the system's behavior. Defaults to None, which uses PROMPTS["rag_response"].

        Returns:
            QueryResult: The result of the query execution.
        """
        loop = always_get_an_event_loop()

        return loop.run_until_complete(self.aquery(query, param, retrieval, system_prompt))  # type: ignore

    async def aquery(
        self,
        query: str,
        param: QueryParam = QueryParam(),
        retrieval: BaseRetrieval = None,
        system_prompt: str | None = None,
    ) -> QueryResult:
        """
        Perform a async query.

        This function is a wrapper around aquery_llm returning the full structured QueryResult.

        Args:
            query (str): The query to be executed.
            param (QueryParam): Configuration parameters for query execution.
                If param.model_func is provided, it will be used instead of the global model.
            retrieval (BaseRetrieval): Retrieval strategy object.
            system_prompt (Optional[str]): Custom prompts for fine-tuned control over the system's behavior. Defaults to None, which uses PROMPTS["rag_response"].

        Returns:
            QueryResult: The structured query result.
        """
        # Call the new aquery_llm function to get complete results
        return await self.aquery_llm(query, param, retrieval, system_prompt)

    def query_data(
        self,
        query: str,
        param: QueryParam = QueryParam(),
    ) -> dict[str, Any]:
        """
        Synchronous data retrieval API: returns structured retrieval results without LLM generation.

        This function is the synchronous version of aquery_data, providing the same functionality
        for users who prefer synchronous interfaces.

        Args:
            query: Query text for retrieval.
            param: Query parameters controlling retrieval behavior (same as aquery).

        Returns:
            dict[str, Any]: Same structured data result as aquery_data.
        """
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.aquery_data(query, param))

    async def aquery_data(
        self,
        query: str,
        param: QueryParam = QueryParam(),
    ) -> dict[str, Any]:
        """
        Asynchronous data retrieval API: returns structured retrieval results without LLM generation.

        This function reuses the same logic as aquery but stops before LLM generation,
        returning the final processed entities, relationships, and chunks data that would be sent to LLM.

        Args:
            query: Query text for retrieval.
            param: Query parameters controlling retrieval behavior (same as aquery).

        Returns:
            dict[str, Any]: Structured data result in the following format:

            **Success Response:**
            ```python
            {
                "status": "success",
                "message": "Query executed successfully",
                "data": {
                    "entities": [
                        {
                            "entity_name": str,      # Entity identifier
                            "entity_type": str,      # Entity category/type
                            "description": str,      # Entity description
                            "source_id": str,        # Source chunk references
                            "file_path": str,        # Origin file path
                            "created_at": str,       # Creation timestamp
                            "reference_id": str      # Reference identifier for citations
                        }
                    ],
                    "relationships": [
                        {
                            "src_id": str,           # Source entity name
                            "tgt_id": str,           # Target entity name
                            "description": str,      # Relationship description
                            "keywords": str,         # Relationship keywords
                            "weight": float,         # Relationship strength
                            "source_id": str,        # Source chunk references
                            "file_path": str,        # Origin file path
                            "created_at": str,       # Creation timestamp
                            "reference_id": str      # Reference identifier for citations
                        }
                    ],
                    "chunks": [
                        {
                            "content": str,          # Document chunk content
                            "file_path": str,        # Origin file path
                            "chunk_id": str,         # Unique chunk identifier
                            "reference_id": str      # Reference identifier for citations
                        }
                    ],
                    "references": [
                        {
                            "reference_id": str,     # Reference identifier
                            "file_path": str         # Corresponding file path
                        }
                    ]
                },
                "metadata": {
                    "query_mode": str,           # Query mode used ("local", "global", "hybrid", "mix", "naive", "bypass")
                    "keywords": {
                        "high_level": List[str], # High-level keywords extracted
                        "low_level": List[str]   # Low-level keywords extracted
                    },
                    "processing_info": {
                        "total_entities_found": int,        # Total entities before truncation
                        "total_relations_found": int,       # Total relations before truncation
                        "entities_after_truncation": int,   # Entities after token truncation
                        "relations_after_truncation": int,  # Relations after token truncation
                        "merged_chunks_count": int,          # Chunks before final processing
                        "final_chunks_count": int            # Final chunks in result
                    }
                }
            }
            ```

            **Query Mode Differences:**
            - **local**: Focuses on entities and their related chunks based on low-level keywords
            - **global**: Focuses on relationships and their connected entities based on high-level keywords
            - **hybrid**: Combines local and global results using round-robin merging
            - **mix**: Includes knowledge graph data plus vector-retrieved document chunks
            - **naive**: Only vector-retrieved chunks, entities and relationships arrays are empty
            - **bypass**: All data arrays are empty, used for direct LLM queries

            ** processing_info is optional and may not be present in all responses, especially when query result is empty**

            **Failure Response:**
            ```python
            {
                "status": "failure",
                "message": str,  # Error description
                "data": {}       # Empty data object
            }
            ```

            **Common Failure Cases:**
            - Empty query string
            - Both high-level and low-level keywords are empty
            - Query returns empty dataset
            - Missing tokenizer or system configuration errors

        Note:
            The function adapts to the new data format from convert_to_user_format where
            actual data is nested under the 'data' field, with 'status' and 'message'
            fields at the top level.
        """

        # Create a copy of param to avoid modifying the original
        data_param = QueryParam(
            mode=param.mode,
            only_need_context=True,  # Skip LLM generation, only get context and data
            only_need_prompt=False,
            response_type=param.response_type,
            stream=False,  # Data retrieval doesn't need streaming
            top_k=param.top_k,
            chunk_top_k=param.chunk_top_k,
            max_entity_tokens=param.max_entity_tokens,
            max_relation_tokens=param.max_relation_tokens,
            max_total_tokens=param.max_total_tokens,
            hl_keywords=param.hl_keywords,
            ll_keywords=param.ll_keywords,
            conversation_history=param.conversation_history,
            history_turns=param.history_turns,
            model_func=param.model_func,
            user_prompt=param.user_prompt,
        )

        query_result = None
        
        # Create retrieval strategy with reranker
        retrieval_strategy = RetrievalFactory.create_retrieval(data_param, self.reranker_service)

        if data_param.mode in ["local", "global", "hybrid", "mix"]:
            logger.debug(f"[aquery_data] Using kg_query for mode: {data_param.mode}")
            query_result = await kg_query(
                query.strip(),
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.relationships_vdb,
                self.text_chunks,
                data_param,  # Use data_param with only_need_context=True
                tokenizer=self.tokenizer,
                llm_model_func=self.llm_model_func,
                enable_llm_cache=self.enable_llm_cache,
                max_total_tokens=data_param.max_total_tokens,
                language=self.language,
                kg_chunk_pick_method=self.kg_chunk_pick_method,
                max_related_chunks=self.related_chunk_number,
                hashing_kv=self.llm_response_cache,
                system_prompt=None,
                chunks_vdb=self.chunks_vdb,
                retrieval=retrieval_strategy,
            )
        elif data_param.mode == "naive":
            logger.debug(f"[aquery_data] Using naive_query for mode: {data_param.mode}")
            query_result = await naive_query(
                query.strip(),
                self.chunks_vdb,
                data_param,  # Use data_param with only_need_context=True
                tokenizer=self.tokenizer,
                llm_model_func=self.llm_model_func,
                max_total_tokens=data_param.max_total_tokens,
                enable_llm_cache=self.enable_llm_cache,
                hashing_kv=self.llm_response_cache,
                system_prompt=None,
                retrieval=retrieval_strategy,
            )
        elif data_param.mode == "bypass":
            logger.debug("[aquery_data] Using bypass mode")
            # bypass mode returns empty data using convert_to_user_format
            empty_raw_data = convert_to_user_format(
                [],  # no entities
                [],  # no relationships
                [],  # no chunks
                [],  # no references
                "bypass",
            )
            query_result = QueryResult(content="", raw_data=empty_raw_data, query=query)
        else:
            raise ValueError(f"Unknown mode {data_param.mode}")

        if query_result is None:
            no_result_message = "Query returned no results"
            if data_param.mode == "naive":
                no_result_message = "No relevant document chunks found."
            final_data: dict[str, Any] = {
                "status": "failure",
                "message": no_result_message,
                "data": {},
                "metadata": {
                    "failure_reason": "no_results",
                    "mode": data_param.mode,
                },
            }
            logger.info("[aquery_data] Query returned no results.")
        else:
            # Extract raw_data from QueryResult
            final_data = query_result.raw_data or {}

            # Log final result counts - adapt to new data format from convert_to_user_format
            if final_data and "data" in final_data:
                data_section = final_data["data"]
                entities_count = len(data_section.get("entities", []))
                relationships_count = len(data_section.get("relationships", []))
                chunks_count = len(data_section.get("chunks", []))
                logger.debug(
                    f"[aquery_data] Final result: {entities_count} entities, {relationships_count} relationships, {chunks_count} chunks"
                )
            else:
                logger.warning("[aquery_data] No data section found in query result")

        await self._query_done()
        return final_data

    def _fill_param_defaults(self, param: QueryParam):
        """
        Fill defaults in QueryParam from EasyKnowledgeRetriever configuration
        if the values in QueryParam are None.
        """
        if param.top_k is None:
            param.top_k = self.top_k
            
        if param.chunk_top_k is None:
            param.chunk_top_k = self.chunk_top_k
            
        if param.max_entity_tokens is None:
            param.max_entity_tokens = self.max_entity_tokens
            
        if param.max_relation_tokens is None:
            param.max_relation_tokens = self.max_relation_tokens
            
        if param.max_total_tokens is None:
            param.max_total_tokens = self.max_total_tokens

    def _parse_raw_data_to_lists(self, raw_data: dict) -> dict:
        """Helper to parse raw_data dict into structured lists."""
        data = raw_data.get("data", {})

        def to_objs(cls, items):
            field_names = {f.name for f in fields(cls)}
            objs = []
            for item in items:
                # Filter keys that are valid for the dataclass
                filtered = {k: v for k, v in item.items() if k in field_names}
                objs.append(cls(**filtered))
            return objs

        return {
            "entities": to_objs(Entity, data.get("entities", [])),
            "relationships": to_objs(Relationship, data.get("relationships", [])),
            "chunks": to_objs(Chunk, data.get("chunks", [])),
            "references": to_objs(Reference, data.get("references", []))
        }

    async def aquery_llm(
        self,
        query: str,
        param: QueryParam = QueryParam(),
        retrieval: BaseRetrieval = None,
        system_prompt: str | None = None,
    ) -> QueryResult:
        """
        Asynchronous complete query API: returns structured retrieval results with LLM generation.

        This function performs a single query operation and returns both structured data and LLM response,
        based on the original aquery logic to avoid duplicate calls.

        Args:
            query: Query text for retrieval and LLM generation.
            param: Query parameters controlling retrieval and LLM behavior.
            retrieval: Retrieval strategy object.
            system_prompt: Optional custom system prompt for LLM generation.

        Returns:
            QueryResult: Complete response with structured data and LLM response.
        """
        logger.debug(f"[aquery_llm] Query param: {param}")

        # Fill default parameters from configuration if they are None
        self._fill_param_defaults(param)


        try:
            if retrieval is None:
                retrieval = RetrievalFactory.create_retrieval(param)

            # Logic specific to Bypass vs others
            from easy_knowledge_retriever.retrieval.bypass import BypassRetrieval
            is_bypass = isinstance(retrieval, BypassRetrieval)

            if is_bypass:
                # Bypass mode: directly use LLM without knowledge retrieval
                use_llm_func = param.model_func or self.llm_model_func
                # Apply higher priority (8) to entity/relation summary tasks
                use_llm_func = partial(use_llm_func, _priority=8)

                param.stream = True if param.stream is None else param.stream
                response = await use_llm_func(
                    query.strip(),
                    system_prompt=system_prompt,
                    history_messages=param.conversation_history,
                    enable_cot=True,
                    stream=param.stream,
                )
                
                is_streaming = not isinstance(response, str)
                
                return QueryResult(
                    status="success",
                    message=f"Bypass mode LLM {'streaming' if is_streaming else 'non streaming'} response",
                    content=response if not is_streaming else None,
                    response_iterator=response if is_streaming else None,
                    is_streaming=is_streaming,
                    metadata={},
                    raw_data={},
                    query=query,
                    system_prompt=system_prompt or "",
                    user_prompt=param.user_prompt or ""
                )

            # Standard RAG mode
            query_context_result = await self.retrieve(query, retrieval)
            
            # Check if query_context_result is effectively empty/None (context string empty)
            if not query_context_result.context and not query_context_result.raw_data:
                 return QueryResult(
                    status="failure",
                    message="Query returned no results",
                    content=PROMPTS["fail_response"],
                    metadata={
                        "failure_reason": "no_results",
                        "mode": retrieval.mode,
                    },
                    query=query,
                    system_prompt="",
                    user_prompt=param.user_prompt or ""
                 )

            parsed_data = self._parse_raw_data_to_lists(query_context_result.raw_data or {})

            # Return different content based on query parameters
            if param.only_need_context and not param.only_need_prompt:
                # Return context as content
                return QueryResult(
                    status="success",
                    message="Context retrieval successful",
                    content=query_context_result.context,
                    context=query_context_result.context,
                    raw_data=query_context_result.raw_data,
                    metadata=query_context_result.raw_data.get("metadata", {}),
                    query=query,
                    system_prompt="",
                    user_prompt=param.user_prompt or "",
                    **parsed_data
                )

            user_prompt = query + (f"\n\n{param.user_prompt}" if param.user_prompt else "")
            response_type = (
                param.response_type
                if param.response_type
                else "Multiple Paragraphs"
            )

            # Build system prompt
            sys_prompt_temp = system_prompt if system_prompt else PROMPTS["rag_response"]
            sys_prompt = sys_prompt_temp.format(
                response_type=response_type,
                user_prompt=user_prompt,
                context_data=query_context_result.context,
            )
            
            if param.only_need_prompt:
                 return QueryResult(
                     status="success",
                     message="Prompt generation successful",
                     content=sys_prompt,
                     raw_data=query_context_result.raw_data,
                     metadata=query_context_result.raw_data.get("metadata", {}),
                     query=query,
                     system_prompt=sys_prompt,
                     user_prompt=param.user_prompt or "",
                     **parsed_data
                 )

            # Generate
            if param.model_func:
                use_model_func = param.model_func
            else:
                use_model_func = self.llm_model_func
                use_model_func = partial(use_model_func, _priority=5)

            param.stream = True if param.stream is None else param.stream
            
            response = await use_model_func(
                sys_prompt,
                history_messages=param.conversation_history,
                stream=param.stream,
            )

            is_streaming = not isinstance(response, str)
            
            result = QueryResult(
                status="success",
                message="Query success",
                content=response if not is_streaming else None,
                response_iterator=response if is_streaming else None,
                is_streaming=is_streaming,
                context=query_context_result.context,
                raw_data=query_context_result.raw_data,
                metadata={
                    "mode": retrieval.mode,
                    "param": asdict(param),
                    **(query_context_result.raw_data.get("metadata", {}) or {})
                },
                query=query,
                system_prompt=sys_prompt,
                user_prompt=param.user_prompt or "",
                **parsed_data
            )
            
            await self._query_done()
            return result

        except Exception as e:
            logger.error(f"Query failed: {e}")
            logger.error(traceback.format_exc())
            # Return error response
            return QueryResult(
                status="failure",
                message=f"Query failed: {str(e)}",
                content=None,
                query=query,
                system_prompt="",
                user_prompt=param.user_prompt or ""
            )

    def query_llm(
        self,
        query: str,
        param: QueryParam = QueryParam(),
        retrieval: BaseRetrieval = None,
        system_prompt: str | None = None,
    ) -> QueryResult:
        """
        Synchronous complete query API: returns structured retrieval results with LLM generation.

        This function is the synchronous version of aquery_llm, providing the same functionality
        for users who prefer synchronous interfaces.

        Args:
            query: Query text for retrieval and LLM generation.
            param: Query parameters controlling retrieval and LLM behavior.
            retrieval: Retrieval strategy object.
            system_prompt: Optional custom system prompt for LLM generation.

        Returns:
            QueryResult: Same complete response format as aquery_llm.
        """
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.aquery_llm(query, param, retrieval, system_prompt))

    async def _query_done(self):
        await self.llm_response_cache.index_done_callback()

    async def aclear_cache(self) -> None:
        """Clear all cache data from the LLM response cache storage.

        This method clears all cached LLM responses regardless of mode.

        Example:
            # Clear all cache
            await rag.aclear_cache()
        """
        if not self.llm_response_cache:
            logger.warning("No cache storage configured")
            return

        try:
            # Clear all cache using drop method
            success = await self.llm_response_cache.drop()
            if success:
                logger.info("Cleared all cache")
            else:
                logger.warning("Failed to clear all cache")

            await self.llm_response_cache.index_done_callback()

        except Exception as e:
            logger.error(f"Error while clearing cache: {e}")

    def clear_cache(self) -> None:
        """Synchronous version of aclear_cache."""
        return always_get_an_event_loop().run_until_complete(self.aclear_cache())

    async def get_docs_by_status(
        self, status: DocStatus
    ) -> dict[str, DocProcessingStatus]:
        """Get documents by status

        Returns:
            Dict with document id is keys and document status is values
        """
        return await self.doc_status.get_docs_by_status(status)

    async def aget_docs_by_ids(
        self, ids: str | list[str]
    ) -> dict[str, DocProcessingStatus]:
        """Retrieves the processing status for one or more documents by their IDs.

        Args:
            ids: A single document ID (string) or a list of document IDs (list of strings).

        Returns:
            A dictionary where keys are the document IDs for which a status was found,
            and values are the corresponding DocProcessingStatus objects. IDs that
            are not found in the storage will be omitted from the result dictionary.
        """
        if isinstance(ids, str):
            # Ensure input is always a list of IDs for uniform processing
            id_list = [ids]
        elif (
            ids is None
        ):  # Handle potential None input gracefully, although type hint suggests str/list
            logger.warning(
                "aget_docs_by_ids called with None input, returning empty dict."
            )
            return {}
        else:
            # Assume input is already a list if not a string
            id_list = ids

        # Return early if the final list of IDs is empty
        if not id_list:
            logger.debug("aget_docs_by_ids called with an empty list of IDs.")
            return {}

        # Create tasks to fetch document statuses concurrently using the doc_status storage
        tasks = [self.doc_status.get_by_id(doc_id) for doc_id in id_list]
        # Execute tasks concurrently and gather the results. Results maintain order.
        # Type hint indicates results can be DocProcessingStatus or None if not found.
        results_list: list[Optional[DocProcessingStatus]] = await asyncio.gather(*tasks)

        # Build the result dictionary, mapping found IDs to their statuses
        found_statuses: dict[str, DocProcessingStatus] = {}
        # Keep track of IDs for which no status was found (for logging purposes)
        not_found_ids: list[str] = []

        # Iterate through the results, correlating them back to the original IDs
        for i, status_obj in enumerate(results_list):
            doc_id = id_list[
                i
            ]  # Get the original ID corresponding to this result index
            if status_obj:
                # If a status object was returned (not None), add it to the result dict
                found_statuses[doc_id] = status_obj
            else:
                # If status_obj is None, the document ID was not found in storage
                not_found_ids.append(doc_id)

        # Log a warning if any of the requested document IDs were not found
        if not_found_ids:
            logger.warning(
                f"Document statuses not found for the following IDs: {not_found_ids}"
            )

        # Return the dictionary containing statuses only for the found document IDs
        return found_statuses

    async def adelete_by_doc_id(
        self, doc_id: str, delete_llm_cache: bool = False
    ) -> DeletionResult:
        """Delete a document and all its related data, including chunks, graph elements.

        This method orchestrates a comprehensive deletion process for a given document ID.
        It ensures that not only the document itself but also all its derived and associated
        data across different storage layers are removed or rebuiled. If entities or relationships
        are partially affected, they will be rebuilded using LLM cached from remaining documents.

        **Concurrency Control Design:**

        This function implements a pipeline-based concurrency control to prevent data corruption:

        1. **Single Document Deletion** (when WE acquire pipeline):
           - Sets job_name to "Single document deletion" (NOT starting with "deleting")
           - Prevents other adelete_by_doc_id calls from running concurrently
           - Ensures exclusive access to graph operations for this deletion

        2. **Batch Document Deletion** (when background_delete_documents acquires pipeline):
           - Sets job_name to "Deleting {N} Documents" (starts with "deleting")
           - Allows multiple adelete_by_doc_id calls to join the deletion queue
           - Each call validates the job name to ensure it's part of a deletion operation

        The validation logic `if not job_name.startswith("deleting") or "document" not in job_name`
        ensures that:
        - adelete_by_doc_id can only run when pipeline is idle OR during batch deletion
        - Prevents concurrent single deletions that could cause race conditions
        - Rejects operations when pipeline is busy with non-deletion tasks

        Args:
            doc_id (str): The unique identifier of the document to be deleted.
            delete_llm_cache (bool): Whether to delete cached LLM extraction results
                associated with the document. Defaults to False.

        Returns:
            DeletionResult: An object containing the outcome of the deletion process.
                - `status` (str): "success", "not_found", "not_allowed", or "failure".
                - `doc_id` (str): The ID of the document attempted to be deleted.
                - `message` (str): A summary of the operation's result.
                - `status_code` (int): HTTP status code (e.g., 200, 404, 403, 500).
                - `file_path` (str | None): The file path of the deleted document, if available.
        """
        # Get pipeline status shared data and lock for validation
        pipeline_status = await get_namespace_data(
            "pipeline_status", workspace=self.workspace
        )
        pipeline_status_lock = get_namespace_lock(
            "pipeline_status", workspace=self.workspace
        )

        # Track whether WE acquired the pipeline
        we_acquired_pipeline = False

        # Check and acquire pipeline if needed
        async with pipeline_status_lock:
            if not pipeline_status.get("busy", False):
                # Pipeline is idle - WE acquire it for this deletion
                we_acquired_pipeline = True
                pipeline_status.update(
                    {
                        "busy": True,
                        "job_name": "Single document deletion",
                        "job_start": datetime.now(timezone.utc).isoformat(),
                        "docs": 1,
                        "batchs": 1,
                        "cur_batch": 0,
                        "request_pending": False,
                        "cancellation_requested": False,
                        "latest_message": f"Starting deletion for document: {doc_id}",
                    }
                )
                # Initialize history messages
                pipeline_status["history_messages"][:] = [
                    f"Starting deletion for document: {doc_id}"
                ]
            else:
                # Pipeline already busy - verify it's a deletion job
                job_name = pipeline_status.get("job_name", "").lower()
                if not job_name.startswith("deleting") or "document" not in job_name:
                    return DeletionResult(
                        status="not_allowed",
                        doc_id=doc_id,
                        message=f"Deletion not allowed: current job '{pipeline_status.get('job_name')}' is not a document deletion job",
                        status_code=403,
                        file_path=None,
                    )
                # Pipeline is busy with deletion - proceed without acquiring

        deletion_operations_started = False
        original_exception = None
        doc_llm_cache_ids: list[str] = []

        async with pipeline_status_lock:
            log_message = f"Starting deletion process for document {doc_id}"
            logger.info(log_message)
            pipeline_status["latest_message"] = log_message
            pipeline_status["history_messages"].append(log_message)

        try:
            # 1. Get the document status and related data
            doc_status_data = await self.doc_status.get_by_id(doc_id)
            file_path = doc_status_data.get("file_path") if doc_status_data else None
            if not doc_status_data:
                logger.warning(f"Document {doc_id} not found")
                return DeletionResult(
                    status="not_found",
                    doc_id=doc_id,
                    message=f"Document {doc_id} not found.",
                    status_code=404,
                    file_path="",
                )

            # Check document status and log warning for non-completed documents
            raw_status = doc_status_data.get("status")
            try:
                doc_status = DocStatus(raw_status)
            except ValueError:
                doc_status = raw_status

            if doc_status != DocStatus.PROCESSED:
                if doc_status == DocStatus.PENDING:
                    warning_msg = (
                        f"Deleting {doc_id} {file_path}(previous status: PENDING)"
                    )
                elif doc_status == DocStatus.PROCESSING:
                    warning_msg = (
                        f"Deleting {doc_id} {file_path}(previous status: PROCESSING)"
                    )
                elif doc_status == DocStatus.PREPROCESSED:
                    warning_msg = (
                        f"Deleting {doc_id} {file_path}(previous status: PREPROCESSED)"
                    )
                elif doc_status == DocStatus.FAILED:
                    warning_msg = (
                        f"Deleting {doc_id} {file_path}(previous status: FAILED)"
                    )
                else:
                    status_text = (
                        doc_status.value
                        if isinstance(doc_status, DocStatus)
                        else str(doc_status)
                    )
                    warning_msg = (
                        f"Deleting {doc_id} {file_path}(previous status: {status_text})"
                    )
                logger.info(warning_msg)
                # Update pipeline status for monitoring
                async with pipeline_status_lock:
                    pipeline_status["latest_message"] = warning_msg
                    pipeline_status["history_messages"].append(warning_msg)

            # 2. Get chunk IDs from document status
            chunk_ids = set(doc_status_data.get("chunks_list", []))

            if not chunk_ids:
                logger.warning(f"No chunks found for document {doc_id}")
                # Mark that deletion operations have started
                deletion_operations_started = True
                try:
                    # Still need to delete the doc status and full doc
                    await self.full_docs.delete([doc_id])
                    await self.doc_status.delete([doc_id])
                except Exception as e:
                    logger.error(
                        f"Failed to delete document {doc_id} with no chunks: {e}"
                    )
                    raise Exception(f"Failed to delete document entry: {e}") from e

                async with pipeline_status_lock:
                    log_message = (
                        f"Document deleted without associated chunks: {doc_id}"
                    )
                    logger.info(log_message)
                    pipeline_status["latest_message"] = log_message
                    pipeline_status["history_messages"].append(log_message)

                return DeletionResult(
                    status="success",
                    doc_id=doc_id,
                    message=log_message,
                    status_code=200,
                    file_path=file_path,
                )

            # Mark that deletion operations have started
            deletion_operations_started = True

            if delete_llm_cache and chunk_ids:
                if not self.llm_response_cache:
                    logger.info(
                        "Skipping LLM cache collection for document %s because cache storage is unavailable",
                        doc_id,
                    )
                elif not self.text_chunks:
                    logger.info(
                        "Skipping LLM cache collection for document %s because text chunk storage is unavailable",
                        doc_id,
                    )
                else:
                    try:
                        chunk_data_list = await self.text_chunks.get_by_ids(
                            list(chunk_ids)
                        )
                        seen_cache_ids: set[str] = set()
                        for chunk_data in chunk_data_list:
                            if not chunk_data or not isinstance(chunk_data, dict):
                                continue
                            cache_ids = chunk_data.get("llm_cache_list", [])
                            if not isinstance(cache_ids, list):
                                continue
                            for cache_id in cache_ids:
                                if (
                                    isinstance(cache_id, str)
                                    and cache_id
                                    and cache_id not in seen_cache_ids
                                ):
                                    doc_llm_cache_ids.append(cache_id)
                                    seen_cache_ids.add(cache_id)
                        if doc_llm_cache_ids:
                            logger.info(
                                "Collected %d LLM cache entries for document %s",
                                len(doc_llm_cache_ids),
                                doc_id,
                            )
                        else:
                            logger.info(
                                "No LLM cache entries found for document %s", doc_id
                            )
                    except Exception as cache_collect_error:
                        logger.error(
                            "Failed to collect LLM cache ids for document %s: %s",
                            doc_id,
                            cache_collect_error,
                        )
                        raise Exception(
                            f"Failed to collect LLM cache ids for document {doc_id}: {cache_collect_error}"
                        ) from cache_collect_error

            # 4. Analyze entities and relationships that will be affected
            entities_to_delete = set()
            entities_to_rebuild = {}  # entity_name -> remaining chunk id list
            relationships_to_delete = set()
            relationships_to_rebuild = {}  # (src, tgt) -> remaining chunk id list
            entity_chunk_updates: dict[str, list[str]] = {}
            relation_chunk_updates: dict[tuple[str, str], list[str]] = {}

            try:
                # Get affected entities and relations from full_entities and full_relations storage
                doc_entities_data = await self.full_entities.get_by_id(doc_id)
                doc_relations_data = await self.full_relations.get_by_id(doc_id)

                affected_nodes = []
                affected_edges = []

                # Get entity data from graph storage using entity names from full_entities
                if doc_entities_data and "entity_names" in doc_entities_data:
                    entity_names = doc_entities_data["entity_names"]
                    # get_nodes_batch returns dict[str, dict], need to convert to list[dict]
                    nodes_dict = await self.chunk_entity_relation_graph.get_nodes_batch(
                        entity_names
                    )
                    for entity_name in entity_names:
                        node_data = nodes_dict.get(entity_name)
                        if node_data:
                            # Ensure compatibility with existing logic that expects "id" field
                            if "id" not in node_data:
                                node_data["id"] = entity_name
                            affected_nodes.append(node_data)

                # Get relation data from graph storage using relation pairs from full_relations
                if doc_relations_data and "relation_pairs" in doc_relations_data:
                    relation_pairs = doc_relations_data["relation_pairs"]
                    edge_pairs_dicts = [
                        {"src": pair[0], "tgt": pair[1]} for pair in relation_pairs
                    ]
                    # get_edges_batch returns dict[tuple[str, str], dict], need to convert to list[dict]
                    edges_dict = await self.chunk_entity_relation_graph.get_edges_batch(
                        edge_pairs_dicts
                    )

                    for pair in relation_pairs:
                        src, tgt = pair[0], pair[1]
                        edge_key = (src, tgt)
                        edge_data = edges_dict.get(edge_key)
                        if edge_data:
                            # Ensure compatibility with existing logic that expects "source" and "target" fields
                            if "source" not in edge_data:
                                edge_data["source"] = src
                            if "target" not in edge_data:
                                edge_data["target"] = tgt
                            affected_edges.append(edge_data)

            except Exception as e:
                logger.error(f"Failed to analyze affected graph elements: {e}")
                raise Exception(f"Failed to analyze graph dependencies: {e}") from e

            try:
                # Process entities
                for node_data in affected_nodes:
                    node_label = node_data.get("entity_id")
                    if not node_label:
                        continue

                    existing_sources: list[str] = []
                    if self.entity_chunks:
                        stored_chunks = await self.entity_chunks.get_by_id(node_label)
                        if stored_chunks and isinstance(stored_chunks, dict):
                            existing_sources = [
                                chunk_id
                                for chunk_id in stored_chunks.get("chunk_ids", [])
                                if chunk_id
                            ]

                    if not existing_sources and node_data.get("source_id"):
                        existing_sources = [
                            chunk_id
                            for chunk_id in node_data["source_id"].split(
                                GRAPH_FIELD_SEP
                            )
                            if chunk_id
                        ]

                    if not existing_sources:
                        # No chunk references means this entity should be deleted
                        entities_to_delete.add(node_label)
                        entity_chunk_updates[node_label] = []
                        continue

                    remaining_sources = subtract_source_ids(existing_sources, chunk_ids)

                    if not remaining_sources:
                        entities_to_delete.add(node_label)
                        entity_chunk_updates[node_label] = []
                    elif remaining_sources != existing_sources:
                        entities_to_rebuild[node_label] = remaining_sources
                        entity_chunk_updates[node_label] = remaining_sources
                    else:
                        logger.info(f"Untouch entity: {node_label}")

                async with pipeline_status_lock:
                    log_message = f"Found {len(entities_to_rebuild)} affected entities"
                    logger.info(log_message)
                    pipeline_status["latest_message"] = log_message
                    pipeline_status["history_messages"].append(log_message)

                # Process relationships
                for edge_data in affected_edges:
                    # source target is not in normalize order in graph db property
                    src = edge_data.get("source")
                    tgt = edge_data.get("target")

                    if not src or not tgt or "source_id" not in edge_data:
                        continue

                    edge_tuple = tuple(sorted((src, tgt)))
                    if (
                        edge_tuple in relationships_to_delete
                        or edge_tuple in relationships_to_rebuild
                    ):
                        continue

                    existing_sources: list[str] = []
                    if self.relation_chunks:
                        storage_key = make_relation_chunk_key(src, tgt)
                        stored_chunks = await self.relation_chunks.get_by_id(
                            storage_key
                        )
                        if stored_chunks and isinstance(stored_chunks, dict):
                            existing_sources = [
                                chunk_id
                                for chunk_id in stored_chunks.get("chunk_ids", [])
                                if chunk_id
                            ]

                    if not existing_sources:
                        existing_sources = [
                            chunk_id
                            for chunk_id in edge_data["source_id"].split(
                                GRAPH_FIELD_SEP
                            )
                            if chunk_id
                        ]

                    if not existing_sources:
                        # No chunk references means this relationship should be deleted
                        relationships_to_delete.add(edge_tuple)
                        relation_chunk_updates[edge_tuple] = []
                        continue

                    remaining_sources = subtract_source_ids(existing_sources, chunk_ids)

                    if not remaining_sources:
                        relationships_to_delete.add(edge_tuple)
                        relation_chunk_updates[edge_tuple] = []
                    elif remaining_sources != existing_sources:
                        relationships_to_rebuild[edge_tuple] = remaining_sources
                        relation_chunk_updates[edge_tuple] = remaining_sources
                    else:
                        logger.info(f"Untouch relation: {edge_tuple}")

                async with pipeline_status_lock:
                    log_message = (
                        f"Found {len(relationships_to_rebuild)} affected relations"
                    )
                    logger.info(log_message)
                    pipeline_status["latest_message"] = log_message
                    pipeline_status["history_messages"].append(log_message)

                current_time = int(time.time())

                if entity_chunk_updates and self.entity_chunks:
                    entity_upsert_payload = {}
                    for entity_name, remaining in entity_chunk_updates.items():
                        if not remaining:
                            # Empty entities are deleted alongside graph nodes later
                            continue
                        entity_upsert_payload[entity_name] = {
                            "chunk_ids": remaining,
                            "count": len(remaining),
                            "updated_at": current_time,
                        }
                    if entity_upsert_payload:
                        await self.entity_chunks.upsert(entity_upsert_payload)

                if relation_chunk_updates and self.relation_chunks:
                    relation_upsert_payload = {}
                    for edge_tuple, remaining in relation_chunk_updates.items():
                        if not remaining:
                            # Empty relations are deleted alongside graph edges later
                            continue
                        storage_key = make_relation_chunk_key(*edge_tuple)
                        relation_upsert_payload[storage_key] = {
                            "chunk_ids": remaining,
                            "count": len(remaining),
                            "updated_at": current_time,
                        }

                    if relation_upsert_payload:
                        await self.relation_chunks.upsert(relation_upsert_payload)

            except Exception as e:
                logger.error(f"Failed to process graph analysis results: {e}")
                raise Exception(f"Failed to process graph dependencies: {e}") from e

            # Data integrity is ensured by allowing only one process to hold pipeline at a time（no graph db lock is needed anymore)

            # 5. Delete chunks from storage
            if chunk_ids:
                try:
                    await self.chunks_vdb.delete(chunk_ids)
                    await self.text_chunks.delete(chunk_ids)

                    async with pipeline_status_lock:
                        log_message = (
                            f"Successfully deleted {len(chunk_ids)} chunks from storage"
                        )
                        logger.info(log_message)
                        pipeline_status["latest_message"] = log_message
                        pipeline_status["history_messages"].append(log_message)

                except Exception as e:
                    logger.error(f"Failed to delete chunks: {e}")
                    raise Exception(f"Failed to delete document chunks: {e}") from e

            # 6. Delete relationships that have no remaining sources
            if relationships_to_delete:
                try:
                    # Delete from relation vdb
                    rel_ids_to_delete = []
                    for src, tgt in relationships_to_delete:
                        rel_ids_to_delete.extend(
                            [
                                compute_mdhash_id(src + tgt, prefix="rel-"),
                                compute_mdhash_id(tgt + src, prefix="rel-"),
                            ]
                        )
                    await self.relationships_vdb.delete(rel_ids_to_delete)

                    # Delete from graph
                    await self.chunk_entity_relation_graph.remove_edges(
                        list(relationships_to_delete)
                    )

                    # Delete from relation_chunks storage
                    if self.relation_chunks:
                        relation_storage_keys = [
                            make_relation_chunk_key(src, tgt)
                            for src, tgt in relationships_to_delete
                        ]
                        await self.relation_chunks.delete(relation_storage_keys)

                    async with pipeline_status_lock:
                        log_message = f"Successfully deleted {len(relationships_to_delete)} relations"
                        logger.info(log_message)
                        pipeline_status["latest_message"] = log_message
                        pipeline_status["history_messages"].append(log_message)

                except Exception as e:
                    logger.error(f"Failed to delete relationships: {e}")
                    raise Exception(f"Failed to delete relationships: {e}") from e

            # 7. Delete entities that have no remaining sources
            if entities_to_delete:
                try:
                    # Batch get all edges for entities to avoid N+1 query problem
                    nodes_edges_dict = (
                        await self.chunk_entity_relation_graph.get_nodes_edges_batch(
                            list(entities_to_delete)
                        )
                    )

                    # Debug: Check and log all edges before deleting nodes
                    edges_to_delete = set()
                    edges_still_exist = 0

                    for entity, edges in nodes_edges_dict.items():
                        if edges:
                            for src, tgt in edges:
                                # Normalize edge representation (sorted for consistency)
                                edge_tuple = tuple(sorted((src, tgt)))
                                edges_to_delete.add(edge_tuple)

                                if (
                                    src in entities_to_delete
                                    and tgt in entities_to_delete
                                ):
                                    logger.warning(
                                        f"Edge still exists: {src} <-> {tgt}"
                                    )
                                elif src in entities_to_delete:
                                    logger.warning(
                                        f"Edge still exists: {src} --> {tgt}"
                                    )
                                else:
                                    logger.warning(
                                        f"Edge still exists: {src} <-- {tgt}"
                                    )
                            edges_still_exist += 1

                    if edges_still_exist:
                        logger.warning(
                            f"⚠️ {edges_still_exist} entities still has edges before deletion"
                        )

                    # Clean residual edges from VDB and storage before deleting nodes
                    if edges_to_delete:
                        # Delete from relationships_vdb
                        rel_ids_to_delete = []
                        for src, tgt in edges_to_delete:
                            rel_ids_to_delete.extend(
                                [
                                    compute_mdhash_id(src + tgt, prefix="rel-"),
                                    compute_mdhash_id(tgt + src, prefix="rel-"),
                                ]
                            )
                        await self.relationships_vdb.delete(rel_ids_to_delete)

                        # Delete from relation_chunks storage
                        if self.relation_chunks:
                            relation_storage_keys = [
                                make_relation_chunk_key(src, tgt)
                                for src, tgt in edges_to_delete
                            ]
                            await self.relation_chunks.delete(relation_storage_keys)

                        logger.info(
                            f"Cleaned {len(edges_to_delete)} residual edges from VDB and chunk-tracking storage"
                        )

                    # Delete from graph (edges will be auto-deleted with nodes)
                    await self.chunk_entity_relation_graph.remove_nodes(
                        list(entities_to_delete)
                    )

                    # Delete from vector vdb
                    entity_vdb_ids = [
                        compute_mdhash_id(entity, prefix="ent-")
                        for entity in entities_to_delete
                    ]
                    await self.entities_vdb.delete(entity_vdb_ids)

                    # Delete from entity_chunks storage
                    if self.entity_chunks:
                        await self.entity_chunks.delete(list(entities_to_delete))

                    async with pipeline_status_lock:
                        log_message = (
                            f"Successfully deleted {len(entities_to_delete)} entities"
                        )
                        logger.info(log_message)
                        pipeline_status["latest_message"] = log_message
                        pipeline_status["history_messages"].append(log_message)

                except Exception as e:
                    logger.error(f"Failed to delete entities: {e}")
                    raise Exception(f"Failed to delete entities: {e}") from e

            # Persist changes to graph database before entity and relationship rebuild
            await self._insert_done()

            # 8. Rebuild entities and relationships from remaining chunks
            if entities_to_rebuild or relationships_to_rebuild:
                try:
                    await rebuild_knowledge_from_chunks(
                        entities_to_rebuild=entities_to_rebuild,
                        relationships_to_rebuild=relationships_to_rebuild,
                        knowledge_graph_inst=self.chunk_entity_relation_graph,
                        entities_vdb=self.entities_vdb,
                        relationships_vdb=self.relationships_vdb,
                        text_chunks_storage=self.text_chunks,
                        llm_response_cache=self.llm_response_cache,
                        tokenizer=self.tokenizer,
                        llm_service=self.llm_service,
                        embedding_service=self.embedding_service,
                        workspace=self.workspace,
                        max_source_ids_per_entity=self.max_source_ids_per_entity,
                        max_source_ids_per_relation=self.max_source_ids_per_relation,
                        max_file_paths=self.max_file_paths,
                        source_ids_limit_method=self.source_ids_limit_method,
                        force_llm_summary_on_merge=getattr(self.llm_service, "force_llm_summary_on_merge", 5),
                        file_path_more_placeholder=self.file_path_more_placeholder,
                        language=self.language,
                        pipeline_status=pipeline_status,
                        pipeline_status_lock=pipeline_status_lock,
                        entity_chunks_storage=self.entity_chunks,
                        relation_chunks_storage=self.relation_chunks,
                    )

                except Exception as e:
                    logger.error(f"Failed to rebuild knowledge from chunks: {e}")
                    raise Exception(f"Failed to rebuild knowledge graph: {e}") from e

            # 9. Delete from full_entities and full_relations storage
            try:
                await self.full_entities.delete([doc_id])
                await self.full_relations.delete([doc_id])
            except Exception as e:
                logger.error(f"Failed to delete from full_entities/full_relations: {e}")
                raise Exception(
                    f"Failed to delete from full_entities/full_relations: {e}"
                ) from e

            # 10. Delete original document and status
            try:
                await self.full_docs.delete([doc_id])
                await self.doc_status.delete([doc_id])
            except Exception as e:
                logger.error(f"Failed to delete document and status: {e}")
                raise Exception(f"Failed to delete document and status: {e}") from e

            if delete_llm_cache and doc_llm_cache_ids and self.llm_response_cache:
                try:
                    await self.llm_response_cache.delete(doc_llm_cache_ids)
                    cache_log_message = f"Successfully deleted {len(doc_llm_cache_ids)} LLM cache entries for document {doc_id}"
                    logger.info(cache_log_message)
                    async with pipeline_status_lock:
                        pipeline_status["latest_message"] = cache_log_message
                        pipeline_status["history_messages"].append(cache_log_message)
                    log_message = cache_log_message
                except Exception as cache_delete_error:
                    log_message = f"Failed to delete Easy Knowledge Retriever cache for document {doc_id}: {cache_delete_error}"
                    logger.error(log_message)
                    logger.error(traceback.format_exc())
                    async with pipeline_status_lock:
                        pipeline_status["latest_message"] = log_message
                        pipeline_status["history_messages"].append(log_message)

            return DeletionResult(
                status="success",
                doc_id=doc_id,
                message=log_message,
                status_code=200,
                file_path=file_path,
            )

        except Exception as e:
            original_exception = e
            error_message = f"Error while deleting document {doc_id}: {e}"
            logger.error(error_message)
            logger.error(traceback.format_exc())
            return DeletionResult(
                status="fail",
                doc_id=doc_id,
                message=error_message,
                status_code=500,
                file_path=file_path,
            )

        finally:
            # ALWAYS ensure persistence if any deletion operations were started
            if deletion_operations_started:
                try:
                    await self._insert_done()
                except Exception as persistence_error:
                    persistence_error_msg = f"Failed to persist data after deletion attempt for {doc_id}: {persistence_error}"
                    logger.error(persistence_error_msg)
                    logger.error(traceback.format_exc())

                    # If there was no original exception, this persistence error becomes the main error
                    if original_exception is None:
                        return DeletionResult(
                            status="fail",
                            doc_id=doc_id,
                            message=f"Deletion completed but failed to persist changes: {persistence_error}",
                            status_code=500,
                            file_path=file_path,
                        )
                    # If there was an original exception, log the persistence error but don't override the original error
                    # The original error result was already returned in the except block
            else:
                logger.debug(
                    f"No deletion operations were started for document {doc_id}, skipping persistence"
                )

            # Release pipeline only if WE acquired it
            if we_acquired_pipeline:
                async with pipeline_status_lock:
                    pipeline_status["busy"] = False
                    pipeline_status["cancellation_requested"] = False
                    completion_msg = (
                        f"Deletion process completed for document: {doc_id}"
                    )
                    pipeline_status["latest_message"] = completion_msg
                    pipeline_status["history_messages"].append(completion_msg)
                    logger.info(completion_msg)

    async def adelete_by_entity(self, entity_name: str) -> DeletionResult:
        """Asynchronously delete an entity and all its relationships.

        Args:
            entity_name: Name of the entity to delete.

        Returns:
            DeletionResult: An object containing the outcome of the deletion process.
        """
        from easy_knowledge_retriever.kg.graph_storage.utils_graph import adelete_by_entity

        return await adelete_by_entity(
            self.chunk_entity_relation_graph,
            self.entities_vdb,
            self.relationships_vdb,
            entity_name,
        )

    def delete_by_entity(self, entity_name: str) -> DeletionResult:
        """Synchronously delete an entity and all its relationships.

        Args:
            entity_name: Name of the entity to delete.

        Returns:
            DeletionResult: An object containing the outcome of the deletion process.
        """
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.adelete_by_entity(entity_name))

    async def adelete_by_relation(
        self, source_entity: str, target_entity: str
    ) -> DeletionResult:
        """Asynchronously delete a relation between two entities.

        Args:
            source_entity: Name of the source entity.
            target_entity: Name of the target entity.

        Returns:
            DeletionResult: An object containing the outcome of the deletion process.
        """
        from easy_knowledge_retriever.kg.graph_storage.utils_graph import adelete_by_relation
        return await adelete_by_relation(
            self.chunk_entity_relation_graph,
            self.relationships_vdb,
            source_entity,
            target_entity,
        )

    def delete_by_relation(
        self, source_entity: str, target_entity: str
    ) -> DeletionResult:
        """Synchronously delete a relation between two entities.

        Args:
            source_entity: Name of the source entity.
            target_entity: Name of the target entity.

        Returns:
            DeletionResult: An object containing the outcome of the deletion process.
        """
        loop = always_get_an_event_loop()
        return loop.run_until_complete(
            self.adelete_by_relation(source_entity, target_entity)
        )

    async def get_processing_status(self) -> dict[str, int]:
        """Get current document processing status counts

        Returns:
            Dict with counts for each status
        """
        return await self.doc_status.get_status_counts()

    async def aget_docs_by_track_id(
        self, track_id: str
    ) -> dict[str, DocProcessingStatus]:
        """Get documents by track_id

        Args:
            track_id: The tracking ID to search for

        Returns:
            Dict with document id as keys and document status as values
        """
        return await self.doc_status.get_docs_by_track_id(track_id)

    async def get_entity_info(
        self, entity_name: str, include_vector_data: bool = False
    ) -> dict[str, str | None | dict[str, str]]:
        """Get detailed information of an entity"""
        from easy_knowledge_retriever.kg.graph_storage.utils_graph import get_entity_info

        return await get_entity_info(
            self.chunk_entity_relation_graph,
            self.entities_vdb,
            entity_name,
            include_vector_data,
        )

    async def get_relation_info(
        self, src_entity: str, tgt_entity: str, include_vector_data: bool = False
    ) -> dict[str, str | None | dict[str, str]]:
        """Get detailed information of a relationship"""
        from easy_knowledge_retriever.kg.graph_storage.utils_graph import get_relation_info

        return await get_relation_info(
            self.chunk_entity_relation_graph,
            self.relationships_vdb,
            src_entity,
            tgt_entity,
            include_vector_data,
        )

    async def aedit_entity(
        self,
        entity_name: str,
        updated_data: dict[str, str],
        allow_rename: bool = True,
        allow_merge: bool = False,
    ) -> dict[str, Any]:
        """Asynchronously edit entity information.

        Updates entity information in the knowledge graph and re-embeds the entity in the vector database.
        Also synchronizes entity_chunks_storage and relation_chunks_storage to track chunk references.

        Args:
            entity_name: Name of the entity to edit
            updated_data: Dictionary containing updated attributes, e.g. {"description": "new description", "entity_type": "new type"}
            allow_rename: Whether to allow entity renaming, defaults to True
            allow_merge: Whether to merge into an existing entity when renaming to an existing name

        Returns:
            Dictionary containing updated entity information
        """
        from easy_knowledge_retriever.kg.graph_storage.utils_graph import aedit_entity

        return await aedit_entity(
            self.chunk_entity_relation_graph,
            self.entities_vdb,
            self.relationships_vdb,
            entity_name,
            updated_data,
            allow_rename,
            allow_merge,
            self.entity_chunks,
            self.relation_chunks,
        )

    def edit_entity(
        self,
        entity_name: str,
        updated_data: dict[str, str],
        allow_rename: bool = True,
        allow_merge: bool = False,
    ) -> dict[str, Any]:
        loop = always_get_an_event_loop()
        return loop.run_until_complete(
            self.aedit_entity(entity_name, updated_data, allow_rename, allow_merge)
        )

    async def aedit_relation(
        self, source_entity: str, target_entity: str, updated_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Asynchronously edit relation information.

        Updates relation (edge) information in the knowledge graph and re-embeds the relation in the vector database.
        Also synchronizes the relation_chunks_storage to track which chunks reference this relation.

        Args:
            source_entity: Name of the source entity
            target_entity: Name of the target entity
            updated_data: Dictionary containing updated attributes, e.g. {"description": "new description", "keywords": "new keywords"}

        Returns:
            Dictionary containing updated relation information
        """
        from easy_knowledge_retriever.kg.graph_storage.utils_graph import aedit_relation

        return await aedit_relation(
            self.chunk_entity_relation_graph,
            self.entities_vdb,
            self.relationships_vdb,
            source_entity,
            target_entity,
            updated_data,
            self.relation_chunks,
        )

    def edit_relation(
        self, source_entity: str, target_entity: str, updated_data: dict[str, Any]
    ) -> dict[str, Any]:
        loop = always_get_an_event_loop()
        return loop.run_until_complete(
            self.aedit_relation(source_entity, target_entity, updated_data)
        )

    async def acreate_entity(
        self, entity_name: str, entity_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Asynchronously create a new entity.

        Creates a new entity in the knowledge graph and adds it to the vector database.

        Args:
            entity_name: Name of the new entity
            entity_data: Dictionary containing entity attributes, e.g. {"description": "description", "entity_type": "type"}

        Returns:
            Dictionary containing created entity information
        """
        from easy_knowledge_retriever.kg.graph_storage.utils_graph import acreate_entity

        return await acreate_entity(
            self.chunk_entity_relation_graph,
            self.entities_vdb,
            self.relationships_vdb,
            entity_name,
            entity_data,
        )

    def create_entity(
        self, entity_name: str, entity_data: dict[str, Any]
    ) -> dict[str, Any]:
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.acreate_entity(entity_name, entity_data))

    async def acreate_relation(
        self, source_entity: str, target_entity: str, relation_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Asynchronously create a new relation between entities.

        Creates a new relation (edge) in the knowledge graph and adds it to the vector database.

        Args:
            source_entity: Name of the source entity
            target_entity: Name of the target entity
            relation_data: Dictionary containing relation attributes, e.g. {"description": "description", "keywords": "keywords"}

        Returns:
            Dictionary containing created relation information
        """
        from easy_knowledge_retriever.kg.graph_storage.utils_graph import acreate_relation

        return await acreate_relation(
            self.chunk_entity_relation_graph,
            self.entities_vdb,
            self.relationships_vdb,
            source_entity,
            target_entity,
            relation_data,
        )

    def create_relation(
        self, source_entity: str, target_entity: str, relation_data: dict[str, Any]
    ) -> dict[str, Any]:
        loop = always_get_an_event_loop()
        return loop.run_until_complete(
            self.acreate_relation(source_entity, target_entity, relation_data)
        )

    async def amerge_entities(
        self,
        source_entities: list[str],
        target_entity: str,
        merge_strategy: dict[str, str] = None,
        target_entity_data: dict[str, Any] = None,
    ) -> dict[str, Any]:
        """Asynchronously merge multiple entities into one entity.

        Merges multiple source entities into a target entity, handling all relationships,
        and updating both the knowledge graph and vector database.

        Args:
            source_entities: List of source entity names to merge
            target_entity: Name of the target entity after merging
            merge_strategy: Merge strategy configuration, e.g. {"description": "concatenate", "entity_type": "keep_first"}
                Supported strategies:
                - "concatenate": Concatenate all values (for text fields)
                - "keep_first": Keep the first non-empty value
                - "keep_last": Keep the last non-empty value
                - "join_unique": Join all unique values (for fields separated by delimiter)
            target_entity_data: Dictionary of specific values to set for the target entity,
                overriding any merged values, e.g. {"description": "custom description", "entity_type": "PERSON"}

        Returns:
            Dictionary containing the merged entity information
        """
        from easy_knowledge_retriever.kg.graph_storage.utils_graph import amerge_entities

        return await amerge_entities(
            self.chunk_entity_relation_graph,
            self.entities_vdb,
            self.relationships_vdb,
            source_entities,
            target_entity,
            merge_strategy,
            target_entity_data,
            self.entity_chunks,
            self.relation_chunks,
        )

    def merge_entities(
        self,
        source_entities: list[str],
        target_entity: str,
        merge_strategy: dict[str, str] = None,
        target_entity_data: dict[str, Any] = None,
    ) -> dict[str, Any]:
        loop = always_get_an_event_loop()
        return loop.run_until_complete(
            self.amerge_entities(
                source_entities, target_entity, merge_strategy, target_entity_data
            )
        )

    async def aexport_data(
        self,
        output_path: str,
        file_format: Literal["csv", "excel", "md", "txt"] = "csv",
        include_vector_data: bool = False,
    ) -> None:
        """
        Asynchronously exports all entities, relations, and relationships to various formats.
        Args:
            output_path: The path to the output file (including extension).
            file_format: Output format - "csv", "excel", "md", "txt".
                - csv: Comma-separated values file
                - excel: Microsoft Excel file with multiple sheets
                - md: Markdown tables
                - txt: Plain text formatted output
                - table: Print formatted tables to console
            include_vector_data: Whether to include data from the vector database.
        """
        from easy_knowledge_retriever.utils.vector_utils import aexport_data as utils_aexport_data

        await utils_aexport_data(
            self.chunk_entity_relation_graph,
            self.entities_vdb,
            self.relationships_vdb,
            output_path,
            file_format,
            include_vector_data,
        )

    def export_data(
        self,
        output_path: str,
        file_format: Literal["csv", "excel", "md", "txt"] = "csv",
        include_vector_data: bool = False,
    ) -> None:
        """
        Synchronously exports all entities, relations, and relationships to various formats.
        Args:
            output_path: The path to the output file (including extension).
            file_format: Output format - "csv", "excel", "md", "txt".
                - csv: Comma-separated values file
                - excel: Microsoft Excel file with multiple sheets
                - md: Markdown tables
                - txt: Plain text formatted output
                - table: Print formatted tables to console
            include_vector_data: Whether to include data from the vector database.
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        loop.run_until_complete(
            self.aexport_data(output_path, file_format, include_vector_data)
        )
