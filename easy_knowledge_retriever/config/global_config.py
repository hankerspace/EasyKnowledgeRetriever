from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List
from .base import BaseConfig
from easy_knowledge_retriever.constants import (
    DEFAULT_TOP_K,
    DEFAULT_CHUNK_TOP_K,
    DEFAULT_MAX_ENTITY_TOKENS,
    DEFAULT_MAX_RELATION_TOKENS,
    DEFAULT_MAX_TOTAL_TOKENS,
    DEFAULT_COSINE_THRESHOLD,
    DEFAULT_RELATED_CHUNK_NUMBER,
    DEFAULT_KG_CHUNK_PICK_METHOD,
    DEFAULT_MAX_GLEANING,
    DEFAULT_FORCE_LLM_SUMMARY_ON_MERGE,
    DEFAULT_MAX_PARALLEL_INSERT,
    DEFAULT_MAX_GRAPH_NODES,
    DEFAULT_MAX_SOURCE_IDS_PER_ENTITY,
    DEFAULT_MAX_SOURCE_IDS_PER_RELATION,
    DEFAULT_SOURCE_IDS_LIMIT_METHOD,
    DEFAULT_MAX_FILE_PATHS,
    DEFAULT_FILE_PATH_MORE_PLACEHOLDER,
    DEFAULT_SUMMARY_LANGUAGE,
    DEFAULT_ENTITY_TYPES,
)
from easy_knowledge_retriever.utils.tokenizer import Tokenizer
from easy_knowledge_retriever.utils.vector_utils import normalize_source_ids_limit_method

@dataclass
class GlobalConfig(BaseConfig):
    working_dir: str = "./rag_storage"
    workspace: str = ""
    
    # Query Params
    top_k: int = DEFAULT_TOP_K
    chunk_top_k: int = DEFAULT_CHUNK_TOP_K
    max_entity_tokens: int = DEFAULT_MAX_ENTITY_TOKENS
    max_relation_tokens: int = DEFAULT_MAX_RELATION_TOKENS
    max_total_tokens: int = DEFAULT_MAX_TOTAL_TOKENS
    cosine_threshold: float = DEFAULT_COSINE_THRESHOLD
    related_chunk_number: int = DEFAULT_RELATED_CHUNK_NUMBER
    kg_chunk_pick_method: str = DEFAULT_KG_CHUNK_PICK_METHOD
    
    # Entity Extraction
    entity_extract_max_gleaning: int = DEFAULT_MAX_GLEANING
    force_llm_summary_on_merge: int = DEFAULT_FORCE_LLM_SUMMARY_ON_MERGE
    
    # Chunking
    chunk_token_size: int = 1200
    chunk_overlap_token_size: int = 100
    tiktoken_model_name: str = "gpt-4o-mini"
    
    # Storage
    
    # Extensions
    max_parallel_insert: int = DEFAULT_MAX_PARALLEL_INSERT
    max_source_ids_per_entity: int = DEFAULT_MAX_SOURCE_IDS_PER_ENTITY
    max_source_ids_per_relation: int = DEFAULT_MAX_SOURCE_IDS_PER_RELATION
    source_ids_limit_method: str = field(
        default_factory=lambda: normalize_source_ids_limit_method(DEFAULT_SOURCE_IDS_LIMIT_METHOD)
    )
    max_file_paths: int = DEFAULT_MAX_FILE_PATHS
    file_path_more_placeholder: str = DEFAULT_FILE_PATH_MORE_PLACEHOLDER
    
    language: str = DEFAULT_SUMMARY_LANGUAGE
    entity_types: List[str] = field(default_factory=lambda: DEFAULT_ENTITY_TYPES)
    
    enable_llm_cache: bool = True
    enable_llm_cache_for_entity_extract: bool = True
