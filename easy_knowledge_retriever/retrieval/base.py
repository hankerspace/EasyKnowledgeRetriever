from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import TYPE_CHECKING, List, Dict, Any

from easy_knowledge_retriever.kg.base import (
    QueryParam, 
    QueryContextResult, 
)
from easy_knowledge_retriever.kg.graph_storage.base import BaseGraphStorage
from easy_knowledge_retriever.kg.vector_storage.base import BaseVectorStorage
from easy_knowledge_retriever.reranker.base import BaseRerankerService
from easy_knowledge_retriever.llm.prompts import PROMPTS


if TYPE_CHECKING:
    from easy_knowledge_retriever.retriever import EasyKnowledgeRetriever

@dataclass
class BaseRetrieval(ABC):
    """Base class for retrieval strategies."""
    
    # Common parameters for all retrievals
    max_total_tokens: int = 2000
    conversation_history: list[dict[str, str]] = field(default_factory=list)
    reranker_service: BaseRerankerService | None = None
    query_decomposition: bool = False
    prompt_template: str = field(default_factory=lambda: PROMPTS["rag_response"])

    @abstractmethod
    def _create_query_param(self) -> QueryParam:
        """Helper to create a QueryParam object from this retrieval config."""
        pass

    @abstractmethod
    async def retrieve(self, query: str, rag: "EasyKnowledgeRetriever") -> QueryContextResult:
        """Execute the retrieval strategy."""
        pass

    @abstractmethod
    async def search(
        self,
        query: str,
        knowledge_graph_inst: BaseGraphStorage,
        entities_vdb: BaseVectorStorage,
        relationships_vdb: BaseVectorStorage,
        chunks_vdb: BaseVectorStorage,
        query_embedding: list[float] = None,
    ) -> dict[str, Any]:
        """
        Perform the specific search logic for this retrieval mode.
        
        Returns:
            dict containing keys: 'local_entities', 'local_relations', 'global_entities', 'global_relations', 'vector_chunks', 'chunk_tracking'
            (some keys might be missing or empty depending on mode)
        """
        pass

async def _common_kg_retrieve(retrieval: BaseRetrieval, query: str, rag: "EasyKnowledgeRetriever") -> QueryContextResult:
    """Helper for KG-based retrieval modes."""
    from easy_knowledge_retriever.retrieval.query_processing import (
        _build_query_context,
        get_keywords_from_query,
    )

    query_param = retrieval._create_query_param()
    
    # 1. Extract keywords
    hl_keywords, ll_keywords = await get_keywords_from_query(
        query, 
        query_param, 
        tokenizer=rag.tokenizer,
        llm_model_func=rag.llm_model_func,
        enable_llm_cache=rag.enable_llm_cache,
        language=rag.language,
        hashing_kv=rag.llm_response_cache
    )
    
    # Update params with extracted keywords if they were empty
    if not getattr(retrieval, "hl_keywords", None) and hl_keywords:
        if hasattr(retrieval, "hl_keywords"):
            retrieval.hl_keywords = hl_keywords
            # Also update query_param to stay in sync
            query_param.hl_keywords = hl_keywords
            
    if not getattr(retrieval, "ll_keywords", None) and ll_keywords:
        if hasattr(retrieval, "ll_keywords"):
            retrieval.ll_keywords = ll_keywords
            query_param.ll_keywords = ll_keywords
    
    # Still update query_param if retrieval didn't have the attribute (fallback)
    if not query_param.hl_keywords and hl_keywords:
        query_param.hl_keywords = hl_keywords
    if not query_param.ll_keywords and ll_keywords:
        query_param.ll_keywords = ll_keywords
        
    ll_keywords_str = ", ".join(ll_keywords) if ll_keywords else ""
    hl_keywords_str = ", ".join(hl_keywords) if hl_keywords else ""
    
    # 2. Build context (now passing retrieval object)
    context_result = await _build_query_context(
        query=query,
        ll_keywords=ll_keywords_str,
        hl_keywords=hl_keywords_str,
        knowledge_graph_inst=rag.chunk_entity_relation_graph,
        entities_vdb=rag.entities_vdb,
        relationships_vdb=rag.relationships_vdb,
        text_chunks_db=rag.text_chunks,
        query_param=query_param,
        tokenizer=rag.tokenizer,
        kg_chunk_pick_method=rag.kg_chunk_pick_method,
        max_related_chunks=rag.related_chunk_number,
        max_total_tokens=rag.max_total_tokens,
        chunks_vdb=rag.chunks_vdb,
        retrieval=retrieval,
        system_prompt_template=retrieval.prompt_template,
    )
    
    if context_result is None:
        return QueryContextResult(context="", raw_data={})
        
    return context_result
