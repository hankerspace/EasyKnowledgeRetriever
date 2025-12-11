from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any


from easy_knowledge_retriever.kg.base import (
    QueryParam, 
    QueryContextResult, 
)
from easy_knowledge_retriever.kg.graph_storage.base import BaseGraphStorage
from easy_knowledge_retriever.kg.vector_storage.base import BaseVectorStorage
from easy_knowledge_retriever.retrieval.base import BaseRetrieval, _common_kg_retrieve
from easy_knowledge_retriever.retrieval.ops import get_node_data, get_edge_data

if TYPE_CHECKING:
    from easy_knowledge_retriever.retriever import EasyKnowledgeRetriever


@dataclass
class HybridRetrieval(BaseRetrieval):
    """Hybrid retrieval strategy combining local (entities) and global (relationships)."""
    mode: str = "hybrid"
    top_k: int = 40
    max_entity_tokens: int = 2000
    max_relation_tokens: int = 3000
    hl_keywords: list[str] = field(default_factory=list)
    ll_keywords: list[str] = field(default_factory=list)

    def _create_query_param(self) -> QueryParam:
        return QueryParam(
            mode="hybrid",
            top_k=self.top_k,
            max_entity_tokens=self.max_entity_tokens,
            max_relation_tokens=self.max_relation_tokens,
            hl_keywords=self.hl_keywords,
            ll_keywords=self.ll_keywords,
            max_total_tokens=self.max_total_tokens,
            conversation_history=self.conversation_history,
            only_need_context=True,
        )

    async def retrieve(self, query: str, rag: "EasyKnowledgeRetriever") -> QueryContextResult:
        return await _common_kg_retrieve(self, query, rag)

    async def search(
        self,
        query: str,
        knowledge_graph_inst: BaseGraphStorage,
        entities_vdb: BaseVectorStorage,
        relationships_vdb: BaseVectorStorage,
        chunks_vdb: BaseVectorStorage,
        query_embedding: list[float] = None,
    ) -> dict[str, Any]:
        local_entities = []
        local_relations = []
        global_entities = []
        global_relations = []
        
        # Local search
        search_query_local = ", ".join(self.ll_keywords) if self.ll_keywords else ""
        if search_query_local:
             local_entities, local_relations = await get_node_data(
                search_query_local,
                knowledge_graph_inst,
                entities_vdb,
                self.top_k
            )

        # Global search
        search_query_global = ", ".join(self.hl_keywords) if self.hl_keywords else ""
        if search_query_global:
            global_relations, global_entities = await get_edge_data(
                search_query_global,
                knowledge_graph_inst,
                relationships_vdb,
                self.top_k
            )
            
        return {
            "local_entities": local_entities,
            "local_relations": local_relations,
            "global_entities": global_entities,
            "global_relations": global_relations,
            "vector_chunks": [],
            "chunk_tracking": {}
        }
