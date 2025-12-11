from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any


from easy_knowledge_retriever.kg.base import (
    QueryParam, 
    QueryContextResult, 
)
from easy_knowledge_retriever.kg.graph_storage.base import BaseGraphStorage
from easy_knowledge_retriever.kg.vector_storage.base import BaseVectorStorage
from easy_knowledge_retriever.retrieval.base import BaseRetrieval, _common_kg_retrieve
from easy_knowledge_retriever.retrieval.ops import get_node_data, get_edge_data, get_vector_context

if TYPE_CHECKING:
    from easy_knowledge_retriever.retriever import EasyKnowledgeRetriever


@dataclass
class MixRetrieval(BaseRetrieval):
    """Mix retrieval strategy combining KG (Local+Global) and Vector (Chunks) retrieval."""
    mode: str = "mix"
    top_k: int = 40
    chunk_top_k: int = 20
    max_entity_tokens: int = 2000
    max_relation_tokens: int = 3000
    hl_keywords: list[str] = field(default_factory=list)
    ll_keywords: list[str] = field(default_factory=list)

    def _create_query_param(self) -> QueryParam:
        return QueryParam(
            mode="mix",
            top_k=self.top_k,
            chunk_top_k=self.chunk_top_k,
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
        vector_chunks = []
        chunk_tracking = {}
        
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
            
        # Vector chunks
        if chunks_vdb:
             vector_chunks = await get_vector_context(
                query,
                chunks_vdb,
                self.chunk_top_k,
                query_embedding
            )
             # Track vector chunks
             for i, chunk in enumerate(vector_chunks):
                chunk_id = chunk.get("chunk_id") or chunk.get("id")
                if chunk_id:
                    chunk_tracking[chunk_id] = {
                        "source": "C",
                        "frequency": 1,
                        "order": i + 1,
                    }
            
        return {
            "local_entities": local_entities,
            "local_relations": local_relations,
            "global_entities": global_entities,
            "global_relations": global_relations,
            "vector_chunks": vector_chunks,
            "chunk_tracking": chunk_tracking
        }
