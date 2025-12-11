from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any


from easy_knowledge_retriever.kg.base import (
    QueryParam, 
    QueryContextResult, 
)
from easy_knowledge_retriever.kg.graph_storage.base import BaseGraphStorage
from easy_knowledge_retriever.kg.vector_storage.base import BaseVectorStorage
from easy_knowledge_retriever.retrieval.base import BaseRetrieval, _common_kg_retrieve
from easy_knowledge_retriever.retrieval.ops import get_node_data

if TYPE_CHECKING:
    from easy_knowledge_retriever.retriever import EasyKnowledgeRetriever


@dataclass
class LocalRetrieval(BaseRetrieval):
    """Local retrieval strategy focusing on entities."""
    mode: str = "local"
    top_k: int = 40
    max_entity_tokens: int = 2000
    ll_keywords: list[str] = field(default_factory=list)

    def _create_query_param(self) -> QueryParam:
        return QueryParam(
            mode="local",
            top_k=self.top_k,
            max_entity_tokens=self.max_entity_tokens,
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
        
        search_query = ", ".join(self.ll_keywords) if self.ll_keywords else ""
        
        if search_query:
            local_entities, local_relations = await get_node_data(
                search_query,
                knowledge_graph_inst,
                entities_vdb,
                self.top_k
            )
            
        return {
            "local_entities": local_entities,
            "local_relations": local_relations,
            "global_entities": [],
            "global_relations": [],
            "vector_chunks": [],
            "chunk_tracking": {}
        }
