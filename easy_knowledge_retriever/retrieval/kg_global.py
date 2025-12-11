from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any


from easy_knowledge_retriever.kg.base import (
    QueryParam, 
    QueryContextResult, 
)
from easy_knowledge_retriever.kg.graph_storage.base import BaseGraphStorage
from easy_knowledge_retriever.kg.vector_storage.base import BaseVectorStorage
from easy_knowledge_retriever.retrieval.base import BaseRetrieval, _common_kg_retrieve
from easy_knowledge_retriever.retrieval.ops import get_edge_data

if TYPE_CHECKING:
    from easy_knowledge_retriever.retriever import EasyKnowledgeRetriever


@dataclass
class GlobalRetrieval(BaseRetrieval):
    """Global retrieval strategy focusing on relationships."""
    mode: str = "global"
    top_k: int = 40
    max_relation_tokens: int = 3000
    hl_keywords: list[str] = field(default_factory=list)

    def _create_query_param(self) -> QueryParam:
        return QueryParam(
            mode="global",
            top_k=self.top_k,
            max_relation_tokens=self.max_relation_tokens,
            hl_keywords=self.hl_keywords,
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
        global_relations = []
        global_entities = []
        
        search_query = ", ".join(self.hl_keywords) if self.hl_keywords else ""
        
        if search_query:
            global_relations, global_entities = await get_edge_data(
                search_query,
                knowledge_graph_inst,
                relationships_vdb,
                self.top_k
            )
            
        return {
            "local_entities": [],
            "local_relations": [],
            "global_entities": global_entities,
            "global_relations": global_relations,
            "vector_chunks": [],
            "chunk_tracking": {}
        }
