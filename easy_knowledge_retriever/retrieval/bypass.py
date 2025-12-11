from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from easy_knowledge_retriever.kg.base import (
    QueryParam, 
    QueryContextResult, 
)
from easy_knowledge_retriever.kg.graph_storage.base import BaseGraphStorage
from easy_knowledge_retriever.kg.vector_storage.base import BaseVectorStorage
from easy_knowledge_retriever.retrieval.base import BaseRetrieval

if TYPE_CHECKING:
    from easy_knowledge_retriever.retriever import EasyKnowledgeRetriever


@dataclass
class BypassRetrieval(BaseRetrieval):
    """Bypass retrieval strategy (no context retrieval)."""
    mode: str = "bypass"

    def _create_query_param(self) -> QueryParam:
        return QueryParam(
            mode="bypass",
            max_total_tokens=self.max_total_tokens,
            conversation_history=self.conversation_history,
            only_need_context=True,
        )

    async def retrieve(self, query: str, rag: "EasyKnowledgeRetriever") -> QueryContextResult:
        return QueryContextResult(context="", raw_data={})

    async def search(
        self,
        query: str,
        knowledge_graph_inst: BaseGraphStorage,
        entities_vdb: BaseVectorStorage,
        relationships_vdb: BaseVectorStorage,
        chunks_vdb: BaseVectorStorage,
        query_embedding: list[float] = None,
    ) -> dict[str, Any]:
        return {
            "local_entities": [],
            "local_relations": [],
            "global_entities": [],
            "global_relations": [],
            "vector_chunks": [],
            "chunk_tracking": {}
        }
