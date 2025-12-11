from dataclasses import dataclass
from typing import TYPE_CHECKING, Any


from easy_knowledge_retriever.kg.base import (
    QueryParam, 
    QueryContextResult, 
)
from easy_knowledge_retriever.kg.graph_storage.base import BaseGraphStorage
from easy_knowledge_retriever.kg.vector_storage.base import BaseVectorStorage
from easy_knowledge_retriever.retrieval.base import BaseRetrieval, _common_kg_retrieve
from easy_knowledge_retriever.retrieval.ops import get_vector_context

if TYPE_CHECKING:
    from easy_knowledge_retriever.retriever import EasyKnowledgeRetriever


@dataclass
class NaiveRetrieval(BaseRetrieval):
    """Naive retrieval strategy using only vector search on text chunks."""
    mode: str = "naive"
    chunk_top_k: int = 20

    def _create_query_param(self) -> QueryParam:
        return QueryParam(
            mode="naive",
            chunk_top_k=self.chunk_top_k,
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
        vector_chunks = []
        chunk_tracking = {}
        
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
            "local_entities": [],
            "local_relations": [],
            "global_entities": [],
            "global_relations": [],
            "vector_chunks": vector_chunks,
            "chunk_tracking": chunk_tracking
        }
