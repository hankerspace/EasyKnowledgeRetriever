from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any


from easy_knowledge_retriever.kg.base import (
    QueryParam, 
    QueryContextResult, 
)
from easy_knowledge_retriever.kg.graph_storage.base import BaseGraphStorage
from easy_knowledge_retriever.kg.vector_storage.base import BaseVectorStorage
from easy_knowledge_retriever.retrieval.base import BaseRetrieval
from easy_knowledge_retriever.retrieval.ops import get_vector_context
from easy_knowledge_retriever.retrieval.query_processing import _build_context_str
from easy_knowledge_retriever.llm.prompts import PROMPTS

if TYPE_CHECKING:
    from easy_knowledge_retriever.retriever import EasyKnowledgeRetriever


@dataclass
class NaiveRetrieval(BaseRetrieval):
    """Naive retrieval strategy using only vector search on text chunks."""
    mode: str = "naive"
    chunk_top_k: int = 20
    prompt_template: str = field(default_factory=lambda: PROMPTS["naive_rag_response"])

    def _create_query_param(self) -> QueryParam:
        return QueryParam(
            mode="naive",
            chunk_top_k=self.chunk_top_k,
            max_total_tokens=self.max_total_tokens,
            conversation_history=self.conversation_history,
            only_need_context=True,
        )

    async def retrieve(self, query: str, rag: "EasyKnowledgeRetriever") -> QueryContextResult:
        query_param = self._create_query_param()

        # Execute Search directly without KG access
        search_result = await self.search(
            query=query,
            knowledge_graph_inst=rag.chunk_entity_relation_graph,
            entities_vdb=rag.entities_vdb,
            relationships_vdb=rag.relationships_vdb,
            chunks_vdb=rag.chunks_vdb,
        )
        
        vector_chunks = search_result.get("vector_chunks", [])
        chunk_tracking = search_result.get("chunk_tracking", {})

        # Build context string directly, skipping keyword extraction and KG context building
        context, final_data = await _build_context_str(
            entities_context=[],
            relations_context=[],
            merged_chunks=vector_chunks,
            query=query,
            query_param=query_param,
            tokenizer=rag.tokenizer,
            max_total_tokens=rag.max_total_tokens,
            system_prompt_template=self.prompt_template,
            chunk_tracking=chunk_tracking
        )

        return QueryContextResult(context=context, raw_data=final_data)

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
