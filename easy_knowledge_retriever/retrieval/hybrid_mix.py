import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, List, Dict, Set
import numpy as np
from rank_bm25 import BM25Okapi

from easy_knowledge_retriever.kg.base import QueryParam, QueryContextResult
from easy_knowledge_retriever.kg.graph_storage.base import BaseGraphStorage
from easy_knowledge_retriever.kg.vector_storage.base import BaseVectorStorage
from easy_knowledge_retriever.retrieval.base import BaseRetrieval, _common_kg_retrieve
from easy_knowledge_retriever.retrieval.ops import get_node_data, get_edge_data
from easy_knowledge_retriever.utils.logger import logger


def simple_tokenize(text: str) -> List[str]:
    return re.findall(r'\w+', text.lower())


@dataclass
class HybridMixRetrieval(BaseRetrieval):
    """
    Hybrid Retrieval (Sparse + Dense + Graph) with RRF Fusion.
    Dense: Vector Search.
    Sparse: BM25 (or simple keyword matching).
    Fusion: RRF (Reciprocal Rank Fusion).
    Graph: Local (Entities) + Global (Relations).
    """
    mode: str = "hybrid_mix"
    top_k: int = 40  # Total chunks to retrieve after fusion
    chunk_top_k: int = 20  # Number of chunks to retrieve from each source (Dense/Sparse) before fusion
    
    # BM25 parameters
    k1: float = 1.5
    b: float = 0.75
    
    # RRF k constant
    rrf_k: int = 60

    hl_keywords: list[str] = field(default_factory=list)
    ll_keywords: list[str] = field(default_factory=list)

    # BM25 Cache
    bm25_index: Any = None
    corpus_chunks: List[Dict[str, Any]] = field(default_factory=list)

    def _create_query_param(self) -> QueryParam:
        return QueryParam(
            mode=self.mode,
            top_k=self.top_k,
            chunk_top_k=self.chunk_top_k,
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
        
        # 1. Dense Search (Vectors)
        dense_results = await self._dense_search(query, chunks_vdb, self.chunk_top_k, query_embedding)
        
        # 2. Sparse Search (BM25)
        sparse_results = await self._sparse_search(query, chunks_vdb, self.chunk_top_k)
        
        # 3. Fusion (RRF)
        fused_chunks = self._rrf_fusion(dense_results, sparse_results, self.rrf_k)
        
        # Limit to top_k
        fused_chunks = fused_chunks[:self.top_k]
        
        chunk_tracking = {}
        for i, chunk in enumerate(fused_chunks):
            chunk_id = chunk.get("chunk_id") or chunk.get("id")
            if chunk_id:
                chunk_tracking[chunk_id] = {
                    "source": "hybrid",
                    "frequency": 1,
                    "order": i + 1,
                }

        # 4. Graph Search (Local + Global)
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
            "vector_chunks": fused_chunks,
            "chunk_tracking": chunk_tracking
        }

    async def _dense_search(
        self, 
        query: str, 
        chunks_vdb: BaseVectorStorage, 
        top_k: int, 
        query_embedding: list[float] = None
    ) -> List[Dict[str, Any]]:
        if not chunks_vdb:
            return []
            
        results = await chunks_vdb.query(
            query, top_k=top_k, query_embedding=query_embedding
        )
        
        valid_chunks = []
        for result in results:
            if "content" in result:
                chunk_with_metadata = {
                    "content": result["content"],
                    "created_at": result.get("created_at", None),
                    "file_path": result.get("file_path", "unknown_source"),
                    "source_type": "vector",
                    "chunk_id": result.get("id"),
                    "id": result.get("id"),
                }
                valid_chunks.append(chunk_with_metadata)
        return valid_chunks

    async def _sparse_search(
        self, 
        query: str, 
        chunks_vdb: BaseVectorStorage, 
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Implement a simple in-memory BM25 search using rank_bm25.
        Optimized to cache index.
        """
        if not chunks_vdb:
            return []

        # 1. Initialize index if not exists
        if not self.bm25_index:
            try:
                storage = await chunks_vdb.client_storage
                all_data = storage.get("data", [])
            except Exception as e:
                logger.warning(f"Could not access internal storage for BM25: {e}")
                return []
            
            if not all_data:
                return []
                
            tokenized_corpus = [simple_tokenize(doc.get("content", "")) for doc in all_data]
            self.bm25_index = BM25Okapi(tokenized_corpus)
            self.corpus_chunks = all_data

        # 2. Query the optimized index
        tokenized_query = simple_tokenize(query)
        if not tokenized_query:
            return []
            
        doc_scores = self.bm25_index.get_scores(tokenized_query)
        
        # 3. Sort and get Top K
        # np.argsort returns indices that would sort the array
        top_indices = np.argsort(doc_scores)[::-1][:top_k]
        
        valid_chunks = []
        for idx in top_indices:
            score = doc_scores[idx]
            if score <= 0:
                continue
            
            result = self.corpus_chunks[idx]
            chunk_with_metadata = {
                "content": result.get("content", ""),
                "created_at": result.get("__created_at__", None), # NanoDB internal key
                "file_path": result.get("file_path", "unknown_source"),
                "source_type": "bm25",
                "chunk_id": result.get("__id__"), # NanoDB internal key
                "id": result.get("__id__"),
                "score": float(score)
            }
            valid_chunks.append(chunk_with_metadata)
            
        return valid_chunks

    def _rrf_fusion(
        self, 
        dense_results: List[Dict[str, Any]], 
        sparse_results: List[Dict[str, Any]], 
        k: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Reciprocal Rank Fusion.
        score = 1 / (k + rank)
        """
        fused_scores = {}
        chunks_map = {}
        
        # Process Dense
        for rank, chunk in enumerate(dense_results):
            chunk_id = chunk.get("chunk_id")
            if not chunk_id:
                continue
            chunks_map[chunk_id] = chunk
            if chunk_id not in fused_scores:
                fused_scores[chunk_id] = 0.0
            fused_scores[chunk_id] += 1.0 / (k + rank + 1)
            
        # Process Sparse
        for rank, chunk in enumerate(sparse_results):
            chunk_id = chunk.get("chunk_id")
            if not chunk_id:
                continue
            # Update chunks map if not present (prefer dense version if both exist, or update metadata)
            if chunk_id not in chunks_map:
                chunks_map[chunk_id] = chunk
            
            if chunk_id not in fused_scores:
                fused_scores[chunk_id] = 0.0
            fused_scores[chunk_id] += 1.0 / (k + rank + 1)
            
        # Sort by fused score
        sorted_ids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)
        
        final_results = []
        for chunk_id in sorted_ids:
            chunk = chunks_map[chunk_id]
            chunk["rrf_score"] = fused_scores[chunk_id]
            final_results.append(chunk)
            
        return final_results
