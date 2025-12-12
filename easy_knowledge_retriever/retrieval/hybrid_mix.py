import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, List, Dict, Set

from easy_knowledge_retriever.kg.base import QueryParam, QueryContextResult
from easy_knowledge_retriever.kg.graph_storage.base import BaseGraphStorage
from easy_knowledge_retriever.kg.vector_storage.base import BaseVectorStorage
from easy_knowledge_retriever.retrieval.base import BaseRetrieval, _common_kg_retrieve
from easy_knowledge_retriever.utils.logger import logger


def simple_tokenize(text: str) -> List[str]:
    return re.findall(r'\w+', text.lower())


@dataclass
class HybridMixRetrieval(BaseRetrieval):
    """
    Hybrid Retrieval (Sparse + Dense) with RRF Fusion.
    Dense: Vector Search.
    Sparse: BM25 (or simple keyword matching).
    Fusion: RRF (Reciprocal Rank Fusion).
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

        return {
            "local_entities": [],
            "local_relations": [],
            "global_entities": [],
            "global_relations": [],
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
        Implement a simple in-memory BM25 search.
        NOTE: This iterates over ALL chunks in the DB, which might be slow for large datasets.
        """
        if not chunks_vdb:
            return []

        # Get all data from storage
        # Depending on implementation, we might need to access the underlying client storage
        try:
            storage = await chunks_vdb.client_storage
            all_data = storage.get("data", [])
        except Exception as e:
            logger.warning(f"Could not access internal storage for BM25: {e}")
            return []
            
        if not all_data:
            return []

        # Simple BM25 Implementation
        query_tokens = simple_tokenize(query)
        if not query_tokens:
            return []

        # Precompute corpus stats
        corpus_size = len(all_data)
        avgdl = 0
        doc_freqs = Counter()
        
        # We need to tokenize all docs. 
        # For performance in a real system, this should be pre-indexed.
        # Here we do it on the fly as per constraints.
        tokenized_corpus = []
        for doc in all_data:
            content = doc.get("content", "")
            tokens = simple_tokenize(content)
            tokenized_corpus.append((doc, tokens))
            avgdl += len(tokens)
            
            # Update doc freqs for query terms
            unique_tokens = set(tokens)
            for token in query_tokens:
                if token in unique_tokens:
                    doc_freqs[token] += 1
                    
        avgdl /= corpus_size if corpus_size > 0 else 1
        
        scores = []
        
        for doc, tokens in tokenized_corpus:
            score = 0
            doc_len = len(tokens)
            doc_counter = Counter(tokens)
            
            for token in query_tokens:
                f = doc_counter[token]
                if f == 0:
                    continue
                
                n_q = doc_freqs[token]
                idf = math.log(1 + (corpus_size - n_q + 0.5) / (n_q + 0.5))
                
                term_score = idf * (f * (self.k1 + 1)) / (f + self.k1 * (1 - self.b + self.b * (doc_len / avgdl)))
                score += term_score
            
            if score > 0:
                scores.append((score, doc))
                
        # Sort by score desc
        scores.sort(key=lambda x: x[0], reverse=True)
        
        top_results = scores[:top_k]
        
        valid_chunks = []
        for score, result in top_results:
            chunk_with_metadata = {
                "content": result.get("content", ""),
                "created_at": result.get("__created_at__", None), # NanoDB internal key
                "file_path": result.get("file_path", "unknown_source"),
                "source_type": "bm25",
                "chunk_id": result.get("__id__"), # NanoDB internal key
                "id": result.get("__id__"),
                "score": score
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
