"""Retrieval pipeline checks: stored vectors reused, rerank order and top-N. Run: python test_retrieval_quality.py"""
import asyncio
from types import SimpleNamespace
from easy_knowledge_retriever.utils.vector_utils import pick_by_vector_similarity, process_retrieved_chunks


class KV:
    async def get_by_ids(self, ids):
        return [{"content": f"text {i}"} for i in ids]


class VDB:
    async def get_vectors_by_ids(self, ids):  # "c" has no stored vector
        return {i: ([1.0, 0.0] if i == "a" else [0.0, 1.0]) for i in ids if i != "c"}


calls = []


async def embed(texts):
    calls.append(list(texts))
    return [[0.7, 0.7] for _ in texts]


picked = asyncio.run(pick_by_vector_similarity(
    query="q", text_chunks_storage=KV(), chunks_vdb=VDB(), num_of_chunks=2,
    entity_info=[{"chunks": ["a", "b", "c"]}], embedding_func=embed, query_embedding=[1.0, 0.0]))
assert calls == [["text c"]], calls  # only the chunk without a stored vector is embedded
assert picked == ["a", "c"], picked


class Reranker:
    async def rerank(self, query, documents, top_n=None):
        scores = {"low": 0.1, "high": 0.9, "mid": 0.5}
        return [{"index": i, "relevance_score": scores[d]} for i, d in enumerate(documents)]


chunks = [{"content": c} for c in ("low", "high", "mid")]
out = asyncio.run(process_retrieved_chunks("q", chunks, SimpleNamespace(chunk_top_k=2), 10_000, Reranker()))
assert [c["content"] for c in out] == ["high", "mid"], out
assert out[0]["rerank_score"] == 0.9
print("ok")

# Rerank requests always carry an integer top_n (a null/missing one is rejected by some backends).
from easy_knowledge_retriever.reranker.generic import build_rerank_payload

assert build_rerank_payload("m", "q", ["a", "b", "c"])["top_n"] == 3
assert build_rerank_payload("m", "q", ["a", "b"], top_n=1, extra_body={"x": 1}) == {"model": "m", "query": "q", "documents": ["a", "b"], "top_n": 1, "x": 1}
print("rerank payload ok")
