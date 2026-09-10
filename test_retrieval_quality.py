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

# The OpenAI-compatible reranker calls generic_rerank_api with arguments it accepts
# (return_documents / request_format made every rerank raise, silently falling back to the unranked order).
import inspect
import easy_knowledge_retriever.reranker.openai as openai_rerank
from easy_knowledge_retriever.reranker.generic import generic_rerank_api

real_signature = inspect.signature(generic_rerank_api)


async def fake_generic(*args, **kwargs):
    real_signature.bind(*args, **kwargs)  # raises TypeError on an unknown argument
    return [{"index": 0, "relevance_score": 1.0}]


openai_rerank.generic_rerank_api = fake_generic
service = openai_rerank.OpenAIRerankerService(model="m", base_url="http://x/v1/rerank", api_key="k")
assert asyncio.run(service.rerank("q", ["doc"])) == [{"index": 0, "relevance_score": 1.0}]
print("reranker call signature ok")

# Context is rendered as tagged text with pages, not JSON lines.
from easy_knowledge_retriever.kg.base import QueryParam
from easy_knowledge_retriever.llm.prompts import PROMPTS
from easy_knowledge_retriever.retrieval.query_processing import _build_context_str
from easy_knowledge_retriever.utils.tokenizer import TiktokenTokenizer

ctx, data = asyncio.run(_build_context_str(
    entities_context=[{"entity": "Fournisseur", "type": "concept", "description": "développe un système d'IA"}],
    relations_context=[{"entity1": "Fournisseur", "entity2": "Système d'IA", "description": "met sur le marché"}],
    merged_chunks=[{"content": "Article 3 Définitions", "chunk_id": "c1", "file_path": "doc.pdf", "page_start": 163}],
    query="q", query_param=QueryParam(mode="hybrid_mix"), tokenizer=TiktokenTokenizer(), max_total_tokens=30000,
    system_prompt_template=PROMPTS["rag_response"]))
assert '<chunk reference_id="1" page="163">\nArticle 3 Définitions\n</chunk>' in ctx, ctx
assert "- Fournisseur (concept): développe un système d'IA" in ctx
assert "- Fournisseur <-> Système d'IA: met sur le marché" in ctx
print("tagged context ok")

# Chunk headings: prepended for embedding, rendered in the context tag.
from easy_knowledge_retriever.kg.vector_storage.nano_vector_db_impl import embedding_text

assert embedding_text({"content": "1. Les fournisseurs...", "heading": "Article 16 — Obligations"}) == "Article 16 — Obligations\n1. Les fournisseurs..."
assert embedding_text({"content": "texte"}) == "texte"
ctx, _ = asyncio.run(_build_context_str(
    entities_context=[], relations_context=[],
    merged_chunks=[{"content": "suite", "chunk_id": "c9", "file_path": "doc.pdf", "page_start": 5, "heading": 'Article 3 — "Définitions"'}],
    query="q", query_param=QueryParam(mode="hybrid_mix"), tokenizer=TiktokenTokenizer(), max_total_tokens=30000,
    system_prompt_template=PROMPTS["rag_response"]))
assert '<chunk reference_id="1" page="5" heading="Article 3 — \'Définitions\'">' in ctx, ctx
print("chunk headings ok")

# The reranker scores the heading with the content.
seen_docs = []


class RecordingReranker:
    async def rerank(self, query, documents, top_n=None):
        seen_docs.append(list(documents))
        return [{"index": i, "relevance_score": 1.0 - i / 10} for i in range(len(documents))]


asyncio.run(process_retrieved_chunks("q", [{"content": "1. Biométrie", "heading": "Annexe III"}, {"content": "sans titre"}],
                                     SimpleNamespace(chunk_top_k=5), 10_000, RecordingReranker()))
assert seen_docs == [["Annexe III\n1. Biométrie", "sans titre"]], seen_docs
print("rerank text ok")

# Chunks following the best hits are scored and replace the tail only when they outrank it.
from easy_knowledge_retriever.retrieval.query_processing import _add_reranked_neighbors


class OrderedKV:
    def __init__(self):
        self._data = {f"c{i}": {"content": f"part {i}", "full_doc_id": "d", "chunk_order_index": i, "file_path": "doc.pdf"} for i in range(5)}

    async def get_by_ids(self, ids):
        return [self._data.get(i) for i in ids]


class NeighbourReranker:
    async def rerank(self, query, documents, top_n=None):
        seen_docs.append(list(documents))
        scores = {"part 1": 0.8, "part 2": 0.1, "part 4": 0.05}
        return [{"index": i, "relevance_score": scores[d]} for i, d in enumerate(documents)]


seen_docs.clear()
best = [{"content": "part 0", "chunk_id": "c0", "rerank_score": 0.9}, {"content": "part 3", "chunk_id": "c3", "rerank_score": 0.2}]
tracking = {}
kept = asyncio.run(_add_reranked_neighbors("q", best, OrderedKV(), NeighbourReranker(), QueryParam(chunk_top_k=2), tracking))
assert seen_docs == [["part 1", "part 2", "part 4"]], seen_docs  # c3 is already kept; c5 does not exist
assert [c["chunk_id"] for c in kept] == ["c0", "c1"], kept
assert kept[1]["file_path"] == "doc.pdf" and tracking["c1"]["source"] == "N"
print("neighbour chunks ok")
