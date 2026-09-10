"""Smoke checks for the pieces changed for the Dedhicated POC.

Runnable without any LLM, network or heavy extra:
    python test_ekr_smoke.py
"""
import asyncio
import os
import subprocess
import sys
import tempfile
import types


def test_core_imports_without_extras():
    """The core must import with no torch / mineru / neo4j / milvus installed."""
    import easy_knowledge_retriever as ekr
    from easy_knowledge_retriever.retrieval import HybridMixRetrieval, NaiveRetrieval
    from easy_knowledge_retriever.llm.service import OpenAILLMService, OpenAIEmbeddingService
    from easy_knowledge_retriever.kg.kv_storage.json_kv_impl import JsonKVStorage
    from easy_knowledge_retriever.kg.vector_storage.nano_vector_db_impl import NanoVectorDBStorage
    from easy_knowledge_retriever.kg.graph_storage.networkx_impl import NetworkXStorage

    assert ekr.EasyKnowledgeRetriever is not None
    assert "torch" not in sys.modules, "core import must not pull torch"


def test_env_tunable_concurrency():
    """Concurrency must be tunable, and a bad value must not break the import."""
    from easy_knowledge_retriever import constants

    assert constants.DEFAULT_MAX_ASYNC >= 1
    assert constants._env_int("EKR__ABSENT", 7) == 7
    os.environ["EKR__TMP"] = "12"
    assert constants._env_int("EKR__TMP", 1) == 12
    os.environ["EKR__TMP"] = "not-a-number"
    assert constants._env_int("EKR__TMP", 1) == 1, "bad value must fall back"
    os.environ["EKR__TMP"] = "0"
    assert constants._env_int("EKR__TMP", 4, minimum=1) == 4, "below minimum must fall back"
    del os.environ["EKR__TMP"]


def test_embedding_encoding_format_is_switchable():
    """OpenAI-compatible gateways often reject base64 with a 422."""
    import importlib
    from easy_knowledge_retriever import constants

    assert constants.DEFAULT_EMBEDDING_ENCODING_FORMAT == "base64", "default must not change"

    os.environ["EKR_EMBEDDING_ENCODING_FORMAT"] = "float"
    try:
        c = importlib.reload(constants)
        assert c.DEFAULT_EMBEDDING_ENCODING_FORMAT == "float"
        os.environ["EKR_EMBEDDING_ENCODING_FORMAT"] = "nonsense"
        c = importlib.reload(constants)
        assert c.DEFAULT_EMBEDDING_ENCODING_FORMAT == "base64", "bad value must fall back"
    finally:
        del os.environ["EKR_EMBEDDING_ENCODING_FORMAT"]
        importlib.reload(constants)


def test_read_text_document_shape():
    """Text files must produce the same structure MinerU yields."""
    from easy_knowledge_retriever.retriever import read_text_document

    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8") as f:
        f.write("# Titre\n\nContenu accentue e.")
        path = f.name
    try:
        data = read_text_document(path)
        assert data["content"].startswith("# Titre")
        assert len(data["pages"]) == 1
        assert data["pages"][0]["page_number"] == 1
        assert data["pages"][0]["content"] == data["content"]
        assert data["pages"][0]["images"] == []
    finally:
        os.unlink(path)


def test_ingest_rejects_unsupported_extension():
    """An unsupported type must fail loudly and early, not deep inside MinerU."""
    from easy_knowledge_retriever.retriever import EasyKnowledgeRetriever

    fake_self = types.SimpleNamespace(working_dir="/tmp")
    try:
        asyncio.run(EasyKnowledgeRetriever.ingest(fake_self, "/tmp/report.docx"))
    except ValueError as exc:
        assert ".docx" in str(exc) and ".pdf" in str(exc), exc
    else:
        raise AssertionError("expected ValueError for .docx")


def test_mineru_subprocess_is_bounded():
    """A hung MinerU run must abort the parse instead of hanging forever."""
    from easy_knowledge_retriever.operations.mineru_parser import MineruParser

    assert MineruParser(output_dir="/tmp", timeout=42).timeout == 42
    assert MineruParser(output_dir="/tmp").timeout >= 1

    # The mechanism itself: subprocess.run must raise, not block.
    try:
        subprocess.run([sys.executable, "-c", "import time; time.sleep(30)"], timeout=1)
    except subprocess.TimeoutExpired:
        pass
    else:
        raise AssertionError("expected TimeoutExpired")


def test_merge_query_results_keeps_chunks_and_references():
    """Decomposed queries used to return chunks=[] because raw_data["data"] was dropped."""
    from easy_knowledge_retriever.kg.base import QueryContextResult
    from easy_knowledge_retriever.retrieval.query_processing import merge_query_results

    def fake(chunks, refs, entities, hl):
        return QueryContextResult(context="ctx", raw_data={
            "status": "success",
            "data": {"chunks": chunks, "references": refs, "entities": entities, "relationships": []},
            "metadata": {"query_mode": "mix", "keywords": {"high_level": hl, "low_level": []},
                         "processing_info": {"final_chunks_count": len(chunks)}},
        })

    a = fake(
        [{"chunk_id": "c1", "reference_id": "1", "file_path": "a.pdf", "content": "x"},
         {"chunk_id": "c2", "reference_id": "1", "file_path": "a.pdf", "content": "y"}],
        [{"reference_id": "1", "file_path": "a.pdf"}],
        [{"entity_name": "AI Act"}], ["risk"],
    )
    # Second sub-query numbers its own references: b.pdf is "1" here, a.pdf is "2".
    b = fake(
        [{"chunk_id": "c3", "reference_id": "1", "file_path": "b.pdf", "content": "z"},
         {"chunk_id": "c2", "reference_id": "2", "file_path": "a.pdf", "content": "y"}],
        [{"reference_id": "1", "file_path": "b.pdf"}, {"reference_id": "2", "file_path": "a.pdf"}],
        [{"entity_name": "AI Act"}, {"entity_name": "GPAI"}], ["risk", "gpai"],
    )

    raw = merge_query_results([a, b]).raw_data
    data = raw["data"]
    assert [c["chunk_id"] for c in data["chunks"]] == ["c1", "c2", "c3"]
    assert data["references"] == [
        {"reference_id": "1", "file_path": "a.pdf"},
        {"reference_id": "2", "file_path": "b.pdf"},
    ]
    refs = {r["reference_id"]: r["file_path"] for r in data["references"]}
    assert all(refs[c["reference_id"]] == c["file_path"] for c in data["chunks"])
    assert [e["entity_name"] for e in data["entities"]] == ["AI Act", "GPAI"]
    assert raw["metadata"]["query_mode"] == "mix"
    assert raw["metadata"]["keywords"]["high_level"] == ["risk", "gpai"]
    assert raw["metadata"]["processing_info"]["final_chunks_count"] == 3
    assert raw["status"] == "success"


def test_naive_retrieval_carries_page_from_text_chunks():
    """The chunks vdb has no page metadata: without the text chunk store lookup the LLM invents pages."""
    import json
    from easy_knowledge_retriever.retrieval import NaiveRetrieval
    from easy_knowledge_retriever.utils.tokenizer import Tokenizer

    class ChunksVdb:
        cosine_better_than_threshold = 0.2

        async def query(self, query, top_k, query_embedding=None):
            return [{"id": "chunk-1", "content": "Article 5 prohibited practices", "file_path": "ai_act.pdf"}]

    class TextChunks:
        async def get_by_ids(self, ids):
            return [{"_id": i, "page_start": 42, "page_end": 43} for i in ids]

    rag = types.SimpleNamespace(
        chunk_entity_relation_graph=None, entities_vdb=None, relationships_vdb=None,
        chunks_vdb=ChunksVdb(), text_chunks=TextChunks(),
        tokenizer=Tokenizer("bytes"), max_total_tokens=100_000,
    )
    result = asyncio.run(NaiveRetrieval(max_total_tokens=100_000).retrieve("q", rag))

    assert '"page_start": 42' in result.context, result.context
    assert '"page_start": 42' in json.dumps(result.raw_data), result.raw_data


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} passed")
