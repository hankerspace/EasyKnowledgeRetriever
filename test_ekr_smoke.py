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


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} passed")
