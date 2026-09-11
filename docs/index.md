---
title: Easy Knowledge Retriever
---

# Easy Knowledge Retriever

A Python library for Retrieval-Augmented Generation (RAG) over your own documents, with an integrated knowledge graph.

- Ingest PDFs (MinerU: layout, tables, images) and plain text / Markdown
- Seven retrieval strategies, from plain vector search to dense + BM25 + knowledge graph (`hybrid_mix`)
- Optional cross-encoder reranker (any OpenAI-compatible `/rerank` endpoint)
- Answers cite the source chunk and its page, and refuse questions the corpus does not cover
- Modular storage backends (JSON, NanoVectorDB, NetworkX, Neo4j, Milvus, PostgreSQL)
- OpenAI-compatible LLMs and embeddings, async-first

Need a user interface? [EasyKnowledgeRetriever-Webapp](https://github.com/hankerspace/EasyKnowledgeRetriever-Webapp) is a ready-to-run web app (chat with sources, document management, graph explorer, Docker image) built on this library.

## Installation

```bash
pip install easy-knowledge-retriever          # core (text / Markdown ingestion)
pip install "easy-knowledge-retriever[pdf]"   # + PDF ingestion with MinerU
```

## Quick Start

```python
import asyncio
import os

from easy_knowledge_retriever import EasyKnowledgeRetriever, QueryParam
from easy_knowledge_retriever.llm.service import OpenAILLMService, OpenAIEmbeddingService
from easy_knowledge_retriever.reranker.openai import OpenAIRerankerService
from easy_knowledge_retriever.kg.kv_storage.json_kv_impl import JsonKVStorage
from easy_knowledge_retriever.kg.kv_storage.json_doc_status_impl import JsonDocStatusStorage
from easy_knowledge_retriever.kg.vector_storage.nano_vector_db_impl import NanoVectorDBStorage
from easy_knowledge_retriever.kg.graph_storage.networkx_impl import NetworkXStorage

WORKING_DIR = "./rag_data"


async def main():
    rag = EasyKnowledgeRetriever(
        working_dir=WORKING_DIR,
        llm_service=OpenAILLMService(model="gpt-4o-mini", api_key=os.environ["OPENAI_API_KEY"]),
        embedding_service=OpenAIEmbeddingService(
            model="text-embedding-3-small",
            base_url="https://api.openai.com/v1",
            api_key=os.environ["OPENAI_API_KEY"],
            embedding_dim=1536,
        ),
        # Optional, but strongly recommended
        reranker_service=OpenAIRerankerService(
            model="bge-reranker-v2-m3",
            base_url="https://my-gateway.example.com/v1/rerank",
            api_key=os.environ["RERANKER_API_KEY"],
        ),
        kv_storage=JsonKVStorage(working_dir=WORKING_DIR),
        vector_storage=NanoVectorDBStorage(working_dir=WORKING_DIR),
        graph_storage=NetworkXStorage(working_dir=WORKING_DIR),
        doc_status_storage=JsonDocStatusStorage(working_dir=WORKING_DIR),
    )

    await rag.initialize_storages()
    try:
        await rag.ingest("./documents/report.pdf")
        result = await rag.aquery(
            "What does the report say about forest fires?",
            param=QueryParam(mode="hybrid_mix"),
        )
        print(result.content)
    finally:
        await rag.finalize_storages()


asyncio.run(main())
```

## Documentation

- [Retrieval Strategies](RetrievalStrategies.md): how each mode works, measured quality and latency, which one to pick
- [Service & Configuration Catalog](ServiceCatalog.md): every service, storage backend, option and environment variable

## Links

- Source code: https://github.com/hankerspace/EasyKnowledgeRetriever
- Web app: https://github.com/hankerspace/EasyKnowledgeRetriever-Webapp
- License: CC BY-NC-SA 4.0
