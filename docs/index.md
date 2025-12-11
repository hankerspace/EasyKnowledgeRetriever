---
title: Easy Knowledge Retriever
---

# Easy Knowledge Retriever

Powerful and flexible library for building Retrieval-Augmented Generation (RAG) systems with integrated Knowledge Graph support.

- Hybrid retrieval: vectors + knowledge graph
- Modular storage backends (JSON, NanoVectorDB, NetworkX, Neo4j, Milvus, Postgres)
- OpenAI-compatible LLMs and embeddings
- Async-first ingestion and querying

## Installation

```bash
pip install easy-knowledge-retriever
```

## Quick Start

Build a local knowledge base and query it using JSON/Nano/NetworkX backends.

```python
import asyncio
from easy_knowledge_retriever import EasyKnowledgeRetriever, QueryParam
from easy_knowledge_retriever.llm.service import OpenAILLMService, OpenAIEmbeddingService
from easy_knowledge_retriever.kg.json_kv_impl import JsonKVStorage
from easy_knowledge_retriever.kg.nano_vector_db_impl import NanoVectorDBStorage
from easy_knowledge_retriever.kg.networkx_impl import NetworkXStorage
from easy_knowledge_retriever.kg.json_doc_status_impl import JsonDocStatusStorage

async def main():
    embedding = OpenAIEmbeddingService(
        model="text-embedding-3-small",
        base_url="https://api.openai.com/v1",
        api_key="...",
        embedding_dim=1536,
    )
    llm = OpenAILLMService(
        model="gpt-4o",
        base_url="https://api.openai.com/v1",
        api_key="...",
    )

    rag = EasyKnowledgeRetriever(
        working_dir="./rag_data",
        llm_service=llm,
        embedding_service=embedding,
        kv_storage=JsonKVStorage(),
        vector_storage=NanoVectorDBStorage(cosine_better_than_threshold=0.2),
        graph_storage=NetworkXStorage(),
        doc_status_storage=JsonDocStatusStorage(),
    )

    await rag.initialize_storages()
    try:
        await rag.ingest("./documents/example.pdf")
        result = await rag.aquery("What does the document say about forest fires?", param=QueryParam(mode="mix"))
        print(result.content)
    finally:
        await rag.finalize_storages()

if __name__ == "__main__":
    asyncio.run(main())
```

## Service & Configuration Catalog

See the complete catalog of services, implementations, options, and environment variables:

- Service & Configuration Catalog: ServiceCatalog.md

## Links

- Source code on GitHub: https://github.com/hankerspace/EasyKnowledgeRetriever
- License: CC BY-NC-SA 4.0