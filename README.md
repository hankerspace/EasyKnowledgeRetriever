<div align="center">
<img src="docs/logo.png" alt="EKR Logo" width="400">
</div>

<div align="center">
    <h1>Easy Knowledge Retriever - The easiest RAG lib ever</h1>
</div>

[![PyPI - Version](https://img.shields.io/pypi/v/easy-knowledge-retriever.svg)](https://pypi.org/project/easy-knowledge-retriever/) [![Python Versions](https://img.shields.io/pypi/pyversions/easy-knowledge-retriever.svg)](https://pypi.org/project/easy-knowledge-retriever/) [![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/) [![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue.svg)](https://hankerspace.github.io/EasyKnowledgeRetriever/)

**Easy Knowledge Retriever** is a powerful and flexible library for building Retrieval-Augmented Generation (RAG) systems with integrated Knowledge Graph support. It allows you to easily ingest documents, build a structured knowledge base (combining vector embeddings and graph relations), and perform advanced queries using Large Language Models (LLMs).

Full documentation (GitHub Pages): https://hankerspace.github.io/EasyKnowledgeRetriever/

## Features~~~~

- **Hybrid Retrieval**: Combines vector similarity search with knowledge graph exploration for more context-aware answers.
- **Knowledge Graph Integration**: Automatically extracts entities and relationships from your text documents.
- **Modular Storage**: Supports various backends for Key-Value pairs, Vector Stores, and Graph Storage (e.g., JSON, NanoVectorDB, NetworkX, Neo4j, Milvus).
- **LLM Agnostic**: Designed to work with OpenAI-compatible LLM APIs (OpenAI, Gemini via OpenAI adapter, etc.).
- **Async Support**: built with `asyncio` for high-performance ingestion and retrieval.

## Installation

You can install the library via pip:

```bash
pip install easy-knowledge-retriever
```

## Quick Start

This guide will show you how to build a database from PDF documents and then query it.

### 1. Build the Database (Ingestion)

During this step, documents are processed, chunked, embedded, and entities/relations are extracted to build the Knowledge Graph options.

```python
import asyncio
import os
from easy_knowledge_retriever import EasyKnowledgeRetriever
from easy_knowledge_retriever.llm.service import OpenAILLMService, OpenAIEmbeddingService
from easy_knowledge_retriever.kg.json_kv_impl import JsonKVStorage
from easy_knowledge_retriever.kg.nano_vector_db_impl import NanoVectorDBStorage
from easy_knowledge_retriever.kg.networkx_impl import NetworkXStorage
from easy_knowledge_retriever.kg.json_doc_status_impl import JsonDocStatusStorage

async def build_database():
    # 1. Configure Services
    # Replace with your actual API keys and endpoints
    embedding_service = OpenAIEmbeddingService(
        api_key="your-embedding-api-key",
        base_url="https://api.openai.com/v1", # or compatible
        model="text-embedding-3-small",
        embedding_dim=1536
    )

    llm_service = OpenAILLMService(
        model="gpt-4o",
        api_key="your-llm-api-key",
        base_url="https://api.openai.com/v1"
    )

    # 2. Initialize Retriever with specific storage backends
    working_dir = "./rag_data"
    rag = EasyKnowledgeRetriever(
        working_dir=working_dir,
        llm_service=llm_service,
        embedding_service=embedding_service,
        kv_storage=JsonKVStorage(),
        vector_storage=NanoVectorDBStorage(cosine_better_than_threshold=0.2),
        graph_storage=NetworkXStorage(),
        doc_status_storage=JsonDocStatusStorage(),
    )

    await rag.initialize_storages()
    
    try:
        # 3. Ingest Documents
        pdf_path = "./documents/example.pdf"
        if os.path.exists(pdf_path):
            print(f"Ingesting {pdf_path}...")
            await rag.ingest(pdf_path)
            print("Ingestion complete.")
        else:
            print("Please provide a valid PDF path.")
            
    finally:
        # Always finalize to save state
        await rag.finalize_storages()

if __name__ == "__main__":
    asyncio.run(build_database())
```

### 2. Retrieve Information (Querying)

Once the database is built, you can query it.

```python
import asyncio
from easy_knowledge_retriever import EasyKnowledgeRetriever, QueryParam
from easy_knowledge_retriever.llm.service import OpenAILLMService, OpenAIEmbeddingService
from easy_knowledge_retriever.kg.json_kv_impl import JsonKVStorage
from easy_knowledge_retriever.kg.nano_vector_db_impl import NanoVectorDBStorage
from easy_knowledge_retriever.kg.networkx_impl import NetworkXStorage
from easy_knowledge_retriever.kg.json_doc_status_impl import JsonDocStatusStorage

async def query_knowledge_base():
    # 1. Re-initialize Services (same config as build)
    embedding_service = OpenAIEmbeddingService(
        api_key="your-embedding-api-key",
        base_url="https://api.openai.com/v1",
        model="text-embedding-3-small",
        embedding_dim=1536
    )
    llm_service = OpenAILLMService(
        model="gpt-4o",
        api_key="your-llm-api-key",
        base_url="https://api.openai.com/v1"
    )

    # 2. Load the existing Retriever
    working_dir = "./rag_data"
    rag = EasyKnowledgeRetriever(
        working_dir=working_dir,
        llm_service=llm_service,
        embedding_service=embedding_service,
        kv_storage=JsonKVStorage(),
        vector_storage=NanoVectorDBStorage(cosine_better_than_threshold=0.2),
        graph_storage=NetworkXStorage(),
        doc_status_storage=JsonDocStatusStorage(),
    )

    await rag.initialize_storages()

    try:
        # 3. Perform a Query
        query_text = "What does the document say about forest fires?"
        
        # 'mix' mode uses both vector search and knowledge graph
        param = QueryParam(mode="mix") 
        
        print(f"Querying: {query_text}")
        result = await rag.aquery(query_text, param=param)
        
        print("\nAnswer:")
        print(result)
        
    finally:
        await rag.finalize_storages()

if __name__ == "__main__":
    asyncio.run(query_knowledge_base())
```

## Advanced Configuration

### Storage Options

You can swap out storage implementations by creating instances of different classes from `easy_knowledge_retriever.kg.*`:

*   **Vector Storage**: `NanoVectorDBStorage` (local, lightweight), `MilvusStorage` (scalable).
*   **Graph Storage**: `NetworkXStorage` (in-memory/json, simple), `Neo4jStorage` (robust graph DB).
*   **KV Storage**: `JsonKVStorage`, `RedisKVStorage` (if available), etc.

Example for Neo4j:

```python
from easy_knowledge_retriever.kg.neo4j_impl import Neo4jStorage

graph_storage = Neo4jStorage(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="password"
)
```

## Service & Configuration Catalog

For a complete, up-to-date list of all services (LLM, Vector/KV/Graph/Doc Status) and their configuration options, see:

- docs/ServiceCatalog.md

## Development

To set up the project for development:

1.  Clone the repository.
2.  Install dependencies: `pip install -r requirements.txt`.
3.  Install the package in editable mode: `pip install -e .`.

### Running Tests

(Instructions for running tests if applicable)
```bash
pytest
```

## RAGAS Evaluation

The provided evaluation example yields an average `answer_relevancy` of 0.78 with Gemini 2.0 Flash Lite, and 0.81 with Gemini 2.5 Flash Lite. This indicates a strong correlation with the model used for knowledge graph generation and retrieval.

## References

This project draws inspiration and references from the following projects:

- [LightRAG](https://github.com/HKUDS/LightRAG)
- [RAG-Anything](https://github.com/HKUDS/RAG-Anything)
- [RagFlow](https://github.com/infiniflow/ragflow)
- [Rag-Stack](https://github.com/finic-ai/rag-stack)

## License

This project is licensed under the Creative Commons Attribution–NonCommercial–ShareAlike 4.0 International (CC BY‑NC‑SA 4.0).

- You must give appropriate credit, provide a link to the license, and indicate if changes were made.
- You may not use the material for commercial purposes.
- If you remix, transform, or build upon the material, you must distribute your contributions under the same license as the original.

Full legal text: https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode  
Summary (EN): https://creativecommons.org/licenses/by-nc-sa/4.0/
