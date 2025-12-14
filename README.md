<div align="center">
<img src="docs/logo.png" alt="EKR Logo" width="400">
</div>

<div align="center">
    <h1>Easy Knowledge Retriever - The easiest RAG lib ever</h1>
</div>

[![PyPI - Version](https://img.shields.io/pypi/v/easy-knowledge-retriever.svg)](https://pypi.org/project/easy-knowledge-retriever/) [![Python Versions](https://img.shields.io/pypi/pyversions/easy-knowledge-retriever.svg)](https://pypi.org/project/easy-knowledge-retriever/) [![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/) [![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue.svg)](https://hankerspace.github.io/EasyKnowledgeRetriever/)

**Easy Knowledge Retriever** is a powerful and flexible library for building Retrieval-Augmented Generation (RAG) systems with integrated Knowledge Graph support. It allows you to easily ingest documents, build a structured knowledge base (combining vector embeddings and graph relations), and perform advanced queries using Large Language Models (LLMs).

![Global Flow](docs/global.png)

## Features

- **Multimodal Ingestion**: Parse & ingest PDF data containing images, tables, equations, ... Based on MinerU.
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
from easy_knowledge_retriever import EasyKnowledgeRetriever
from easy_knowledge_retriever.retrieval import MixRetrieval
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
        # Use MixRetrieval strategy
        
        print(f"Querying: {query_text}")
        result = await rag.aquery(query_text, retrieval=MixRetrieval())
        
        print("\nAnswer:")
        print(result)
        
    finally:
        await rag.finalize_storages()

if __name__ == "__main__":
    asyncio.run(query_knowledge_base())
```

## PDF Ingestion with Mineru (Images & Complex Layouts)

Easy Knowledge Retriever integrates **Mineru** (based on magic-pdf) to handle complex PDF documents, preserving layouts and extracting images.

### Ingestion Pipeline Details

The ingestion process orchestrates several advanced steps to transform raw documents into a rich knowledge base:

![Ingestion Flow](docs/ingest.png)

1.  **Parsing with Mineru**: The system uses **Mineru** (based on *Magic-PDF*) to extract text, tables, and images with high structural fidelity.
2.  **Multimodal Enrichment**: Extracted images are processed by a Vision Language Model (VLM, e.g., GPT-4o). The VLM generates descriptive summaries which are injected directly into the text context, making visual data searchable.
3.  **Page-Aware Chunking**: The text is split into chunks using a sliding window approach that preserves the mapping to original page numbers for precise citations.
4.  **Knowledge Graph Extraction**: An LLM extracts entities and relationships from the chunks. It performs iterative gleaning to ensure no details are missed, building a structured graph of knowledge alongside vector embeddings.

### Prerequisites
Ensure that the `mineru` dependencies are installed (included in `requirements.txt`) and that you are using a Vision-capable LLM model.

### Example
The usage remains simple:

```python
# ... Initialize RAG as shown in Quick Start ...

# Ingest a complex PDF with images
await rag.ingest("./documents/complex_report_with_charts.pdf")

# The system automatically handles image extraction and summarization.
```

## Retrieval Workflow

The retrieval process employs a hybrid strategy that orchestrates parallel searches to capture both semantic meaning and explicit knowledge connections.

![Retrieval Workflow](docs/retrieve.png)

1.  **Keyword Extraction**: An LLM extracts both high-level concepts (for thematic search) and low-level entities (for specific details) from the user query.
2.  **Parallel Search**:
    *   **Local Search**: Navigates the Knowledge Graph using low-level entities to find direct neighbors and details.
    *   **Global Search**: Explores broader relationships in the Knowledge Graph using high-level concepts.
    *   **Vector Search**: Finds semantically similar text chunks from the vector database using the query embedding.
3.  **Fusion & Context Building**: Results from all sources are merged, deduplicated, and mapped back to their original source text chunks. This comprehensive context is then provided to the LLM to generate an accurate, grounded response.

## Available Retrieval Strategies

Easy Knowledge Retriever offers flexible retrieval strategies to suit different use cases:

- **Naive** (`naive`): Standard Vector Search on text chunks. Best for simple exact matches.
- **Local** (`local`): Entity-focused Graph Retrieval. Best for specific details about entities.
- **Global** (`global`): Relation-focused Graph Retrieval. Best for broad thematic questions.
- **Hybrid** (`hybrid`): Combines Local and Global Graph Retrieval.
- **Mix** (`mix`): Combines Graph (Hybrid) and Vector (Naive). **Recommended default** for best performance.
- **HybridMix** (`hybrid_mix`): Advanced Chunk Search combining Dense (Vector) and Sparse (BM25) search with Fusion.

For detailed workflows and comparisons, see [Retrieval Strategies Documentation](docs/RetrievalStrategies.md).

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

## Evaluation

The system is evaluated using the RAGAS framework on two distinct datasets to assess performance across different domains and document types.

### Dataset 1: Text Files (General Knowledge)

This dataset consists of plain text files covering general topics such as **Forest Fires** and **Childbirth**. It tests the system's ability to retrieve information from unstructured text.

**Results (Dataset 1):**

*   **Faithfulness: 0.99**
    The Faithfulness metric measures how factually consistent a response is with the retrieved context. It ranges from 0 to 1, with higher scores indicating better consistency.

*   **Context Recall: 1.0**
    Context Recall measures how many of the relevant documents (or pieces of information) were successfully retrieved. It focuses on not missing important results. Higher recall means fewer relevant documents were left out. In short, recall is about not missing anything important.

*   **Answer Relevancy: 0.78 (Gemini 2.0 Flash Lite) / 0.81 (Gemini 2.5 Flash Lite)**
    Answer Relevancy focuses on assessing how pertinent the generated answer is to the given prompt. A lower score is assigned to answers that are incomplete or contain redundant information, and higher scores indicate better relevancy.

### Dataset 2: Scientific PDFs (Technical Domain)

This dataset comprises scientific research papers in PDF format, specifically focusing on **Deep Reinforcement Learning (DRL) for Autonomous Vehicle Intersection Management**. It evaluates the system's capability to handle complex documents, scientific terminology, and multi-modal content.

**Results (Dataset 2):**

*   **Faithfulness: 1.0** (in both cases)
*   **Context Recall: 1.0** (in both cases)
*   **Answer Relevancy**:
    *   **0.92** with Gemini 2.5 Flash Lite (Standard Retrieval without reranker).
    *   **0.96** with Gemini 2.5 Flash Lite using a **Reranker** and **HybridMixRetrieval** (combining Hybrid Vector Search + Knowledge Graph + Query Decomposition).

## References

This project draws inspiration and references from the following projects:

- [LightRAG](https://github.com/HKUDS/LightRAG)
- [RAG-Anything](https://github.com/HKUDS/RAG-Anything)
- [RagFlow](https://github.com/infiniflow/ragflow)
- [Rag-Stack](https://github.com/finic-ai/rag-stack)
- [Mineru](https://github.com/opendatalab/MinerU)
- [RAGas](https://github.com/vibrantlabsai/ragas)

## License

This project is licensed under the Creative Commons Attribution–NonCommercial–ShareAlike 4.0 International (CC BY‑NC‑SA 4.0).

- You must give appropriate credit, provide a link to the license, and indicate if changes were made.
- You may not use the material for commercial purposes.
- If you remix, transform, or build upon the material, you must distribute your contributions under the same license as the original.

Full legal text: https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode  
Summary (EN): https://creativecommons.org/licenses/by-nc-sa/4.0/
