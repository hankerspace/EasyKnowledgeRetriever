# Easy Knowledge Retriever — Service and Configuration Catalog

Last updated: 2025-12-11

This document lists all services available in Easy Knowledge Retriever (EKR) and explains how to configure them. It covers:
- What each service does
- Available implementations
- Constructor/configuration options and defaults
- Required environment variables per implementation

Note: Class and option names here follow the codebase. Examples are in Python.

## 1. Overview of Services

EKR is composed of the following service layers:

1) LLM Services (easy_knowledge_retriever.llm.service)
- OpenAILLMService: Calls an OpenAI‑compatible Chat/Completions API.
- OpenAIEmbeddingService: Calls an OpenAI‑compatible Embeddings API.

2) Storage Services (easy_knowledge_retriever.kg.services)
- KVStorageService: Key–Value metadata storage (per namespace).
- VectorStorageService: Vector storage for chunks, entities, relations.
- GraphStorageService: Graph storage for entities and relationships.
- DocStatusStorageService: Tracks document processing status.

Each StorageService is a thin factory around a concrete storage implementation (e.g., NanoVectorDBStorage, NetworkXStorage). Implementations are selected by name via an internal registry.

## 2. LLM Services Configuration

Module: `easy_knowledge_retriever.llm.service`

### 2.1 OpenAILLMService
- **Purpose**: Text generation / summaries / reasoning via an OpenAI‑compatible API.
- **Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | str | **Required** | Model name (e.g., "gpt-4o") |
| `base_url` | str | `None` | API base URL (e.g. "https://api.openai.com/v1") |
| `api_key` | str | `None` | API key |
| `temperature` | float | `1.0` | Sampling temperature |
| `max_async` | int | `1` | Max concurrent requests |
| `timeout` | int | `1_000_000` | Request timeout (ms) |
| `summary_max_tokens` | int | `1200` | Max tokens in summarization output |
| `summary_context_size` | int | `12000` | Context window size for summarization |
| `summary_length_recommended` | int | `600` | Target length for summary |

### 2.2 OpenAIEmbeddingService
- **Purpose**: Generate embeddings via an OpenAI‑compatible API.
- **Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | str | **Required** | Model name (e.g. "text-embedding-3-small") |
| `base_url` | str | `None` | API base URL |
| `api_key` | str | `None` | API key |
| `embedding_dim` | int | `1536` | Vector dimension |
| `batch_num` | int | `1` | Batch size for embedding requests |
| `max_async` | int | `1` | Max concurrent requests |
| `timeout` | int | `1_000_000` | Request timeout (ms) |

## 3. Storage Implementations Configuration

Module: `easy_knowledge_retriever.kg`

**Important Note on `working_dir`**: When instantiating storage classes directly (e.g., `JsonKVStorage(working_dir="/data")`), you **MUST** provide the `working_dir` argument if you want data to be persisted in a specific location. If omitted, it defaults to an empty string `""`.

### 3.1 KV Storage (`KVStorageService`)

#### **JsonKVStorage**
- **Type**: Local File (JSON)
- **Env Vars**: None
- **Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `working_dir` | str | `""` | Root directory for data storage |
| `namespace` | str | `None` | Storage namespace (set by `create()`) |
| `workspace` | str | `""` | Workspace name (subdirectory) |
| `embedding_func` | callable | `None` | Not used by KV storage |

#### **PGKVStorage**
- **Type**: PostgreSQL Database
- **Env Vars**: `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DATABASE`, `POSTGRES_HOST`, `POSTGRES_PORT` (managed by global ClientManager)
- **Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `db` | PostgreSQLDB | `None` | Helper db instance (auto-initialized if None) |
| `max_batch_size` | int | `1` | Batch size for ops |
| `working_dir` | str | `""` | Not used (uses DB) |
| `workspace` | str | `""` | Workspace name (controls table filtering) |

---

### 3.2 Vector Storage (`VectorStorageService`)

#### **NanoVectorDBStorage**
- **Type**: Local File (NanoVectorDB)
- **Env Vars**: `NANO_VECTOR_DB_WORKSPACE` (Overwrites `workspace` if set)
- **Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `working_dir` | str | `""` | Root directory for data storage |
| `embedding_dim` | int | `None` | Vector dimension (e.g. 1536) |
| `cosine_better_than_threshold` | float | `None` | Similarity threshold (typ. 0.2) |
| `meta_fields` | set | `set()` | Fields to store alongside vectors |
| `workspace` | str | `""` | Workspace name |
| `namespace` | str | `None` | Storage namespace |

#### **MilvusVectorDBStorage**
- **Type**: Milvus Database
- **Env Vars**: `MILVUS_WORKSPACE` (Overwrites `workspace` if set)
- **Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `milvus_uri` | str | `None` | Connection URI (e.g. "http://localhost:19530") |
| `milvus_token` | str | `None` | Auth token |
| `milvus_user` | str | `None` | Username |
| `milvus_password` | str | `None` | Password |
| `milvus_db_name` | str | `"default"` | Database name |
| `workspace` | str | `""` | Workspace prefix for collections |
| `embedding_dim` | int | `1536` | Vector dimension |
| `cosine_better_than_threshold` | float | `0.2` | Similarity threshold |

#### **PGVectorStorage**
- **Type**: PostgreSQL with pgvector
- **Env Vars**: Standard Postgres vars (see PGKVStorage)
- **Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `db` | PostgreSQLDB | `None` | DB Client |
| `max_batch_size` | int | `1` | Batch size |
| `workspace` | str | `""` | Workspace filter |

---

### 3.3 Graph Storage (`GraphStorageService`)

#### **NetworkXStorage**
- **Type**: Local File (GraphML)
- **Env Vars**: None
- **Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `working_dir` | str | `""` | Root directory for GraphML files |
| `max_graph_nodes` | int | `DEFAULT` | Max nodes returned in BFS searches (default ~1000) |
| `workspace` | str | `""` | Workspace name |

#### **Neo4JStorage**
- **Type**: Neo4j Database
- **Env Vars**: `NEO4J_WORKSPACE` (Overwrites `workspace` if set)
- **Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `neo4j_uri` | str | `None` | Bolt URI (e.g. "bolt://localhost:7687") |
| `neo4j_username` | str | `None` | Username |
| `neo4j_password` | str | `None` | Password |
| `neo4j_database` | str | `None` | Database name |
| `neo4j_connection_pool_size` | int | `100` | Pool size |
| `max_graph_nodes` | int | `DEFAULT` | Max nodes in search |
| `workspace` | str | `"base"` | Workspace name |

#### **PGGraphStorage**
- **Type**: PostgreSQL with Apache AGE
- **Env Vars**: Standard Postgres vars
- **Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `db` | PostgreSQLDB | `None` | DB Client |
| `max_graph_nodes` | int | `DEFAULT` | Max nodes in search |
| `workspace` | str | `""` | Workspace name (determines graph name) |

---

### 3.4 DocStatus Storage (`DocStatusStorageService`)

#### **JsonDocStatusStorage**
- **Type**: Local File (JSON)
- **Env Vars**: None
- **Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `working_dir` | str | `""` | Root directory |
| `workspace` | str | `""` | Workspace name |

#### **PGDocStatusStorage**
- **Type**: PostgreSQL
- **Env Vars**: Standard Postgres vars
- **Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `db` | PostgreSQLDB | `None` | DB Client |
| `workspace` | str | `""` | Workspace filter |

## 4. End-to-End Example

```python
from easy_knowledge_retriever import EasyKnowledgeRetriever
from easy_knowledge_retriever.llm.service import OpenAILLMService, OpenAIEmbeddingService
from easy_knowledge_retriever.kg.kv_storage.json_kv_impl import JsonKVStorage
from easy_knowledge_retriever.kg.vector_storage.nano_vector_db_impl import NanoVectorDBStorage
from easy_knowledge_retriever.kg.graph_storage.networkx_impl import NetworkXStorage
from easy_knowledge_retriever.kg.kv_storage.json_doc_status_impl import JsonDocStatusStorage

# 1. Setup Services
llm_service = OpenAILLMService(
    model="gpt-4o", 
    base_url="https://api.openai.com/v1", 
    api_key="sk-..."
)
embedding_service = OpenAIEmbeddingService(
    model="text-embedding-3-small", 
    base_url="https://api.openai.com/v1", 
    api_key="sk-...", 
    embedding_dim=1536
)

# 2. Initialize Retriever with Explicit Storage Configuration
# Note: Explicit working_dir is required for local storages
rag = EasyKnowledgeRetriever(
    working_dir="./rag_data",
    llm_service=llm_service,
    embedding_service=embedding_service,
    
    # KV Storage
    kv_storage=JsonKVStorage(working_dir="./rag_data"),
    
    # Vector Storage
    vector_storage=NanoVectorDBStorage(
        working_dir="./rag_data", 
        embedding_dim=1536,
        cosine_better_than_threshold=0.2
    ),
    
    # Graph Storage
    graph_storage=NetworkXStorage(working_dir="./rag_data"),
    
    # Doc Status Storage
    doc_status_storage=JsonDocStatusStorage(working_dir="./rag_data"),
)
```
