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

## 2. How Implementations Are Selected

The registry (easy_knowledge_retriever.kg.registry) maps class names to modules:
- STORAGES: name → module path
- STORAGE_IMPLEMENTATIONS: valid implementations per storage type
- STORAGE_ENV_REQUIREMENTS: environment variables required by certain implementations

You typically construct a service with either `storage_name` (string) or `storage_cls` (class) and then call `create(namespace=...)` to obtain a storage instance.

Example (Vector storage with NanoVectorDB):
```python
from easy_knowledge_retriever.kg.services import VectorStorageService
from easy_knowledge_retriever.config.global_config import GlobalConfig

g = GlobalConfig(working_dir="./rag_storage", workspace="demo")
vectors = VectorStorageService(
    global_config=g,
    storage_name="NanoVectorDBStorage",
    workspace=g.workspace,
    cosine_better_than_threshold=0.2,
)
vstore = vectors.create(
    namespace="entities",
    embedding_func=my_embedding_func,
    meta_fields={"file_path", "created_at"},
    embedding_dim=1536,
)
```

## 3. LLM Services

Module: easy_knowledge_retriever.llm.service

### 3.1 OpenAILLMService
- Purpose: Text generation / summaries / reasoning via an OpenAI‑compatible API.
- Constructor parameters:
  - model (str): Model name (e.g., "gpt-4o", "gpt-4o-mini").
  - base_url (str | None): API base URL (OpenAI or compatible gateway).
  - api_key (str | None): API key.
  - temperature (float): Default sampling temperature. Default: 1.0.
  - max_async (int): Max concurrent requests. Default: 1.
  - timeout (int): Request timeout (ms). Default: 1_000_000.
  - summary_max_tokens (int): Max tokens in summarization. Default: 1200.
  - summary_context_size (int): Context window used for summarization. Default: 12000.
  - summary_length_recommended (int): Target summary length. Default: 600.

Usage:
```python
from easy_knowledge_retriever.llm.service import OpenAILLMService
llm = OpenAILLMService(model="gpt-4o", base_url="https://api.openai.com/v1", api_key="...")
# inside an async function:
# text = await llm("Summarize this text ...")
```

### 3.2 OpenAIEmbeddingService
- Purpose: Generate embeddings via an OpenAI‑compatible API.
- Constructor parameters:
  - model (str)
  - base_url (str | None)
  - api_key (str | None)
  - batch_num (int): Default: 1.
  - max_async (int): Default: 1.
  - timeout (int): Default: 1_000_000.
  - embedding_dim (int): Vector dimension. Default: 1536.
  - cache_config (dict | None)

Usage:
```python
from easy_knowledge_retriever.llm.service import OpenAIEmbeddingService
embed = OpenAIEmbeddingService(model="text-embedding-3-small", base_url="https://api.openai.com/v1", api_key="...")
# inside an async function:
# vectors = await embed(["hello", "world"])  # returns np.ndarray
```

## 4. Storage Services and Implementations

Module: easy_knowledge_retriever.kg.services

All storage services share base dataclass fields through `Base*Storage` classes, such as:
- namespace (str)
- working_dir (str)
- workspace (str)
- embedding_func (callable | None)
Additional fields exist for vector stores (e.g., `embedding_dim`, `cosine_better_than_threshold`, `meta_fields`).

### 4.1 KV Storage
Service: `KVStorageService`

Implementations:
- JsonKVStorage (no special env vars)
- PGKVStorage (requires Postgres env)

Constructor (via service.create):
```python
kv = KVStorageService(global_config=g, storage_name="JsonKVStorage", workspace=g.workspace)
kv_store = kv.create(namespace="kv_cache", embedding_func=None)
```

Environment requirements:
- JsonKVStorage: none
- PGKVStorage: POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DATABASE

### 4.2 Vector Storage
Service: `VectorStorageService`

Service options:
- cosine_better_than_threshold (float): default 0.2; can be overridden per `create()`.

Implementations:
- NanoVectorDBStorage (local, lightweight; no special env vars)
- MilvusVectorDBStorage (requires Milvus settings)
- PGVectorStorage (requires Postgres settings)

Constructor (via service.create):
```python
vectors = VectorStorageService(global_config=g, storage_name="NanoVectorDBStorage", workspace=g.workspace, cosine_better_than_threshold=0.2)
vstore = vectors.create(namespace="entities", embedding_func=embed_func, meta_fields={"file_path", "created_at"}, embedding_dim=1536)
```

Implementation‑specific constructor notes:
- MilvusVectorDBStorage(namespace, embedding_func, meta_fields, working_dir=None, embedding_dim=None, cosine_better_than_threshold=None, workspace=None, milvus_uri=None, milvus_user=None, milvus_password=None, milvus_token=None, milvus_db_name="default")
  - Honors `MILVUS_WORKSPACE` env var to override `workspace`.
  - Common meta field `created_at` is always included.

Environment requirements:
- NanoVectorDBStorage: none
- MilvusVectorDBStorage: MILVUS_URI, MILVUS_DB_NAME (plus auth if applicable)
- PGVectorStorage: POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DATABASE

### 4.3 Graph Storage
Service: `GraphStorageService`

Implementations:
- NetworkXStorage (JSON on disk; no special env)
- Neo4JStorage (Neo4j driver)
- PGGraphStorage (Postgres + AGE extension)
- AGEStorage (Apache AGE specific)

Selected constructor highlights:
- Neo4JStorage(namespace, embedding_func, working_dir=None, workspace=None, neo4j_uri=None, neo4j_username=None, neo4j_password=None, neo4j_connection_pool_size=100, neo4j_connection_timeout=30.0, neo4j_connection_acquisition_timeout=30.0, neo4j_max_transaction_retry_time=30.0, neo4j_max_connection_lifetime=300.0, neo4j_liveness_check_timeout=30.0, neo4j_keep_alive=True, neo4j_database=None, max_graph_nodes=DEFAULT_MAX_GRAPH_NODES)

Environment requirements:
- NetworkXStorage: none
- Neo4JStorage: NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD
- PGGraphStorage: POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DATABASE
- AGEStorage: AGE_POSTGRES_DB, AGE_POSTGRES_USER, AGE_POSTGRES_PASSWORD

### 4.4 Document Status Storage
Service: `DocStatusStorageService`

Implementations:
- JsonDocStatusStorage (no special env)
- PGDocStatusStorage (Postgres)

Environment requirements:
- JsonDocStatusStorage: none
- PGDocStatusStorage: (Postgres standard vars; see PG entries above)

## 5. Global Configuration (GlobalConfig)

Module: easy_knowledge_retriever.config.global_config

The `GlobalConfig` dataclass centralizes defaults used across ingestion and querying. Key fields and defaults:
- working_dir: "./rag_storage"
- workspace: ""
- top_k: 40
- chunk_top_k: 20
- max_entity_tokens: 6000
- max_relation_tokens: 8000
- max_total_tokens: 30000
- cosine_threshold: 0.2
- related_chunk_number: 5
- kg_chunk_pick_method: "VECTOR"
- entity_extract_max_gleaning: 1
- force_llm_summary_on_merge: 8
- chunk_token_size: 1200
- chunk_overlap_token_size: 100
- tiktoken_model_name: "gpt-4o-mini"
- max_parallel_insert: 1
- max_source_ids_per_entity: 300
- max_source_ids_per_relation: 300
- source_ids_limit_method: "FIFO" (KEEP | FIFO)
- max_file_paths: 100
- file_path_more_placeholder: "truncated"
- language: "French" (default processing language)
- entity_types: ["Person", "Creature", "Organization", "Location", "Event", "Concept", "Method", "Content", "Data", "Artifact", "NaturalObject"]
- enable_llm_cache: True
- enable_llm_cache_for_entity_extract: True

Tip: `GlobalConfig.to_dict()` converts the config into a plain dict for services.

## 6. Putting It All Together

End‑to‑end example using JSON/Nano/NetworkX (fully local):
```python
from easy_knowledge_retriever import EasyKnowledgeRetriever
from easy_knowledge_retriever.llm.service import OpenAILLMService, OpenAIEmbeddingService
from easy_knowledge_retriever.kg.json_kv_impl import JsonKVStorage
from easy_knowledge_retriever.kg.nano_vector_db_impl import NanoVectorDBStorage
from easy_knowledge_retriever.kg.networkx_impl import NetworkXStorage
from easy_knowledge_retriever.kg.json_doc_status_impl import JsonDocStatusStorage
from easy_knowledge_retriever.config.global_config import GlobalConfig

g = GlobalConfig(working_dir="./rag_data", workspace="demo")

llm_service = OpenAILLMService(model="gpt-4o", base_url="https://api.openai.com/v1", api_key="...")
embedding_service = OpenAIEmbeddingService(model="text-embedding-3-small", base_url="https://api.openai.com/v1", api_key="...", embedding_dim=1536)

rag = EasyKnowledgeRetriever(
    working_dir=g.working_dir,
    llm_service=llm_service,
    embedding_service=embedding_service,
    kv_storage=JsonKVStorage(),
    vector_storage=NanoVectorDBStorage(cosine_better_than_threshold=0.2),
    graph_storage=NetworkXStorage(),
    doc_status_storage=JsonDocStatusStorage(),
)
```

Switching to Milvus/Neo4j/Postgres only changes the storage classes and their connection options or environment variables as listed above.

## 7. Troubleshooting

- Unknown storage name: ensure the `storage_name` matches one of the registry keys (see Section 4). The service will raise `ValueError` if the name is not registered.
- Missing environment variables: set the required variables listed for your implementation before starting your app/tests.
- Vector dimension mismatches: set `embedding_dim` in `VectorStorageService.create(...)` to match your embedding model (e.g., 1536 for `text-embedding-3-small`).
