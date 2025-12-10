# Developer Guide: Factories and Configurations

This project now uses the **Abstract Factory Pattern** to manage configurations and storage backends. This approach improves modularity, testability, and flexibility.

## Overview

The core components are:
*   **Configs** (`config/`): Dataclasses that hold configuration parameters (e.g., `GlobalConfig`, `LLMConfig`).
*   **Factories** (`factories/`): Classes responsible for creating instances of storages and the main `EasyKnowledgeRetriever` objects.
*   **EasyKnowledgeRetriever**: The main entry point, which now consumes `GlobalConfig` and uses `StorageFactory` internally (or injected).

## Configurations

Instead of passing massive dictionaries or kwargs, use specific Config objects.

### GlobalConfig
Controls general settings like working directory, storage types, and query parameters.
```python
from config.global_config import GlobalConfig

global_config = GlobalConfig(
    working_dir="./my_data",
    kv_storage="JsonKVStorage",
    vector_storage="MilvusVectorDBStorage",
    # ...
)
```

### LLMConfig
Encapsulates LLM settings.
```python
from config.llm_config import LLMConfig
from llm.service import OpenAILLMService

llm_service = OpenAILLMService(
    model="gpt-4o",
    api_key="...",
    base_url="..."
)

llm_config = LLMConfig(
    model_func=llm_service,
    model_name="gpt-4o",
    api_config={"api_key": "..."}
)
```

### EmbeddingConfig
```python
from config.embedding_config import EmbeddingConfig
from llm.service import OpenAIEmbeddingService
from llm.utils import EmbeddingFunc

embedding_service = OpenAIEmbeddingService(
    model="text-embedding-3-small",
    api_key="..."
)

# Wrapping for dimension check/consistency
embedding_func = EmbeddingFunc(
    embedding_dim=1536,
    func=embedding_service,
    send_dimensions=True
)

embedding_config = EmbeddingConfig(embedding_func=embedding_func)
```

## using Factories

### RetrieverFactory
The recommended way to instantiate `EasyKnowledgeRetriever`.

```python
from factories.retriever_factory import RetrieverFactory

factory = RetrieverFactory()
rag = factory.create(
    global_config=global_config,
    llm_config=llm_config,
    embedding_config=embedding_config
)
```

### StorageFactory
Used internally by `EasyKnowledgeRetriever`, but can be used directly if you need to create storage instances manually.

```python
from factories.storage_factory import StorageFactory

storage_factory = StorageFactory(global_config)
kv_store = storage_factory.create_kv_storage("JsonKVStorage", namespace="test")
```

## Adding New Storage Types
1. Implement the storage class in `kg/`.
2. Register it in `kg/registry.py`.
3. Validated that `StorageFactory` can load it (it uses the registry dynamically).

## Backward Compatibility
`EasyKnowledgeRetriever` still accepts legacy kwargs in `__init__`, but it is recommended to migrate to the Factory pattern.
