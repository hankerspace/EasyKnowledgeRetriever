STORAGE_IMPLEMENTATIONS = {
    "KV_STORAGE": {
        "implementations": [
            "JsonKVStorage",
            "PGKVStorage",
        ],
        "required_methods": ["get_by_id", "upsert"],
    },
    "GRAPH_STORAGE": {
        "implementations": [
            "NetworkXStorage",
            "Neo4JStorage",
            "PGGraphStorage",
            "AGEStorage",
        ],
        "required_methods": ["upsert_node", "upsert_edge"],
    },
    "VECTOR_STORAGE": {
        "implementations": [
            "NanoVectorDBStorage",
            "MilvusVectorDBStorage",
            "PGVectorStorage",
        ],
        "required_methods": ["query", "upsert"],
    },
    "DOC_STATUS_STORAGE": {
        "implementations": [
            "JsonDocStatusStorage",
            "PGDocStatusStorage",
        ],
        "required_methods": ["get_docs_by_status"],
    },
}

# Storage implementation environment variable without default value
STORAGE_ENV_REQUIREMENTS: dict[str, list[str]] = {
    # KV Storage Implementations
    "JsonKVStorage": [],

    "PGKVStorage": ["POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DATABASE"],
    # Graph Storage Implementations
    "NetworkXStorage": [],
    "Neo4JStorage": ["NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD"],


    "AGEStorage": [
        "AGE_POSTGRES_DB",
        "AGE_POSTGRES_USER",
        "AGE_POSTGRES_PASSWORD",
    ],
    "PGGraphStorage": [
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
        "POSTGRES_DATABASE",
    ],
    # Vector Storage Implementations
    "MilvusVectorDBStorage": [
        "MILVUS_URI",
        "MILVUS_DB_NAME",
    ],
    "PGVectorStorage": ["POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DATABASE"],

    # Document Status Storage Implementations
    "JsonDocStatusStorage": [],

}

# Storage implementation module mapping
STORAGES = {
    "NetworkXStorage": "easy_knowledge_retriever.kg.graph_storage.networkx_impl",
    "NanoVectorDBStorage": "easy_knowledge_retriever.kg.vector_storage.nano_vector_db_impl",
    "JsonKVStorage": "easy_knowledge_retriever.kg.kv_storage.json_kv_impl",
    "JsonDocStatusStorage": "easy_knowledge_retriever.kg.kv_storage.json_doc_status_impl",
    "Neo4JStorage": "easy_knowledge_retriever.kg.graph_storage.neo4j_impl",
    "MilvusVectorDBStorage": "easy_knowledge_retriever.kg.vector_storage.milvus_impl",
    "PGKVStorage": "easy_knowledge_retriever.kg.kv_storage.postgres_impl",
    "PGVectorStorage": "easy_knowledge_retriever.kg.vector_storage.postgres_impl",
    "AGEStorage": "easy_knowledge_retriever.kg.age_impl",
    "PGGraphStorage": "easy_knowledge_retriever.kg.graph_storage.postgres_impl",
    "PGDocStatusStorage": "easy_knowledge_retriever.kg.kv_storage.postgres_impl",


}


def verify_storage_implementation(storage_type: str, storage_name: str) -> None:
    """Verify if storage implementation is compatible with specified storage type

    Args:
        storage_type: Storage type (KV_STORAGE, GRAPH_STORAGE etc.)
        storage_name: Storage implementation name

    Raises:
        ValueError: If storage implementation is incompatible or missing required methods
    """
    if storage_type not in STORAGE_IMPLEMENTATIONS:
        raise ValueError(f"Unknown storage type: {storage_type}")

    storage_info = STORAGE_IMPLEMENTATIONS[storage_type]
    if storage_name not in storage_info["implementations"]:
        raise ValueError(
            f"Storage implementation '{storage_name}' is not compatible with {storage_type}. "
            f"Compatible implementations are: {', '.join(storage_info['implementations'])}"
        )
