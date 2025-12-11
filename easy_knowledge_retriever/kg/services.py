from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, Set


from easy_knowledge_retriever.kg.kv_storage.base import BaseKVStorage, DocStatusStorage
from easy_knowledge_retriever.kg.vector_storage.base import BaseVectorStorage
from easy_knowledge_retriever.kg.graph_storage.base import BaseGraphStorage
from easy_knowledge_retriever.llm.utils import EmbeddingFunc
from easy_knowledge_retriever.kg.registry import STORAGES
from easy_knowledge_retriever.utils.common_utils import lazy_external_import

class BaseStorageService(ABC):
    """Abstract Base Class for Storage Services."""
    pass

class BaseKVStorageService(BaseStorageService):
    @abstractmethod
    def create(self, namespace: str, embedding_func: EmbeddingFunc | None = None) -> BaseKVStorage:
        """Creates a KV Storage instance for a specific namespace."""
        pass

class BaseVectorStorageService(BaseStorageService):
    @abstractmethod
    def create(
        self, 
        namespace: str, 
        embedding_func: EmbeddingFunc | None = None,
        meta_fields: Set[str] | None = None,
        cosine_better_than_threshold: float | None = None,
        embedding_dim: int | None = None
    ) -> BaseVectorStorage:
        """Creates a Vector Storage instance for a specific namespace."""
        pass

class BaseGraphStorageService(BaseStorageService):
    @abstractmethod
    def create(self, namespace: str, embedding_func: EmbeddingFunc | None = None) -> BaseGraphStorage:
        """Creates a Graph Storage instance for a specific namespace."""
        pass

class BaseDocStatusStorageService(BaseStorageService):
    @abstractmethod
    def create(self, namespace: str, embedding_func: EmbeddingFunc | None = None) -> DocStatusStorage:
        """Creates a Doc Status Storage instance for a specific namespace."""
        pass

# --- Concrete Implementations ---


class GenericStorageServiceMixin:
    """Mixin to handle storage class loading and initialization."""
    def __init__(
        self, 
        storage_name: str | None = None, 
        storage_cls: Any = None, 
        workspace: str = "", 
        working_dir: str = "./rag_storage",
    ):
        self.workspace = workspace
        self.working_dir = working_dir
        
        if storage_cls:
            self.storage_cls = storage_cls
            self.storage_name = storage_cls.__name__
        elif storage_name:
            self.storage_name = storage_name
            self.storage_cls = self._get_storage_class(storage_name)
        else:
            raise ValueError("Either storage_name or storage_cls must be provided")

    def _get_storage_class(self, storage_name: str):
        if storage_name not in STORAGES:
            raise ValueError(f"Unknown storage name: {storage_name}")
            
        module_path = STORAGES[storage_name]
        try:
            if module_path.startswith('.'):
                module_path = module_path[1:]
            
            module = lazy_external_import(module_path)
            return getattr(module, storage_name)
        except ImportError as e:
            raise ImportError(f"Could not import storage {storage_name}: {e}")

class KVStorageService(BaseKVStorageService, GenericStorageServiceMixin):
    def __init__(
        self,
        storage_name: str | None = None,
        storage_cls: Any = None,
        workspace: str = "",
        working_dir: str = "./rag_storage",
        # KV specific params if any (e.g. max_batch_size for PG)
        max_batch_size: int | None = None,
    ):
        super().__init__(storage_name, storage_cls, workspace, working_dir)
        self.max_batch_size = max_batch_size

    def create(self, namespace: str, embedding_func: EmbeddingFunc | None = None) -> BaseKVStorage:
        # Prepare kwargs explicitly
        kwargs = {}
        if self.max_batch_size is not None:
             kwargs["max_batch_size"] = self.max_batch_size

        return self.storage_cls(
            namespace=namespace,
            workspace=self.workspace,
            working_dir=self.working_dir,
            embedding_func=embedding_func,
            **kwargs
        )

class VectorStorageService(BaseVectorStorageService, GenericStorageServiceMixin):
    def __init__(
        self, 
        storage_name: str | None = None,
        storage_cls: Any = None,
        workspace: str = "", 
        working_dir: str = "./rag_storage",
        cosine_better_than_threshold: float = 0.2,
        embedding_dim: int = 1536,
        # Milvus specific
        milvus_uri: str | None = None,
        milvus_user: str | None = None,
        milvus_password: str | None = None,
        milvus_token: str | None = None,
        milvus_db_name: str = "default",
        # Postgres specific
        max_batch_size: int | None = None,
    ):
        super().__init__(
            storage_name=storage_name, 
            storage_cls=storage_cls, 
            workspace=workspace, 
            working_dir=working_dir, 
        )
        self.cosine_better_than_threshold = cosine_better_than_threshold
        self.embedding_dim = embedding_dim
        self.milvus_uri = milvus_uri
        self.milvus_user = milvus_user
        self.milvus_password = milvus_password
        self.milvus_token = milvus_token
        self.milvus_db_name = milvus_db_name
        self.max_batch_size = max_batch_size

    def create(
        self, 
        namespace: str, 
        embedding_func: EmbeddingFunc | None = None,
        meta_fields: Set[str] | None = None,
        cosine_better_than_threshold: float | None = None,
        embedding_dim: int | None = None
    ) -> BaseVectorStorage:
        
        # Use provided threshold or fallback to service default
        threshold = cosine_better_than_threshold if cosine_better_than_threshold is not None else self.cosine_better_than_threshold
        fields = meta_fields or set()
        dim = embedding_dim if embedding_dim is not None else self.embedding_dim

        # Prepare kwargs explicitly based on what's set
        kwargs = {}
        if self.milvus_uri is not None:
            kwargs["milvus_uri"] = self.milvus_uri
        if self.milvus_user is not None:
            kwargs["milvus_user"] = self.milvus_user
        if self.milvus_password is not None:
            kwargs["milvus_password"] = self.milvus_password
        if self.milvus_token is not None:
            kwargs["milvus_token"] = self.milvus_token
        if self.milvus_db_name != "default":
             kwargs["milvus_db_name"] = self.milvus_db_name
        if self.max_batch_size is not None:
             kwargs["max_batch_size"] = self.max_batch_size

        return self.storage_cls(
            namespace=namespace,
            workspace=self.workspace,
            working_dir=self.working_dir,
            embedding_func=embedding_func,
            meta_fields=fields,
            cosine_better_than_threshold=threshold,
            embedding_dim=dim,
            **kwargs
        )

class GraphStorageService(BaseGraphStorageService, GenericStorageServiceMixin):
    def __init__(
        self,
        storage_name: str | None = None,
        storage_cls: Any = None,
        workspace: str = "",
        working_dir: str = "./rag_storage",
        # Graph specific
        max_graph_nodes: int | None = None,
        # Neo4j specific
        neo4j_uri: str | None = None,
        neo4j_username: str | None = None,
        neo4j_password: str | None = None,
        neo4j_database: str | None = None,
    ):
        super().__init__(storage_name, storage_cls, workspace, working_dir)
        self.max_graph_nodes = max_graph_nodes
        self.neo4j_uri = neo4j_uri
        self.neo4j_username = neo4j_username
        self.neo4j_password = neo4j_password
        self.neo4j_database = neo4j_database

    def create(self, namespace: str, embedding_func: EmbeddingFunc | None = None) -> BaseGraphStorage:
        kwargs = {}
        if self.max_graph_nodes is not None:
            kwargs["max_graph_nodes"] = self.max_graph_nodes
        if self.neo4j_uri is not None:
            kwargs["neo4j_uri"] = self.neo4j_uri
        if self.neo4j_username is not None:
            kwargs["neo4j_username"] = self.neo4j_username
        if self.neo4j_password is not None:
            kwargs["neo4j_password"] = self.neo4j_password
        if self.neo4j_database is not None:
            kwargs["neo4j_database"] = self.neo4j_database

        return self.storage_cls(
            namespace=namespace,
            workspace=self.workspace,
            working_dir=self.working_dir,
            embedding_func=embedding_func,
            **kwargs
        )

class DocStatusStorageService(BaseDocStatusStorageService, GenericStorageServiceMixin):
    def __init__(
        self,
        storage_name: str | None = None,
        storage_cls: Any = None,
        workspace: str = "",
        working_dir: str = "./rag_storage",
        # Docs specific
    ):
        super().__init__(storage_name, storage_cls, workspace, working_dir)

    def create(self, namespace: str, embedding_func: EmbeddingFunc | None = None) -> DocStatusStorage:
        return self.storage_cls(
            namespace=namespace,
            workspace=self.workspace,
            working_dir=self.working_dir,
            embedding_func=embedding_func,
        )

