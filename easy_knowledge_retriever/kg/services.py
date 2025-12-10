from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, Set

from easy_knowledge_retriever.config.global_config import GlobalConfig
from easy_knowledge_retriever.kg.base import BaseKVStorage, BaseVectorStorage, BaseGraphStorage, DocStatusStorage
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
    def __init__(self, global_config: GlobalConfig | Dict[str, Any], storage_name: str | None = None, storage_cls: Any = None, workspace: str = "", **storage_kwargs):
        self.global_config = global_config if isinstance(global_config, dict) else global_config.to_dict()
        self.workspace = workspace
        self.storage_kwargs = storage_kwargs
        
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
    def create(self, namespace: str, embedding_func: EmbeddingFunc | None = None) -> BaseKVStorage:
        return self.storage_cls(
            namespace=namespace,
            workspace=self.workspace,
            working_dir=self.global_config.get("working_dir"),
            embedding_func=embedding_func,
            **self.storage_kwargs
        )

class VectorStorageService(BaseVectorStorageService, GenericStorageServiceMixin):
    def __init__(
        self, 
        global_config: GlobalConfig | Dict[str, Any], 
        storage_name: str | None = None,
        storage_cls: Any = None,
        workspace: str = "", 
        cosine_better_than_threshold: float = 0.2,
        **storage_kwargs
    ):
        super().__init__(global_config, storage_name, storage_cls, workspace, **storage_kwargs)
        self.cosine_better_than_threshold = cosine_better_than_threshold

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
        dim = embedding_dim if embedding_dim is not None else 1536

        return self.storage_cls(
            namespace=namespace,
            workspace=self.workspace,
            working_dir=self.global_config.get("working_dir"),
            embedding_func=embedding_func,
            meta_fields=fields,
            cosine_better_than_threshold=threshold,
            embedding_dim=dim,
            **self.storage_kwargs
        )

class GraphStorageService(BaseGraphStorageService, GenericStorageServiceMixin):
    def create(self, namespace: str, embedding_func: EmbeddingFunc | None = None) -> BaseGraphStorage:
        return self.storage_cls(
            namespace=namespace,
            workspace=self.workspace,
            working_dir=self.global_config.get("working_dir"),
            embedding_func=embedding_func,
            **self.storage_kwargs
        )

class DocStatusStorageService(BaseDocStatusStorageService, GenericStorageServiceMixin):
    def create(self, namespace: str, embedding_func: EmbeddingFunc | None = None) -> DocStatusStorage:
        return self.storage_cls(
            namespace=namespace,
            workspace=self.workspace,
            working_dir=self.global_config.get("working_dir"),
            embedding_func=embedding_func,
            **self.storage_kwargs
        )
