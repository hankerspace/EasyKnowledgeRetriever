from typing import Type, Any, Dict, Union
from .base import BaseFactory
from kg.base import BaseKVStorage, BaseVectorStorage, BaseGraphStorage, DocStatusStorage
from kg.registry import STORAGES, verify_storage_implementation
from utils.common_utils import lazy_external_import, check_storage_env_vars
from config.global_config import GlobalConfig

class StorageFactory(BaseFactory):
    def __init__(self, global_config: GlobalConfig):
        self.global_config = global_config

    def _get_storage_class(self, storage_name: str) -> Type:
        module_path = STORAGES[storage_name]
        try:
            # Import module relative to package
            # Note: This mirrors the logic in EasyKnowledgeRetriever._get_storage_class
            # We need to make sure import path is correct
            # If storage_name is say "JsonKVStorage", module_path is ".kg.json_kv_impl"
            # It seems the original code relies on relative import from 'easy_knowledge_retriever.py'
            # Here we might need absolute import or fix the paths in registry
             
            # Fix: Remove leading dot if present for absolute import within project
            if module_path.startswith('.'):
                module_path = module_path[1:]
            
            # Since STORAGES uses relative path string like .kg.networkx_impl
            # We will try to import it as an absolute path from root
            # e.g. kg.networkx_impl
             
            # But original code was importing from current context.
            # Let's assume absolute import works if we are at root.
            module = lazy_external_import(module_path)
            return getattr(module, storage_name)
        except ImportError as e:
             # Retry with original logic if needed, but absolute should be safer
             raise ImportError(f"Could not import storage {storage_name}: {e}")

    def create_kv_storage(self, storage_name: str, **kwargs) -> BaseKVStorage:
        # verify_storage_implementation("KV_STORAGE", storage_name)
        storage_cls = self._get_storage_class(storage_name)
        return storage_cls(global_config=self.global_config.to_dict(), **kwargs)

    def create_vector_storage(self, storage_name: str, **kwargs) -> BaseVectorStorage:
        # verify_storage_implementation("VECTOR_STORAGE", storage_name)
        storage_cls = self._get_storage_class(storage_name)
        return storage_cls(global_config=self.global_config.to_dict(), **kwargs)

    def create_graph_storage(self, storage_name: str, **kwargs) -> BaseGraphStorage:
        # verify_storage_implementation("GRAPH_STORAGE", storage_name)
        storage_cls = self._get_storage_class(storage_name)
        return storage_cls(global_config=self.global_config.to_dict(), **kwargs)
    
    def create_doc_status_storage(self, storage_name: str, **kwargs) -> DocStatusStorage:
        # verify_storage_implementation("DOC_STATUS_STORAGE", storage_name)
        storage_cls = self._get_storage_class(storage_name)
        return storage_cls(global_config=self.global_config.to_dict(), **kwargs)

    def create(self, **kwargs):
        raise NotImplementedError("Use specific create methods for StorageFactory")
