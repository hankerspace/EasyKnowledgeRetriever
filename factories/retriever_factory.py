from typing import Optional, Dict, Any, Callable
from .base import BaseFactory
from config.global_config import GlobalConfig
from config.llm_config import LLMConfig
from config.embedding_config import EmbeddingConfig
from factories.storage_factory import StorageFactory
from easy_knowledge_retriever import EasyKnowledgeRetriever
from llm.utils import EmbeddingFunc
from functools import partial

class RetrieverFactory(BaseFactory[EasyKnowledgeRetriever]):
    def create(
        self,
        global_config: GlobalConfig,
        llm_config: LLMConfig,
        embedding_config: EmbeddingConfig,
        storage_factory: Optional[StorageFactory] = None
    ) -> EasyKnowledgeRetriever:
        
        # This factory method insulates the user from manual EKR instantiation
        # It handles the wiring of configs and components
        
        if storage_factory is None:
            storage_factory = StorageFactory(global_config)
            
        # We need to bridge the gap between new config usage and existing EKR init
        # Ideally EKR would be refactored to take these configs directly.
        # For now, we unpack into the constructor which we will update shortly.
        
        # Prepare params
        params = global_config.to_dict()
        
        # Prepare LLM func
        # If llm_config has model_func, use it, otherwise we might need to create it
        # EKR expects model_func to be passed or it uses defaults. 
        # Here we assume the user has set up the configs correctly or we use EKR defaults.
        
        # Prepare Embedding Func
        # EKR expects embedding_func object
        embedding_func = embedding_config.embedding_func
        
        # Instantiate
        # Note: We will modify EKR __init__ to accept these configs optionally
        # Or we pass strict args. 
        
        return EasyKnowledgeRetriever(
            working_dir=global_config.working_dir,
            workspace=global_config.workspace,
            kv_storage=global_config.kv_storage,
            vector_storage=global_config.vector_storage,
            graph_storage=global_config.graph_storage,
            doc_status_storage=global_config.doc_status_storage,
            llm_model_func=llm_config.model_func,
            llm_model_name=llm_config.model_name,
            embedding_func=embedding_func,
            llm_config=llm_config.api_config, # Passing dict for backward compat
            # We will inject the configs directly into EKR in the next step
            # _global_config=global_config,
            # _llm_config=llm_config, 
            # _embedding_config=embedding_config
        )
