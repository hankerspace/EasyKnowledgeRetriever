from easy_knowledge_retriever.retrieval.base import BaseRetrieval
from easy_knowledge_retriever.kg.base import QueryParam
from easy_knowledge_retriever.reranker.base import BaseRerankerService

class RetrievalFactory:
    """Factory to create retrieval strategies from QueryParam."""
    
    @staticmethod
    def create_retrieval(param: QueryParam, reranker_service: BaseRerankerService = None) -> BaseRetrieval:
        from easy_knowledge_retriever.retrieval.naive import NaiveRetrieval
        from easy_knowledge_retriever.retrieval.local import LocalRetrieval
        from easy_knowledge_retriever.retrieval.kg_global import GlobalRetrieval
        from easy_knowledge_retriever.retrieval.hybrid import HybridRetrieval
        from easy_knowledge_retriever.retrieval.mix import MixRetrieval
        from easy_knowledge_retriever.retrieval.hybrid_mix import HybridMixRetrieval
        from easy_knowledge_retriever.retrieval.bypass import BypassRetrieval

        # Common params
        base_params = {
            "max_total_tokens": param.max_total_tokens,
            "conversation_history": param.conversation_history,
            "reranker_service": reranker_service,
            "query_decomposition": param.query_decomposition,
        }

        if param.mode == "local":
            return LocalRetrieval(
                **base_params,
                top_k=param.top_k,
                max_entity_tokens=param.max_entity_tokens,
                ll_keywords=param.ll_keywords
            )
        elif param.mode == "global":
            return GlobalRetrieval(
                **base_params,
                top_k=param.top_k,
                max_relation_tokens=param.max_relation_tokens,
                hl_keywords=param.hl_keywords
            )
        elif param.mode == "hybrid":
            return HybridRetrieval(
                **base_params,
                top_k=param.top_k,
                max_entity_tokens=param.max_entity_tokens,
                max_relation_tokens=param.max_relation_tokens,
                hl_keywords=param.hl_keywords,
                ll_keywords=param.ll_keywords
            )
        elif param.mode == "mix":
            return MixRetrieval(
                **base_params,
                top_k=param.top_k,
                chunk_top_k=param.chunk_top_k,
                max_entity_tokens=param.max_entity_tokens,
                max_relation_tokens=param.max_relation_tokens,
                hl_keywords=param.hl_keywords,
                ll_keywords=param.ll_keywords,
            )
        elif param.mode == "hybrid_mix":
             return HybridMixRetrieval(
                **base_params,
                top_k=param.top_k,
                chunk_top_k=param.chunk_top_k,
                hl_keywords=param.hl_keywords,
                ll_keywords=param.ll_keywords,
            )
        elif param.mode == "naive":
            # Handle chunk_top_k fallback if needed
            chunk_top_k = param.chunk_top_k if param.chunk_top_k is not None else param.top_k
            return NaiveRetrieval(
                **base_params,
                chunk_top_k=chunk_top_k,
            )
        elif param.mode == "bypass":
            return BypassRetrieval(**base_params)
        else:
            raise ValueError(f"Unknown retrieval mode: {param.mode}")
