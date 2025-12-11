import asyncio
import os
import sys
# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from easy_knowledge_retriever import EasyKnowledgeRetriever, QueryParam
from easy_knowledge_retriever.retrieval.mix import MixRetrieval
from easy_knowledge_retriever.llm.service import OpenAILLMService, OpenAIEmbeddingService


from easy_knowledge_retriever.kg.kv_storage.json_kv_impl import JsonKVStorage
from easy_knowledge_retriever.kg.vector_storage.nano_vector_db_impl import NanoVectorDBStorage
from easy_knowledge_retriever.kg.graph_storage.networkx_impl import NetworkXStorage
from easy_knowledge_retriever.kg.kv_storage.json_doc_status_impl import JsonDocStatusStorage


async def main():
    # LLM Configuration
    llm_api_key = "AIzaSyCXNwdAhQ8D39yntVVXXrMcLhYWVtnCrpE"
    llm_base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
    llm_model = "gemini-2.5-flash-lite"

    # Embedding Configuration
    embedding_api_key = "AIzaSyCXNwdAhQ8D39yntVVXXrMcLhYWVtnCrpE"
    embedding_base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
    embedding_model = "gemini-embedding-001"
    embedding_dim = 3072

    # Directory setup
    working_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

    if not os.path.exists(working_dir):
        print(f"Warning: Data directory {working_dir} does not exist. Please run build_index.py first.")
        os.makedirs(working_dir)

    print(f"Initializing RAG with existing storage in: {working_dir}")

    # Initialize EasyKnowledgeRetriever
    # Services
    embedding_service = OpenAIEmbeddingService(
        api_key=embedding_api_key,
        base_url=embedding_base_url,
        model=embedding_model,
        embedding_dim=embedding_dim
    )


    
    llm_service = OpenAILLMService(
        model=llm_model,
        api_key=llm_api_key,
        base_url=llm_base_url
    )

    # Initialize EasyKnowledgeRetriever
    rag = EasyKnowledgeRetriever(
        working_dir=working_dir,
        llm_service=llm_service,
        embedding_service=embedding_service,
        kv_storage=JsonKVStorage(working_dir=working_dir),
        vector_storage=NanoVectorDBStorage(working_dir=working_dir, cosine_better_than_threshold=0.2),
        graph_storage=NetworkXStorage(working_dir=working_dir),
        # For Neo4j:
        # graph_storage=Neo4jStorage(uri="bolt://...", ...),
        doc_status_storage=JsonDocStatusStorage(working_dir=working_dir),
    )

    await rag.initialize_storages()
    
    try:
        # Perform the query using existing data
        # Note: We rely on data having been inserted by build_index.py
        # If 'que peux tu me dire sur les feux de foret' query is intended, 
        # ensure build_index.py put something relevant or just test general retrieval.
        query_text = "Crée un QCM de 20 questions avec les justifications concernant la prise en charge d'un accouchement."

        print(f"\nQuerying: '{query_text}'")
        
        # Use Mix retrieval strategy
        # param = QueryParam(mode="mix",only_need_prompt=False )
        retrieval = MixRetrieval()
        
        # We can still pass param for generation settings if needed, or rely on defaults
        # But aquery uses retrieval for retrieval strategy.
        result = await rag.aquery(query_text, retrieval=retrieval)
        print("\nResult:")
        print(result)
        
    finally:
        await rag.finalize_storages()

if __name__ == "__main__":
    asyncio.run(main())
