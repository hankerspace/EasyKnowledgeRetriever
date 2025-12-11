import asyncio
import os
import sys
from functools import partial

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from easy_knowledge_retriever import EasyKnowledgeRetriever
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

    # Directories
    working_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    pdf_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pdf")

    if not os.path.exists(working_dir):
        os.makedirs(working_dir)
        
    if not os.path.exists(pdf_dir):
        print(f"PDF directory not found: {pdf_dir}")
        print(f"Creating it... Please put your .pdf files there.")
        os.makedirs(pdf_dir)
        return

    print(f"Building RAG Index in: {working_dir}")
    print(f"Reading PDFs from: {pdf_dir}")

    # Initialize EasyKnowledgeRetriever with JsonKVStorage explicitly
    # Initialize Services
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

    # Initialize EasyKnowledgeRetriever with JsonKVStorage explicitly
    rag = EasyKnowledgeRetriever(
        working_dir=working_dir,
        llm_service=llm_service,
        embedding_service=embedding_service,
        kv_storage=JsonKVStorage(working_dir=working_dir),
        vector_storage=NanoVectorDBStorage(working_dir=working_dir),
        graph_storage=NetworkXStorage(working_dir=working_dir),
        doc_status_storage=JsonDocStatusStorage(working_dir=working_dir),
    )

    await rag.initialize_storages()
    
    try:
        # Collect documents from PDF folder
        pdf_files = [f for f in os.listdir(pdf_dir) if f.casefold().endswith('.pdf')]
        
        if not pdf_files:
            print("No PDF files found in the directory.")
        
        for filename in pdf_files:
            file_path = os.path.join(pdf_dir, filename)
            print(f"Processing {filename}...")
            
            try:
                # Ingest document
                await rag.ingest(file_path)
                print(f"  - Successfully ingested {filename}")
                    
            except Exception as e:
                print(f"  - Error processing {filename}: {e}")
                import traceback
                traceback.print_exc()
        
    finally:
        await rag.finalize_storages()

if __name__ == "__main__":
    asyncio.run(main())
