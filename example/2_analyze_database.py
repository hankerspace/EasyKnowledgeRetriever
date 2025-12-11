import asyncio
import os
import sys
from functools import partial

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from easy_knowledge_retriever import EasyKnowledgeRetriever
from easy_knowledge_retriever.llm.service import BaseLLMService, BaseEmbeddingService
from easy_knowledge_retriever.llm.utils import EmbeddingFunc


# Dummy embedding service
class DummyEmbeddingService(BaseEmbeddingService):
    def __init__(self, embedding_dim: int):
        super().__init__(embedding_dim=embedding_dim)
        
    async def __call__(self, texts: list[str]) -> list[list[float]]:
        # Return dummy vectors
        # Note: EmbeddingFunc wraper expects np.ndarray or list of list
        # We return list of list, wrapper handles it.
        # Although type hint says np.ndarray, list is often accepted by downstream. 
        # But let's return numpy if feasible or list.
        # Actually EmbeddingFunc converts to numpy.
        return [[0.0] * self.embedding_dim for _ in texts]

# Dummy LLM Service
class DummyLLMService(BaseLLMService):
    async def __call__(self, prompt: str, **kwargs) -> str:
        return "Dummy response"

# Old functions removed

from easy_knowledge_retriever.kg.kv_storage.json_kv_impl import JsonKVStorage
from easy_knowledge_retriever.kg.vector_storage.nano_vector_db_impl import NanoVectorDBStorage
from easy_knowledge_retriever.kg.graph_storage.networkx_impl import NetworkXStorage
from easy_knowledge_retriever.kg.kv_storage.json_doc_status_impl import JsonDocStatusStorage


async def main():
    # Configuration
    working_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    embedding_dim = 3072

    print(f"Analyzing RAG data in: {working_dir} with dim {embedding_dim}")

    if not os.path.exists(working_dir):
        print(f"Error: Directory {working_dir} does not exist.")
        return

    # Initialize EasyKnowledgeRetriever
    embedding_service = DummyEmbeddingService(embedding_dim)
    llm_service = DummyLLMService()

    embedding_func = EmbeddingFunc(
        embedding_dim=embedding_dim,
        func=embedding_service,
        send_dimensions=True
    )

    rag = EasyKnowledgeRetriever(
        working_dir=working_dir,
        llm_service=llm_service,
        embedding_service=embedding_service,
        kv_storage=JsonKVStorage(working_dir=working_dir),
        vector_storage=NanoVectorDBStorage(working_dir=working_dir, cosine_better_than_threshold=0.2),
        graph_storage=NetworkXStorage(working_dir=working_dir),
        doc_status_storage=JsonDocStatusStorage(working_dir=working_dir),
    )

    await rag.initialize_storages()
    
    try:
        print("\n" + "="*50)
        print("RAG Data Analysis")
        print("="*50)

        # 1. Documents (KV Storage)
        print("\n--- Documents (KV Storage) ---")
        if hasattr(rag.full_docs, "_data"):
            docs = rag.full_docs._data
            print(f"Total Documents: {len(docs)}")
            for doc_id, doc_data in list(docs.items())[:5]: # Show first 5
                 print(f"  ID: {doc_id}")
                 # doc_data details might vary, let's print keys or raw
                 # print(f"    Keys: {doc_data.keys()}")
                 # Typical fields: content, raw_content, etc.
                 # Let's try to print a snippet
                 content = doc_data.get('content', '')
                 print(f"    Content Snippet: {content[:100]}..." if content else "    No content")
        else:
            print("  [Reference to rag.full_docs._data failed]")

        # 2. Text Chunks (KV Storage)
        print("\n--- Text Chunks (KV Storage) ---")
        if hasattr(rag.text_chunks, "_data"):
            chunks = rag.text_chunks._data
            print(f"Total Text Chunks: {len(chunks)}")
            for chunk_id, chunk_data in list(chunks.items())[:5]:
                print(f"  ID: {chunk_id}")
                content = chunk_data.get('content', '')
                print(f"    Content Snippet: {content[:100]}..." if content else "    No content")
                print(f"    Full Doc ID: {chunk_data.get('full_doc_id', 'N/A')}")
        else:
             print("  [Reference to rag.text_chunks._data failed]")

        # 3. Entities (KV Storage)
        print("\n--- Entities (KV Storage) ---")
        if hasattr(rag.full_entities, "_data"):
            entities = rag.full_entities._data
            print(f"Total Entities: {len(entities)}")
            for entity_id, entity_data in list(entities.items())[:5]:
                print(f"  ID: {entity_id}")
                print(f"    Name: {entity_data.get('entity_name', 'N/A')}")
                print(f"    Description: {entity_data.get('description', '')[:100]}...")
        else:
             print("  [Reference to rag.full_entities._data failed]")

        # 4. Relations (KV Storage)
        print("\n--- Relations (KV Storage) ---")
        if hasattr(rag.full_relations, "_data"):
            relations = rag.full_relations._data
            print(f"Total Relations: {len(relations)}")
            for rel_id, rel_data in list(relations.items())[:5]:
                print(f"  ID: {rel_id}")
                print(f"    Source: {rel_data.get('src_id', 'N/A')}")
                print(f"    Target: {rel_data.get('tgt_id', 'N/A')}")
                print(f"    Description: {rel_data.get('description', '')[:100]}...")
        else:
             print("  [Reference to rag.full_relations._data failed]")
             
        # 5. Vector Databases
        print("\n--- Vector Databases ---")
        
        # Entities Vector DB
        if rag.entities_vdb:
             client = await rag.entities_vdb._get_client()
             print(f"Entities Vector DB Count: {len(client)}")
        
        # Chunks Vector DB
        if rag.chunks_vdb:
            client = await rag.chunks_vdb._get_client()
            print(f"Chunks Vector DB Count: {len(client)}")

        # Relationships Vector DB
        if rag.relationships_vdb:
             client = await rag.relationships_vdb._get_client()
             print(f"Relationships Vector DB Count: {len(client)}")

    finally:
        # Avoid saving if we are just analyzing (although we are only reading)
        # But finalize might trigger save if we modified anything (we didn't)
        await rag.finalize_storages()

if __name__ == "__main__":
    asyncio.run(main())
