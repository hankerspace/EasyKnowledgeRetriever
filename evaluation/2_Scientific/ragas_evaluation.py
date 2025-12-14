import asyncio
import os
import sys
import shutil

from easy_knowledge_retriever.reranker.openai import OpenAIRerankerService
from easy_knowledge_retriever.retrieval import MixRetrieval, HybridMixRetrieval

# --- Patch for multiprocess bug on Python 3.12+ (Windows) ---
# See: https://github.com/uqfoundation/multiprocess/issues/185
try:
    import multiprocess.resource_tracker
    
    def _stop_locked_patched(
        self,
        close=os.close,
        waitpid=os.waitpid,
        waitstatus_to_exitcode=os.waitstatus_to_exitcode,
    ):
        # Safe retrieval of recursion count for Py3.12+ compatibility
        try:
            recursion_count = self._lock._recursion_count()
        except AttributeError:
            recursion_count = 0
            
        if recursion_count > 1:
            return self._reentrant_call_error()
            
        if self._fd is None:
            return
        if self._pid is None:
            return

        close(self._fd)
        self._fd = None

        waitpid(self._pid, 0)
        self._pid = None

    multiprocess.resource_tracker.ResourceTracker._stop_locked = _stop_locked_patched
except ImportError:
    pass
# ------------------------------------------------------------

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from easy_knowledge_retriever import EasyKnowledgeRetriever, QueryParam
from easy_knowledge_retriever.llm.service import OpenAILLMService, OpenAIEmbeddingService
from easy_knowledge_retriever.kg.kv_storage.json_kv_impl import JsonKVStorage
from easy_knowledge_retriever.kg.vector_storage.nano_vector_db_impl import NanoVectorDBStorage
from easy_knowledge_retriever.kg.graph_storage.networkx_impl import NetworkXStorage
from easy_knowledge_retriever.kg.kv_storage.json_doc_status_impl import JsonDocStatusStorage

# Configuration
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(EVAL_DIR, "data")
WORK_DIR = os.path.join(EVAL_DIR, "work_dir")

# LLM Configuration (using same keys as example for now - user should replace or env vars)
LLM_API_KEY = "AIzaSyAfd4owmFzZMjTP0-ByWqxH_XGGdhAfSaM"
LLM_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
LLM_MODEL = "gemini-2.5-flash-lite"
EMBEDDING_API_KEY = "AIzaSyAfd4owmFzZMjTP0-ByWqxH_XGGdhAfSaM"
EMBEDDING_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIM = 3072
RERANKER_API_KEY = "gpustack_a446e348d5180a0b_a90a04f559a3aee4e0a24bc7ab4aef2e"
RERANKER_BASE_URL = "https://llm.isaratech.com/v1"
RERANKER_MODEL = "bge-reranker-v2-m3"

async def setup_rag():
    if not os.path.exists(WORK_DIR):
        os.makedirs(WORK_DIR)

    embedding_service = OpenAIEmbeddingService(
        api_key=EMBEDDING_API_KEY,
        base_url=EMBEDDING_BASE_URL,
        model=EMBEDDING_MODEL,
        embedding_dim=3072 
    )

    llm_service = OpenAILLMService(
        model=LLM_MODEL,
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL
    )

    reranker_service = OpenAIRerankerService(
        api_key=RERANKER_API_KEY,
        base_url=RERANKER_BASE_URL,
        model=RERANKER_MODEL
    )

    rag = EasyKnowledgeRetriever(
        working_dir=WORK_DIR,
        llm_service=llm_service,
        embedding_service=embedding_service,
        kv_storage=JsonKVStorage(working_dir=WORK_DIR),
        vector_storage=NanoVectorDBStorage(working_dir=WORK_DIR, cosine_better_than_threshold=0.2),
        graph_storage=NetworkXStorage(working_dir=WORK_DIR),
        doc_status_storage=JsonDocStatusStorage(working_dir=WORK_DIR),
        reranker_service=reranker_service
    )
    
    await rag.initialize_storages()
    return rag

async def ingest_data(rag):
    # Collect documents from PDF folder
    pdf_files = [f for f in os.listdir(DATA_DIR) if f.casefold().endswith('.pdf')]

    if not pdf_files:
        print("No PDF files found in the directory.")

    for filename in pdf_files:
        file_path = os.path.join(DATA_DIR, filename)
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


async def collect_data():
    rag = await setup_rag()
    
    data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }
    
    try:
        await ingest_data(rag)
        
        # Test Data
        test_questions = [
            "Is it possible to use DRL to manage autonomous intersections of vehicles?",
            "In Deep Reinforcement Learning Approach for V2X Managed Intersections of Connected Vehicles by Lombard et al., how is represented the environment?",
            "What is a possible representation of the action space for DRL applied to autonomous intersection management?",
            "Are there issues related to safety when using DRL to control autonomous intersections of vehicles?",
            "What are some potential benefits of applying DRL to control intersections of autonomous vehicles?"
        ]
        
        ground_truths = [
            [ "Yes, it is indeed possible. As proposed by Lombard et al., the intersection can be modeled with the agent being the intersection controller, in charge of distributing the right of way, and the environment being composed of the vehicles approaching the intersection. A possible implementation is to let the agent learn how to distribute the right-of-way according to the positions and directions of vehicles approaching the intersection."],
            ["In Lombard et al. (2023), the environment is represented as a 2D picture, in which pixels represents vehicles: the color of the pixel is an indication of the route followed by vehicle, while the luminosity of the pixel is a visualisation of its speed. One advantage of this representation is that the size of the state space does not depend on the number of vehicles."],
            ["In Lombard et al. (2023), a possible representation of the action space is a vector of size 4^l with l being the number of lanes approaching the intersection. For each lane, the agent can decide to give the right-of-way to the first vehicle, or the first two vehicles, of the lane."],
            ["Yes, controlling an intersection by learning a policy using Deep Reinforcement Learning can lead to safety issues if it is not managed correctly. Possible ways to mitigate these issues are: 1) Integrating the safety concerns in the DRL process (e.g. ensuring that the action space is inherently safe), 2) Submit a negative reward when the produced behavior is considered unsafe, 3) Filter the decision before its application (by not applying it if it is unsafe)."],
            ["According to Lombard et al. (2023), applying DRL to control intersections of autonomous vehicles can lead to and improved performance in terms of throughput of the intersection (reduced average waiting time and increased number of evacuated vehicles), as well as reduction of CO2 emissions (compared to intersection managed by traffic lights, or managed using standard policy like First-Come First-Served or Distributed Clearing Policy)."]
        ]
        
        print("\nStarting Evaluation Queries...")
        
        for i, question in enumerate(test_questions):
            print(f"Querying: {question}")
            result = await rag.aquery(question, retrieval=HybridMixRetrieval())
            

            answer = result.content or ""
            
            # Extract contexts from chunks, entities and relationships
            chunks = result.chunks
            entities = result.entities
            relationships = result.relationships

            contexts = [chunk.content for chunk in chunks]

            # Add entities to context
            for entity in entities:
                if entity.description:
                    contexts.append(f"Entity {entity.entity_name}: {entity.description}")

            # Add relationships to context
            for rel in relationships:
                if rel.description:
                    contexts.append(f"Relation {rel.src_id} -> {rel.tgt_id}: {rel.description}")
            
            data["question"].append(question)
            data["answer"].append(answer)
            data["contexts"].append(contexts)
            data["ground_truth"].append(ground_truths[i][0])
            
            print(f"Answer: {answer[:100]}...")
            
    finally:
        await rag.finalize_storages()
        
    return data


def run_evaluation(data):
    print("\nRunning RAGAS Evaluation...")
    dataset = Dataset.from_dict(data)
    

    # Configure Ragas to use the same Gemini/OpenAI-compat endpoint
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas.run_config import RunConfig
    
    # Note: langchain-openai uses 'openai_api_key' and 'openai_api_base' 
    # but strictly it might need 'api_key' depending on version. 
    # ChatOpenAI args: api_key, base_url.
    
    llm = ChatOpenAI(
        model=LLM_MODEL,
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL
    )
    
    # OpenAIEmbeddings might need dimensions check or specific params
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=EMBEDDING_API_KEY,
        base_url=EMBEDDING_BASE_URL,
        check_embedding_ctx_length=False # skip check for custom models
    )
    

    results = evaluate(
        dataset=dataset,
        llm=llm,
        embeddings=embeddings,
        run_config=RunConfig(max_workers=1, timeout=120),
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
    )
    
    print("\nEvaluation Results:")
    print(results)
    
    results_df = results.to_pandas()
    output_path = os.path.join(EVAL_DIR, "ragas_results.csv")
    results_df.to_csv(output_path)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    # Collect data asynchronously
    data = asyncio.run(collect_data())
    

    # Run evaluation synchronously (avoids nesting loops/threads issues)
    if len(data["question"]) > 0:
        run_evaluation(data)
    else:
        print("No data collected, skipping evaluation.")

