import asyncio
import os
import sys
import shutil

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
LLM_API_KEY = ""
LLM_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
LLM_MODEL = "gemini-2.0-flash-lite"
EMBEDDING_API_KEY = ""
EMBEDDING_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIM = 3072

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

    rag = EasyKnowledgeRetriever(
        working_dir=WORK_DIR,
        llm_service=llm_service,
        embedding_service=embedding_service,
        kv_storage=JsonKVStorage(working_dir=WORK_DIR),
        vector_storage=NanoVectorDBStorage(working_dir=WORK_DIR, cosine_better_than_threshold=0.2),
        graph_storage=NetworkXStorage(working_dir=WORK_DIR),
        doc_status_storage=JsonDocStatusStorage(working_dir=WORK_DIR),
    )
    
    await rag.initialize_storages()
    return rag

async def ingest_data(rag):
    print("Ingesting data...")
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".txt")]
    
    contents = []
    file_paths = []
    
    for filename in files:
        path = os.path.join(DATA_DIR, filename)
        with open(path, "r", encoding="utf-8") as f:
            contents.append(f.read())
        file_paths.append(path)
        
    if contents:
        await rag.ainsert(input=contents, file_paths=file_paths)
        print(f"Ingested {len(contents)} documents.")
    else:
        print("No documents found to ingest.")


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
            "What are the main causes of forest fires?",
            "How does climate change affect wildfires?",
            "Describe the stages of labor.",
            "What is an Apgar Score?",
            "What are the risks of wildfire smoke?"
        ]
        
        ground_truths = [
            ["Forest fires are caused by natural factors like lightning, but mostly by human activities such as discarded cigarettes, unattended campfires, widespread burning, and equipment malfunctions."],
            ["Climate change causes hotter and drier conditions, earlier snowmelts, and prolonged dry seasons, which increase vegetation flammability and lead to more frequent and severe fires."],
            ["Labor has three stages: 1) Cervix thinning (effacement) and opening (dilation). 2) Delivery of the baby. 3) Delivery of the placenta."],
            ["Apgar Score is a quick test at 1 and 5 minutes after birth assessing Activity, Pulse, Grimace, Appearance, and Respiration on a scale of 0-2."],
            ["Wildfire smoke contains fine particulate matter (PM2.5) that can penetrate lungs and enter the bloodstream, aggravating respiratory conditions like asthma and COPD, and increasing heart attack and stroke risks."]
        ]
        
        print("\nStarting Evaluation Queries...")
        
        for i, question in enumerate(test_questions):
            print(f"Querying: {question}")
            # Use hybrid mode for best retrieval
            param = QueryParam(mode="hybrid") 
            result = await rag.aquery_llm(question, param=param)
            

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

