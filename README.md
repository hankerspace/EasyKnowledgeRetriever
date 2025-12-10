# Easy Knowledge Retreiver

Easy Knowledge Retreiver is a simple and fast Retrieval-Augmented Generation (RAG) system designed content retrieval and generation.

## Features

- **Efficient Retrieval**: Optimizes retrieval speed and accuracy.
- **Knowledge Graph Integration**: Supports graph-based knowledge representation.
- **Flexible Storage**: Compatible with various storage backends including JSON, Redis, Neo4j, Milvus, and more.
- **LLM Support**: Supports multiple LLM providers like OpenAI, Azure, Gemini, etc.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Initialization

```python
from easy_knowledge_retriever import EasyKnowledgeRetriever, QueryParam

data_path = "./data.txt"

rag = EasyKnowledgeRetriever(
    working_dir="./rag_storage",
    llm_model_func=...,  # Your LLM model function
)

with open(data_path, "r", encoding="utf-8") as f:
    rag.insert(f.read())

# Perform a query
print(rag.query("What is the main theme of the document?", param=QueryParam(mode="naive")))
```

## API Server

To run the API server:

```bash
python -m api.server
```

## Visualizer

A visualizer tool is available in `tools/easy_knowledge_retriever_visualizer`.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)
