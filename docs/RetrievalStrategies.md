# Retrieval Strategies & Workflows

Easy Knowledge Retriever provides multiple retrieval strategies to cater to different use cases, balancing between precision (specific details) and recall (broad themes), as well as between structured knowledge (Graph) and unstructured text (Vector).

## Overview

| Strategy | Mode Name | Description | Best For |
|----------|-----------|-------------|----------|
| **Naive** | `naive` | Classic Vector Search on text chunks. | Simple queries, exact matches in text. |
| **Local** | `local` | Entity-focused Graph Retrieval. | Specific questions about entities and their direct attributes. |
| **Global** | `global` | Relation-focused Graph Retrieval. | Thematic questions, understanding connections between concepts. |
| **Hybrid** | `hybrid` | Combines Local (Entity) and Global (Relation) Graph Retrieval. | Comprehensive Graph-based understanding. |
| **Mix** | `mix` | Combines Graph (Hybrid) and Vector (Naive) Retrieval. | The most robust strategy, leveraging both structured and unstructured data. |
| **HybridMix** | `hybrid_mix` | Hybrid Search on Chunks (Vector + BM25) with RRF Fusion. | Advanced text search without Graph overhead. |

## Detailed Workflows

### 1. Naive Retrieval (`naive`)
**Type**: Vector Search (Unstructured)

This is the standard RAG approach.
1.  **Embedding**: The user query is embedded using the `EmbeddingService`.
2.  **Vector Search**: The system searches for the most similar text chunks in `VectorStorage` (Cosine Similarity).
3.  **Context Building**: Retrieved chunks are concatenated to form the context.
4.  **Generation**: LLM generates the answer based on chunks.

### 2. Local Retrieval (`local`)
**Type**: Graph Retrieval (Entity-Centric)

Focuses on specific entities mentioned in the query.
1.  **Keyword Extraction**: LLM extracts "Low-Level" keywords (entities) from the query.
2.  **Entity Lookup**: System looks up these entities in the Knowledge Graph.
3.  **Neighbor Retrieval**: Retrieves direct neighbors (relations) of these entities.
4.  **Context Building**: Constructs a context description from the matched entities and their relations.

### 3. Global Retrieval (`global`)
**Type**: Graph Retrieval (Relation-Centric)

Focuses on broader relationships and themes.
1.  **Keyword Extraction**: LLM extracts "High-Level" keywords from the query.
2.  **Relation Lookup**: System searches for relationships in the Graph that match these keywords.
3.  **Context Building**: Constructs a context description from the matched relationships and involved entities.

### 4. Hybrid Retrieval (`hybrid`)
**Type**: Graph Retrieval (Combined)

Combines **Local** and **Global** strategies.
1.  Performs **Local Retrieval** to get entity details.
2.  Performs **Global Retrieval** to get relational context.
3.  **Fusion**: Merges both contexts to provide a comprehensive view of the structured knowledge.

### 5. Mix Retrieval (`mix`)
**Type**: Composite (Graph + Vector)

Combines the best of both worlds: Structured Graph knowledge and Unstructured Vector search.
1.  **Graph Retrieval**: Performs **Hybrid Retrieval** (Local + Global).
2.  **Vector Retrieval**: Performs **Naive Retrieval** (Vector Search on Chunks).
3.  **Fusion**: Concatenates contexts from both sources.
4.  **Benefit**: Handles queries where the answer might be in the graph structure OR in specific text details not captured by the graph.

### 6. HybridMix Retrieval (`hybrid_mix`)
**Type**: Advanced Chunk Search (Dense + Sparse)

Does **not** use the Knowledge Graph. It improves upon Naive Retrieval by adding keyword matching.
1.  **Dense Search**: Vector search on chunks (Semantic match).
2.  **Sparse Search**: BM25 (or similar) keyword search on chunks (Exact match).
3.  **RRF Fusion**: Combines results using Reciprocal Rank Fusion to rank the best chunks.
4.  **Benefit**: Better retrieval accuracy than simple Vector Search, especially for specific terms, without the complexity of building a Graph.

## How to use

Pass the specific Retrieval class instance to the `retrieval` parameter when calling the `query` method.

```python
from easy_knowledge_retriever.retrieval import MixRetrieval

# Example of querying with Mix strategy
response = await rag.query("What is the impact of X?", retrieval=MixRetrieval())
```

Or you can initialize the Retrieval classes directly if using them in a custom flow.
