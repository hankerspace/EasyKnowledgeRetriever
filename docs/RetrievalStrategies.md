# Retrieval Strategies

Easy Knowledge Retriever offers several retrieval strategies. They differ in what they search (text chunks, knowledge graph, or both), how they rank what they find, and how long they take.

## Overview

| Strategy | Mode | What it searches | Best for |
|---|---|---|---|
| **HybridMix** | `hybrid_mix` | Dense vectors + BM25 on chunks, knowledge graph (entities and relations), one rerank | **Recommended default.** Best evidence recall and page citations, graph context for cross-references |
| **Naive** | `naive` | Dense vectors on chunks (+ rerank) | Direct factual questions when latency matters |
| **Mix** | `mix` | Knowledge graph (hybrid) + dense vectors on chunks | Robust alternative; slowest tail latency |
| **Hybrid** | `hybrid` | Knowledge graph only (local + global) | Questions about how concepts relate |
| **Local** | `local` | Entities and their direct relations | Specific questions about named entities |
| **Global** | `global` | Relations matching high-level concepts | Broad thematic questions |
| **Bypass** | `bypass` | Nothing: the LLM answers directly | Chit-chat, or comparing with/without RAG |

When a `reranker_service` is configured, every strategy that returns chunks reranks them before building the context.

## Measured performance

Benchmark on a regulatory corpus (the EU AI Act, a long PDF in French):

- 51 questions: definitions, obligations, deadlines, cross-references, and 7 *trap* questions (out of scope or false premise)
- Generator `qwen-3.6-35b-instruct`, reranker `bge-reranker-v2-m3`, `chunk_top_k = 20`
- Answers graded 1 to 5 by an independent judge model (`gpt-oss-120b`) against reference answers

| Mode | Correctness /5 | Hallucinated answers | Traps handled | Evidence in top 5 | Reference facts in context | Correct page cited | Latency p50 | Latency p95 |
|---|---|---|---|---|---|---|---|---|
| `hybrid_mix` | 4.75 – 4.92 ¹ | 3.9 – 7.8 % ¹ | 7/7 | 87.5 % | 98.8 % | 95.8 % | 4.7 s | 8.4 s |
| `naive` | 4.96 | 3.9 % | 7/7 | 89.6 % | 97.8 % | 93.8 % | 3.9 s | 7.2 s |
| `mix` | 4.84 | 9.8 % | 7/7 | 87.5 % | 97.7 % | 97.9 % | 6.0 s | 13.3 s |
| `hybrid` | 4.71 | 7.8 % | 6/7 | 87.5 % | 95.7 % | 93.8 % | 5.1 s | 9.1 s |
| `hybrid_mix` + `query_decomposition` | 4.92 | 3.9 % | 7/7 | 87.5 % | 97.5 % | 95.8 % | 6.2 s | 11.2 s |

¹ Range over three identical runs. Re-running the same configuration moves the mean correctness by up to ~0.2 and one trap question either way, so differences smaller than that are noise, not a ranking.

What the numbers say:

- **All chunk-based modes answer well** once the reranker is in place; the remaining errors are mostly partial answers, not invented ones.
- **`hybrid_mix`** puts the most reference facts in front of the LLM and cites the right page 96 % of the time, for about one second more than `naive`.
- **`naive`** is the fastest and scored as well on this single-document corpus. Prefer it when latency matters more than graph context.
- **`hybrid`** (graph only) sends more, less relevant chunks (33 on average) and missed one trap question.
- **`query_decomposition`** brought no measurable gain and added 1.5 s at the median: leave it off.
- **The generator matters as much as the retrieval.** With the same `hybrid_mix` context, `mistral-small-4-119b` scored 4.63, hallucinated in 19.6 % of answers, handled 5/7 traps and cited the right page 37.5 % of the time.
- `local` and `global` were not benchmarked on their own.

## Detailed workflows

### Naive (`naive`)

Classic vector search on text chunks.

1. **Embedding**: the query is embedded with the `EmbeddingService`.
2. **Dense search**: the most similar chunks are fetched from `VectorStorage`. With a reranker, twice `chunk_top_k` candidates are fetched so the reranker can promote a relevant chunk that dense search ranked low.
3. **Rerank**: the reranker scores heading + content and keeps the best `chunk_top_k`.
4. **Context**: chunks are rendered with their reference id, page and heading.

### Local (`local`)

Entity-centric graph retrieval.

1. **Keyword extraction**: the LLM extracts low-level keywords (entities) from the query.
2. **Entity lookup**: matching entities are found in the knowledge graph.
3. **Neighbours**: their direct relations are retrieved.
4. **Smart edge filtering**: a local induced subgraph is built, its PageRank centrality identifies structurally important nodes relative to the query anchors, and edges connecting important nodes or two query entities are kept first.
5. **Context**: matched entities, filtered relations and their source chunks.

### Global (`global`)

Relation-centric graph retrieval.

1. **Keyword extraction**: the LLM extracts high-level keywords (themes) from the query.
2. **Relation lookup**: relations matching these keywords are found in the graph.
3. **Context**: matched relations, the entities they involve and their source chunks.

### Hybrid (`hybrid`)

Runs **local** and **global** and merges both contexts: a graph-only view of the corpus.

### Mix (`mix`)

Runs **hybrid** (graph) and **naive** (dense vectors) and merges the contexts, so an answer can come from the graph structure or from text details the graph did not capture.

### HybridMix (`hybrid_mix`)

Dense + sparse chunk search, knowledge graph, and a single rerank.

1. **Keyword extraction**: the LLM extracts high- and low-level keywords (cached).
2. **Dense search** on the stored chunk vectors (no re-embedding at query time) and **BM25** on chunk text (heading + content), fused with Reciprocal Rank Fusion (`rrf_k = 60`).
3. **Neighbour chunks**: the chunks that follow the best dense and BM25 hits join the pool, because an article or a list often continues in the next chunk.
4. **Graph**: local (entities) and global (relations) retrieval with a deliberately small budget (`max_entity_tokens` and `max_relation_tokens` = 2000), so graph text does not crowd out the passages.
5. **Rerank**: the whole pool is reranked once (heading + content) and cut to `chunk_top_k`, keeping relevance order. If the reranker is down, the first `chunk_top_k` chunks are kept in fusion order instead of the whole pool.
6. **Context**: tagged passages (`<chunk reference_id page heading>`) plus entities and relations, with the question placed after the context.

### Bypass (`bypass`)

Sends the query straight to the LLM, without retrieval.

## How to use

By mode name: the retriever's `reranker_service` is passed to the strategy automatically.

```python
from easy_knowledge_retriever import QueryParam

result = await rag.aquery(
    "Which AI practices are prohibited?",
    param=QueryParam(mode="hybrid_mix", chunk_top_k=20),
)
print(result.content)      # the answer
print(result.references)   # the cited sources
```

With a strategy object, for per-strategy options (BM25 `k1`/`b`, `rrf_k`, token budgets…). Pass the reranker yourself:

```python
from easy_knowledge_retriever.retrieval import HybridMixRetrieval

retrieval = HybridMixRetrieval(reranker_service=reranker, chunk_top_k=20, rrf_k=60)
result = await rag.aquery("Which AI practices are prohibited?", retrieval=retrieval)
```

Available classes in `easy_knowledge_retriever.retrieval`: `NaiveRetrieval`, `LocalRetrieval`, `GlobalRetrieval`, `HybridRetrieval`, `MixRetrieval`, `HybridMixRetrieval`, `BypassRetrieval`.
