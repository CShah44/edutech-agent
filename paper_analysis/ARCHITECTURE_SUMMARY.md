# Architecture Comparison Summary - Quick Reference

## System Overview

| Dimension | Architecture 1 (main.py) | Architecture 2 (arch_1/simple_agent.py) |
|-----------|--------------------------|------------------------------------------|
| **Primary Technique** | Web Search + Multi-Model | Local RAG + Hybrid Search |
| **Knowledge Source** | Real-time web (Tavily API) | OpenThoughts-114k (RAG) + Wikipedia |
| **Model Strategy** | 5 configurations, task-optimized | Single fixed model (llama3.2:1b) |
| **Offline Capability** | ❌ No (requires internet) | ✅ Yes (after initial setup) |
| **Reproducibility** | 🔴 Low (web changes) | 🟢 High (fixed dataset) |
| **Inference Speed** | Medium (network I/O) | Fast (local only) |
| **Complexity** | High | Low |

---

## Pipeline Comparison

### Execution Flow

**Architecture 1:**
```
breakdown → [reasoning ∥ scientific] → synthesis → creative → output
            └─ parallel ─┘
            (reasoning fast, scientific has I/O)
```

**Architecture 2:**
```
breakdown → parallel_analysis_node → synthesis → creative → output
           ┌─ reasoning
           └─ scientific (with batch encoding)
```

---

## Information Retrieval Comparison

### Architecture 1: Web Search Approach
```
Query decomposition (Breakdown Agent)
        ↓
Search Queries extracted
        ↓
Tavily API: "advanced" depth web search
        ↓
Max 10 results per query
        ↓
Parallel ThreadPoolExecutor
        ↓
Deduplicate by URL
        ↓
Select top 10 sources
        ↓
LLM extracts 12 facts
```

**Characteristics:**
- Real-time, always current
- Variable quality (depends on search results)
- API dependencies (cost, rate limits)
- No caching of web results

---

### Architecture 2: Hybrid RAG Approach
```
Query decomposition (Breakdown Agent)
        ↓
Search Queries extracted
        ↓
[Parallel Execution]
├─ BM25 (keyword search) on OpenThoughts
├─ Semantic (embedding similarity) on OpenThoughts
└─ Wikipedia API (parallel, max 2 queries)
        ↓
Merge results (no duplicates)
        ↓
Select top 5 sources combined
        ↓
LLM extracts 12 facts
```

**Characteristics:**
- Consistent, deterministic results
- Optimized batch encoding (3-5x speedup)
- Persistent disk caching
- Hybrid recall (BM25 + semantic)

---

## Model Configuration Comparison

### Architecture 1: Task-Optimized

```python
MODEL_CONFIGS = {
    "config1": {llama3.2:1b, llama3.2:1b, llama3.2:1b},      # All light
    "config2": {llama3.2:latest, llama3.2:latest, llama3.2:latest},
    "config3": {mistral:7b, llama3.2:latest, llama3.2:latest},  # Strong reasoning
    "config4": {mistral:7b, llama3.2:latest, mistral:7b},    # Hybrid
    "config5": {mistral:7b, mistral:7b, mistral:7b}          # All strong
}
```

**Use Case:** Optimize model size for each agent role:
- Reasoning: Can be lightweight (mistral:7b when needed)
- Scientific: Fact extraction (balanced)
- Creative: Quality output (mistral:7b for large config)

**Temperature Tuning:**
- Breakdown/Reasoning/Scientific: 0.1 (deterministic)
- Synthesis: 0.2 (some flexibility)
- Creative: 0.3 (more creative)

---

### Architecture 2: Fixed Model

```python
MODEL_NAME = "llama3.2:1b"  # Fixed for all agents
OLLAMA_NUM_THREADS = 8      # CPU parallelization
OLLAMA_REQUEST_TIMEOUT = 120  # Prevent hanging
```

**Use Case:** Simplicity and consistency
- Same model for all agents
- Faster deployment
- Simpler testing/debugging
- Lower resource overhead

**Temperature Tuning:**
- Breakdown/Reasoning/Scientific: 0.1 (deterministic)
- Creative: 0.5 (more creative)

---

## Performance Optimizations

### Architecture 1
| Optimization | Method | Benefit |
|--------------|--------|---------|
| **Parallel reasoning + scientific** | ThreadPoolExecutor with 2 threads | Reasoning doesn't block on web I/O |
| **Context management** | 3900 char limit, per-source truncation | Prevent token overflow |
| **Deduplication** | URL-based dedup before processing | Reduce redundant processing |
| **Caching** | LLM, Agent, Graph caches | Reuse expensive compilations |
| **Batch Google Sheets** | Batch update API calls | Reduce sheet I/O |

---

### Architecture 2
| Optimization | Method | Benefit |
|--------------|--------|---------|
| **Batch RAG encoding** | Encode ALL queries at once | 3-5x speedup for multiple queries |
| **Hybrid search merge** | BM25 union with semantic | Better recall without explosion |
| **Parallel Wikipedia** | ThreadPoolExecutor, max_workers=2 | Non-blocking Wikipedia calls |
| **Disk-persistent RAG** | Pickle BM25 + embeddings to disk | Skip rebuild on next run |
| **Device consistency** | Ensure CPU/CUDA alignment | Avoid device mismatch errors |
| **Thread pooling** | ThreadPoolExecutor for batch Q&A | 3 concurrent questions |

---

## Caching Strategy Comparison

### Architecture 1
```python
llm_cache[(model, temp, system)] = LLM instance
agent_cache[(type, model)] = compiled agent
graph_cache["main_workflow"] = StateGraph
# No disk persistence for retrieval results
```

**Benefit:** Faster LLM reuse, graph compilation cached
**Limitation:** Web results not cached (always fresh but potentially slower)

---

### Architecture 2
```python
llm_cache[(model, temp, system)] = LLM instance
agent_cache[type] = compiled agent
graph_cache["workflow"] = StateGraph
rag_cache = {dataset, bm25, corpus, model, embeddings}
# Disk persist: ./rag_cache/rag_resources_hybrid.pkl
```

**Benefit:**
- LLM/agent/graph caching (same as Arch 1)
- Plus disk-persistent RAG (avoids rebuild)
- First run: ~1-2 min (build embeddings)
- Subsequent runs: Near-instant (load cache)

---

## Synthesis Strategy (Both Architectures)

Both use the same synthesis approach:

```
Evaluate Quality:
├─ Reasoning: Logically sound? Helpful?
└─ Facts: Concrete? Accurate? Relevant?

Select Strategy:
├─ reasoning_heavy (70/30): Facts weak
├─ facts_heavy (70/30): Facts excellent
└─ balanced (50/50): Both good

Curate Points:
├─ Arch 1: 6-8 points
└─ Arch 2: 4-6 points (more conservative)
```

---

## Research Paper Methodology Implications

### Use Architecture 1 When...
✅ Studying real-world Q&A quality
✅ Evaluating current information retrieval
✅ Testing multi-model orchestration
✅ Measuring search depth effectiveness
✅ Need latest web information

### Use Architecture 2 When...
✅ Needing reproducible benchmarking
✅ Testing offline systems
✅ Evaluating hybrid search methods
✅ Measuring batch processing throughput
✅ Cost-sensitive deployment research
✅ Testing with fixed knowledge cutoff

---

## Key Implementation Differences - Code Level

### State Management
| Aspect | Arch 1 | Arch 2 |
|--------|--------|--------|
| `model_config` in state | ✅ Yes | ❌ No |
| Multiple model support | ✅ Yes | ❌ No |
| Per-question model selection | ✅ Yes | ❌ No |

### Tool Implementation
| Tool | Arch 1 | Arch 2 |
|------|--------|--------|
| Primary search | Tavily API | BM25 + Semantic |
| Encoding optimization | None | Batch encoding |
| Caching strategy | Memory only | Memory + Disk |
| Fallback sources | Web only | RAG + Wikipedia |

### Structured Outputs
| Agent | Arch 1 | Arch 2 |
|-------|--------|--------|
| Breakdown | ✅ Pydantic | ✅ Pydantic |
| Reasoning | ✅ Pydantic | ✅ Pydantic |
| Scientific | ✅ Pydantic | ✅ Pydantic |
| Synthesis | ✅ Pydantic | ✅ Pydantic |
| Creative | ✅ Pydantic | ✅ Pydantic |

---

## Deployment Readiness

### Architecture 1
**Prerequisites:**
- Ollama server running
- Tavily API key
- Internet connection
- Python dependencies (langchain, tavily)

**Advantages:** Latest information, model flexibility
**Disadvantages:** Network dependency, variable results

---

### Architecture 2
**Prerequisites:**
- Ollama server running
- ~500MB+ disk for embeddings
- Internet (initial dataset download only)
- Python dependencies (langchain, sentence-transformers)

**Advantages:** Fast, reproducible, offline-capable
**Disadvantages:** Knowledge cutoff, setup time

---

## Performance Characteristics

### Latency Comparison (Typical)

| Component | Architecture 1 | Architecture 2 |
|-----------|----------------|----------------|
| Breakdown | ~2-3s | ~1-2s |
| Reasoning | ~2-3s | ~1-2s |
| Scientific | ~8-15s (I/O) | ~3-5s (local) |
| Synthesis | ~2-3s | ~1-2s |
| Creative | ~2-3s | ~2-3s |
| **Total** | ~16-27s | ~8-15s |

**Architecture 2 is ~2x faster** due to local retrieval

---

## Scalability Analysis

### Throughput (Batch Processing)

**Architecture 1:**
- Single query: ~20-30s
- Batch (3 parallel workers): ~100 queries/hour
- Bottleneck: Tavily API rate limits

**Architecture 2:**
- Single query: ~8-15s
- Batch (3 parallel workers): ~720 queries/hour
- Bottleneck: CPU (Ollama with num_thread=8)

---

## Flexibility Matrix

| Dimension | Flexibility |
|-----------|-------------|
| **Architecture 1** |
| Model selection | 🔴🔴🔴 High (5 configs + custom) |
| Temperature tuning | 🔴🔴 Medium (can modify per-agent) |
| Knowledge sources | 🔴 Low (web search only) |
| Reproducibility | 🟢 Low (web changes constantly) |
| |
| **Architecture 2** |
| Model selection | 🟢 Low (fixed model) |
| Temperature tuning | 🟢 Medium (can modify per-agent) |
| Knowledge sources | 🔴🔴 Medium (RAG + Wiki, hybrid search) |
| Reproducibility | 🔴🔴🔴 High (fixed dataset) |

---

## Conclusion Matrix

| Use Case | Arch 1 | Arch 2 |
|----------|--------|--------|
| Production real-time QA | ✅ Best | ⚠️ Limited |
| Research benchmarking | ⚠️ Variable | ✅ Best |
| Offline operation | ❌ No | ✅ Yes |
| Cost-sensitive | ❌ API costs | ✅ Free |
| Latest information | ✅ Yes | ❌ Knowledge cutoff |
| Model experimentation | ✅ Yes | ⚠️ Limited |
| Fast inference | ⚠️ Medium | ✅ Fast |
| Reproducibility | ❌ Low | ✅ High |

