# ELI5 Multi-Agent Architecture Documentation Index

## Quick Navigation

### For Researchers Writing Papers
**Start Here:**
1. **[ARCHITECTURE_SUMMARY.md](ARCHITECTURE_SUMMARY.md)** - Quick reference tables and comparison matrix (5 min read)
2. **[ARCHITECTURE_COMPARISON.md](ARCHITECTURE_COMPARISON.md)** - Detailed technical comparison (15 min read)
3. **[RESEARCH_METHODOLOGY.md](RESEARCH_METHODOLOGY.md)** - How to design experiments (20 min read)

### For Engineers Implementing Systems
**Start Here:**
1. **[ARCHITECTURE_SUMMARY.md](ARCHITECTURE_SUMMARY.md)** - Performance characteristics & deployment considerations
2. **[main.py](main.py)** - Web search + multi-model architecture implementation
3. **[arch_1/simple_agent.py](arch_1/simple_agent.py)** - RAG + hybrid search architecture implementation

### For Educators / Students
**Start Here:**
1. **[ARCHITECTURE_SUMMARY.md](ARCHITECTURE_SUMMARY.md)** - High-level comparison (easiest entry point)
2. **Execution Flow** section for pipeline diagrams

---

## Document Overview

### ARCHITECTURE_SUMMARY.md
**Purpose:** Quick reference guide with tables and matrices
**Length:** ~500 lines
**Key Sections:**
- System overview table
- Pipeline execution flows
- Information retrieval comparison
- Model configuration comparison
- Performance optimizations
- Deployment readiness
- Use case matrix

**Best For:** Quick lookup, comparing specific aspects, presentations

---

### ARCHITECTURE_COMPARISON.md
**Purpose:** Comprehensive technical comparison for research papers
**Length:** ~1,200 lines
**Key Sections:**
1. Agent Pipeline Flow - Detailed agent descriptions and execution models
2. State Management - Schema definitions, caching strategies
3. Tool Usage & Information Retrieval - Search algorithms, knowledge bases
4. LLM Configuration & Model Usage - Model strategies, temperature tuning
5. Key Architectural Differences - Comparison matrix
6. Performance Optimization Techniques - Specific optimization implementations
7. Comparison Matrix - Design tradeoffs
8. Synthesis & Quality Gate Strategy - Decision-making logic
9. Production Deployment Considerations - Pros/cons for each
10. Methodological Implications - Research guidance
11. Structured Output Enforcement - Schema progression
12. Testing & Evaluation Readiness - Evaluation points

**Best For:** Methodology section of research papers, detailed understanding

---

### RESEARCH_METHODOLOGY.md
**Purpose:** Guide for designing experiments using both architectures
**Length:** ~800 lines
**Key Sections:**
1. Comparative Study Framework - 3 main research questions
2. Benchmarking Framework - Reproducible baseline protocol
3. Real-World Evaluation Framework - Current information testing
4. Cost-Effectiveness Analysis - TCO comparison
5. Scalability & Throughput Analysis - Performance scaling
6. Ablation Study Framework - Component impact testing
7. Human Evaluation Protocol - ELI5 quality rubric
8. Publication Guidelines - Reproducibility checklists
9. Recommended Research Roadmap - 5-phase plan
10. Expected Research Contributions - Novel insights

**Best For:** Planning experiments, evaluating systems, publication guidance

---

## Architecture Choice Quick Reference

### Choose Architecture 1 (main.py) if:
- ✅ Need real-time, current information
- ✅ Studying multi-model orchestration
- ✅ Testing with different model configurations
- ✅ Can afford API costs (Tavily)
- ✅ Internet connectivity available
- ✅ Need model flexibility per agent

### Choose Architecture 2 (arch_1/simple_agent.py) if:
- ✅ Need reproducible results
- ✅ Offline operation required
- ✅ Cost-sensitive deployment
- ✅ Studying hybrid search methods
- ✅ Large-scale batch processing
- ✅ Deterministic benchmarking

---

## Key Metrics Comparison

| Metric | Arch 1 | Arch 2 |
|--------|--------|--------|
| **Latency (single query)** | 16-27s | 8-15s |
| **Throughput (batch)** | ~100 q/hour | ~720 q/hour |
| **Cost per query** | $0.015-0.055 | <$0.00001 |
| **Reproducibility** | Low (web varies) | High (fixed dataset) |
| **Offline capable** | No | Yes |
| **Model flexibility** | High (5 configs) | Low (fixed) |
| **Setup time** | Minutes | 1-2 minutes |

---

## File Structure

```
edutech-agent/
├── main.py                              # Architecture 1: Web Search + Multi-Model
├── arch_1/simple_agent.py               # Architecture 2: RAG + Hybrid Search
├── ARCHITECTURE_COMPARISON.md           # Detailed technical comparison
├── ARCHITECTURE_SUMMARY.md              # Quick reference tables
├── RESEARCH_METHODOLOGY.md              # Experiment design guide
├── DOCUMENTATION_INDEX.md               # This file
├── AGENTS.md                            # Project overview & entry points
│
├── evaluation.py                        # Evaluation harness (non-LLM + Ollama judge)
├── ragas_evaluator.py                   # Full RAGAS metrics pipeline
├── ragas_evaluator_llm_only.py          # LLM-only RAGAS (answer_accuracy)
├── load_dataset.py                      # ELI5 dataset utilities
├── run_batch_incremental.py             # Batch answer generation (resumable)
├── baseline_llama.py                    # Single-LLM baseline
│
├── run_gptoss_judge_all.py              # ★ NEW: NVIDIA NIM LLM-as-judge batch runner
├── evaluate_with_gptoss_judge.py        # Single-file NVIDIA NIM evaluator
├── batch_evaluate_gptoss.py             # Multi-file NVIDIA NIM evaluator
├── compare_gptoss_results.py            # Comparison table from judge summaries
├── find_missing.py                      # Audit missing experimental data points
├── analyze_results.py                   # Results analysis utilities
│
├── outputs_llm_final/                   # ★ 14 CSVs (500 rows each, seed=22)
│   └── {baseline|arch_1}_{model}_0_30000_ragas_llm.csv
│
├── llm-metrics-50-samples-gpt4.1/      # ★ GPT-4.1 judge results (50 samples, seed=18)
│   └── {config}_ragas_llm_summary.json
│
├── llm_metrics_gptoss/                  # ★ Llama-3.3-70B judge results (50 samples, seed=42)
│   └── {config}_gptoss_judge_summary.json
│
├── llm_metrics_output/                  # Old Llama-2-13b judge summaries (deprecated)
├── non_llm_metrics_output/             # ROUGE/BERTScore/perplexity summaries
├── results_new/                         # Misc evaluation CSVs
├── evaluation_results/                  # Non-LLM per-question metrics
├── generated_answers/                   # Raw generated answer CSVs
├── paper/                               # LaTeX paper source
├── interesting_findings/                # Research notes & gap analysis
└── vllm/                                # vLLM-based inference variants
```

---

## Code Implementation Details

### Architecture 1: main.py
**Key Functions:**
- `create_llm()` - Creates configurable LLM instances with model selection
- `create_breakdown_agent()` - Decomposes query into search + reasoning
- `create_reasoning_agent()` - Pure logical analysis
- `create_scientific_agent()` - Web search tool integration
- `create_synthesis_agent()` - Quality gate with strategy selection
- `create_creative_agent()` - ELI5 explanation generation
- `create_graph()` - Assembles multi-agent workflow
- `answer_question()` - Main entry point

**Configuration:**
- `MODEL_CONFIGS` - 5 predefined model combinations
- `MAX_SOURCES = 10` - Web search result limits
- `FACTS_TARGET_COUNT = 12` - Fact extraction count
- System prompts for each agent

---

### Architecture 2: arch_1/simple_agent.py
**Key Functions:**
- `get_rag_resources()` - Loads/caches RAG dataset with embeddings
- `rag_search()` - Hybrid BM25 + semantic search
- `batch_rag_search()` - Optimized batch query encoding
- `wikipedia_search()` - Wikipedia API integration
- `parallel_analysis_node()` - Concurrent reasoning + scientific
- `create_graph()` - Assembles workflow
- `answer_question()` - Main entry point
- `generate_answers_batch()` - Batch processing with resume capability

**Configuration:**
- `MODEL_NAME = "llama3.2:1b"` - Fixed model
- `OLLAMA_NUM_THREADS = 8` - CPU parallelization
- `FACTS_TARGET = 12` - Fact extraction count
- `MAX_SOURCES = 5` - Combined retrieval limit
- `RAG_CACHE_DIR = "./rag_cache"` - Disk cache location

---

## State Schema Comparison

### Architecture 1
Includes `model_config` for per-question model selection
```python
model_config: Dict[str, str]  # Specifies model for each agent
```

### Architecture 2
Fixed model, simplified state
No `model_config` field

Both include:
- `search_queries`, `reasoning_points` (from breakdown)
- `extracted_facts` (from scientific)
- `synthesis_strategy`, `final_points` (from synthesis)
- `final_answer` (from creative)

---

## Tool/API Comparison

### Architecture 1 Tools
1. **Tavily Web Search API**
   - Requires: API key in environment
   - Cost: $0.01-0.05 per query
   - Depth: "advanced" (default)

### Architecture 2 Tools
1. **BM25 Search** - Local, free
2. **Semantic Search** - SentenceTransformer embeddings, local
3. **Wikipedia API** - Free, public
4. No API keys required

---

## Integration Points

### With Evaluation Framework
Both architectures integrate with:
- `evaluation.py` - Main evaluation harness
- `ragas_evaluator.py` - RAGAS quality metrics
- `load_dataset.py` - Dataset utilities

### With External Services
**Architecture 1:**
- Tavily API (web search)
- Ollama server (LLM inference)

**Architecture 2:**
- HuggingFace (dataset download)
- Wikipedia API (definitions)
- Ollama server (LLM inference)

---

## Performance Profiling

### Where to Measure
1. `breakdown_node()` - Query decomposition speed
2. `reasoning_node()` / `scientific_node()` - Individual agent speed
3. `parallel_analysis_node()` - Parallel efficiency
4. `synthesis_node()` - Strategy selection speed
5. `creative_node()` - ELI5 generation speed

### Timing Instrumentation
**Architecture 2 includes:**
```python
ENABLE_TIMING = True  # Control timing logs
def log_time(message, start_time)  # Reusable timer

Usage: log_time("Agent name", start_time)
```

---

## Version Information

### Architecture 1
- Tested with: langchain, ollama, tavily-python
- Model: mistral:7b, llama3.2:latest, llama3.2:1b
- Python: 3.8+

### Architecture 2
- Tested with: langchain, ollama, sentence-transformers, rank-bm25
- Model: llama3.2:1b
- Dataset: open-thoughts/OpenThoughts-114k
- Python: 3.8+

---

## For Literature Review

### Key Concepts Implemented
- **Multi-Agent Orchestration**: Both use LangGraph
- **Prompt Engineering**: System prompts for each agent role
- **Structured Outputs**: Pydantic models for reliable parsing
- **RAG (Retrieval Augmented Generation)**: Architecture 2
- **Hybrid Search**: BM25 + semantic embeddings
- **Information Retrieval**: Web search vs. local retrieval
- **Synthesis**: Quality gate pattern
- **ELI5 Generation**: Adapted prompts for child-friendly output

### Research Topics
- Multi-agent systems
- Information retrieval methods
- LLM orchestration
- Educational QA systems
- Explainable AI
- Cost-quality tradeoffs
- Reproducible ML systems

---

## Getting Started

### For Quick Understanding (5 minutes)
1. Read this file (you're here!)
2. Skim ARCHITECTURE_SUMMARY.md tables

### For Detailed Understanding (30 minutes)
1. Read ARCHITECTURE_SUMMARY.md fully
2. Read ARCHITECTURE_COMPARISON.md sections 1-5

### For Implementing Changes (1-2 hours)
1. Read relevant architecture (main.py or arch_1/simple_agent.py)
2. Read ARCHITECTURE_COMPARISON.md sections 3-4
3. Trace through code for target components

### For Research Paper Writing (2-4 hours)
1. Read ARCHITECTURE_COMPARISON.md fully
2. Read RESEARCH_METHODOLOGY.md fully
3. Scan code implementations
4. Design experiment protocol

---

## Questions to Ask

### Which architecture has...
- **Faster inference?** Architecture 2 (~2x faster, local only)
- **More flexibility?** Architecture 1 (5 model configs)
- **Better reproducibility?** Architecture 2 (fixed dataset)
- **Current information?** Architecture 1 (web search)
- **Lower cost?** Architecture 2 (no API costs)
- **Easier deployment?** Architecture 2 (simpler setup)
- **Better quality?** Depends on question domain

---

## Contact & Support

For questions about:
- **Architecture design**: See ARCHITECTURE_COMPARISON.md
- **Experiments**: See RESEARCH_METHODOLOGY.md
- **Implementation**: Read source code with comments
- **Deployment**: See ARCHITECTURE_SUMMARY.md deployment section

---

## Evaluation Results Summary

### LLM-as-Judge Evaluations Completed (June 2026)

All 14 experiment configs (7 baseline + 7 arch_1) have been evaluated by two independent strong judges:

| Judge | Location | Sample | Seed | Key Finding |
|---|---|---|---|---|
| GPT-4.1 | `llm-metrics-50-samples-gpt4.1/` | 50/config | 18 | Multi-agent ▲ +10.4% answer_accuracy |
| Llama-3.3-70B (NVIDIA NIM) | `llm_metrics_gptoss/` | 50/config | 42 | Multi-agent ▲ +19.3% correctness |
| Llama-2-13b (old, deprecated) | `llm_metrics_output/` | 500/config | 22 | Multi-agent ▼ -38% (weak judge artifact) |

**Key insight:** The original -38% accuracy drop (Llama-2-13b judge) was a weak-judge artifact. Both GPT-4.1 and Llama-3.3-70B agree the multi-agent system improves factual correctness. See `interesting_findings/gaps_in_research.md` for full analysis.

### Scripts for Reproducing LLM Judge Evaluation

```bash
# Run all 14 configs (50 samples each) with NVIDIA NIM judge:
export NVIDIA_API_KEY="nvapi-..."
python3 run_gptoss_judge_all.py

# Resume if interrupted (skips already-completed configs):
python3 run_gptoss_judge_all.py

# Force re-run all:
python3 run_gptoss_judge_all.py --force

# Change model:
python3 run_gptoss_judge_all.py --model meta/llama-3.3-70b-instruct
```

---

## Document History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-04-07 | Initial comprehensive documentation |
| 1.1 | 2026-06-17 | Added evaluation results, updated file structure, new scripts |

---

**Last Updated:** June 17, 2026

