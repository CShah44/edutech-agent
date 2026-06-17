# EduTech Agent — Project Documentation

## Executive Summary

EduTech Agent is a multi-agent LLM system that generates ELI5 (Explain Like I'm 5) answers by decomposing questions into logical reasoning and scientific fact-gathering tasks. The system compares two architectures — web search (via Tavily) and RAG (Retrieval-Augmented Generation with BM25 + semantic search) — evaluates generated answers using RAGAS, ROUGE, BERTScore, and LLM-as-judge metrics, and produces a LaTeX research paper documenting the findings.

---

## 1. Project Overview

### Purpose

The project investigates how multi-agent LLM architectures can generate high-quality educational explanations. It compares:

1. **Web Search Architecture** (`main.py`): Uses Tavily API for real-time web search
2. **RAG Architecture** (`vllm/simple_agent_vllm.py`): Uses BM25 + semantic search over OpenThoughts dataset + Wikipedia

### Key Capabilities

- 5-agent LangGraph pipeline with configurable LLM backends
- vLLM-optimized staged batching for high-throughput generation
- Comprehensive evaluation with 8+ metrics (RAGAS, ROUGE, BERTScore, perplexity, semantic similarity)
- LaTeX paper compilation with automated results integration
- Resumable batch processing with checkpoint saving

---

## 2. Architecture Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        EduTech Agent System                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │   Input      │    │  Generation  │    │   Output     │          │
│  │   Layer      │───▶│  Pipeline    │───▶│   Layer      │          │
│  └──────────────┘    └──────────────┘    └──────────────┘          │
│         │                   │                   │                   │
│         ▼                   ▼                   ▼                   │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │ ELI5 Dataset │    │ 5-Agent      │    │ CSV Answers  │          │
│  │ (HuggingFace)│    │ LangGraph    │    │ JSON Metrics │          │
│  └──────────────┘    │ Pipeline     │    │ LaTeX Paper  │          │
│                      └──────────────┘    └──────────────┘          │
│                             │                                       │
│              ┌──────────────┼──────────────┐                       │
│              ▼              ▼              ▼                        │
│       ┌──────────┐   ┌──────────┐   ┌──────────┐                  │
│       │ Tavily   │   │ RAG      │   │ Ollama/  │                  │
│       │ Web      │   │ (BM25 +  │   │ vLLM     │                  │
│       │ Search   │   │ Semantic)│   │ LLM      │                  │
│       └──────────┘   └──────────┘   └──────────┘                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Processing Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                    5-Agent LangGraph Pipeline                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│                        ┌─────────────┐                              │
│                        │   START     │                              │
│                        └──────┬──────┘                              │
│                               │                                     │
│                               ▼                                     │
│                        ┌─────────────┐                              │
│                        │ Breakdown   │                              │
│                        │ Agent       │                              │
│                        └──────┬──────┘                              │
│                               │                                     │
│              ┌────────────────┼────────────────┐                    │
│              ▼                                 ▼                    │
│       ┌─────────────┐                   ┌─────────────┐            │
│       │ Reasoning   │                   │ Scientific  │            │
│       │ Agent       │                   │ Agent       │            │
│       └──────┬──────┘                   └──────┬──────┘            │
│              │                                 │                    │
│              └────────────────┬────────────────┘                    │
│                               ▼                                     │
│                        ┌─────────────┐                              │
│                        │ Synthesis   │                              │
│                        │ (Quality    │                              │
│                        │  Gate)      │                              │
│                        └──────┬──────┘                              │
│                               │                                     │
│                               ▼                                     │
│                        ┌─────────────┐                              │
│                        │ Creative    │                              │
│                        │ Agent       │                              │
│                        └──────┬──────┘                              │
│                               │                                     │
│                               ▼                                     │
│                        ┌─────────────┐                              │
│                        │    END      │                              │
│                        └─────────────┘                              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### vLLM Staged Batching Optimization

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Staged Batch Processing                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Original: N questions × 5 sequential LLM calls = N×5 calls        │
│  Staged:   5 stages × 1 batched vLLM call = 5 calls total         │
│                                                                     │
│  Stage 1: Breakdown       →  Batch N prompts                       │
│           ↓                                                        │
│  Stage 2: Retrieval       →  Parallel ThreadPool (no LLM)          │
│           Reasoning       →  Combined as one 2N-prompt              │
│           Extraction      →  batch vLLM call                       │
│           ↓                                                        │
│  Stage 3: Synthesis       →  Batch N prompts                       │
│           ↓                                                        │
│  Stage 4: Creative        →  Batch N prompts                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Key Components

### 3.1 Generation Scripts

| Script | Architecture | Backend | Use Case |
|--------|--------------|---------|----------|
| `main.py` | 5-agent LangGraph | Ollama + Tavily | Web search comparison |
| `vllm/simple_agent_vllm.py` | 4-agent optimized | vLLM + RAG + Wikipedia | High-throughput generation |
| `vllm/baseline_vllm.py` | Single LLM | vLLM | Baseline comparison |
| `run_batch_incremental.py` | 5-agent (broken) | Ollama | Deprecated — imports missing module |

### 3.2 Agent Roles

| Agent | Input | Output | Responsibility |
|-------|-------|--------|----------------|
| **Breakdown** | User question | Search queries, reasoning points | Decompose question into actionable tasks |
| **Reasoning** | Reasoning points | Logical analysis, conclusions | Causal reasoning without external facts |
| **Scientific** | Search queries | Extracted facts (12 target) | Gather facts via web search or RAG |
| **Synthesis** | Reasoning + facts | Strategy (balanced/reasoning_heavy/facts_heavy), curated points | Quality gate — evaluate and merge |
| **Creative** | Curated points | Final ELI5 answer | Generate child-friendly explanation |

### 3.3 Evaluation Scripts

| Script | Metrics | Input Format |
|--------|---------|--------------|
| `ragas_evaluator.py` | RAGAS (Factual Correctness, BLEU, CHRF, ROUGE, Answer Accuracy), BERTScore, Semantic Similarity, Perplexity | CSV with `question`, `generated_answer`, `reference_answers` |
| `evaluation.py` | ROUGE, Perplexity, Semantic Similarity, Entailment, LLM Judge | Same CSV format |
| `analyze_results.py` | Statistical summaries, visualizations | JSON evaluation results |

### 3.4 Data Structures

**AgentState (LangGraph)**
```python
class AgentState(TypedDict):
    query: str                    # Original question
    breakdown_output: str         # Breakdown agent summary
    reasoning_output: str         # Reasoning agent analysis
    scientific_output: str        # Scientific agent output
    final_answer: str             # Final ELI5 answer
    search_queries: List[str]     # From breakdown
    reasoning_points: List[str]   # From breakdown
    extracted_facts: List[Dict]   # From scientific
    synthesis_strategy: str       # "balanced" | "reasoning_heavy" | "facts_heavy"
    final_points: List[str]       # Curated points for creative
```

**Structured Output Schemas**
```python
class BreakdownOutput(BaseModel):
    summary: str
    search_queries: List[str]
    reasoning_points: List[str]

class ReasoningOutput(BaseModel):
    reasoning_analysis: List[str]
    conclusions: List[str]

class ScientificOutput(BaseModel):
    facts: List[Dict[str, str]]  # {"fact": "...", "text": "..."}

class SynthesisOutput(BaseModel):
    synthesis_strategy: Literal["reasoning_heavy", "facts_heavy", "balanced"]
    final_points: List[str]

class CreativeOutput(BaseModel):
    final_answer: str
```

---

## 4. Setup and Installation

### Prerequisites

- Python 3.10+
- Ollama (for local LLM inference)
- GPU with CUDA (recommended for vLLM)
- `pdflatex` and `bibtex` (for paper compilation)

### Installation Steps

```bash
# 1. Clone the repository
git clone <repository-url>
cd edutech-agent

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Cache the ELI5 dataset (required before first run)
python load_dataset.py

# 5. Start Ollama (in separate terminal)
ollama serve

# 6. Pull required models
ollama pull llama3.2:1b
ollama pull llama3.2:latest
ollama pull mistral:7b
```

### Environment Configuration

Create a `.env` file in the project root:

```bash
# Required for main.py (web search architecture)
TAVILY_API_KEY=your_tavily_api_key_here

# Optional: Google Sheets integration
# GOOGLE_SHEETS_CREDENTIALS=path/to/credentials.json
```

---

## 5. Usage Guide

### Quick Start

```bash
# Single question with vLLM + RAG
python vllm/simple_agent_vllm.py "Why is the sky blue?"

# Single question with LangGraph + Web Search
python main.py single config1 "Why is the sky blue?"
```

### Batch Processing

```bash
# vLLM staged batching (recommended)
python vllm/simple_agent_vllm.py --batch --start 0 --end 3000
python vllm/simple_agent_vllm.py --batch --start 0 --end 3000 --chunk-size 100

# Split processing across multiple runs
python vllm/simple_agent_vllm.py --batch --split 0  # First third
python vllm/simple_agent_vllm.py --batch --split 1  # Second third
python vllm/simple_agent_vllm.py --batch --split 2  # Final third
```

### Evaluation

```bash
# RAGAS evaluation
python ragas_evaluator.py --input generated_answers/answers.csv --output eval_ragas/

# Unified evaluation
python evaluation.py --input generated_answers/answers.csv --output evaluation_results/

# Quick test
python ragas_evaluator.py --test --response "The sky is blue because of Rayleigh scattering." --reference "The sky appears blue due to Rayleigh scattering of sunlight."
```

### Analysis and Paper

```bash
# Analyze results
python analyze_results.py evaluation_results/results.json

# Compile paper
cd paper && ./compile.sh
```

---

## 6. Dependencies

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `langgraph` | latest | Multi-agent workflow orchestration |
| `langchain-ollama` | latest | Ollama LLM integration |
| `vllm` | latest | High-throughput offline inference |
| `sentence-transformers` | latest | Semantic embeddings |
| `rank-bm25` | latest | BM25 retrieval |
| `wikipediaapi` | latest | Wikipedia search |
| `ragas` | latest | Evaluation framework |
| `bert-score` | latest | BERTScore metrics |
| `rouge-score` | latest | ROUGE metrics |
| `transformers` | latest | HuggingFace models |
| `tavily-python` | latest | Web search API |
| `datasets` | latest | HuggingFace dataset loading |
| `pandas` | latest | Data manipulation |
| `numpy` | latest | Numerical operations |
| `torch` | latest | PyTorch backend |

### Optional Dependencies

| Package | Purpose |
|---------|---------|
| `gspread` | Google Sheets integration |
| `google-oauth2` | Google API authentication |
| `pygraphviz` | Graph visualization (PNG export) |
| `matplotlib` | Result visualization |
| `seaborn` | Statistical plots |

---

## 7. File Structure

```
edutech-agent/
├── AGENTS.md                        # Agent rules and contracts
├── CLAUDE.md                        # Claude-specific instructions
├── DOCUMENTATION_INDEX.md           # Documentation index
├── RESEARCH_METHODOLOGY.md          # Research methodology docs
├── .env                             # Environment variables (secrets)
├── .gitignore                       # Git ignore rules
├── requirements.txt                 # Python dependencies
│
├── main.py                          # 5-agent LangGraph + Tavily web search
├── load_dataset.py                  # ELI5 dataset loader/cache
├── run_batch_incremental.py         # Batch runner (deprecated - broken import)
├── evaluation.py                    # Unified evaluation metrics
├── ragas_evaluator.py               # RAGAS + BERTScore evaluation
├── ragas_evaluator_llm_only.py      # RAGAS LLM-only evaluation
├── analyze_results.py               # Results analysis and visualization
│
├── vllm/
│   ├── simple_agent_vllm.py         # vLLM + RAG + Wikipedia (optimized)
│   └── baseline_vllm.py             # vLLM baseline (no agents)
│
├── paper/
│   ├── main.tex                     # LaTeX source
│   ├── references.bib               # Bibliography
│   ├── compile.sh                   # Compilation script
│   ├── sections/                    # LaTeX sections
│   ├── README.md                    # Paper documentation
│   ├── PAPER_OUTLINE.md             # Paper structure
│   ├── RESULTS_PRESENTATION.md      # Results formatting
│   ├── ANALYSIS_DOCUMENT.md         # Analysis documentation
│   └── COMPLETION_REPORT.md         # Project completion report
│
├── generated_answers/               # CSV outputs from generation
│   └── answers_0_1000.csv
│
├── evaluation_results/              # Consolidated metric reports
│
├── llm_metrics_output/              # LLM-as-judge evaluation JSONs
│
├── non_llm_metrics_output/          # Automated text metrics
│
├── outputs_llm_final/               # Final LLM judge summaries
│
├── human_evaluation/                # Human evaluation scripts
│   ├── analyze_human_eval.py
│   ├── prepare_human_eval.py
│   └── instructions.md
│
├── paper_analysis/                  # Paper analysis documents
│   ├── ARCHITECTURE_COMPARISON.md
│   ├── RESULTS_SUMMARY.md
│   └── ...
│
├── plans/                           # Project planning docs
│   ├── priority_list.md
│   └── gaps_in_research.md
│
└── .opencode/
    └── skills/
        ├── edutech-agent/
        │   └── SKILL.md             # This skill file
        └── research-paper-writing/
            └── SKILL.md             # Paper writing skill
```

---

## 8. Rules and Anti-Patterns

### Do's

- Always run `python load_dataset.py` before first generation
- Use `vllm/simple_agent_vllm.py --batch` for batch processing (not `run_batch_incremental.py`)
- Start Ollama with `ollama serve` before running any scripts
- Save after every question for resumability (built into batch scripts)
- Use `--chunk-size` to control memory usage during batch processing

### Don'ts

- Don't use `arch_1/simple_agent.py` — it doesn't exist
- Don't use `run_batch_incremental.py` — it imports from missing module
- Don't commit `.env` file or `.venv` directory
- Don't hardcode metrics in the paper — use output from evaluation scripts
- Don't run multiple Ollama instances simultaneously

---

## 9. Troubleshooting

| Problem | Cause | Solution |
|---------|-------|----------|
| `FileNotFoundError: eli5_dataset_cache.pkl` | Dataset not cached | Run `python load_dataset.py` |
| `ConnectionRefusedError: localhost:11434` | Ollama not running | Run `ollama serve` |
| `TAVILY_API_KEY not found` | Missing API key | Add to `.env` file |
| `ModuleNotFoundError: simple_agent` | Using deprecated script | Use `vllm/simple_agent_vllm.py` |
| GPU OOM with vLLM | Insufficient GPU memory | Reduce `GPU_MEMORY_UTILIZATION` or use smaller model |
| LaTeX compilation fails | Missing LaTeX tools | Install `texlive` or MacTeX |
| Timeout during batch processing | Ollama overloaded | Increase `TIMEOUT_SECONDS` or reduce `chunk_size` |

---

## 10. Performance Metrics

### Generation Speed Comparison

| Method | Questions/Minute | Notes |
|--------|------------------|-------|
| `main.py` (sequential) | ~2-5 | Depends on web search latency |
| `vllm/simple_agent_vllm.py` (batch) | ~20-50 | Staged batching optimization |
| `vllm/baseline_vllm.py` | ~30-60 | No agent overhead |

### Model Size vs Quality Tradeoff

| Model | Size | Speed | Quality |
|-------|------|-------|---------|
| `llama3.2:1b` | 1B | Fastest | Baseline |
| `llama3.2:3b` | 3B | Fast | Good |
| `mistral:7b` | 7B | Moderate | Best |

---

**Last Updated**: June 2026  
**Version**: 1.0
