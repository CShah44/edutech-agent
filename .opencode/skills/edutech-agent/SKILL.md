---
name: edutech-agent
description: Multi-agent LLM system for generating ELI5-style answers. Use when running generation, evaluation, analysis, or paper compilation for the edutech-agent project. Covers vLLM optimization, RAG pipelines, RAGAS/ROUGE/BERTScore evaluation, and LaTeX paper output.
---

# EduTech Agent — Project Skill

## Overview

A multi-agent LLM system that generates ELI5 (Explain Like I'm 5) answers by comparing architectures (web search vs RAG), evaluates with RAGAS/ROUGE/BERTScore, and produces a LaTeX research paper.

**Repository**: `edutech-agent`  
**Primary Language**: Python  
**Key Frameworks**: LangGraph, vLLM, Ollama, RAGAS, HuggingFace Transformers

---

## Project Structure

```
edutech-agent/
├── main.py                          # 5-agent LangGraph pipeline (web search)
├── vllm/
│   ├── simple_agent_vllm.py         # vLLM + RAG + Wikipedia (staged batching)
│   └── baseline_vllm.py             # vLLM baseline (no agents)
├── run_batch_incremental.py         # Batch runner (Ollama-based)
├── load_dataset.py                  # Dataset loader/cache creator
├── evaluation.py                    # Unified evaluation (ROUGE, perplexity, similarity, LLM judge)
├── ragas_evaluator.py               # RAGAS + BERTScore evaluation
├── analyze_results.py               # Results visualization and analysis
├── paper/
│   ├── main.tex                     # LaTeX source
│   ├── compile.sh                   # Paper compilation script
│   └── sections/                    # LaTeX section files
├── generated_answers/               # CSV outputs from generation
├── evaluation_results/              # Consolidated metric reports
├── llm_metrics_output/              # LLM-as-judge evaluation JSONs
├── non_llm_metrics_output/          # Automated text metrics (ROUGE, BERT-F1)
├── outputs_llm_final/               # Final LLM judge summaries
└── .env                             # Secrets (TAVILY_API_KEY)
```

---

## Critical Gotchas

| Issue | Solution |
|-------|----------|
| `arch_1/simple_agent.py` does NOT exist | Use `vllm/simple_agent_vllm.py` instead |
| Ollama required on port 11434 | Run `ollama serve` before any generation |
| No `.venv` in git | Create virtual environment manually |
| Dataset cache missing | Run `python load_dataset.py` once first |
| `run_batch_incremental.py` imports missing `simple_agent` | Use `vllm/simple_agent_vllm.py --batch` instead |

---

## Execution Workflow

### Step 1: Setup (Once)

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Cache the ELI5 dataset
python load_dataset.py

# Start Ollama (in separate terminal)
ollama serve
```

### Step 2: Generate Answers

**Option A: vLLM + RAG (Recommended)**
```bash
# Single question
python vllm/simple_agent_vllm.py "Why is the sky blue?"

# Batch processing (staged batching for speed)
python vllm/simple_agent_vllm.py --batch --start 0 --end 3000
python vllm/simple_agent_vllm.py --batch --start 0 --end 3000 --chunk-size 100
python vllm/simple_agent_vllm.py --batch --start 0 --end 3000 --model llama3.2:1b
```

**Option B: LangGraph + Web Search (requires Tavily API)**
```bash
# Single question
python main.py single config1 "Why is the sky blue?"

# Google Sheets integration
python main.py sheets config1

# Generate workflow visualization
python main.py graph
```

**Option C: Baseline (No Agents)**
```bash
python vllm/baseline_vllm.py --start 0 --end 1000 --output baseline_answers/llama3b_0_1000.csv
```

### Step 3: Evaluate

```bash
# RAGAS evaluation (comprehensive metrics)
python ragas_evaluator.py --input generated_answers/answers.csv --output eval_ragas/

# Unified evaluation (ROUGE, perplexity, similarity, LLM judge)
python evaluation.py --input generated_answers/answers.csv --output evaluation_results/

# Limit evaluation rows
python ragas_evaluator.py --input generated_answers/answers.csv --output eval_ragas/ --max-rows 50
```

### Step 4: Analyze

```bash
python analyze_results.py evaluation_results/results.json
```

### Step 5: Compile Paper

```bash
cd paper && ./compile.sh
# OR manually:
cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

---

## Architecture

### 5-Agent LangGraph Pipeline (main.py)

```
START → Breakdown → Reasoning ─┐
               │                ├→ Synthesis → Creative → END
               └→ Scientific ──┘
```

| Agent | Role | Model |
|-------|------|-------|
| **Breakdown** | Decomposes question into search queries + reasoning points | Configurable |
| **Reasoning** | Logical analysis of reasoning points | Configurable |
| **Scientific** | Web search (Tavily) + fact extraction | Configurable |
| **Synthesis** | Quality gate — evaluates reasoning vs facts, creates curated points | Configurable |
| **Creative** | Generates ELI5 explanation from curated points | Configurable |

### vLLM Optimized Pipeline (vllm/simple_agent_vllm.py)

```
START → Breakdown → Parallel Analysis → Synthesis → Creative → END
                   (RAG + Wikipedia)
```

**Key Optimization**: Staged batching — N questions pass through each stage as a single batched vLLM call, turning N×5 sequential calls into 5 batch calls.

| Stage | Batch Size | Description |
|-------|------------|-------------|
| 1. Breakdown | N prompts | Decompose all questions |
| 2. Parallel Analysis | 2N prompts (reasoning + extraction) | RAG/Wikipedia retrieval + LLM extraction |
| 3. Synthesis | N prompts | Quality gate + curation |
| 4. Creative | N prompts | ELI5 generation |

### Model Configurations (main.py)

| Config | Reasoning | Scientific | Creative |
|--------|-----------|------------|----------|
| `config1` | llama3.2:1b | llama3.2:1b | llama3.2:1b |
| `config2` | llama3.2:latest | llama3.2:latest | llama3.2:latest |
| `config3` | mistral:7b | llama3.2:latest | llama3.2:latest |
| `config4` | mistral:7b | llama3.2:latest | mistral:7b |
| `config5` | mistral:7b | mistral:7b | mistral:7b |

---

## Evaluation Metrics

### RAGAS Metrics (ragas_evaluator.py)
- Factual Correctness (LLM-based)
- BLEU Score
- CHRF Score
- ROUGE (1, 2, L)
- Answer Accuracy (LLM-based)
- BERTScore (P/R/F1)
- Semantic Similarity (SentenceTransformer)
- Perplexity (GPT-2)

### Unified Metrics (evaluation.py)
- ROUGE (1, 2, L)
- Perplexity (GPT-2)
- Semantic Similarity (Sentence Transformers)
- Entailment (DeBERTa)
- LLM as a Judge (Llama 2 via Ollama)

---

## Output Directories

| Directory | Contents |
|-----------|----------|
| `generated_answers/` | Raw CSV outputs from generation scripts |
| `evaluation_results/` | Consolidated metric reports |
| `llm_metrics_output/` | LLM-as-judge evaluation JSONs |
| `non_llm_metrics_output/` | Automated text metrics (ROUGE, BERT-F1) |
| `outputs_llm_final/` | Final LLM judge summaries (14 configs) |

---

## Key Dependencies

| Package | Purpose |
|---------|---------|
| `langgraph` | Multi-agent workflow orchestration |
| `langchain-ollama` | Ollama LLM integration |
| `vllm` | High-throughput offline inference |
| `sentence-transformers` | Semantic embeddings for RAG |
| `rank-bm25` | BM25 retrieval for RAG |
| `wikipediaapi` | Wikipedia search |
| `ragas` | Evaluation metrics framework |
| `bert-score` | BERTScore evaluation |
| `rouge-score` | ROUGE metrics |
| `transformers` | HuggingFace models |
| `tavily-python` | Web search API |
| `gspread` | Google Sheets integration |
| `datasets` | HuggingFace dataset loading |

---

## Environment Variables

```bash
# .env file
TAVILY_API_KEY=your_tavily_api_key_here  # Required for main.py web search
```

---

## Common Commands Reference

```bash
# Dataset
python load_dataset.py                    # Cache ELI5 dataset

# Generation
python vllm/simple_agent_vllm.py "query"  # Single question (vLLM)
python vllm/simple_agent_vllm.py --batch --start 0 --end 1000  # Batch (vLLM)
python main.py single config1 "query"     # Single question (LangGraph)
python main.py sheets config1             # Google Sheets batch

# Evaluation
python ragas_evaluator.py --input FILE --output DIR
python evaluation.py --input FILE --output DIR

# Analysis
python analyze_results.py RESULTS.json

# Paper
cd paper && ./compile.sh
```

---

## Extension Points

1. **Add new model configuration**: Edit `MODEL_CONFIGS` dict in `main.py`
2. **Add new evaluation metric**: Add to `evaluation.py` or `ragas_evaluator.py`
3. **Modify agent prompts**: Edit system prompts in `main.py` or `vllm/simple_agent_vllm.py`
4. **Add new agent**: Add node function and wire into `create_graph()`
5. **Customize batching**: Adjust `chunk_size` in vLLM batch commands

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `FileNotFoundError: eli5_dataset_cache.pkl` | Run `python load_dataset.py` |
| `ConnectionRefusedError: localhost:11434` | Run `ollama serve` in separate terminal |
| `TAVILY_API_KEY not found` | Add key to `.env` file |
| `ModuleNotFoundError: simple_agent` | Use `vllm/simple_agent_vllm.py` instead |
| GPU OOM with vLLM | Reduce `GPU_MEMORY_UTILIZATION` or use smaller model |
| LaTeX compilation fails | Ensure `pdflatex` and `bibtex` are installed |
