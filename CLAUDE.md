# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an educational AI agent research project that generates ELI5 (Explain Like I'm 5) style answers using multi-agent systems built on LangGraph and LangChain. The project compares different agent architectures and LLM configurations, evaluates them using multiple metrics (RAGAS, ROUGE, BERTScore, etc.), and is being used to write a research paper for academic publication.

## Architecture

### Multi-Agent System (main.py)
The core system uses a **LangGraph state machine** with specialized agents:

1. **Breakdown Agent** - Decomposes questions into search queries and reasoning points
2. **Scientific Agent** - Retrieves factual information using Tavily search API and knowledge bases
3. **Reasoning Agent** - Performs logical analysis using parametric knowledge
4. **Synthesis Agent** - Combines facts and reasoning into structured points
5. **Creative Agent** - Transforms synthesized content into ELI5-style answers

**Key architectural patterns:**
- State flows through `AgentState` TypedDict with structured fields
- LLM responses use Pydantic schemas for structured output parsing
- Caching mechanisms (`llm_cache`, `graph_cache`, `agent_cache`) reduce redundant LLM calls
- Model configurations support testing different LLM combinations (see `MODEL_CONFIGS` dict)

### Alternative Architectures
- **arch_1/** - Original 5-agent architecture (breakdown → scientific → reasoning → synthesis → creative)
- **vllm/** - Versions using vLLM for faster inference
- **baseline_llama.py** - Simple single-LLM baseline without agent framework

### Data Flow
```
ELI5 Dataset → Agent System → Generated Answers → Evaluation → Results Analysis → LaTeX Paper
```

## Common Development Commands

### Dataset Setup
```bash
# First-time setup: download and cache ELI5 dataset
python load_dataset.py
```

### Generate Answers

#### Multi-agent system
```bash
# Generate answers with specific model config and question range
python run_batch_incremental.py
# Outputs to generated_answers/answers_0_1000.csv
```

#### Baseline (no agents)
```bash
python baseline_llama.py --start 0 --end 1000 --output baseline_answers/llama3b_0_1000.csv
```

### Evaluation

#### Full evaluation with all metrics
```bash
python ragas_evaluator.py --input generated_answers/answers.csv --output eval_ragas/
```

#### Non-LLM metrics only (faster)
```bash
python evaluation.py --input generated_answers/answers.csv --output evaluation_results/
```

#### Quick single-pair test
```bash
python ragas_evaluator.py --test --response "answer text" --reference "reference text"
```

### Results Analysis
```bash
python analyze_results.py evaluation_results/results.json
# Generates plots and statistical summaries in results_analysis/ subdirectory
```

## Key Dependencies

- **LangChain/LangGraph** - Agent orchestration framework
- **Ollama** - Local LLM inference (expects localhost:11434)
- **Tavily API** - Web search for scientific facts (requires TAVILY_API_KEY in .env)
- **RAGAS** - LLM-based evaluation metrics
- **HuggingFace Transformers** - BERTScore, perplexity calculations
- **Google Sheets API** - Question ingestion (requires credentials.json)

## Environment Setup

Required environment variables (`.env`):
```
TAVILY_API_KEY=your_key_here
```

Required files:
- `credentials.json` - Google Service Account credentials (for Sheets API)
- `eli5_dataset_cache.pkl` - Created by load_dataset.py

## Ollama Configuration

The system expects Ollama running locally with these models:
- `llama3.2:1b`, `llama3.2:latest` (3B)
- `mistral:7b`
- `llama2:13b` (for evaluation)

Start Ollama before running experiments:
```bash
ollama serve
```

## Research Paper Context

This codebase is being analyzed to write a research paper. When working on LaTeX output:

1. **Extract all results from actual data** - Never hardcode metrics
2. **Follow academic format** - Use IEEEtran or ACM templates
3. **Structure output as**:
   ```
   paper/
     main.tex
     sections/introduction.tex
     sections/methodology.tex
     sections/results.tex
     sections/conclusion.tex
     references.bib
   ```
4. **Cite all libraries** - langchain, ragas, sentence-transformers, etc.
5. **Results locations**:
   - Generated answers: `generated_answers/`
   - Evaluation metrics: `evaluation_results/`, `llm_metrics_output/`, `non_llm_metrics_output/`
   - Analysis plots: Created by analyze_results.py

## File Organization

- **Main implementations**: `main.py`, `arch_1/simple_agent.py`, `baseline_llama.py`
- **Evaluation scripts**: `evaluation.py`, `ragas_evaluator.py`, `ragas_evaluator_llm_only.py`
- **Analysis**: `analyze_results.py`
- **Utilities**: `load_dataset.py`, `run_batch_incremental.py`
- **Results**: `generated_answers/`, `evaluation_results/`, `results_new/`, `outputs_llm_final/`
- **Architecture variants**: `arch_1/`, `vllm/`

## Performance Notes

- **Timeouts**: Questions have 120s timeout with retry logic in run_batch_incremental.py
- **Batch processing**: Saves after every question for resumability
- **Rate limiting**: 10s delay after every 10 questions to prevent Ollama overload
- **Caching**: Multiple cache layers reduce redundant LLM calls
- **Concurrency**: ThreadPoolExecutor used for parallel LLM requests in evaluation

## Known Constraints

- Context limit: MAX_CONTEXT_CHARS = 3900 characters to stay under token limits
- Max sources per query: 5-10 (configurable)
- Ollama must be running locally on port 11434
- Google Sheets integration requires valid service account credentials



## Project Structure
Here is the project structure logically organized by function:
1. Core Architecture & Generation
- main.py: Architecture 1. A 5-stage LangGraph pipeline (Breakdown → Scientific → Reasoning → Synthesis → Creative) using Tavily web search and configurable LLMs.
- vllm/simple_agent_vllm.py: Architecture 2. A fully local, vLLM-optimized pipeline using RAG (BM25 + semantic) and Wikipedia API, featuring highly efficient "staged batching".
- baseline_llama.py & vllm/baseline_vllm.py: Single-pass generation scripts used as experimental baselines to compare against the multi-agent systems.
- run_batch_incremental.py: Handles batch processing with auto-resumability and rate-limiting protections.
- load_dataset.py: Downloads and caches the HuggingFace sentence-transformers/eli5 dataset locally.
2. Evaluation Pipeline
- evaluation.py: Computes traditional NLP metrics (ROUGE, BERTScore, GPT-2 Perplexity, Semantic Similarity) and runs a basic Ollama LLM judge.
- ragas_evaluator.py: Runs comprehensive RAGAS metrics (Factual Correctness, Answer Accuracy, BLEU, CHRF) via an OpenAI-compatible API.
- analyze_results.py: Aggregates the JSON/CSV evaluation outputs to calculate the final statistics used in the paper.
3. Data & Results Directories
- generated_answers/ & results_new/: Stores raw CSV outputs containing the generated ELI5 answers.
- outputs_llm_final/: Contains JSON summaries of LLM-as-a-judge evaluations (Answer Accuracy) for the 14 model configurations.
- non_llm_metrics_output/: Contains JSON summaries of automated text metrics (ROUGE, BERT-F1, etc.) for all configurations.
- evaluation_results/: Stores consolidated metric reports.
4. Research Paper & Documentation
- paper/: LaTeX source files (main.tex, sections/, references.bib) for the compiled 6-page research paper.
- paper_analysis/: Detailed markdown notes comparing architectures, outlining results, and planning methodologies.
- DOCUMENTATION_INDEX.md & RESEARCH_METHODOLOGY.md: Detailed guides on the experimental framework and architectural differences.
- AGENTS.md & CLAUDE.md: Operational instructions and commands for AI assistants working in the repository.