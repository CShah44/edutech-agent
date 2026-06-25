# AGENTS.md

Multi-agent LLM system for generating ELI5-style answers. Compares architectures (web search vs RAG), evaluates with RAGAS/ROUGE/BERTScore, and produces a LaTeX research paper.

## Critical Gotchas

- **Missing file**: `arch_1/simple_agent.py` does NOT exist. `run_batch_incremental.py` imports from it and will fail. Use `vllm/simple_agent_vllm.py` instead.
- **Ollama required**: Must be running locally on port 11434 before any generation or evaluation. Start with `ollama serve`.
- **No `.venv` in git**: Virtual environment setup is not in the repo. Ensure dependencies are installed before running.
- **Dataset cache**: Run `python load_dataset.py` once to create `eli5_dataset_cache.pkl`. Without it, generation scripts fail.
- **baseline_llama.py**: Does NOT exist. Use `vllm/baseline_vllm.py` instead.

## Architecture

| Entry Point | What It Does | Output |
|-------------|--------------|--------|
| `main.py` | 5-agent LangGraph pipeline (breakdown → scientific → reasoning → synthesis → creative) with Tavily web search | Direct answers |
| `vllm/simple_agent_vllm.py` | Local vLLM + RAG (BM25 + semantic) + Wikipedia, staged batching | CSV answers |
| `vllm/baseline_vllm.py` | Single-LLM baseline (no agents) via vLLM | CSV answers |
| `run_batch_incremental.py` | Batch runner (imports missing `simple_agent`) | `generated_answers/answers_0_1000.csv` |

## Execution Order

```bash
# 1. Setup (once)
python load_dataset.py
ollama serve  # In separate terminal

# 2. Generate answers (pick one)
python vllm/simple_agent_vllm.py --batch --start 0 --end 1000
# OR
python vllm/baseline_vllm.py --start 0 --end 1000 --output baseline_answers/llama3b_0_1000.csv

# 3. Evaluate (non-LLM metrics + LLM judge)
python ragas_evaluator.py --input generated_answers/answers.csv --output eval_ragas/
python evaluation.py --input generated_answers/answers.csv --output evaluation_results/

# 4. Analyze
python analyze_results.py evaluation_results/results.json

# 5. Paper (from paper/ directory)
cd paper && ./compile.sh
```

## Key Dependencies

- **Tavily API**: Requires `TAVILY_API_KEY` in `.env` (for `main.py` web search only)
- **Ollama**: Local LLM inference on `localhost:11434`
- **HuggingFace models**: Downloaded automatically for BERTScore, perplexity
- **Google Sheets API**: Optional, requires `credentials.json`

## Evaluation Output Directories

- `generated_answers/` — Raw CSV outputs from generation
- `evaluation_results/` — Consolidated metric reports (Llama2:13b judge, ROUGE, similarity)
- `non_llm_metrics_output/` — Automated text metrics for all 14 configs (ROUGE, BERT-F1, BLEU, CHRF, similarity, perplexity; 30,000 samples each)
- `llm-metrics-50-samples-gpt4.1/` — GPT-4.1 LLM judge evaluation (answer accuracy, 50 samples per config, seed=18)
- `llm_metrics_llama3.3/` — Llama-3.3-70B LLM judge evaluation (correctness, completeness, ELI5 quality, overall; 50 samples per config, seed=42)
- `human_evaluation/` — Human evaluation data (2 evaluators, 150 blind samples, 4 metrics, 7 models)

## Paper Compilation

Run from `paper/` directory:
```bash
./compile.sh
# OR manually: pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

**All numbers in the paper must come from `llm-metrics-50-samples-gpt4.1/`, `llm_metrics_llama3.3/`, `non_llm_metrics_output/`, and `evaluation_results/`. Never hardcode metrics.**

## Batch Processing Notes

- `run_batch_incremental.py` saves after every question (resumable)
- 10s delay after every 10 questions to avoid Ollama overload
- 120s timeout per question with 2 retries
- Editable at top of file: `TOTAL_QUESTIONS`, `OUTPUT_FILE`, `TIMEOUT_SECONDS`

## File Conventions

- `*.py` — Scripts are run directly, not imported as modules
- `paper/` — LaTeX source, compile with `./compile.sh`
- `vllm/` — vLLM-optimized implementations
- `.env` — Secrets (TAVILY_API_KEY, never committed)
