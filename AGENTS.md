# AGENTS.md

Research project: multi-agent LLM system that generates ELI5-style answers using LangGraph/LangChain, with evaluation pipelines and a LaTeX research paper.

## Architecture

- `main.py` — **5-agent LangGraph pipeline** (breakdown → scientific → reasoning → synthesis → creative). Uses Tavily web search. Configurable per-agent models via `MODEL_CONFIGS`.
- `run_batch_incremental.py` — **Batch runner** that calls `simple_agent` (not `main.py`). Processes ELI5 dataset questions incrementally, saves after each question, handles timeouts/restarts.
- `evaluation.py` — Non-LLM metrics: ROUGE, perplexity (GPT-2), semantic similarity, entailment, plus LLM-as-judge via Ollama.
- `ragas_evaluator.py` — Full evaluation including RAGAS metrics (factual correctness, BLEU, CHRF, ROUGE, answer accuracy, BERTScore, semantic similarity, perplexity).
- `baseline_llama.py` — Single-LLM baseline (no agents). Generates answers directly.
- `vllm/` — vLLM-based implementations (`baseline_vllm.py`, `simple_agent_vllm.py`).
- `paper/` — LaTeX paper source (`main.tex`, `sections/`, `references.bib`).

## Entry Points & Execution

### 1. Dataset Setup (one-time)
```bash
python load_dataset.py          # Downloads ELI5, caches to eli5_dataset_cache.pkl
```

### 2. Generate Answers
**Multi-agent (main.py):**
```bash
python main.py single config1 "Why is the sky blue?"
python main.py sheets config1   # Reads from Google Sheets
python main.py graph            # Generate workflow visualization
```

**Batch processing (run_batch_incremental.py):**
```bash
python run_batch_incremental.py # Calls simple_agent.answer_question(), outputs to generated_answers/answers_0_1000.csv
```
- Editable at top of file: `TOTAL_QUESTIONS`, `OUTPUT_FILE`, `TIMEOUT_SECONDS`, `DELAY_DURATION`
- Resumes automatically if output CSV exists
- 10s rest after every 10 questions to avoid Ollama overload

**Baseline:**
```bash
python baseline_llama.py --start 0 --end 1000 --output baseline_answers/llama3b_0_1000.csv
```

### 3. Evaluate
```bash
# Full突然被要求完成这个任务用户说：

《AGENTS.md》文件已创建。内容概览：

- **架构**：5 智能体 LangGraph 管道、批处理运行器、评估脚本、基线和 vLLM 变体、LaTeX 论文。
- **执行流程**：数据集设置 → 生成答案 → 评估 → 分析 → 论文；包含具体命令和输出路径。
- **关键依赖项**：Tavily API、Ollama（端口 11434）、.env 凭证、Hugging Face 模型缓存。
- **操作注意事项**：无虚拟环境设置步骤；无测试/检查/CI 配置；开发工作流程的直接脚本调用；批处理可恢复并配置超时；三个结果目录用于指标输出。
- **论文说明**：所有数字必须从 `llm_metrics_output/`、`non_llm_metrics_output/` 和 `evaluation_results/` 中提取，切勿硬编码。构建流程：`pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex`

需要我进行任何调整吗？