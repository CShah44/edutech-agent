# Paper Outline: Multi-Agent ELI5 System
## 4-6 Page ArXiv Preprint Format

**Title**: Multi-Agent Orchestration for Simplified Explanations: A Comparative Study of Architectures

**Target**: 4-6 pages, ArXiv preprint format

---

## Section Breakdown (with Page Allocations)

### Abstract (150-200 words, 0.2 pages)

**Content**:
- **Problem**: ELI5 explanations require balancing accuracy and accessibility
- **Approach**: Multi-agent architecture with 4-stage pipeline (breakdown → parallel analysis → synthesis → creative) + RAG integration
- **Method**: Comparative evaluation across 7 models (1B-7B params), 30K samples, 11+ metrics
- **Key Finding**: Accuracy-quality trade-off: -38.2% LLM accuracy BUT +34.2% ROUGE1, +10.0% BERT-F1
- **Contribution**: (1) Novel multi-agent architecture with staged batching, (2) Comprehensive empirical study, (3) First documentation of accuracy-quality paradox

**Key Numbers**: 7 models, 30K samples, -38.2%, +34.2%

---

### 1. Introduction (0.75 pages, ~450 words)

**Paragraph 1: Motivation** (100 words)
- ELI5 task: simplify complex topics for non-experts
- Critical for education, science communication, accessibility
- Challenge: balance factual accuracy with simplicity
- Current approaches: single-pass prompting with large LLMs

**Paragraph 2: Limitations & Gap** (100 words)
- Small models struggle with complex reasoning
- No external grounding → potential hallucinations
- Single-pass lacks explicit reasoning decomposition
- Trade-offs between accuracy and accessibility unclear

**Paragraph 3: Our Approach** (150 words)
- Multi-agent orchestration with 4-stage pipeline
- RAG integration (hybrid BM25 + semantic search)
- Staged batching for efficiency (100x reduction in vLLM calls)
- Comparative study: baseline vs. multi-agent across 7 models

**Paragraph 4: Key Findings** (100 words)
- Accuracy-quality paradox: lower LLM accuracy, higher text quality
- Consistent across all models (architectural, not model-specific)
- 100% success rate, practical latency (29.64s)
- Staged batching demonstrates computational efficiency

**Paragraph 5: Contributions** (100 words summarizing 3 contributions)
1. Novel multi-agent architecture with adaptive synthesis
2. Comprehensive evaluation (7 models, 14 configs, 66K samples, 11+ metrics)
3. Discovery and analysis of accuracy-quality trade-off

---

### 2. Related Work (0.5 pages, ~300 words)

**Paragraph 1: Multi-Agent LLM Systems** (100 words)
- LangGraph, AutoGPT, MetaGPT
- Decomposition-based approaches
- Our contribution: staged batching, RAG integration for ELI5

**Paragraph 2: RAG and Question Answering** (100 words)
- Retrieval-augmented generation (Lewis et al.)
- Hybrid retrieval (BM25 + dense)
- Question answering benchmarks
- Our contribution: adaptive synthesis strategy

**Paragraph 3: ELI5 and Evaluation** (100 words)
- ELI5 dataset (Fan et al. 2019)
- RAGAS evaluation framework
- ROUGE, BERT-Score, LLM-as-judge
- Our contribution: comprehensive metric comparison

---

### 3. Methodology (1.5 pages, ~900 words)

**3.1 Task and Dataset** (150 words)
- ELI5 task definition
- Dataset: sentence-transformers/eli5
- Sample characteristics (questions, reference answers)
- Evaluation split (30K samples)

**3.2 Baseline Architecture** (150 words)
- Single-pass prompting approach
- Prompt template structure (high-level)
- vLLM batch inference
- Configuration: gpu_mem=0.60, batch_size=20, temp=0.4

**3.3 Multi-Agent Architecture** (400 words)
- **Stage 1**: Breakdown node
  - Question decomposition
  - Search queries + reasoning points generation
  - Structured output (Pydantic schema)
  
- **Stage 2**: Parallel analysis
  - RAG retrieval (hybrid BM25 + semantic, OpenThoughts-114k)
  - Wikipedia integration (parallel)
  - Reasoning analysis (vLLM)
  - Scientific extraction (vLLM)
  
- **Stage 3**: Synthesis node
  - Quality evaluation
  - Strategy selection: reasoning_heavy | facts_heavy | balanced
  - Point curation (4-6 final points)
  
- **Stage 4**: Creative node
  - ELI5 transformation
  - Simple language generation

- **Key Innovation**: Staged batching (5 vLLM calls for N questions)

**3.4 Implementation Details** (200 words)
- LangGraph orchestration
- vLLM configuration comparison (Table)
- Temperature schedule (0.1 → 0.5)
- RAG: hybrid BM25 + semantic (all-MiniLM-L6-v2)
- Query deduplication (50% reduction)

---

### 4. Experiments (0.75 pages, ~450 words)

**4.1 Experimental Setup** (200 words)
- 7 models tested (Table 1)
- Parameter range: 1B-7B
- Model families: LLaMA, Qwen, Gemma, Mistral
- 14 configurations total (baseline + multi-agent each)
- Scale: 30,000 samples (full), 1,000 (comprehensive)
- Deterministic sampling (seed=22)
- Hardware: vLLM on GPU

**4.2 Evaluation Metrics** (250 words)

**LLM-Based**:
- Answer accuracy (0-1): Llama-2-13b judge
- Correctness, completeness, overall (0-10)

**Automatic Text Metrics**:
- Lexical: ROUGE1/2/L, BLEU, CHRF
- Semantic: BERT-Score (F1), sentence similarity
- LM-based: Perplexity (GPT-2)

**Configuration**:
- Judge: Llama-2-13b-chat-hf
- Similarity: all-MiniLM-L6-v2
- BERTScore: microsoft/deberta-xlarge-mnli
- Temperature: 0.0 (evaluation), seed: 22

---

### 5. Results (1.0 pages, ~600 words)

**5.1 Primary Finding: Accuracy-Quality Trade-off** (250 words)
- Table 2: LLM accuracy comparison
  - All 7 models show decline
  - Average: -38.2% (0.557 → 0.341)
  - Range: -17.4% to -45.9%
  
- Table 3: Text quality metrics (Gemma-2-2b-it)
  - ROUGE1: +34.2% ⬆️
  - BERT-F1: +10.0% ⬆️
  - RougeL: +31.6% ⬆️
  
- Paradox: Lower accuracy, higher text quality

**5.2 Pattern Consistency** (150 words)
- Architectural, not model-specific
- Consistent across 1B-7B parameters
- All model families show same pattern
- Suggests fundamental trade-off

**5.3 Text Quality Analysis** (100 words)
- Lexical overlap improvements (ROUGE)
- Semantic similarity gains (BERT)
- Perplexity increase (+92.7%)
- More diverse/creative generations

**5.4 Performance and Robustness** (100 words)
- Table 4: Summary statistics
- 100% success rate (1,000/1,000)
- Avg. generation time: 29.64s
- Staged batching: 100x vLLM call reduction

---

### 6. Discussion (0.5 pages, ~300 words)

**6.1 Interpreting the Trade-off** (150 words)

**Why Lower LLM Accuracy?**
- Style mismatch with judge model
- Structured outputs vs. natural flow
- Simplicity emphasis may sacrifice nuance
- Over-grounding constrains creativity

**Why Higher Text Quality?**
- Consistent structure → higher ROUGE
- RAG grounding → better semantic alignment
- Explicit curation → more focused explanations
- Multi-stage refinement → coherent text

**6.2 Practical Implications** (150 words)

**When to Use Baseline**:
- Simple questions, direct answers needed
- Low-latency requirements (<1s)
- LLM judge accuracy is primary metric

**When to Use Multi-Agent**:
- Complex questions requiring reasoning
- External grounding important
- Text structure/consistency matters
- Computational efficiency at scale

**Future Work**:
- Hybrid approach (multi-agent retrieval + single-pass generation)
- Fine-tuning to recover accuracy
- Alternative evaluation metrics (human judges)

---

### 7. Conclusion (0.25 pages, ~150 words)

**Paragraph 1: Summary** (75 words)
- Multi-agent architecture for ELI5
- 4-stage pipeline with RAG integration
- Comprehensive evaluation (7 models, 66K samples)
- Accuracy-quality trade-off discovered

**Paragraph 2: Key Takeaways** (75 words)
- Trade-off is architectural, not model-specific
- Choice depends on use case priorities
- Staged batching provides efficiency gains
- Opens questions about evaluation metrics alignment
- Future work: hybrid approaches, human evaluation

---

## References (0.3 pages)

**Essential Citations**:

1. **Datasets**:
   - ELI5: Fan et al. 2019
   - OpenThoughts-114k: HuggingFace
   - sentence-transformers/eli5

2. **Frameworks**:
   - vLLM: Kwon et al. 2023
   - LangChain/LangGraph: Harrison Chase
   - RAGAS: Exploding Gradients

3. **Evaluation Metrics**:
   - ROUGE: Lin 2004
   - BLEU: Papineni et al. 2002
   - BERT-Score: Zhang et al. 2020
   - Sentence-Transformers: Reimers & Gurevych 2019

4. **Models**:
   - LLaMA: Touvron et al. 2023
   - Mistral: Jiang et al. 2023
   - Qwen: Bai et al. 2023
   - Gemma: Google 2024

5. **Retrieval**:
   - BM25: Robertson & Zaragoza 2009
   - RAG: Lewis et al. 2020

---

## Page Budget Summary

| Section | Pages | Words |
|---------|-------|-------|
| Abstract | 0.2 | 175 |
| Introduction | 0.75 | 450 |
| Related Work | 0.5 | 300 |
| Methodology | 1.5 | 900 |
| Experiments | 0.75 | 450 |
| Results | 1.0 | 600 |
| Discussion | 0.5 | 300 |
| Conclusion | 0.25 | 150 |
| References | 0.3 | - |
| **TOTAL** | **5.75** | **~3,325** |

Plus 4 tables (0.25 pages combined) = **~6 pages total**

---

## Tables to Include

1. **Table 1**: Model configurations (7 models with parameter counts)
2. **Table 2**: LLM accuracy comparison (primary finding)
3. **Table 3**: Text quality metrics (representative model)
4. **Table 4**: Evaluation summary statistics

All tables ready in RESULTS_PRESENTATION.md

---

## Key Messages (One-Sentence Summary per Section)

- **Abstract**: Multi-agent ELI5 architecture shows -38.2% LLM accuracy but +34.2% text quality
- **Intro**: We propose a 4-stage multi-agent system for ELI5 with RAG integration
- **Related**: Builds on multi-agent LLMs, RAG, and ELI5 evaluation frameworks
- **Method**: Baseline (single-pass) vs. multi-agent (4-stage with staged batching)
- **Experiments**: 7 models, 14 configs, 66K samples, 11+ metrics
- **Results**: Consistent accuracy-quality trade-off across all models
- **Discussion**: Trade-off is architectural; choice depends on use case
- **Conclusion**: Novel architecture and empirical finding; future work on hybrid approaches

---

## Writing Principles

1. **Evidence-based**: Every claim backed by data from ANALYSIS_DOCUMENT.md
2. **Concise**: 4-6 pages means every sentence counts
3. **Clear**: High-level focus (no code, per user request)
4. **Reproducible**: Sufficient detail for replication
5. **Honest**: Acknowledge limitations and trade-offs

---

**Status**: ✅ Outline complete and ready for section writing

**Next Step**: Write individual LaTeX section files using this outline
