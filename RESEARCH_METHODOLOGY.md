# Research Methodology Guide: Using Both Architectures

## Overview

This guide helps researchers understand when and how to use each architecture for different research objectives.

---

## 1. COMPARATIVE STUDY FRAMEWORK

### Research Question 1: "How does information retrieval method affect ELI5 quality?"

**Experiment Setup:**
```
Same questions → Architecture 1 (web search) vs Architecture 2 (RAG hybrid)
              ↓
Compare outputs on:
├─ Factual accuracy (against ground truth)
├─ Comprehensiveness (coverage of key concepts)
├─ Simplicity (ELI5 language appropriateness)
└─ Relevance (answering the original question)
```

**Expected Findings:**
- Arch 1: More current, potentially more coverage, but variable quality
- Arch 2: More consistent, focused on reasoning, potentially missing recent info

**Metrics:**
- BLEU/ROUGE against reference answers
- Factual consistency (verify facts against Wikipedia)
- Human evaluation on ELI5 quality

---

### Research Question 2: "What is the impact of multi-model orchestration?"

**Experiment Setup:**
```
Architecture 1 with different MODEL_CONFIGS:
├─ config1 (all lightweight)
├─ config2 (all medium)
├─ config3 (strong reasoning only)
├─ config4 (hybrid)
└─ config5 (all large)

Same questions → Each config
              ↓
Compare outputs on quality + latency
```

**Expected Findings:**
- Lightweight: Fastest but potentially lower quality
- Strong reasoning: Better logical analysis
- Strong creative: Better ELI5 explanations
- Trade-off between quality and latency

**Metrics:**
- Answer quality (human evaluation)
- Inference latency per agent
- Total pipeline time
- Cost-quality Pareto frontier

---

### Research Question 3: "How effective is hybrid search vs. pure web search?"

**Experiment Setup:**
```
Architecture 2 with ablation:
├─ Full hybrid (BM25 + semantic + Wikipedia)
├─ BM25 only
├─ Semantic only
├─ RAG only (no Wikipedia)
└─ Wikipedia only

Same questions → Each variant
              ↓
Compare on fact quality + retrieval speed
```

**Expected Findings:**
- BM25: Fast keyword matching, good for direct answers
- Semantic: Better at conceptual relevance
- Hybrid: Best recall
- Wikipedia addition: Better for definitions

**Metrics:**
- Retrieval precision (fact validation)
- Retrieval recall (coverage)
- Retrieval latency
- Final answer quality

---

## 2. BENCHMARKING FRAMEWORK

### Use Case: Establishing Reproducible Baseline

**Why Architecture 2:**
- Fixed dataset ensures reproducibility
- No web variance
- Others can replicate exact results

**Experimental Protocol:**
```
1. Setup:
   ├─ Fixed random seed
   ├─ Single model (llama3.2:1b)
   ├─ Loaded rag_cache (disk-persistent)
   └─ Test questions: n=100+ from ELI5 dataset

2. Execution:
   ├─ Run architecture 2 on all questions
   ├─ Log generation times
   ├─ Save outputs to CSV
   └─ Document exact versions (langchain, ollama, etc.)

3. Evaluation:
   ├─ Human evaluation (for sample)
   ├─ Automated metrics (ROUGE, factual consistency)
   ├─ Performance analysis (latency, throughput)
   └─ Reproducibility check (rerun, compare exact matches)

4. Publication:
   ├─ Provide exact reproducibility instructions
   ├─ Share rag_cache (disk cache file)
   ├─ Document environment (Python versions, etc.)
   └─ Enable community replication
```

**Expected Reproducibility:**
- 100% identical outputs on rerun (deterministic)
- Others get same results in same environment
- Enables fair comparison with other methods

---

## 3. REAL-WORLD EVALUATION FRAMEWORK

### Use Case: Measuring Performance on Current Information

**Why Architecture 1:**
- Real web results reflect current state
- Tests system's ability to handle latest information
- Measures quality with dynamic knowledge

**Experimental Protocol:**
```
1. Dataset:
   ├─ Questions about recent events (last 30 days)
   ├─ Questions about evergreen concepts (science, history)
   ├─ Mix of difficulty levels
   └─ n=50-100 questions

2. Execution:
   ├─ Run with each MODEL_CONFIG
   ├─ Log all web search results
   ├─ Record synthesis strategy (reasoning vs facts)
   └─ Measure end-to-end latency

3. Evaluation:
   ├─ Expert review (correctness, currency, clarity)
   ├─ Fact checking against recent news
   ├─ ELI5 appropriateness scoring
   └─ Model performance correlation

4. Analysis:
   ├─ Which model config performs best?
   ├─ Does web search depth matter?
   ├─ How often does synthesis choose "facts_heavy"?
   └─ Correlation: question complexity → model choice
```

**Expected Insights:**
- Best model config for real-world deployment
- Trade-off curves (quality vs. latency)
- When web search depth helps

---

## 4. COST-EFFECTIVENESS ANALYSIS

### Comparison: API Costs vs. Local Inference

**Architecture 1: Web Search Costs**
```
Cost per query:
├─ Tavily API: $0.01-0.05 per query (depends on plan)
├─ OpenAI (synthesis): ~$0.001 per query
├─ Data transfer: Minimal
└─ Total: ~$0.015-0.055 per query

For 10,000 queries/month:
├─ Tavily: ~$150-500/month
├─ Ollama (local): $0 (one-time hardware)
└─ Total: $150-500/month + hardware
```

**Architecture 2: Local RAG Costs**
```
Cost per query:
├─ Dataset setup (one-time): ~2 hours
├─ Disk space: ~500MB ($0.10 amortized)
├─ Ollama (local): $0 per query
└─ Total: ~$0.00001 per query

For 10,000 queries/month:
├─ Additional data transfer: Minimal
├─ Electricity: ~$5-20/month
└─ Total: ~$5-30/month
```

**Research Application:**
```
Cost comparison study:
├─ Calculate TCO for both
├─ Factor in quality differences
├─ Determine break-even point
└─ Recommend for different deployment scales
```

---

## 5. SCALABILITY & THROUGHPUT ANALYSIS

### Framework: How Systems Scale to Large Question Sets

**Architecture 1 Testing:**
```
Test with question batches:
├─ Small batch (10 questions): ~200-300s
├─ Medium batch (100 questions): ~2000-3000s (potential rate limits)
├─ Large batch (1000 questions): ???

Measure:
├─ Actual throughput (questions/hour)
├─ API rate limit hit points
├─ Cost per query (scales?)
└─ Bottleneck identification
```

**Architecture 2 Testing:**
```
Test with question batches:
├─ Small batch (10 questions): ~80-150s
├─ Medium batch (100 questions): ~800-1500s
├─ Large batch (1000 questions): ~8000-15000s

Measure:
├─ Actual throughput (questions/hour)
├─ CPU utilization (num_thread=8)
├─ Memory usage (embeddings cached)
└─ Bottleneck identification (CPU bound)
```

**Analysis:**
```
Create scaling curves:
├─ Latency vs. batch size
├─ Cost vs. throughput
├─ Resource utilization
└─ Break-even analysis
```

---

## 6. ABLATION STUDY FRAMEWORK

### Architecture 1 Ablations

**Variable 1: Search Depth**
```
Compare:
├─ Tavily "advanced" (current)
├─ Tavily "basic" (faster, less deep)
├─ Custom limit (fewer results)

Measure: Quality vs. Speed trade-off
```

**Variable 2: Number of Search Queries**
```
Use breakdown agent output:
├─ Use all 3-5 queries (current)
├─ Limit to top 2 queries
├─ Limit to top 1 query

Measure: Diminishing returns
```

**Variable 3: Facts Target Count**
```
Change FACTS_TARGET_COUNT:
├─ 8 facts
├─ 12 facts (current)
├─ 16 facts
├─ 20 facts

Measure: Quality vs. context size
```

---

### Architecture 2 Ablations

**Variable 1: Search Strategy**
```
Compare:
├─ BM25 + Semantic (current)
├─ BM25 only
├─ Semantic only
└─ Union vs. Intersection merge

Measure: Precision, Recall, F1
```

**Variable 2: Knowledge Sources**
```
Compare:
├─ RAG + Wikipedia (current)
├─ RAG only
├─ Wikipedia only
└─ Different RAG datasets

Measure: Fact quality by domain
```

**Variable 3: Batch Encoding**
```
Compare:
├─ Current: batch encode all
├─ Individual: encode each query
├─ Chunked: batch size=2

Measure: Speed-up factor
```

---

## 7. HUMAN EVALUATION PROTOCOL

### Recommended Rubric for ELI5 Quality

**Accuracy (0-5 scale)**
- 5: All facts correct, no errors
- 4: Mostly correct, minor factual issue
- 3: Some facts correct, some wrong
- 2: Mostly incorrect
- 1: Completely wrong

**Simplicity (0-5 scale)**
- 5: Perfect for 5-year-old, uses analogies
- 4: Good, mostly simple words
- 3: OK, some complex concepts
- 2: Too technical, hard to follow
- 1: Not suitable for children

**Completeness (0-5 scale)**
- 5: Fully answers the question
- 4: Good coverage, minor gaps
- 3: Covers main points
- 2: Incomplete, significant gaps
- 1: Barely addresses question

**Clarity (0-5 scale)**
- 5: Very clear flow, engaging
- 4: Clear and easy to follow
- 3: Understandable, some confusion
- 2: Hard to follow
- 1: Confusing, unclear

**Overall Quality (0-5 scale)**
- Aggregate of above

---

## 8. PUBLICATION GUIDELINES

### For Architecture 1 Studies

**Reproducibility Checklist:**
```
☐ Document Tavily API plan used
☐ Note exact model versions (ollama, langchain)
☐ Provide model configurations tested
☐ List Python version + dependencies
☐ Include exact prompts used (in appendix)
☐ Provide test dataset (or reference)
☐ Document random seeds if applicable
☐ Share code publicly
```

**Limitations to Acknowledge:**
```
☐ Web results may vary over time
☐ Tavily API changes may affect reproducibility
☐ Rate limits may impact large-scale runs
☐ Model selection complexity
☐ Dependency on internet connectivity
```

---

### For Architecture 2 Studies

**Reproducibility Checklist:**
```
☐ Document llama3.2:1b version
☐ Provide rag_cache dump (or rebuild instructions)
☐ Note SentenceTransformer version
☐ Python version + dependencies
☐ Include exact prompts used (in appendix)
☐ Document num_thread and timeout settings
☐ Provide all code and configs
☐ Exact test dataset (hard-coded or referenced)
```

**Strengths to Highlight:**
```
☐ 100% reproducible across runs
☐ No external dependencies (offline-capable)
☐ Fixed knowledge base (no staleness)
☐ Deterministic results enable comparison
☐ Low cost for scaling
```

---

## 9. RECOMMENDED RESEARCH ROADMAP

### Phase 1: Baseline & Characterization
1. Run Architecture 2 on 100 ELI5 questions
2. Establish reproducible baseline
3. Characterize performance (speed, quality)
4. Document exact setup

### Phase 2: Comparative Analysis
1. Run Architecture 1 on same 100 questions
2. Compare outputs on multiple dimensions
3. Analyze synthesis strategy choices
4. Measure cost-quality trade-offs

### Phase 3: Ablation Studies
1. Test model configurations (Architecture 1)
2. Test hybrid search variants (Architecture 2)
3. Measure impact of each component
4. Determine optimal configurations

### Phase 4: Scaling & Production
1. Test on large question sets (1000+)
2. Measure throughput and costs
3. Profile bottlenecks
4. Make deployment recommendations

### Phase 5: Publication & Release
1. Document findings
2. Release code + data
3. Enable reproducibility
4. Propose future improvements

---

## 10. EXPECTED RESEARCH CONTRIBUTIONS

### Unique Insights from Architecture 1
- Multi-model orchestration effectiveness
- Real-time information retrieval in ELI5
- Synthesis strategy selection patterns
- Cost-quality curves for multi-agent systems

### Unique Insights from Architecture 2
- Hybrid search effectiveness (BM25 + semantic)
- Offline ELI5 systems feasibility
- Batch encoding optimizations
- Reproducible benchmarking methodology

### Comparative Insights
- When to use local vs. web retrieval
- Information retrieval method impact on ELI5
- Model vs. retrieval bottlenecks
- Cost-effectiveness analysis for different scales

