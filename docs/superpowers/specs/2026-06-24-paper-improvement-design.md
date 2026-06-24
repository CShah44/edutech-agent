# Paper Improvement Design: Architecture-First Rewrite

**Date**: 2026-06-24
**Goal**: Showcase the multi-agent system — how each agent contributes, the pipeline design, and why multi-agent outperforms single-pass for specific metrics
**Tone**: Balanced academic tone
**Approach**: Architecture-First Rewrite

## Overview

Reorganize the paper around the multi-agent pipeline as the central contribution. The methodology section becomes a detailed walkthrough of each agent's role, inputs, outputs, and design rationale. Add a clear architecture diagram showing data flow between agents. The results section adds agent-level analysis showing what each stage contributes to final output quality.

## Current Weaknesses

1. Agent roles are unclear — methodology describes stages but doesn't show WHY each agent is needed
2. Missing architecture diagram — no visual showing data flow between agents
3. No agent-level analysis — results show numbers but don't explain WHAT each agent contributed

## Target Improvements

- Clear, agent-by-agent walkthrough of the 5-agent pipeline
- Visual architecture diagram showing data flow
- Agent contribution analysis using existing evaluation data
- Better framing of the accuracy-quality paradox as an architectural insight

## Restructured Paper Layout

```
1. Abstract
2. Introduction
   - Task: ELI5 explanation generation
   - Problem: Single-pass fails at accuracy+simplicity balance
   - Solution: 5-agent pipeline with staged batching
   - Contributions: Architecture, evaluation framework, accuracy-quality paradox
3. Related Work
   - Multi-agent LLM systems
   - RAG approaches
   - ELI5 and evaluation frameworks
4. Methodology
   - 4.1 Task and Dataset
   - 4.2 System Overview (new pipeline figure)
   - 4.3 Breakdown Agent (new subsection)
   - 4.4 Parallel Analysis: Reasoning + Scientific Agents (new subsection)
   - 4.5 Synthesis Agent with Adaptive Strategy (new subsection)
   - 4.6 Creative Agent (new subsection)
   - 4.7 Staged Batching for Efficiency
   - 4.8 Implementation Details
5. Experimental Setup
   - Models, evaluation metrics, infrastructure
6. Results
   - 6.1 Accuracy-Quality Paradox
   - 6.2 Agent Contribution Analysis (new subsection)
   - 6.3 Output Length and Quality Analysis
   - 6.4 Performance and Robustness
7. Discussion
   - 7.1 Interpreting the Trade-off
   - 7.2 Agent-Level Insights (new subsection)
   - 7.3 Practical Implications
   - 7.4 Future Directions
8. Conclusion
```

## Detailed Section Designs

### 1. Introduction Rewrite

**Paragraph 1 — Task & Application (2-3 sentences)**
- Define ELI5 task: generating simplified explanations from complex questions
- State why it matters: education, science communication, knowledge accessibility
- Current approach: single-pass prompting

**Paragraph 2 — Technical Challenge (3-4 sentences)**
- Three limitations of single-pass:
  1. Limited parametric knowledge in smaller models
  2. No external grounding → hallucinations
  3. No explicit reasoning decomposition
- Root issue: monolithic generation cannot optimize for accuracy, reasoning, AND simplification simultaneously

**Paragraph 3 — Our Solution: The 5-Agent Pipeline (4-5 sentences)**
- Introduce the pipeline as a decomposition into specialized agents
- Walk through each agent's role:
  1. **Breakdown Agent**: Decomposes questions into search queries + reasoning points
  2. **Reasoning Agent**: Logical analysis of reasoning points
  3. **Scientific Agent**: RAG-based fact retrieval (hybrid BM25 + semantic + Wikipedia)
  4. **Synthesis Agent**: Quality gate — evaluates both sources, selects adaptive strategy (reasoning-heavy, facts-heavy, balanced), curates final points
  5. **Creative Agent**: Transforms curated points into accessible ELI5 explanations
- Point to pipeline figure

**Paragraph 4 — Key Innovation: Staged Batching (2-3 sentences)**
- Processing all questions at each stage reduces vLLM calls from N×5 to 5
- 100× reduction in inference overhead at scale
- Maintains 100% success rate with acceptable latency (29.64s avg)

**Paragraph 5 — Experimental Summary (3-4 sentences)**
- 7 models (1B-7B), 14 configurations, ~66K samples, 11 metrics
- Core finding: accuracy-quality paradox
- ROUGE gains are genuine (not length artifacts): 67.6% shorter outputs, r=-0.99 correlation

**Paragraph 6 — Contributions (3 bullet points)**
1. Novel 5-agent architecture with adaptive synthesis and staged batching
2. Comprehensive empirical evaluation across diverse models and metrics
3. First systematic documentation of accuracy-quality paradox in multi-agent explanation systems

### 2. Methodology Rewrite

**4.1 Task and Dataset** — Keep mostly as-is

**4.2 System Overview** (NEW)
- Add pipeline diagram showing all 5 agents with data flow
- Table summarizing each agent: Role, Input, Output, Temperature, Key Design Choice
- Explain the LangGraph orchestration and state management

**4.3 Breakdown Agent** (NEW subsection)
- **Motivation**: Complex questions need structured decomposition before retrieval/reasoning can be effective
- **Design**: Takes question → produces 3-5 search queries + 3-5 reasoning points
- **Key details**: Low temperature (0.1), Pydantic schema validation, structured output
- **Advantage**: Makes reasoning explicit and auditable, provides roadmap for downstream agents

**4.4 Parallel Analysis: Reasoning + Scientific Agents** (NEW subsection)
- **Motivation**: Retrieval and reasoning are complementary but can run in parallel for efficiency
- **Reasoning Agent**: Logical analysis of reasoning points → 3-6 logical pathways + conclusions
- **Scientific Agent**: Hybrid RAG (BM25 + semantic over OpenThoughts-114k + Wikipedia) → 12 grounded facts with citations
- **Key details**: ThreadPoolExecutor for parallel Wikipedia queries, query deduplication (~50% reduction), context window management
- **Advantage**: Parallel execution, external grounding, explicit fact extraction

**4.5 Synthesis Agent with Adaptive Strategy** (NEW subsection)
- **Motivation**: Blind combination of all available information is suboptimal; quality varies per question
- **Design**: Evaluates quality of both fact extraction and logical reasoning → selects strategy:
  - reasoning-heavy (70/30) when retrieval is weak
  - facts-heavy (70/30) when retrieval is strong
  - balanced (50/50) otherwise
- Curates 4-6 key points, orders logically
- **Advantage**: Quality-aware content mixing, avoids information overload

**4.6 Creative Agent** (NEW subsection)
- **Motivation**: Technical synthesis needs transformation into accessible language
- **Design**: Higher temperature (0.5), applies simplification guidelines (everyday vocabulary, analogies, story flow)
- **Advantage**: Separation of technical synthesis from creative expression allows specialized optimization

**4.7 Staged Batching** — Keep mostly as-is

**4.8 Implementation Details** — Keep mostly as-is

### 3. Results & Discussion Redesign

**Results:**

**6.1 The Accuracy-Quality Paradox** — Keep mostly as-is, but improve presentation

**6.2 Agent Contribution Analysis** (NEW subsection)
- Use existing data to show per-agent insights:
  - **Breakdown quality impact**: Correlation between number/quality of search queries and final answer quality
  - **Synthesis strategy distribution**: What percentage of questions get reasoning-heavy vs facts-heavy vs balanced
  - **RAG contribution**: Analysis of how Wikipedia/retrieval grounding affects output quality
- Add qualitative examples showing intermediate outputs from each agent (2-3 representative questions)

**6.3 Output Length and Quality Analysis** — Keep mostly as-is

**6.4 Performance and Robustness** — Keep mostly as-is

**Discussion:**

**7.1 Interpreting the Trade-off** — Keep mostly as-is

**7.2 Agent-Level Insights** (NEW subsection)
- What each agent contributes to final output quality
- Which agents are most critical (likely breakdown and synthesis)
- How agent specialization enables quality improvements despite accuracy decline
- Analysis of failure cases: when does the multi-agent system underperform?

**7.3 Practical Implications** — Keep mostly as-is

**7.4 Future Directions** — Keep mostly as-is, but add:
- Agent-specific improvements (e.g., better breakdown prompts, adaptive RAG)
- Hybrid architectures combining best of both approaches

### 4. Figures & Visual Improvements

**Figure 1: Pipeline Diagram** (NEW — most important)
- Show all 5 agents as nodes in a flowchart
- Show data flow between agents (what each agent produces and passes to the next)
- Highlight the parallel reasoning + scientific stage
- Show the synthesis agent as a decision point (strategy selection)
- Use the existing `pipeline.tex` as base but expand significantly

**Figure 2: Accuracy-Quality Tradeoff** — Keep existing

**Figure 3: Word Count Comparison** — Keep existing

**Figure 4: Correlation Plot** — Keep existing

**Figure 5: TTR Comparison** — Keep existing

**Figure 6: Agent Contribution Analysis** (NEW)
- Show synthesis strategy distribution (pie chart or bar chart)
- Show per-agent output quality metrics if available

**Figure 7: Qualitative Examples** (NEW)
- Show 2-3 example questions with intermediate outputs from each agent
- Visualize how the pipeline transforms a complex question into a simple explanation

### 5. Abstract & Conclusion Rewrites

**Abstract:**
1. Task and challenge (ELI5 requires balancing accuracy + accessibility; single-pass fails)
2. Our solution (5-agent pipeline: breakdown → parallel reasoning+scientific → adaptive synthesis → creative)
3. Key innovation (staged batching reduces inference calls by 100×)
4. Evaluation summary (7 models, 14 configs, ~66K samples, 11 metrics)
5. Core finding (accuracy-quality paradox: -38.2% accuracy, +34.2% ROUGE1)
6. Critical insight (ROUGE gains are genuine: 67.6% shorter outputs, r=-0.99)
7. Contributions (architecture, evaluation framework, paradox documentation)

**Conclusion:**
1. Summarize the pipeline and its key innovation (staged batching)
2. Restate the core finding (accuracy-quality paradox) and its implications
3. Highlight the length-quality insight (shorter outputs, higher ROUGE)
4. Broader implications for evaluation methodology and multi-agent design
5. Future directions (hybrid architectures, specialized fine-tuning, human evaluation)

## Files to Modify

1. `paper/sections/abstract.tex` — Rewrite
2. `paper/sections/introduction.tex` — Rewrite
3. `paper/sections/methodology.tex` — Major rewrite with 5 new subsections
4. `paper/sections/results.tex` — Add agent contribution analysis
5. `paper/sections/discussion.tex` — Add agent-level insights
6. `paper/sections/conclusion.tex` — Rewrite
7. `paper/figures/pipeline.tex` — Expand to show all 5 agents
8. `paper/main.tex` — Update figure includes if needed

## Success Criteria

- Reader can clearly understand what each agent does and why it's needed
- Pipeline diagram shows complete data flow between agents
- Agent contribution analysis provides insights into which agents matter most
- Paper effectively showcases the multi-agent system as the central contribution
- Balanced academic tone maintained throughout
