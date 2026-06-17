# Research Gaps & Status

To elevate this project from a strong preprint to a rigorous, top-tier publication (journal or premium conference), you need to address the methodological gaps that peer reviewers will immediately flag.
Here are the top priority items you should improve:

---

## ✅ RESOLVED — Gap 2: Upgrade the LLM Judge Model

> *Originally: "You are using Llama-2-13b-chat-hf as your judge. Reviewers will likely reject the LLM-judged accuracy drop as an artifact of a weak judge."*

**Status: RESOLVED (June 2026)**

Two independent strong-judge evaluations have been completed, both on 50 samples per config (all 14 configs):

### GPT-4.1 Judge (completed ~May 2026)
- **Location:** `llm-metrics-50-samples-gpt4.1/`
- **Model:** GPT-4.1 via local proxy (`http://localhost:4141/v1`)
- **Sample:** 50 rows, seed=18
- **Metrics:** `answer_accuracy` [0–1], `factual_correctness` [0–1]
- **Result:**
  - answer_accuracy: baseline=0.316 → arch=0.349 → **▲ +10.4%**
  - factual_correctness: baseline=0.144 → arch=0.151 → **▲ +5.3%**

### Llama-3.3-70B Judge via NVIDIA NIM (completed June 15, 2026)
- **Location:** `llm_metrics_gptoss/`
- **Script:** `run_gptoss_judge_all.py`
- **Model:** `meta/llama-3.3-70b-instruct` (NVIDIA NIM free tier)
- **Sample:** 50 rows, seed=42
- **Metrics:** correctness, completeness, eli5_quality, overall [1–10]
- **Result:**

| Metric | Baseline avg | Arch avg | Δ |
|---|---|---|---|
| correctness | 5.00 | 5.96 | **▲ +0.97 (+19.3%)** |
| completeness | 3.90 | 4.27 | **▲ +0.37 (+9.5%)** |
| eli5_quality | 8.11 | 7.92 | ▼ -0.19 (-2.4%) |
| overall | 5.53 | 5.94 | **▲ +0.41 (+7.5%)** |

### Three-Judge Summary

| Judge | Metric | Multi-Agent vs Baseline |
|---|---|---|
| Llama-2-13b (old, weak) | answer_accuracy | **▼ -38%** (artifact of weak judge) |
| GPT-4.1 (strong) | answer_accuracy | **▲ +10.4%** |
| GPT-4.1 (strong) | factual_correctness | **▲ +5.3%** |
| Llama-3.3-70B (strong) | correctness | **▲ +19.3%** |

**Conclusion:** The -38% drop was entirely a weak-judge artifact. All strong judges agree that the multi-agent system improves factual correctness. The two independent strong judges corroborate each other — ideal for peer review.

---

## 🔴 OPEN — Gap 1: Resolve the "Accuracy-Quality Paradox" with Human Evaluation (Critical)

Your most interesting finding is that the multi-agent system drops LLM-judged accuracy by ~38% but increases ROUGE/BERT-F1 by 10-34%. However, reviewers will ask: Which metric is right?

**UPDATE (June 2026):** The new strong-judge evaluations (Gap 2 above) have substantially clarified this: strong judges now agree the multi-agent system *improves* factual correctness. However, a small-scale human evaluation (100–200 questions) would still strengthen the paper significantly.

- The Fix: Conduct a blind human evaluation on a small, statistically significant subset (e.g., 100-200 questions). Have humans rate the baseline vs. multi-agent outputs for factual correctness and simplicity.
- Why it matters: If humans agree with the new strong LLM judges, the case becomes airtight for the paper.

---

## 🔴 OPEN — Gap 3: Conduct Architectural Ablation Studies (High)

Your multi-agent system has 4-5 stages. If the accuracy is dropping, the paper needs to explain where the system breaks down.

- The Fix: Execute the ablation studies outlined in RESEARCH_METHODOLOGY.md. Test the pipeline by disabling one component at a time:
  - What happens if you bypass the Synthesis Quality Gate?
  - What happens if you use RAG facts but skip the Reasoning Agent?
- Why it matters: Top venues don't just want to know that a system performs a certain way; they want to know why.

---

## 🔴 OPEN — Gap 4: Investigate Output Length and Style Bias (Medium-High)

Automatic metrics like ROUGE and BERTScore can be easily manipulated by output length or specific vocabulary.

- The Fix: Calculate the average word count of the Baseline vs. Multi-Agent outputs.
- Why it matters: If the multi-agent system just writes much longer answers, ROUGE naturally goes up, and LLM judges naturally penalize it. You need to prove the +34% ROUGE gain isn't just a "length hack."

---

## 🟡 PARTIALLY RESOLVED — Gap 5: Fill in the Missing Data Points (Medium)

When I inventoried the results, the matrix was not perfectly symmetric.

- **Status:** The new Llama-3.3-70B judge covers all 14 configs symmetrically (7 baseline + 7 arch). The GPT-4.1 judge also covers all 14.
- **Remaining:** Ensure non-LLM metrics (ROUGE, BERTScore) also cover all 7 model pairs. `arch_1_llama3.2_3b` was previously missing in non-LLM summaries.
- Why it matters: Journals expect perfectly symmetric experimental grids without unexplained missing data points.

---

## Status Summary

| Gap | Priority | Status |
|---|---|---|
| 1. Human Evaluation | Critical | 🔴 Open |
| 2. Upgrade LLM Judge | Critical | ✅ Resolved |
| 3. Ablation Studies | High | 🔴 Open |
| 4. Length/Style Bias | Medium-High | 🔴 Open |
| 5. Missing Data Points | Medium | 🟡 Partial |