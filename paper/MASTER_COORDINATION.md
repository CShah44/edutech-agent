# Master Coordination Document

## Paper Writing Improvement - Sub-Agent Coordination

### Overall Paper Story

**Core Narrative**: Multi-agent orchestration for ELI5 explanations shows a consistent accuracy-quality trade-off: lower LLM-judged accuracy (-38.2%) but higher automatic text quality (+34.2% ROUGE1, +10.0% BERT-F1).

**Key Message**: The choice between architectures should be guided by specific use case priorities, not universal superiority claims.

### Terminology Dictionary

| Term | Definition | Usage |
|------|------------|-------|
| **ELI5** | Explain Like I'm 5 - simplified explanations for non-experts | First use: "ELI5 (Explain Like I'm 5)" |
| **Multi-agent architecture** | Four-stage pipeline with specialized agents | Use consistently throughout |
| **Single-pass prompting** | Baseline approach with one generation step | Use instead of "baseline" alone |
| **Staged batching** | Processing batches through stages together | Key innovation term |
| **Accuracy-quality paradox** | Lower LLM accuracy but higher text quality | Core finding term |
| **RAG** | Retrieval-augmented generation | First use: "retrieval-augmented generation (RAG)" |

### Claim-Evidence Map

#### Abstract Claims
1. **Claim**: Multi-agent architecture shows -38.2% LLM accuracy
   - **Evidence**: Table 2 (Accuracy scores)
   - **Status**: ✅ Supported

2. **Claim**: Multi-agent achieves +34.2% ROUGE1 improvement
   - **Evidence**: Table 3 (Text quality metrics)
   - **Status**: ✅ Supported

3. **Claim**: Trade-off is architectural, not model-specific
   - **Evidence**: All 7 models show same pattern
   - **Status**: ✅ Supported

#### Introduction Claims
1. **Claim**: Current approaches rely on single-pass prompting
   - **Evidence**: Methodology section describes baseline
   - **Status**: ✅ Supported

2. **Claim**: Smaller models struggle with complex reasoning
   - **Evidence**: Literature review + results showing model differences
   - **Status**: ✅ Supported

3. **Claim**: Multi-agent reduces vLLM calls by 100×
   - **Evidence**: Methodology section (staged batching explanation)
   - **Status**: ✅ Supported

#### Results Claims
1. **Claim**: All 7 models show accuracy decline
   - **Evidence**: Table 2 (all rows show negative change)
   - **Status**: ✅ Supported

2. **Claim**: ROUGE improvements are genuine (not length bias)
   - **Evidence**: Output length analysis (multi-agent shorter but higher ROUGE)
   - **Status**: ✅ Supported (NEW FINDING)

### Section Coordination Matrix

| Section | Depends On | Provides To | Key Transitions |
|---------|------------|-------------|-----------------|
| Abstract | All sections | Introduction | Problem → Solution → Evidence |
| Introduction | Abstract | Related Work | Motivation → Gap → Our approach |
| Related Work | Introduction | Methodology | Prior art → Our position |
| Methodology | Related Work | Experiments | Approach → Implementation |
| Experiments | Methodology | Results | Setup → Metrics |
| Results | Experiments | Discussion | Findings → Analysis |
| Discussion | Results | Conclusion | Interpretation → Implications |
| Conclusion | Discussion | - | Summary → Future work |

### Flow Requirements

#### Abstract → Introduction
- Abstract ends with contribution statement
- Introduction begins with motivation that connects to abstract

#### Introduction → Related Work
- Introduction ends with "Our contributions are threefold"
- Related Work begins with "Multi-Agent LLM Systems"

#### Related Work → Methodology
- Related Work ends with positioning our work
- Methodology begins with task definition

#### Methodology → Experiments
- Methodology ends with implementation details
- Experiments begins with models and configurations

#### Experiments → Results
- Experiments ends with evaluation metrics
- Results begins with "The Accuracy-Quality Paradox"

#### Results → Discussion
- Results ends with performance summary
- Discussion begins with "Interpreting the Trade-off"

#### Discussion → Conclusion
- Discussion ends with future directions
- Conclusion begins with summary

### Quality Checkpoints

#### Per-Section Quality
1. **Paragraph Clarity**: One message per paragraph
2. **First Sentence**: States paragraph message
3. **Flow**: Clear transitions between paragraphs
4. **Terminology**: Consistent use of terms
5. **Evidence**: Claims supported by data

#### Cross-Section Quality
1. **Terminology Consistency**: Same terms used throughout
2. **Claim Alignment**: Claims in abstract/intro supported in results
3. **Flow Transitions**: Smooth connections between sections
4. **Figure/Table References**: All references match actual figures/tables
5. **Citation Consistency**: Same citation style throughout

#### Overall Paper Quality
1. **Story Coherence**: Narrative makes sense from start to finish
2. **Contribution Clarity**: Three contributions clear and distinct
3. **Evidence Strength**: All major claims well-supported
4. **Reviewer Appeal**: First impression quality
5. **Reproducibility**: Sufficient detail for replication

### Sub-Agent Instructions

#### General Instructions for All Agents
1. Read the section-specific reference from `references/`
2. Apply global principles from research-paper-writing skill
3. Use terminology dictionary for consistent terms
4. Follow paragraph clarity check for each paragraph
5. Ensure claim-evidence alignment

#### Section-Specific Instructions

**Abstract Agent**:
- Apply Version 1 template: Challenge → Contribution → Evidence
- Ensure 150-200 words
- Include key numbers: 7 models, 30K samples, -38.2%, +34.2%
- Add contribution statement

**Introduction Agent**:
- Five paragraphs: Motivation → Gap → Approach → Findings → Contributions
- One message per paragraph
- First sentence states paragraph message
- Add explicit transitions

**Related Work Agent**:
- Three paragraphs: Multi-Agent LLMs → RAG → ELI5 Evaluation
- Position our work vs. prior art
- Clarify unique contributions

**Methodology Agent**:
- Four subsections: Task → Baseline → Multi-Agent → Implementation
- Apply motivation → design → advantage pattern
- Ensure reproducibility details

**Experiments Agent**:
- Four subsections: Models → Scale → Metrics → Infrastructure
- Ensure reproducibility
- Add justification for choices

**Results Agent**:
- Four subsections: Trade-off → Consistency → Quality → Performance
- Integrate new length analysis findings
- Strengthen claim support

**Discussion Agent**:
- Three subsections: Interpretation → Implications → Future Work
- Deepen analysis
- Add practical recommendations

**Conclusion Agent**:
- One paragraph: Summary + Takeaways
- Keep concise but impactful
- Mention future work

### Coordination Protocol

#### Step 1: Independent Improvement
Each agent improves its section independently using:
- Section-specific reference
- Global principles
- Terminology dictionary
- Paper outline for context

#### Step 2: Cross-Section Review
Master agent reviews all sections for:
- Terminology consistency
- Claim-evidence alignment
- Flow transitions
- Figure/table references

#### Step 3: Final Polish
Master agent runs:
- Adversarial review
- Five-dimension self-review
- Address any issues

### Success Criteria

1. ✅ All sections improved with reviewer-friendly writing
2. ✅ Consistent terminology across paper
3. ✅ All major claims supported by evidence
4. ✅ Clear flow between sections
5. ✅ New findings integrated
6. ✅ Adversarial review passed
7. ✅ Ready for submission
