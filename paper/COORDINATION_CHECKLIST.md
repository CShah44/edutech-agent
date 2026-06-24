# Coordination Checklist

## Master Agent Review Checklist

### Phase 1: Independent Section Review

#### Abstract Agent
- [ ] Abstract follows Version 1 template (Challenge → Contribution → Evidence)
- [ ] Length is 150-200 words
- [ ] Key numbers included (7 models, 30K, -38.2%, +34.2%)
- [ ] Contribution statement clear
- [ ] Claim-evidence map created
- [ ] Self-review completed

#### Introduction Agent
- [ ] Five paragraphs: Motivation → Gap → Approach → Findings → Contributions
- [ ] One message per paragraph
- [ ] First sentence states paragraph message
- [ ] Explicit transitions between paragraphs
- [ ] Three contributions clearly stated
- [ ] Claim-evidence map created
- [ ] Self-review completed

#### Related Work Agent
- [ ] Three paragraphs: Multi-Agent LLMs → RAG → ELI5 Evaluation
- [ ] Our work positioned vs. prior art
- [ ] Unique contributions clarified
- [ ] Recent references included
- [ ] Claim-evidence map created
- [ ] Self-review completed

#### Methodology Agent
- [ ] Four subsections: Task → Baseline → Multi-Agent → Implementation
- [ ] Motivation → design → advantage pattern applied
- [ ] Reproducibility details included
- [ ] Multi-agent architecture clearly described
- [ ] Staged batching explained
- [ ] Claim-evidence map created
- [ ] Self-review completed

#### Experiments Agent
- [ ] Four subsections: Models → Scale → Metrics → Infrastructure
- [ ] Reproducibility ensured
- [ ] Justification for choices added
- [ ] Hardware/software details included
- [ ] Evaluation metrics clearly described
- [ ] Claim-evidence map created
- [ ] Self-review completed

#### Results Agent
- [ ] Four subsections: Trade-off → Consistency → Quality → Performance
- [ ] New length analysis findings integrated
- [ ] Claim support strengthened
- [ ] Tables properly referenced
- [ ] Statistical significance discussed
- [ ] Claim-evidence map created
- [ ] Self-review completed

#### Discussion Agent
- [ ] Three subsections: Interpretation → Implications → Future Work
- [ ] Analysis deepened
- [ ] Practical recommendations added
- [ ] Limitations addressed
- [ ] Future work clearly stated
- [ ] Claim-evidence map created
- [ ] Self-review completed

#### Conclusion Agent
- [ ] One paragraph: Summary + Takeaways
- [ ] Concise but impactful
- [ ] Future work mentioned
- [ ] Key contributions restated
- [ ] Claim-evidence map created
- [ ] Self-review completed

### Phase 2: Cross-Section Coordination

#### Terminology Consistency
- [ ] ELI5 defined on first use
- [ ] Multi-agent architecture used consistently
- [ ] Single-pass prompting used (not "baseline" alone)
- [ ] Staged batching used for innovation
- [ ] Accuracy-quality paradox used for finding
- [ ] RAG defined on first use

#### Claim-Evidence Alignment
- [ ] Abstract claims supported in results
- [ ] Introduction claims supported in results
- [ ] Methodology claims supported in implementation
- [ ] Experiments claims supported in setup
- [ ] All major claims have evidence

#### Flow Transitions
- [ ] Abstract → Introduction: Smooth
- [ ] Introduction → Related Work: Smooth
- [ ] Related Work → Methodology: Smooth
- [ ] Methodology → Experiments: Smooth
- [ ] Experiments → Results: Smooth
- [ ] Results → Discussion: Smooth
- [ ] Discussion → Conclusion: Smooth

#### Figure/Table References
- [ ] Table 1 (Models) referenced in text
- [ ] Table 2 (Accuracy) referenced in text
- [ ] Table 3 (Text Quality) referenced in text
- [ ] Table 4 (Summary) referenced in text
- [ ] All references match actual tables

#### Citation Consistency
- [ ] Same citation style throughout
- [ ] All citations in references.bib
- [ ] No missing citations
- [ ] Citation order logical

### Phase 3: Final Review

#### Adversarial Review
- [ ] Run adversarial review using `references/paper-review.md`
- [ ] Address any high-risk questions
- [ ] Resolve any major rejection risks

#### Five-Dimension Self-Review
- [ ] Contribution clarity: [Pass/Fail]
- [ ] Writing clarity: [Pass/Fail]
- [ ] Experimental strength: [Pass/Fail]
- [ ] Evaluation completeness: [Pass/Fail]
- [ ] Method design soundness: [Pass/Fail]

#### Final Polish
- [ ] Formatting consistent
- [ ] Typos fixed
- [ ] Grammar corrected
- [ ] References formatted
- [ ] Ready for submission

## Coordination Notes

### Dependencies Between Sections
1. **Abstract** → Introduction: Problem statement flows to motivation
2. **Introduction** → Related Work: Gap statement flows to prior art
3. **Related Work** → Methodology: Positioning flows to approach
4. **Methodology** → Experiments: Implementation flows to setup
5. **Experiments** → Results: Setup flows to findings
6. **Results** → Discussion: Findings flow to interpretation
7. **Discussion** → Conclusion: Implications flow to summary

### Information Flow
- **Abstract provides**: Problem, contribution, key numbers
- **Introduction provides**: Motivation, gap, approach, contributions
- **Related Work provides**: Prior art, our position
- **Methodology provides**: Approach, implementation details
- **Experiments provides**: Setup, metrics, justification
- **Results provides**: Findings, evidence
- **Discussion provides**: Interpretation, implications
- **Conclusion provides**: Summary, future work

### Quality Checkpoints
1. **Independent Review**: Each agent improves its section
2. **Cross-Section Review**: Master agent ensures consistency
3. **Final Review**: Adversarial review and self-review

## Success Criteria

✅ All 8 sections improved
✅ Terminology consistent
✅ All claims supported
✅ Flow smooth between sections
✅ Tables properly referenced
✅ Adversarial review passed
✅ Five-dimension self-review passed
✅ Ready for submission
