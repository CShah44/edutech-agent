#!/usr/bin/env python3
"""
Paper Writing Coordination Demo
Shows how sub-agents would coordinate to improve the paper
"""

import json
from pathlib import Path
from typing import Dict, List, Any

class PaperCoordination:
    """Master coordination agent for paper improvement"""
    
    def __init__(self):
        self.sections = [
            "abstract", "introduction", "related_work", 
            "methodology", "experiments", "results", 
            "discussion", "conclusion"
        ]
        
        self.terminology = {
            "ELI5": "ELI5 (Explain Like I'm 5)",
            "multi_agent": "multi-agent architecture",
            "baseline": "single-pass prompting",
            "staged_batching": "staged batching",
            "accuracy_quality": "accuracy-quality paradox",
            "RAG": "retrieval-augmented generation (RAG)"
        }
        
        self.claims_evidence = {
            "abstract": [
                {
                    "claim": "Multi-agent shows -38.2% LLM accuracy",
                    "evidence": "Table 2 (Accuracy scores)",
                    "status": "supported"
                },
                {
                    "claim": "Multi-agent achieves +34.2% ROUGE1",
                    "evidence": "Table 3 (Text quality metrics)",
                    "status": "supported"
                },
                {
                    "claim": "Trade-off is architectural, not model-specific",
                    "evidence": "All 7 models show same pattern",
                    "status": "supported"
                }
            ],
            "introduction": [
                {
                    "claim": "Current approaches rely on single-pass prompting",
                    "evidence": "Methodology section describes baseline",
                    "status": "supported"
                },
                {
                    "claim": "Smaller models struggle with complex reasoning",
                    "evidence": "Literature review + results",
                    "status": "supported"
                },
                {
                    "claim": "Multi-agent reduces vLLM calls by 100×",
                    "evidence": "Methodology section (staged batching)",
                    "status": "supported"
                }
            ],
            "results": [
                {
                    "claim": "All 7 models show accuracy decline",
                    "evidence": "Table 2 (all rows show negative change)",
                    "status": "supported"
                },
                {
                    "claim": "ROUGE improvements are genuine (not length bias)",
                    "evidence": "Output length analysis (multi-agent shorter but higher ROUGE)",
                    "status": "supported"
                }
            ]
        }
        
        self.flow_transitions = {
            "abstract": "introduction",
            "introduction": "related_work",
            "related_work": "methodology",
            "methodology": "experiments",
            "experiments": "results",
            "results": "discussion",
            "discussion": "conclusion"
        }
    
    def check_terminology_consistency(self, section_content: str) -> Dict[str, bool]:
        """Check if terminology is used consistently"""
        issues = []
        
        # Check for inconsistent terms
        if "baseline" in section_content.lower() and "single-pass prompting" not in section_content.lower():
            issues.append("Use 'single-pass prompting' instead of 'baseline'")
        
        if "multi-agent" in section_content.lower() and "multi-agent architecture" not in section_content.lower():
            issues.append("Use 'multi-agent architecture' consistently")
        
        return {
            "consistent": len(issues) == 0,
            "issues": issues
        }
    
    def check_claim_evidence(self, section: str, section_content: str) -> Dict[str, Any]:
        """Check if claims are supported by evidence"""
        if section not in self.claims_evidence:
            return {"checked": False, "message": "No claims defined for this section"}
        
        claims = self.claims_evidence[section]
        results = []
        
        for claim in claims:
            # Simple check: see if evidence keywords appear in content
            evidence_keywords = claim["evidence"].lower().split()
            content_lower = section_content.lower()
            
            keywords_found = sum(1 for keyword in evidence_keywords if keyword in content_lower)
            support_ratio = keywords_found / len(evidence_keywords)
            
            results.append({
                "claim": claim["claim"],
                "evidence": claim["evidence"],
                "status": claim["status"],
                "support_ratio": support_ratio,
                "needs_review": support_ratio < 0.5
            })
        
        return {
            "checked": True,
            "results": results,
            "all_supported": all(r["status"] == "supported" for r in results)
        }
    
    def check_flow_transitions(self, current_section: str, next_section: str, 
                              current_content: str, next_content: str) -> Dict[str, Any]:
        """Check if flow between sections is smooth"""
        # Simple heuristic: check if last sentence of current section
        # connects to first sentence of next section
        
        current_sentences = [s.strip() for s in current_content.split('.') if s.strip()]
        next_sentences = [s.strip() for s in next_content.split('.') if s.strip()]
        
        if not current_sentences or not next_sentences:
            return {"smooth": False, "message": "Insufficient content to check flow"}
        
        last_sentence = current_sentences[-1].lower()
        first_sentence = next_sentences[0].lower()
        
        # Check for transition words
        transition_words = ["therefore", "however", "moreover", "furthermore", "in addition"]
        has_transition = any(word in first_sentence for word in transition_words)
        
        return {
            "smooth": has_transition,
            "last_sentence": last_sentence[:100] + "...",
            "first_sentence": first_sentence[:100] + "...",
            "suggestion": "Add transition sentence if not smooth"
        }
    
    def run_coordination_review(self, section_contents: Dict[str, str]) -> Dict[str, Any]:
        """Run full coordination review on all sections"""
        results = {
            "terminology": {},
            "claim_evidence": {},
            "flow": {},
            "overall": {}
        }
        
        # Check terminology consistency
        for section, content in section_contents.items():
            results["terminology"][section] = self.check_terminology_consistency(content)
        
        # Check claim-evidence alignment
        for section, content in section_contents.items():
            results["claim_evidence"][section] = self.check_claim_evidence(section, content)
        
        # Check flow transitions
        for i in range(len(self.sections) - 1):
            current_section = self.sections[i]
            next_section = self.sections[i + 1]
            
            if current_section in section_contents and next_section in section_contents:
                results["flow"][f"{current_section}_to_{next_section}"] = self.check_flow_transitions(
                    current_section, next_section,
                    section_contents[current_section],
                    section_contents[next_section]
                )
        
        # Overall assessment
        terminology_issues = sum(1 for t in results["terminology"].values() if not t["consistent"])
        claims_supported = all(
            ce.get("all_supported", True) 
            for ce in results["claim_evidence"].values()
        )
        flow_smooth = all(
            f.get("smooth", False) 
            for f in results["flow"].values()
        )
        
        results["overall"] = {
            "terminology_issues": terminology_issues,
            "all_claims_supported": claims_supported,
            "flow_smooth": flow_smooth,
            "ready_for_review": terminology_issues == 0 and claims_supported and flow_smooth
        }
        
        return results
    
    def generate_improvement_suggestions(self, review_results: Dict[str, Any]) -> List[str]:
        """Generate suggestions for improvement"""
        suggestions = []
        
        # Terminology suggestions
        for section, term_result in review_results["terminology"].items():
            if not term_result["consistent"]:
                for issue in term_result["issues"]:
                    suggestions.append(f"[{section.upper()}] {issue}")
        
        # Claim-evidence suggestions
        for section, ce_result in review_results["claim_evidence"].items():
            if ce_result.get("checked", False):
                for result in ce_result.get("results", []):
                    if result.get("needs_review", False):
                        suggestions.append(f"[{section.upper()}] Review claim: {result['claim']}")
        
        # Flow suggestions
        for transition, flow_result in review_results["flow"].items():
            if not flow_result.get("smooth", False):
                suggestions.append(f"[FLOW] Improve transition: {transition}")
        
        return suggestions

def demo_coordination():
    """Demo the coordination process"""
    print("=" * 80)
    print("PAPER WRITING COORDINATION DEMO")
    print("=" * 80)
    
    coordinator = PaperCoordination()
    
    # Sample section contents (abbreviated)
    sample_contents = {
        "abstract": "We present a multi-agent architecture for ELI5 explanations. Our approach shows -38.2% LLM accuracy but +34.2% ROUGE1 improvement.",
        "introduction": "ELI5 explanations require balancing accuracy and simplicity. Current approaches use single-pass prompting. We propose a multi-agent architecture with staged batching that reduces vLLM calls by 100×.",
        "related_work": "Multi-agent LLM systems have been explored. RAG approaches enhance generation. ELI5 evaluation remains challenging.",
        "methodology": "We use the sentence-transformers/eli5 dataset. Our baseline uses single-pass prompting. The multi-agent architecture has four stages with staged batching.",
        "experiments": "We evaluate 7 models from 1B-7B parameters. We use 30,000 samples and 11+ metrics including ROUGE and BERT-Score.",
        "results": "All 7 models show accuracy decline (average -38.2%). ROUGE scores improve (+34.2%). The trade-off is architectural, not model-specific.",
        "discussion": "The accuracy-quality paradox suggests different evaluation approaches measure different aspects. Multi-agent produces more structured outputs.",
        "conclusion": "We presented a multi-agent architecture for ELI5 with staged batching. The accuracy-quality trade-off is architectural. Future work includes hybrid approaches."
    }
    
    print("\n1. Running coordination review...")
    review_results = coordinator.run_coordination_review(sample_contents)
    
    print("\n2. Terminology consistency:")
    for section, result in review_results["terminology"].items():
        status = "✓" if result["consistent"] else "✗"
        print(f"   {section}: {status}")
        if not result["consistent"]:
            for issue in result["issues"]:
                print(f"      - {issue}")
    
    print("\n3. Claim-evidence alignment:")
    for section, result in review_results["claim_evidence"].items():
        if result.get("checked", False):
            status = "✓" if result.get("all_supported", False) else "✗"
            print(f"   {section}: {status}")
    
    print("\n4. Flow transitions:")
    for transition, result in review_results["flow"].items():
        status = "✓" if result.get("smooth", False) else "✗"
        print(f"   {transition}: {status}")
    
    print("\n5. Overall assessment:")
    overall = review_results["overall"]
    print(f"   Terminology issues: {overall['terminology_issues']}")
    print(f"   All claims supported: {overall['all_claims_supported']}")
    print(f"   Flow smooth: {overall['flow_smooth']}")
    print(f"   Ready for review: {overall['ready_for_review']}")
    
    print("\n6. Improvement suggestions:")
    suggestions = coordinator.generate_improvement_suggestions(review_results)
    for i, suggestion in enumerate(suggestions, 1):
        print(f"   {i}. {suggestion}")
    
    if not suggestions:
        print("   No suggestions - paper is ready for review!")
    
    print("\n" + "=" * 80)
    print("COORDINATION DEMO COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    demo_coordination()
