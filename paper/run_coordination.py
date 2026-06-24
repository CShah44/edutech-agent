#!/usr/bin/env python3
"""
Run Paper Coordination
Simple script to demonstrate coordination workflow
"""

import json
from pathlib import Path

def load_section_content(section_file: Path) -> str:
    """Load content from a section file"""
    if section_file.exists():
        return section_file.read_text()
    return ""

def check_terminology(content: str) -> dict:
    """Check terminology consistency"""
    issues = []
    
    # Check for common inconsistencies
    if "baseline" in content.lower() and "single-pass prompting" not in content.lower():
        issues.append("Use 'single-pass prompting' instead of 'baseline'")
    
    if "multi-agent" in content.lower() and "multi-agent architecture" not in content.lower():
        issues.append("Use 'multi-agent architecture' consistently")
    
    if "rag" in content.lower() and "retrieval-augmented generation" not in content.lower():
        issues.append("Define RAG on first use")
    
    return {
        "consistent": len(issues) == 0,
        "issues": issues
    }

def main():
    """Main coordination function"""
    print("=" * 80)
    print("PAPER COORDINATION RUNNER")
    print("=" * 80)
    
    # Define section files
    sections = {
        "abstract": Path("sections/abstract.tex"),
        "introduction": Path("sections/introduction.tex"),
        "related_work": Path("sections/related_work.tex"),
        "methodology": Path("sections/methodology.tex"),
        "experiments": Path("sections/experiments.tex"),
        "results": Path("sections/results.tex"),
        "discussion": Path("sections/discussion.tex"),
        "conclusion": Path("sections/conclusion.tex")
    }
    
    # Load all sections
    section_contents = {}
    for section_name, section_file in sections.items():
        content = load_section_content(section_file)
        section_contents[section_name] = content
        print(f"\n✓ Loaded {section_name}: {len(content)} characters")
    
    # Check terminology consistency
    print("\n" + "=" * 80)
    print("TERMINOLOGY CONSISTENCY CHECK")
    print("=" * 80)
    
    all_consistent = True
    for section_name, content in section_contents.items():
        result = check_terminology(content)
        status = "✓" if result["consistent"] else "✗"
        print(f"\n{section_name}: {status}")
        
        if not result["consistent"]:
            all_consistent = False
            for issue in result["issues"]:
                print(f"  - {issue}")
    
    # Summary
    print("\n" + "=" * 80)
    print("COORDINATION SUMMARY")
    print("=" * 80)
    
    print(f"\nTotal sections: {len(section_contents)}")
    print(f"Total characters: {sum(len(c) for c in section_contents.values())}")
    print(f"Terminology consistent: {'✓' if all_consistent else '✗'}")
    
    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    if all_consistent:
        print("\n✓ Terminology is consistent across all sections")
        print("✓ Ready for flow and claim-evidence review")
    else:
        print("\n✗ Fix terminology inconsistencies before proceeding")
        print("  Use 'single-pass prompting' instead of 'baseline'")
        print("  Use 'multi-agent architecture' consistently")
        print("  Define RAG on first use")
    
    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    
    print("\n1. Review each section for paragraph clarity")
    print("2. Ensure one message per paragraph")
    print("3. Add explicit transitions between sections")
    print("4. Verify claim-evidence alignment")
    print("5. Run adversarial review")
    
    print("\n" + "=" * 80)
    print("COORDINATION COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
