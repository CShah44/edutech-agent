"""
Identify Missing Experimental Data Points
=========================================

Scans the results directories and reports which configurations
are missing LLM metrics, non-LLM metrics, or GPT-OSS evaluations.

Usage:
    python find_missing.py
    python find_missing.py --check-gptoss  # Also check for GPT-OSS outputs
"""

import argparse
import json
import re
from pathlib import Path
from collections import defaultdict


def parse_config_name(filename: str) -> tuple:
    """Extract architecture, model, and range from a filename.

    Examples:
        baseline_llama3b_0_30000_ragas_summary.json
        arch_1_mistral7b_0_30000_ragas_llm_summary.json
        arch1_qwen2.5_3b_0_30000_ragas_summary.json
    """
    name = filename.replace("_ragas_summary.json", "").replace("_ragas_llm_summary.json", "")
    name = name.replace("_gptoss_judge_summary.json", "")

    # Detect architecture
    if name.startswith("baseline_"):
        arch = "baseline"
        model = name.replace("baseline_", "")
    elif name.startswith("arch_1_") or name.startswith("arch1_"):
        arch = "arch_1"
        model = name.replace("arch_1_", "").replace("arch1_", "")
    else:
        arch = "unknown"
        model = name

    return arch, model


def scan_directory(directory: Path, pattern: str):
    """Scan a directory for files matching a pattern."""
    if not directory.exists():
        return []
    return sorted(directory.glob(pattern))


def extract_model_range(filename: str) -> tuple:
    """Extract model name and question range from a filename."""
    # Pattern: {arch}_{model}_0_{range}[_ragas_summary].json
    # or: {arch}_{model}_0_{range}_ragas_llm_summary.json
    stem = Path(filename).stem
    return stem


def main():
    parser = argparse.ArgumentParser(description="Find missing experimental data")
    parser.add_argument("--check-gptoss", action="store_true",
                        help="Also check for GPT-OSS evaluation outputs")
    args = parser.parse_args()

    # Directories
    non_llm_dir = Path("non_llm_metrics_output")
    llm_dir = Path("llm_metrics_output")
    gptoss_dir = Path("llm_metrics_gptoss") if args.check_gptoss else None

    # Collect configs
    configs = defaultdict(lambda: {"non_llm": False, "llm": False, "gptoss": False})

    # Scan non-LLM outputs
    if non_llm_dir.exists():
        print(f"Scanning {non_llm_dir}/ for non-LLM metrics...")
        for f in scan_directory(non_llm_dir, "*_ragas_summary.json"):
            arch, model = parse_config_name(f.name)
            if arch != "unknown":
                configs[f"{arch}_{model}"]["non_llm"] = True

    # Scan LLM outputs
    if llm_dir.exists():
        print(f"Scanning {llm_dir}/ for LLM metrics...")
        for f in scan_directory(llm_dir, "*_ragas_llm_summary.json"):
            arch, model = parse_config_name(f.name)
            if arch != "unknown":
                configs[f"{arch}_{model}"]["llm"] = True

    # Scan GPT-OSS outputs
    if gptoss_dir and gptoss_dir.exists():
        print(f"Scanning {gptoss_dir}/ for GPT-OSS evaluations...")
        for f in scan_directory(gptoss_dir, "*_gptoss_judge_summary.json"):
            arch, model = parse_config_name(f.name)
            if arch != "unknown":
                configs[f"{arch}_{model}"]["gptoss"] = True

    # Report findings
    print("\n" + "=" * 80)
    print("EXPERIMENTAL MATRIX STATUS")
    print("=" * 80)
    print(f"{'Config':<35} {'Non-LLM':<10} {'LLM':<10} {'GPT-OSS':<10}")
    print("-" * 80)

    missing_non_llm = []
    missing_llm = []
    missing_gptoss = []

    for config_name in sorted(configs.keys()):
        status = configs[config_name]
        has_non_llm = "✓" if status["non_llm"] else "✗"
        has_llm = "✓" if status["llm"] else "✗"
        has_gptoss = "✓" if status["gptoss"] else "✗"

        print(f"{config_name:<35} {has_non_llm:<10} {has_llm:<10} {has_gptoss:<10}")

        if not status["non_llm"]:
            missing_non_llm.append(config_name)
        if not status["llm"]:
            missing_llm.append(config_name)
        if args.check_gptoss and not status["gptoss"]:
            missing_gptoss.append(config_name)

    print("=" * 80)

    # Report gaps
    print("\nGAPS IDENTIFIED:")
    print("-" * 80)
    print(f"Missing Non-LLM metrics: {len(missing_non_llm)}")
    for item in missing_non_llm:
        print(f"  - {item}")

    print(f"\nMissing LLM metrics: {len(missing_llm)}")
    for item in missing_llm:
        print(f"  - {item}")

    if args.check_gptoss:
        print(f"\nMissing GPT-OSS evaluations: {len(missing_gptoss)}")
        for item in missing_gptoss:
            print(f"  - {item}")

    # Summary stats
    total_configs = len(configs)
    complete = sum(1 for c in configs.values() if c["non_llm"] and c["llm"])
    print(f"\nSummary: {complete}/{total_configs} configs have both Non-LLM and LLM metrics complete")


if __name__ == "__main__":
    main()
