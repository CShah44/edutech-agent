"""
Compare GPT-OSS Judge Results
============================
Reads multiple GPT-OSS evaluation results and produces a comparison table.

Usage:
    python compare_gptoss_results.py --summaries llm_metrics_gptoss/*_summary.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List
import numpy as np


def load_summary(filepath: Path) -> dict:
    with open(filepath, "r") as f:
        return json.load(f)


def compare(summaries: Dict[str, dict]) -> None:
    """Display comparison table."""
    metrics = ["correctness", "completeness", "eli5_quality", "overall"]
    name_width = max(len(n) for n in summaries.keys()) + 4

    print("\n" + "=" * 80)
    print("GPT-OSS Judge Comparison Table")
    print("=" * 80)
    header = f"{'Config':<{name_width}}" + "".join(f"{m[:8]:>10}" for m in metrics)
    print(header)
    print("-" * 80)

    for name, summary in summaries.items():
        row = f"{name:<{name_width}}"
        for metric in metrics:
            if metric in summary:
                mean = summary[metric]["mean"]
                std = summary[metric]["std"]
                row += f"{mean:>7.2f} ±{std:.2f}"
            else:
                row += "   N/A    "
        print(row)

    print("=" * 80)
    print("\nKey Findings:")
    # Compare baseline vs multi-agent
    baseline_scores = {m: [] for m in metrics}
    multiagent_scores = {m: [] for m in metrics}

    for name, summary in summaries.items():
        for metric in metrics:
            if metric in summary:
                score = summary[metric]["mean"]
                if "baseline" in name.lower():
                    baseline_scores[metric].append(score)
                elif "agent" in name.lower() or "arch" in name.lower():
                    multiagent_scores[metric].append(score)

    for metric in metrics:
        if baseline_scores[metric] and multiagent_scores[metric]:
            b_avg = np.mean(baseline_scores[metric])
            m_avg = np.mean(multiagent_scores[metric])
            diff = m_avg - b_avg
            pct = (diff / b_avg * 100) if b_avg else float('inf')
            direction = "higher" if diff > 0 else "lower"
            print(f"  {metric}: Multi-Agent avg {m_avg:.2f} vs Baseline avg {b_avg:.2f} "
                  f"({diff:+.2f}, {pct:+.1f}% {direction})")


def main():
    parser = argparse.ArgumentParser(description="Compare GPT-OSS evaluation results")
    parser.add_argument("--summaries", nargs="+", required=True, help="Summary JSON files")
    args = parser.parse_args()

    summaries = {}
    for path in args.summaries:
        p = Path(path)
        if p.exists():
            summaries[p.stem] = load_summary(p)

    if not summaries:
        print("No summary files found.")
        return

    compare(summaries)


if __name__ == "__main__":
    main()
