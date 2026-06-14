"""Batch GPT-OSS Judge Evaluation - runs across multiple CSV files."""

import argparse
import time
import json
from pathlib import Path
from typing import List

import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()

from evaluate_with_gptoss_judge import (
    GPTOSSJudge, pick_column, sample_rows, generate_summary, parse_reference_answers
)

# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_CONFIG = {
    "model": "nvidia/gpt-oss-120b",
    "base_url": "https://integrate.api.nvidia.com/v1",
    "sample_size": 50,
    "sample_seed": 42,
    "rate_limit": 0.5,
}


def find_csv_files(patterns: List[str]) -> List[Path]:
    """Find CSV files matching given glob patterns."""
    files = set()
    for pattern in patterns:
        for f in Path(".").rglob(pattern):
            if f.suffix == ".csv":
                files.add(f)
    return sorted(files)


def process_single_file(file_path: Path, judge: GPTOSSJudge, args) -> dict:
    """Evaluate a single file and return summary."""
    print(f"\n{'='*60}")
    print(f"File: {file_path.name}")
    print(f"{'='*60}")

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"  Error reading CSV: {e}")
        return None

    # Resolve columns
    question_col = pick_column(df, ["question", "user_input", "query", "prompt"])
    answer_col = pick_column(df, ["generated_answer", "response", "answer"])
    reference_col = pick_column(df, ["reference_answers", "reference"])

    if not all([question_col, answer_col, reference_col]):
        print(f"  Skipped: missing required columns")
        return None

    # Filter
    if "status" in df.columns:
        df = df[df["status"] == "success"]

    if len(df) == 0:
        print("  No valid rows found")
        return None

    # Sample
    sampled = sample_rows(df, args.sample_size, args.sample_seed)
    print(f"  Sampled: {len(sampled)}/{len(df)} rows")

    # Evaluate
    results = []
    t0 = time.time()

    for i, (_, row) in enumerate(tqdm(sampled.iterrows(), total=len(sampled),
            desc="  Judging", leave=False)):
        generated = str(row[answer_col])
        question = str(row[question_col])
        references = parse_reference_answers(row[reference_col])

        if not generated or not references:
            continue

        result = judge.evaluate_single(question, generated, references)
        result["question_id"] = row.get("question_id", i)
        result["_model"] = args.model
        results.append(result)

        if i < len(sampled) - 1:
            time.sleep(args.rate_limit)

    elapsed = time.time() - t0
    summary = generate_summary(results)
    summary["_meta"] = {
        "file": str(file_path),
        "sample_size": len(sampled),
        "elapsed_seconds": round(elapsed, 2),
    }

    print(f"  Done: {len(results)} evals in {elapsed:.1f}s")
    for key in ["correctness", "completeness", "eli5_quality", "overall"]:
        if key in summary:
            print(f"    {key:15s} mean={summary[key]['mean']:.2f}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Batch GPT-OSS evaluation")
    parser.add_argument("--files", nargs="+", required=True, help="CSV files to evaluate")
    parser.add_argument("--output", default="llm_metrics_gptoss/", help="Output directory")
    parser.add_argument("--model", default=DEFAULT_CONFIG["model"], help="NVIDIA model name")
    parser.add_argument("--base-url", default=DEFAULT_CONFIG["base_url"], help="API base URL")
    parser.add_argument("--api-key", default=None, help="NVIDIA API key")
    parser.add_argument("--sample-size", type=int, default=DEFAULT_CONFIG["sample_size"])
    parser.add_argument("--sample-seed", type=int, default=DEFAULT_CONFIG["sample_seed"])
    parser.add_argument("--rate-limit", type=float, default=DEFAULT_CONFIG["rate_limit"])
    args = parser.parse_args()

    # Find files
    file_paths = [Path(f) for f in args.files if Path(f).exists()]
    if not file_paths:
        print("No valid files found.")
        return

    print(f"Files to evaluate ({len(file_paths)}):")
    for f in file_paths:
        print(f"  - {f}")

    # Initialize judge
    try:
        judge = GPTOSSJudge(model=args.model, base_url=args.base_url, api_key=args.api_key)
    except (ValueError, ImportError) as e:
        print(f"Error: {e}")
        return

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_summaries = {}

    # Process each file
    for file_path in file_paths:
        summary = process_single_file(file_path, judge, args)
        if summary:
            all_summaries[file_path.stem] = summary
            # Save individual summary
            summary_file = output_dir / f"{file_path.stem}_gptoss_judge_summary.json"
            with open(summary_file, "w") as f:
                json.dump(summary, f, indent=2)

    # Master comparison summary
    if len(all_summaries) > 1:
        comparison = {
            "num_configs": len(all_summaries),
            "configs": {}
        }
        for name, summary in all_summaries.items():
            comparison["configs"][name] = {
                key: summary[key] for key in ["correctness", "completeness", "eli5_quality", "overall"]
                if key in summary
            }
        # Save comparison
        comparison_file = output_dir / "gptoss_comparison_summary.json"
        with open(comparison_file, "w") as f:
            json.dump(comparison, f, indent=2)
        print(f"\nComparison saved: {comparison_file}")

    print(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    main()
