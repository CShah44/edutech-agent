"""
Prepare Human Evaluation Samples
=================================

Generates blind evaluation pairs from baseline vs. multi-agent outputs.
Stratified sampling ensures diverse quality levels across all 7 models.

Output:
  - evaluation_sheet.csv  (for evaluators, no scores)
  - scoring_key.csv       (hidden scores for post-hoc analysis)

Usage:
    python human_evaluation/prepare_human_eval.py
"""

import os
import random
import pandas as pd
from pathlib import Path
from collections import defaultdict

# Configuration
INPUT_DIR = Path(__file__).parent.parent / "llm_metrics_gptoss"
OUTPUT_DIR = Path(__file__).parent
TOTAL_SAMPLES = 150
RANDOM_SEED = 42

# Model name mapping (filename prefix → friendly name)
MODEL_MAPPING = {
    "baseline_llama1b": {"model": "llama1b", "arch": "arch1_llama1b"},
    "baseline_llama3b": {"model": "llama3b", "arch": "arch_1_llama3.2_3b"},
    "baseline_gemma-2-2b-it": {"model": "gemma-2-2b-it", "arch": "arch_1_gemma-2-2b-it"},
    "baseline_gemma-7b-it": {"model": "gemma-7b", "arch": "arch1_gemma_7b"},
    "baseline_mistral7b": {"model": "mistral7b", "arch": "arch_1_mistral7b"},
    "baseline_qwen2.5_3b": {"model": "qwen2.5-3b", "arch": "arch1_qwen2.5_3b"},
    "baseline_qwen2.5_7b": {"model": "qwen2.5-7b", "arch": "arch1_qwen2.5_7b"},
}


def load_model_data():
    """Load all baseline and arch CSVs, matched by model."""
    models = {}

    for baseline_prefix, info in MODEL_MAPPING.items():
        model_name = info["model"]
        arch_prefix = info["arch"]

        baseline_path = INPUT_DIR / f"{baseline_prefix}_0_30000_ragas_llm_gptoss_judge.csv"
        arch_path = INPUT_DIR / f"{arch_prefix}_0_30000_ragas_llm_gptoss_judge.csv"

        if not baseline_path.exists():
            print(f"Warning: Missing {baseline_path}")
            continue
        if not arch_path.exists():
            print(f"Warning: Missing {arch_path}")
            continue

        baseline_df = pd.read_csv(baseline_path)
        arch_df = pd.read_csv(arch_path)

        # Match on question_id
        common_ids = set(baseline_df["question_id"]) & set(arch_df["question_id"])

        if not common_ids:
            print(f"Warning: No common question_ids for {model_name}")
            continue

        baseline_matched = baseline_df[baseline_df["question_id"].isin(common_ids)].copy()
        arch_matched = arch_df[arch_df["question_id"].isin(common_ids)].copy()

        # Merge on question_id
        merged = pd.merge(
            baseline_matched,
            arch_matched,
            on="question_id",
            suffixes=("_baseline", "_arch"),
            how="inner",
        )

        # Calculate average correctness for stratification
        merged["avg_correctness"] = (
            merged["correctness_baseline"] + merged["correctness_arch"]
        ) / 2

        models[model_name] = merged
        print(f"Loaded {model_name}: {len(merged)} matched pairs")

    return models


def stratified_sample(df, n_total, n_models, model_idx, seed):
    """Sample with stratification by quality tier."""
    random.seed(seed)

    # Calculate samples per model (distribute remainder across first models)
    base_count = n_total // n_models
    remainder = n_total % n_models
    n_per_model = base_count + (1 if model_idx < remainder else 0)

    # Define quality tiers
    def tier(score):
        if score >= 7:
            return "high"
        elif score >= 4:
            return "medium"
        else:
            return "low"

    df["tier"] = df["avg_correctness"].apply(tier)

    # Calculate samples per tier (proportional)
    tier_counts = {"high": 8, "medium": 9, "low": 8}
    total_tier = sum(tier_counts.values())
    tier_samples = {t: max(1, int(n_per_model * count / total_tier)) for t, count in tier_counts.items()}

    sampled = []
    for t, n in tier_samples.items():
        tier_df = df[df["tier"] == t]
        n_available = min(n, len(tier_df))
        if n_available > 0:
            sampled.append(tier_df.sample(n=n_available, random_state=seed))

    result = pd.concat(sampled, ignore_index=True)

    # If we don't have enough, fill from remaining
    if len(result) < n_per_model:
        remaining = df[~df.index.isin(result.index)]
        deficit = n_per_model - len(result)
        if len(remaining) >= deficit:
            result = pd.concat([result, remaining.sample(n=deficit, random_state=seed)])

    return result.head(n_per_model)


def randomize_ab_order(row, rng):
    """Randomly swap answer A and B to prevent position bias."""
    if rng.random() < 0.5:
        return {
            "answer_A": row["generated_baseline"],
            "answer_B": row["generated_arch"],
            "answer_A_source": "baseline",
            "answer_B_source": "multi-agent",
        }
    else:
        return {
            "answer_A": row["generated_arch"],
            "answer_B": row["generated_baseline"],
            "answer_A_source": "multi-agent",
            "answer_B_source": "baseline",
        }


def main():
    print("Loading model data...")
    models = load_model_data()

    print(f"\nSampling {TOTAL_SAMPLES} pairs total ({TOTAL_SAMPLES // len(models)} per model)...")
    all_samples = []

    for model_idx, (model_name, df) in enumerate(models.items()):
        sampled = stratified_sample(df, TOTAL_SAMPLES, len(models), model_idx, RANDOM_SEED)
        sampled["model"] = model_name
        all_samples.append(sampled)

    combined = pd.concat(all_samples, ignore_index=True)
    print(f"Total samples before shuffle: {len(combined)}")

    # Shuffle all samples
    combined = combined.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    # Randomize A/B order
    rng = random.Random(RANDOM_SEED)
    ab_data = combined.apply(lambda row: randomize_ab_order(row, rng), axis=1)
    ab_df = pd.DataFrame(ab_data.tolist())

    # Build evaluation sheet (for evaluators)
    eval_sheet = pd.DataFrame({
        "sample_id": range(1, len(combined) + 1),
        "model": combined["model"].values,
        "question": combined["question_baseline"].values,
        "answer_A": ab_df["answer_A"].values,
        "answer_B": ab_df["answer_B"].values,
        "reference": combined["reference_answers_baseline"].values
            if "reference_answers_baseline" in combined.columns
            else [""] * len(combined),
        "reminder": "Score 1-5: Accuracy, Simplicity, Completeness, Clarity",
        "score_A_accuracy": [""] * len(combined),
        "score_A_simplicity": [""] * len(combined),
        "score_A_completeness": [""] * len(combined),
        "score_A_clarity": [""] * len(combined),
        "score_B_accuracy": [""] * len(combined),
        "score_B_simplicity": [""] * len(combined),
        "score_B_completeness": [""] * len(combined),
        "score_B_clarity": [""] * len(combined),
        "notes": [""] * len(combined),
    })

    # Build scoring key (hidden scores for analysis)
    scoring_key = pd.DataFrame({
        "sample_id": range(1, len(combined) + 1),
        "model": combined["model"].values,
        "question_id": combined["question_id"].values,
        "answer_A_source": ab_df["answer_A_source"].values,
        "answer_B_source": ab_df["answer_B_source"].values,
        "baseline_correctness": combined["correctness_baseline"].values,
        "agent_correctness": combined["correctness_arch"].values,
        "baseline_completeness": combined["completeness_baseline"].values,
        "agent_completeness": combined["completeness_arch"].values,
    })

    # Write outputs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    eval_path = OUTPUT_DIR / "evaluation_sheet.csv"
    eval_sheet.to_csv(eval_path, index=False)
    print(f"\nEvaluation sheet: {eval_path}")

    key_path = OUTPUT_DIR / "scoring_key.csv"
    scoring_key.to_csv(key_path, index=False)
    print(f"Scoring key: {key_path}")

    # Print summary
    print(f"\nSummary:")
    print(f"  Total samples: {len(eval_sheet)}")
    print(f"  Models: {eval_sheet['model'].nunique()}")
    print(f"  Samples per model:")
    for model, count in eval_sheet["model"].value_counts().items():
        print(f"    {model}: {count}")


if __name__ == "__main__":
    main()
