"""
Analyze Human Evaluation Results
=================================

Compares human evaluators' scores, calculates inter-annotator agreement,
and compares human evaluation with LLM judge scores.

Excludes evaluator_3 (automated, all scores identical).

Usage:
    python human_evaluation/analyze_human_eval.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import cohen_kappa_score
from scipy import stats

# Configuration
INPUT_DIR = Path(__file__).parent
SCORING_KEY_PATH = INPUT_DIR / "scoring_key.csv"
OUTPUT_PATH = INPUT_DIR / "analysis_report.txt"

# Human evaluators (excluding evaluator_1 and evaluator_3)
EVALUATOR_FILES = {
    "evaluator_2": INPUT_DIR / "evaluator_2.csv",
    "evaluator_4": INPUT_DIR / "evaluator_4.csv",
}

# Score columns
SCORE_COLS = [
    "score_A_accuracy", "score_A_simplicity", "score_A_completeness", "score_A_clarity",
    "score_B_accuracy", "score_B_simplicity", "score_B_completeness", "score_B_clarity",
]


def load_data():
    """Load evaluator data and scoring key."""
    evaluators = {}
    for name, path in EVALUATOR_FILES.items():
        evaluators[name] = pd.read_csv(path)
    scoring_key = pd.read_csv(SCORING_KEY_PATH)
    return evaluators, scoring_key


def calculate_pairwise_agreement(evaluators):
    """Calculate Cohen's kappa for each pair of evaluators."""
    results = {}
    evaluator_names = list(evaluators.keys())
    
    for i in range(len(evaluator_names)):
        for j in range(i + 1, len(evaluator_names)):
            name1, name2 = evaluator_names[i], evaluator_names[j]
            eval1, eval2 = evaluators[name1], evaluators[name2]
            
            pair_kappas = {}
            for col in SCORE_COLS:
                scores1 = eval1[col].values
                scores2 = eval2[col].values
                kappa = cohen_kappa_score(scores1, scores2, weights="quadratic")
                pair_kappas[col] = kappa
            
            results[f"{name1} vs {name2}"] = pair_kappas
    
    return results


def calculate_overall_agreement(evaluators):
    """Calculate average agreement across all evaluators."""
    all_kappas = []
    pairwise = calculate_pairwise_agreement(evaluators)
    
    for pair, kappas in pairwise.items():
        all_kappas.extend(kappas.values())
    
    return np.mean(all_kappas)


def compare_baseline_vs_agent(evaluators, scoring_key):
    """Compare baseline vs multi-agent answers using human scores."""
    results = {"baseline": {}, "multi-agent": {}}
    
    for evaluator_name, eval_df in evaluators.items():
        merged = pd.merge(eval_df, scoring_key, on="sample_id", suffixes=("", "_key"))
        
        for prefix, source_col in [("A", "answer_A_source"), ("B", "answer_B_source")]:
            for score_type in ["accuracy", "simplicity", "completeness", "clarity"]:
                col = f"score_{prefix}_{score_type}"
                
                # Baseline scores
                baseline_mask = merged[source_col] == "baseline"
                baseline_scores = merged.loc[baseline_mask, col].dropna()
                
                # Multi-agent scores
                agent_mask = merged[source_col] == "multi-agent"
                agent_scores = merged.loc[agent_mask, col].dropna()
                
                key = f"{evaluator_name}_{score_type}"
                results["baseline"][key] = baseline_scores.tolist()
                results["multi-agent"][key] = agent_scores.tolist()
    
    return results


def compare_with_llm_judge(evaluators, scoring_key):
    """Compare human scores with LLM judge scores."""
    all_human_accuracy = []
    all_llm_correctness = []
    
    for evaluator_name, eval_df in evaluators.items():
        merged = pd.merge(eval_df, scoring_key, on="sample_id")
        
        # Calculate average human accuracy for A and B
        human_accuracy = (merged["score_A_accuracy"] + merged["score_B_accuracy"]) / 2
        all_human_accuracy.extend(human_accuracy.tolist())
        
        # Calculate average LLM correctness
        llm_correctness = (merged["baseline_correctness"] + merged["agent_correctness"]) / 2
        all_llm_correctness.extend(llm_correctness.tolist())
    
    # Correlation
    correlation, p_value = stats.pearsonr(all_human_accuracy, all_llm_correctness)
    
    return {
        "correlation": correlation,
        "p_value": p_value,
        "avg_human_accuracy": np.mean(all_human_accuracy),
        "avg_llm_correctness": np.mean(all_llm_correctness),
    }


def generate_report(evaluators, pairwise_agreement, overall_agreement, comparison, llm_comparison):
    """Generate analysis report."""
    report = []
    report.append("=" * 70)
    report.append("HUMAN EVALUATION ANALYSIS REPORT")
    report.append("=" * 70)
    
    report.append(f"\n1. INTER-ANNOTATOR AGREEMENT (Cohen's Kappa)")
    report.append("-" * 50)
    report.append(f"   Evaluators: {', '.join(evaluators.keys())}")
    report.append(f"   Overall Average Kappa: {overall_agreement:.3f}")
    
    for pair, kappas in pairwise_agreement.items():
        avg_kappa = np.mean(list(kappas.values()))
        report.append(f"\n   {pair}: {avg_kappa:.3f}")
        for col, kappa in kappas.items():
            interpretation = "poor" if kappa < 0.2 else "fair" if kappa < 0.4 else "moderate" if kappa < 0.6 else "substantial" if kappa < 0.8 else "almost perfect"
            report.append(f"     {col:30s}: {kappa:.3f} ({interpretation})")
    
    report.append("\n2. BASELINE VS MULTI-AGENT COMPARISON")
    report.append("-" * 50)
    
    for evaluator_name in evaluators.keys():
        report.append(f"\n   {evaluator_name.upper()}:")
        for score_type in ["accuracy", "simplicity", "completeness", "clarity"]:
            baseline_key = f"{evaluator_name}_{score_type}"
            agent_key = f"{evaluator_name}_{score_type}"
            
            baseline_scores = comparison["baseline"].get(baseline_key, [])
            agent_scores = comparison["multi-agent"].get(agent_key, [])
            
            baseline_mean = np.mean(baseline_scores) if baseline_scores else 0
            agent_mean = np.mean(agent_scores) if agent_scores else 0
            diff = agent_mean - baseline_mean
            diff_pct = (diff / baseline_mean * 100) if baseline_mean > 0 else 0
            
            report.append(f"    {score_type:15s}: baseline={baseline_mean:.2f}, agent={agent_mean:.2f}, diff={diff:+.2f} ({diff_pct:+.1f}%)")
    
    report.append("\n3. HUMAN vs LLM JUDGE COMPARISON")
    report.append("-" * 50)
    report.append(f"  Average Human Accuracy: {llm_comparison['avg_human_accuracy']:.2f}/5")
    report.append(f"  Average LLM Correctness: {llm_comparison['avg_llm_correctness']:.2f}/10")
    report.append(f"  Pearson Correlation: {llm_comparison['correlation']:.3f} (p={llm_comparison['p_value']:.4f})")
    
    report.append("\n4. SUMMARY")
    report.append("-" * 50)
    report.append(f"  Total samples evaluated: {len(list(evaluators.values())[0])}")
    report.append(f"  Number of evaluators: {len(evaluators)}")
    report.append(f"  Models covered: {list(evaluators.values())[0]['model'].nunique()}")
    
    report.append("\n" + "=" * 70)
    
    return "\n".join(report)


def main():
    print("Loading data...")
    evaluators, scoring_key = load_data()
    
    print("Calculating pairwise inter-annotator agreement...")
    pairwise_agreement = calculate_pairwise_agreement(evaluators)
    
    print("Calculating overall agreement...")
    overall_agreement = calculate_overall_agreement(evaluators)
    
    print("Comparing baseline vs multi-agent...")
    comparison = compare_baseline_vs_agent(evaluators, scoring_key)
    
    print("Comparing with LLM judge...")
    llm_comparison = compare_with_llm_judge(evaluators, scoring_key)
    
    print("Generating report...")
    report = generate_report(evaluators, pairwise_agreement, overall_agreement, comparison, llm_comparison)
    
    # Save report
    with open(OUTPUT_PATH, "w") as f:
        f.write(report)
    
    print(f"\nReport saved to: {OUTPUT_PATH}")
    print("\n" + report)


if __name__ == "__main__":
    main()
