#!/usr/bin/env python3
"""
Output Length Analysis Script
Compares word counts between baseline and multi-agent outputs to check if
ROUGE gains are genuine or just from longer outputs.
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import re

# Configuration
OUTPUTS_DIR = Path("outputs_llm_final")
SUMMARY_DIR = Path("non_llm_metrics_output")

# Model name mapping (baseline -> multi-agent)
MODEL_MAPPING = {
    "llama3b": "llama3.2_3b",
    "llama1b": "llama1b",
    "mistral7b": "mistral7b",
    "gemma-7b-it": "gemma_7b",
    "gemma-2-2b-it": "gemma-2-2b-it",
    "qwen2.5_3b": "qwen2.5_3b",
    "qwen2.5_7b": "qwen2.5_7b"
}

def count_words(text):
    """Count words in text, handling NaN and non-string values."""
    if pd.isna(text) or not isinstance(text, str):
        return 0
    # Split on whitespace and filter empty strings
    words = text.split()
    return len(words)

def type_token_ratio(text):
    """Calculate type-token ratio for vocabulary diversity."""
    if pd.isna(text) or not isinstance(text, str):
        return 0.0
    # Tokenize and normalize
    words = re.findall(r'\b\w+\b', text.lower())
    if len(words) == 0:
        return 0.0
    unique_words = set(words)
    return len(unique_words) / len(words)

def load_csv_data(filepath):
    """Load CSV and extract word counts and other metrics."""
    try:
        df = pd.read_csv(filepath)
        if 'generated_answer' not in df.columns:
            print(f"  Warning: No 'generated_answer' column in {filepath.name}")
            return None
        
        # Calculate word counts
        df['word_count'] = df['generated_answer'].apply(count_words)
        df['ttr'] = df['generated_answer'].apply(type_token_ratio)
        
        return df
    except Exception as e:
        print(f"  Error loading {filepath.name}: {e}")
        return None

def load_summary_data(filepath):
    """Load JSON summary and extract metrics."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"  Error loading {filepath.name}: {e}")
        return None

def extract_model_name(filename):
    """Extract model name from filename."""
    # Remove prefix and suffix
    name = filename.replace("baseline_", "").replace("arch1_", "").replace("arch_1_", "")
    name = name.replace("_0_30000_ragas_llm.csv", "").replace("_0_30000_ragas_llm_summary.json", "")
    return name

def main():
    print("=" * 80)
    print("OUTPUT LENGTH ANALYSIS")
    print("Comparing baseline vs multi-agent outputs to check for length bias")
    print("=" * 80)
    
    # Collect all baseline and multi-agent files
    baseline_files = {}
    agent_files = {}
    
    for f in OUTPUTS_DIR.glob("baseline_*.csv"):
        model = extract_model_name(f.name)
        baseline_files[model] = f
    
    for f in OUTPUTS_DIR.glob("arch1_*.csv"):
        model = extract_model_name(f.name)
        agent_files[model] = f
    
    for f in OUTPUTS_DIR.glob("arch_1_*.csv"):
        model = extract_model_name(f.name)
        agent_files[model] = f
    
    print(f"\nFound {len(baseline_files)} baseline files")
    print(f"Found {len(agent_files)} multi-agent files")
    
    # Analyze each model pair
    results = []
    
    for baseline_model, baseline_path in baseline_files.items():
        # Find corresponding agent file
        agent_model = MODEL_MAPPING.get(baseline_model, baseline_model)
        agent_path = agent_files.get(agent_model)
        
        if not agent_path:
            print(f"\nNo multi-agent file found for baseline model: {baseline_model}")
            continue
        
        print(f"\nAnalyzing {baseline_model}...")
        
        # Load data
        baseline_df = load_csv_data(baseline_path)
        agent_df = load_csv_data(agent_path)
        
        if baseline_df is None or agent_df is None:
            continue
        
        # Load summary data for ROUGE scores
        baseline_summary_path = SUMMARY_DIR / f"baseline_{baseline_model}_0_30000_ragas_summary.json"
        agent_summary_path = SUMMARY_DIR / f"arch1_{agent_model}_0_30000_ragas_summary.json"
        if not agent_summary_path.exists():
            agent_summary_path = SUMMARY_DIR / f"arch_1_{agent_model}_0_30000_ragas_summary.json"
        
        baseline_summary = load_summary_data(baseline_summary_path) if baseline_summary_path.exists() else None
        agent_summary = load_summary_data(agent_summary_path) if agent_summary_path.exists() else None
        
        # Calculate statistics
        baseline_stats = {
            'model': baseline_model,
            'type': 'baseline',
            'count': len(baseline_df),
            'word_count_mean': baseline_df['word_count'].mean(),
            'word_count_std': baseline_df['word_count'].std(),
            'word_count_median': baseline_df['word_count'].median(),
            'ttr_mean': baseline_df['ttr'].mean(),
            'rouge1_mean': baseline_summary.get('rouge1', {}).get('mean', 0) if baseline_summary else 0,
            'rougeL_mean': baseline_summary.get('rougeL', {}).get('mean', 0) if baseline_summary else 0,
        }
        
        agent_stats = {
            'model': agent_model,
            'type': 'multi-agent',
            'count': len(agent_df),
            'word_count_mean': agent_df['word_count'].mean(),
            'word_count_std': agent_df['word_count'].std(),
            'word_count_median': agent_df['word_count'].median(),
            'ttr_mean': agent_df['ttr'].mean(),
            'rouge1_mean': agent_summary.get('rouge1', {}).get('mean', 0) if agent_summary else 0,
            'rougeL_mean': agent_summary.get('rougeL', {}).get('mean', 0) if agent_summary else 0,
        }
        
        results.append(baseline_stats)
        results.append(agent_stats)
        
        # Print comparison
        print(f"  Baseline: {baseline_stats['word_count_mean']:.1f} ± {baseline_stats['word_count_std']:.1f} words")
        print(f"  Multi-agent: {agent_stats['word_count_mean']:.1f} ± {agent_stats['word_count_std']:.1f} words")
        print(f"  Difference: {agent_stats['word_count_mean'] - baseline_stats['word_count_mean']:+.1f} words ({((agent_stats['word_count_mean'] / baseline_stats['word_count_mean']) - 1) * 100:+.1f}%)")
        print(f"  ROUGE-1: Baseline={baseline_stats['rouge1_mean']:.4f}, Agent={agent_stats['rouge1_mean']:.4f}")
        print(f"  ROUGE-L: Baseline={baseline_stats['rougeL_mean']:.4f}, Agent={agent_stats['rougeL_mean']:.4f}")
        print(f"  TTR: Baseline={baseline_stats['ttr_mean']:.4f}, Agent={agent_stats['ttr_mean']:.4f}")
    
    # Create summary DataFrame
    df_results = pd.DataFrame(results)
    
    # Save detailed results
    output_file = Path("output_length_analysis.csv")
    df_results.to_csv(output_file, index=False)
    print(f"\nDetailed results saved to: {output_file}")
    
    # Generate summary table for paper
    print("\n" + "=" * 80)
    print("SUMMARY TABLE FOR PAPER")
    print("=" * 80)
    
    # Group by model and calculate averages
    summary_data = []
    for i in range(0, len(results), 2):
        if i + 1 < len(results):
            baseline = results[i]
            agent = results[i + 1]
            
            word_count_change = ((agent['word_count_mean'] / baseline['word_count_mean']) - 1) * 100
            rouge1_change = agent['rouge1_mean'] - baseline['rouge1_mean']
            rougeL_change = agent['rougeL_mean'] - baseline['rougeL_mean']
            
            summary_data.append({
                'Model': baseline['model'],
                'Baseline Words': f"{baseline['word_count_mean']:.1f} ± {baseline['word_count_std']:.1f}",
                'Agent Words': f"{agent['word_count_mean']:.1f} ± {agent['word_count_std']:.1f}",
                'Word Change (%)': f"{word_count_change:+.1f}%",
                'Baseline ROUGE-1': f"{baseline['rouge1_mean']:.4f}",
                'Agent ROUGE-1': f"{agent['rouge1_mean']:.4f}",
                'ROUGE-1 Change': f"{rouge1_change:+.4f}",
                'Baseline ROUGE-L': f"{baseline['rougeL_mean']:.4f}",
                'Agent ROUGE-L': f"{agent['rougeL_mean']:.4f}",
                'ROUGE-L Change': f"{rougeL_change:+.4f}",
                'Baseline TTR': f"{baseline['ttr_mean']:.4f}",
                'Agent TTR': f"{agent['ttr_mean']:.4f}",
            })
    
    df_summary = pd.DataFrame(summary_data)
    print(df_summary.to_string(index=False))
    
    # Save summary table
    summary_file = Path("output_length_summary.csv")
    df_summary.to_csv(summary_file, index=False)
    print(f"\nSummary table saved to: {summary_file}")
    
    # Calculate overall statistics
    print("\n" + "=" * 80)
    print("OVERALL STATISTICS")
    print("=" * 80)
    
    all_baseline_words = [r['word_count_mean'] for r in results if r['type'] == 'baseline']
    all_agent_words = [r['word_count_mean'] for r in results if r['type'] == 'multi-agent']
    
    avg_baseline = np.mean(all_baseline_words)
    avg_agent = np.mean(all_agent_words)
    avg_change = ((avg_agent / avg_baseline) - 1) * 100
    
    print(f"Average baseline word count: {avg_baseline:.1f}")
    print(f"Average multi-agent word count: {avg_agent:.1f}")
    print(f"Average word count change: {avg_change:+.1f}%")
    
    # Check for correlation between length and ROUGE
    print("\n" + "=" * 80)
    print("LENGTH vs ROUGE CORRELATION")
    print("=" * 80)
    
    word_counts = []
    rouge_scores = []
    for r in results:
        word_counts.append(r['word_count_mean'])
        rouge_scores.append(r['rouge1_mean'])
    
    if len(word_counts) > 1:
        correlation = np.corrcoef(word_counts, rouge_scores)[0, 1]
        print(f"Correlation between word count and ROUGE-1: {correlation:.4f}")
        if abs(correlation) > 0.7:
            print("⚠️  Strong correlation detected - length may be influencing ROUGE scores")
        elif abs(correlation) > 0.4:
            print("⚠️  Moderate correlation detected - length may partially influence ROUGE scores")
        else:
            print("✓  Weak correlation - ROUGE gains appear genuine, not just from length")

if __name__ == "__main__":
    main()
