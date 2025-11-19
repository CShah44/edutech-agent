import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List
import argparse

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

class ResultsAnalyzer:
    """Analyze and visualize evaluation results"""
    
    def __init__(self, results_file: str):
        """Load results from JSON file"""
        self.results_file = Path(results_file)
        
        with open(self.results_file, 'r', encoding='utf-8') as f:
            raw_results = json.load(f)

        self.results = self._normalize_results(raw_results)
        
        # Filter successful evaluations
        self.successful = [r for r in self.results if r.get('evaluation_status') == 'success']
        
        print(f"📊 Loaded {len(self.results)} results ({len(self.successful)} successful)")
        
        # Create output directory for plots
        self.output_dir = self.results_file.parent / f"{self.results_file.stem}_analysis"
        self.output_dir.mkdir(exist_ok=True)

    def _normalize_results(self, raw_results):
        """Ensure we end up with a list of per-question evaluation dicts."""
        if isinstance(raw_results, list):
            return raw_results

        if isinstance(raw_results, dict):
            # Some reports may embed the per-question rows under a key.
            candidates = [
                raw_results.get('results'),
                raw_results.get('evaluations'),
            ]
            for candidate in candidates:
                if isinstance(candidate, list):
                    return candidate

            # Handle aggregated *.report.json files by looking for the sibling detailed file.
            if self.results_file.name.endswith('.report.json'):
                detailed_name = self.results_file.name.replace('.report.json', '.json')
                detailed_path = self.results_file.parent / detailed_name
                if detailed_path.exists():
                    with open(detailed_path, 'r', encoding='utf-8') as f:
                        detailed_results = json.load(f)
                    if isinstance(detailed_results, list):
                        print(
                            f"INFO: '{self.results_file.name}' is a summary report. "
                            f"Falling back to '{detailed_name}' for per-question data."
                        )
                        return detailed_results

        raise ValueError(
            "Results file must contain a list of evaluation entries. "
            "If you passed a '.report.json' summary, use the matching '.json' file instead."
        )
    
    def create_summary_statistics(self) -> pd.DataFrame:
        """Create summary statistics table"""
        if not self.successful:
            return pd.DataFrame()
        
        stats = {
            'Metric': [],
            'Mean': [],
            'Std': [],
            'Min': [],
            'Max': [],
            'Median': []
        }
        
        # LLM Judge metrics (simplified to 3 main metrics)
        for metric in ['correctness_score', 'completeness_score', 'overall_score']:
            values = [r['llm_judge'][metric] for r in self.successful]
            stats['Metric'].append(f"LLM Judge - {metric.replace('_', ' ').title()}")
            stats['Mean'].append(np.mean(values))
            stats['Std'].append(np.std(values))
            stats['Min'].append(np.min(values))
            stats['Max'].append(np.max(values))
            stats['Median'].append(np.median(values))
        
        # Similarity metrics
        for metric in ['max_similarity', 'mean_similarity']:
            values = [r['similarity'][metric] for r in self.successful]
            stats['Metric'].append(f"Similarity - {metric.replace('_', ' ').title()}")
            stats['Mean'].append(np.mean(values))
            stats['Std'].append(np.std(values))
            stats['Min'].append(np.min(values))
            stats['Max'].append(np.max(values))
            stats['Median'].append(np.median(values))
        
        # Entailment
        values = [r['entailment']['entailment_ratio'] for r in self.successful]
        stats['Metric'].append("Entailment Ratio")
        stats['Mean'].append(np.mean(values))
        stats['Std'].append(np.std(values))
        stats['Min'].append(np.min(values))
        stats['Max'].append(np.max(values))
        stats['Median'].append(np.median(values))
        
        # Generation time
        values = [r['generation_time'] for r in self.successful]
        stats['Metric'].append("Generation Time (s)")
        stats['Mean'].append(np.mean(values))
        stats['Std'].append(np.std(values))
        stats['Min'].append(np.min(values))
        stats['Max'].append(np.max(values))
        stats['Median'].append(np.median(values))
        
        df = pd.DataFrame(stats)
        
        # Save to CSV
        csv_path = self.output_dir / "summary_statistics.csv"
        df.to_csv(csv_path, index=False)
        print(f"✅ Summary statistics saved to {csv_path}")
        
        return df
    
    def plot_llm_judge_scores(self):
        """Create visualization of LLM judge scores"""
        if not self.successful:
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('LLM Judge Score Distributions', fontsize=16, fontweight='bold')
        
        metrics = ['correctness_score', 'completeness_score', 'overall_score']
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            values = [r['llm_judge'][metric] for r in self.successful]
            
            # Histogram
            ax.hist(values, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
            ax.axvline(np.mean(values), color='red', linestyle='--', linewidth=2, 
                      label=f'Mean: {np.mean(values):.2f}')
            ax.axvline(np.median(values), color='green', linestyle='--', linewidth=2,
                      label=f'Median: {np.median(values):.2f}')
            
            ax.set_title(metric.replace('_', ' ').title())
            ax.set_xlabel('Score')
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.output_dir / "llm_judge_distributions.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✅ LLM judge plot saved to {plot_path}")
        plt.close()
    
    def plot_similarity_scores(self):
        """Create visualization of similarity scores"""
        if not self.successful:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Semantic Similarity Score Distributions', fontsize=16, fontweight='bold')
        
        # Max similarity
        max_sim = [r['similarity']['max_similarity'] for r in self.successful]
        axes[0].hist(max_sim, bins=20, alpha=0.7, color='coral', edgecolor='black')
        axes[0].axvline(np.mean(max_sim), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {np.mean(max_sim):.3f}')
        axes[0].set_title('Max Similarity with References')
        axes[0].set_xlabel('Cosine Similarity')
        axes[0].set_ylabel('Frequency')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Mean similarity
        mean_sim = [r['similarity']['mean_similarity'] for r in self.successful]
        axes[1].hist(mean_sim, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[1].axvline(np.mean(mean_sim), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {np.mean(mean_sim):.3f}')
        axes[1].set_title('Mean Similarity with References')
        axes[1].set_xlabel('Cosine Similarity')
        axes[1].set_ylabel('Frequency')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.output_dir / "similarity_distributions.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✅ Similarity plot saved to {plot_path}")
        plt.close()
    
    def plot_entailment_analysis(self):
        """Create visualization of entailment scores"""
        if not self.successful:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Entailment Analysis', fontsize=16, fontweight='bold')
        
        # Entailment ratio distribution
        ratios = [r['entailment']['entailment_ratio'] for r in self.successful]
        axes[0].hist(ratios, bins=20, alpha=0.7, color='plum', edgecolor='black')
        axes[0].axvline(np.mean(ratios), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {np.mean(ratios):.2%}')
        axes[0].set_title('Entailment Ratio Distribution')
        axes[0].set_xlabel('Entailment Ratio')
        axes[0].set_ylabel('Frequency')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Entailment count distribution
        counts = [r['entailment']['entailment_count'] for r in self.successful]
        axes[1].hist(counts, bins=range(0, max(counts)+2), alpha=0.7, 
                    color='lightblue', edgecolor='black')
        axes[1].axvline(np.mean(counts), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {np.mean(counts):.1f}')
        axes[1].set_title('Entailment Count Distribution')
        axes[1].set_xlabel('Number of Entailed References')
        axes[1].set_ylabel('Frequency')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.output_dir / "entailment_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✅ Entailment plot saved to {plot_path}")
        plt.close()
    
    def plot_correlation_heatmap(self):
        """Create correlation heatmap of all metrics"""
        if not self.successful:
            return
        
        # Extract all metrics (simplified to 3 LLM metrics)
        data = {
            'LLM Correctness': [r['llm_judge']['correctness_score'] for r in self.successful],
            'LLM Completeness': [r['llm_judge']['completeness_score'] for r in self.successful],
            'LLM Overall': [r['llm_judge']['overall_score'] for r in self.successful],
            'Max Similarity': [r['similarity']['max_similarity'] for r in self.successful],
            'Mean Similarity': [r['similarity']['mean_similarity'] for r in self.successful],
            'Entailment Ratio': [r['entailment']['entailment_ratio'] for r in self.successful],
        }
        
        df = pd.DataFrame(data)
        correlation = df.corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0,
                   fmt='.2f', square=True, linewidths=1)
        plt.title('Correlation Between Evaluation Metrics', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        plot_path = self.output_dir / "correlation_heatmap.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✅ Correlation heatmap saved to {plot_path}")
        plt.close()
    
    def plot_performance_over_time(self):
        """Plot metrics over question index to see trends"""
        if not self.successful:
            return
        
        # Sort by question_id
        sorted_results = sorted(self.successful, key=lambda x: x['question_id'])
        
        question_ids = [r['question_id'] for r in sorted_results]
        llm_overall = [r['llm_judge']['overall_score'] for r in sorted_results]
        max_sim = [r['similarity']['max_similarity'] for r in sorted_results]
        entailment = [r['entailment']['entailment_ratio'] for r in sorted_results]
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        fig.suptitle('Performance Trends Over Dataset', fontsize=16, fontweight='bold')
        
        # LLM Overall Score
        axes[0].scatter(question_ids, llm_overall, alpha=0.5, s=10, color='blue')
        axes[0].plot(question_ids, np.convolve(llm_overall, np.ones(50)/50, mode='same'), 
                    'r-', linewidth=2, label='Moving Average (50)')
        axes[0].set_ylabel('LLM Overall Score')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Max Similarity
        axes[1].scatter(question_ids, max_sim, alpha=0.5, s=10, color='green')
        axes[1].plot(question_ids, np.convolve(max_sim, np.ones(50)/50, mode='same'),
                    'r-', linewidth=2, label='Moving Average (50)')
        axes[1].set_ylabel('Max Similarity')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Entailment Ratio
        axes[2].scatter(question_ids, entailment, alpha=0.5, s=10, color='purple')
        axes[2].plot(question_ids, np.convolve(entailment, np.ones(50)/50, mode='same'),
                    'r-', linewidth=2, label='Moving Average (50)')
        axes[2].set_ylabel('Entailment Ratio')
        axes[2].set_xlabel('Question Index')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.output_dir / "performance_trends.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✅ Performance trends plot saved to {plot_path}")
        plt.close()
    
    def create_top_bottom_examples(self, n: int = 10):
        """Extract top and bottom performing examples"""
        if not self.successful:
            return
        
        # Sort by LLM overall score
        sorted_results = sorted(self.successful, 
                              key=lambda x: x['llm_judge']['overall_score'],
                              reverse=True)
        
        top_examples = sorted_results[:n]
        bottom_examples = sorted_results[-n:]
        
        # Create markdown report
        report = "# Top and Bottom Performing Examples\n\n"
        
        report += f"## Top {n} Performing Examples\n\n"
        for i, ex in enumerate(top_examples, 1):
            report += f"### Example {i}\n"
            report += f"**Question:** {ex['question']}\n\n"
            report += f"**Generated Answer:** {ex['generated_answer'][:500]}...\n\n"
            report += f"**Scores:**\n"
            report += f"- LLM Overall: {ex['llm_judge']['overall_score']}/10\n"
            report += f"- Max Similarity: {ex['similarity']['max_similarity']:.3f}\n"
            report += f"- Entailment Ratio: {ex['entailment']['entailment_ratio']:.2%}\n\n"
            report += f"**LLM Judge Reasoning:** {ex['llm_judge']['reasoning']}\n\n"
            report += "---\n\n"
        
        report += f"\n## Bottom {n} Performing Examples\n\n"
        for i, ex in enumerate(bottom_examples, 1):
            report += f"### Example {i}\n"
            report += f"**Question:** {ex['question']}\n\n"
            report += f"**Generated Answer:** {ex['generated_answer'][:500]}...\n\n"
            report += f"**Scores:**\n"
            report += f"- LLM Overall: {ex['llm_judge']['overall_score']}/10\n"
            report += f"- Max Similarity: {ex['similarity']['max_similarity']:.3f}\n"
            report += f"- Entailment Ratio: {ex['entailment']['entailment_ratio']:.2%}\n\n"
            report += f"**LLM Judge Reasoning:** {ex['llm_judge']['reasoning']}\n\n"
            report += "---\n\n"
        
        report_path = self.output_dir / "top_bottom_examples.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"✅ Examples report saved to {report_path}")
    
    def generate_full_report(self):
        """Generate complete analysis report"""
        print(f"\n{'='*80}")
        print("GENERATING COMPREHENSIVE ANALYSIS REPORT")
        print(f"{'='*80}\n")
        
        # Summary statistics
        print("1. Creating summary statistics...")
        stats_df = self.create_summary_statistics()
        print(stats_df.to_string(index=False))
        print()
        
        # Visualizations
        print("2. Generating LLM judge score plots...")
        self.plot_llm_judge_scores()
        
        print("3. Generating similarity score plots...")
        self.plot_similarity_scores()
        
        print("4. Generating entailment analysis plots...")
        self.plot_entailment_analysis()
        
        print("5. Generating correlation heatmap...")
        self.plot_correlation_heatmap()
        
        print("6. Generating performance trends...")
        self.plot_performance_over_time()
        
        print("7. Extracting top/bottom examples...")
        self.create_top_bottom_examples()
        
        print(f"\n{'='*80}")
        print(f"✅ ANALYSIS COMPLETE!")
        print(f"📁 All files saved to: {self.output_dir}")
        print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze evaluation results")
    parser.add_argument('results_file', help='Path to JSON results file')
    parser.add_argument('--stats-only', action='store_true',
                       help='Only generate statistics, no plots')
    
    args = parser.parse_args()
    
    analyzer = ResultsAnalyzer(args.results_file)
    
    if args.stats_only:
        analyzer.create_summary_statistics()
    else:
        analyzer.generate_full_report()


if __name__ == "__main__":
    main()
