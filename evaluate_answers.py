"""
Evaluate Pre-Generated Answers
================================

This script evaluates pre-generated answers from CSV files using:
1. LLM as a Judge (Gemini)
2. Sentence Transformer Similarity
3. Entailment Detection

Much faster than evaluation.py since it doesn't wait for answer generation.
"""

import os
import json
import time
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
from tqdm import tqdm
from pydantic import BaseModel
from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import torch
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================================================
# PYDANTIC MODELS FOR STRUCTURED OUTPUT
# ============================================================================

class LLMJudgeEvaluation(BaseModel):
    """Structured output schema for LLM judge evaluation"""
    correctness_score: int  # 1-10: Factual accuracy compared to references
    completeness_score: int  # 1-10: Coverage of key points from references
    overall_score: int  # 1-10: Overall quality of the answer
    reasoning: str  # Brief explanation of the scores

# ============================================================================
# CONFIGURATION
# ============================================================================

EVALUATION_CONFIG = {
    "results_dir": "evaluation_results",
    "similarity_model": "all-MiniLM-L6-v2",
    "entailment_model": "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
    "max_retries": 3,
    "retry_delay": 10,
    "llm_retry_backoff_factor": 2,
    "llm_request_cooldown": 5,  # 15 RPM 
    "save_interval": 10,  # Save results every N questions
}

# ============================================================================
# EVALUATION SYSTEM
# ============================================================================

class AnswerEvaluator:
    """Evaluates pre-generated answers from CSV files"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize evaluation system"""
        self.config = {**EVALUATION_CONFIG, **(config or {})}
        
        # Create results directory
        self.results_dir = Path(self.config["results_dir"])
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize models
        print("🔧 Initializing evaluation models...")
        self._init_similarity_model()
        self._init_entailment_model()
        self._init_gemini()
        self._last_llm_request_ts = 0.0
        print("✅ All models initialized successfully!\n")
    
    def _init_similarity_model(self):
        """Initialize sentence transformer for similarity evaluation"""
        print(f"  Loading similarity model: {self.config['similarity_model']}")
        self.similarity_model = SentenceTransformer(self.config['similarity_model'])
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.similarity_model.to(self.device)
        print(f"  ✓ Similarity model loaded on {self.device}")
    
    def _init_entailment_model(self):
        """Initialize entailment model"""
        print(f"  Loading entailment model: {self.config['entailment_model']}")
        self.entailment_pipeline = pipeline(
            "text-classification",
            model=self.config['entailment_model'],
            device=0 if torch.cuda.is_available() else -1
        )
        print(f"  ✓ Entailment model loaded")
    
    def _init_gemini(self):
        """Initialize Gemini API for LLM-as-judge"""
        print("  Configuring Gemini API...")
        self.gemini_api_key_primary = os.getenv('GEMINI_API_KEY')
        self.gemini_api_key_backup = os.getenv('GEMINI_API_KEY_2')
        
        if not self.gemini_api_key_primary:
            raise ValueError("GEMINI_API_KEY not found in environment variables!")
        
        self.current_api_key = self.gemini_api_key_primary
        self.gemini_model = genai.Client(api_key=self.current_api_key)
        self.using_backup_key = False
        print("  ✓ Gemini API configured")
        if self.gemini_api_key_backup:
            print("  ✓ Backup API key available")

    def _respect_llm_rate_limit(self):
        """Ensure we leave breathing room between judge calls"""
        cooldown = self.config.get('llm_request_cooldown', 0)
        if cooldown <= 0:
            return
        elapsed = time.time() - getattr(self, '_last_llm_request_ts', 0)
        if elapsed < cooldown:
            time.sleep(cooldown - elapsed)

    # ========================================================================
    # EVALUATION METRICS
    # ========================================================================

    def evaluate_with_llm_judge(self, question: str, generated_answer: str, 
                                reference_answers: List[str]) -> Dict[str, Any]:
        """Evaluate answer quality using Gemini as LLM judge"""
        # Prepare reference answers
        ref_text = "\n\n".join([f"Reference {i+1}: {ans}" 
                                for i, ans in enumerate(reference_answers[:3])])
        
        prompt = f"""Evaluate the following generated answer against the reference answers.

Question: {question}

Generated Answer:
{generated_answer}

Reference Answers:
{ref_text}

Provide your evaluation with scores (1-10) for correctness, completeness, and overall quality."""

        for attempt in range(self.config['max_retries']):
            try:
                self._respect_llm_rate_limit()
                response = self.gemini_model.models.generate_content(
                    model="gemini-2.5-flash-lite",
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=LLMJudgeEvaluation
                    ),
                    contents=prompt
                )

                if response.text is None:
                    raise ValueError("Response text is None")
                
                result = LLMJudgeEvaluation.model_validate_json(response.text)
                self._last_llm_request_ts = time.time()
                return result.model_dump()
                
            except Exception as e:
                error_str = str(e).lower()
                is_rate_limit = 'rate limit' in error_str or 'quota' in error_str or '429' in error_str
                
                # Try switching to backup key on rate limit
                if is_rate_limit and not self.using_backup_key and self.gemini_api_key_backup:
                    print(f"⚠️  Rate limit hit on primary key, switching to backup key...")
                    self.current_api_key = self.gemini_api_key_backup
                    self.gemini_model = genai.Client(api_key=self.current_api_key)
                    self.using_backup_key = True
                    time.sleep(2)  # Brief pause before retry
                    continue
                
                wait_seconds = self.config['retry_delay'] * (self.config['llm_retry_backoff_factor'] ** attempt)
                print(f"⚠️  Gemini evaluation failed (attempt {attempt + 1}/{self.config['max_retries']}): {e}")
                if attempt < self.config['max_retries'] - 1:
                    sleep_time = min(wait_seconds, 60)
                    print(f"   Sleeping {sleep_time:.1f}s before retrying to respect rate limits...")
                    time.sleep(sleep_time)
                self._last_llm_request_ts = time.time()
        
        return {
            'correctness_score': 0,
            'completeness_score': 0,
            'overall_score': 0,
            'reasoning': "All retry attempts failed"
        }
    
    def evaluate_similarity(self, generated_answer: str, 
                           reference_answers: List[str]) -> Dict[str, Any]:
        """Evaluate semantic similarity using sentence transformers"""
        try:
            gen_embedding = self.similarity_model.encode(
                generated_answer, 
                convert_to_tensor=True,
                device=self.device
            )
            
            ref_embeddings = self.similarity_model.encode(
                reference_answers,
                convert_to_tensor=True,
                device=self.device
            )
            
            similarities = util.cos_sim(gen_embedding, ref_embeddings)[0]
            similarities_list = similarities.cpu().numpy().tolist()
            
            return {
                'max_similarity': float(max(similarities_list)),
                'mean_similarity': float(np.mean(similarities_list)),
                'min_similarity': float(min(similarities_list)),
                'all_similarities': similarities_list,
            }
        except Exception as e:
            return {
                'max_similarity': 0.0,
                'mean_similarity': 0.0,
                'min_similarity': 0.0,
                'all_similarities': [],
                'error': str(e)
            }
    
    def evaluate_entailment(self, generated_answer: str, 
                           reference_answers: List[str]) -> Dict[str, Any]:
        """Evaluate entailment between generated and reference answers"""
        try:
            # Validate inputs
            if not isinstance(reference_answers, list):
                reference_answers = [str(reference_answers)]
            if not reference_answers:
                return {
                    'entailment_ratio': 0.0,
                    'entailment_count': 0,
                    'total_comparisons': 0,
                    'detailed_results': [],
                    'error': 'No reference answers provided'
                }
            
            entailment_results = []
            entailment_probs = []
            
            for ref_answer in reference_answers:
                raw_scores = self.entailment_pipeline(
                    f"{generated_answer} [SEP] {ref_answer}",
                    truncation=True,
                    max_length=512,
                    top_k=None
                )
                scores = self._normalize_entailment_scores(raw_scores)
                
                entail_prob = next(
                    (item['score'] for item in scores if 'entail' in item['label'].lower()),
                    0.0
                )
                top_prediction = max(scores, key=lambda item: item['score']) if scores else {'label': 'unknown', 'score': 0.0}
                entailment_results.append({
                    'predicted_label': top_prediction['label'],
                    'predicted_score': top_prediction['score'],
                    'entailment_probability': entail_prob,
                    'scores': scores
                })
                entailment_probs.append(entail_prob)
            
            entailment_ratio = float(np.mean(entailment_probs)) if entailment_probs else 0.0
            predicted_entailments = sum(
                1 for r in entailment_results if 'entail' in r['predicted_label'].lower()
            )
            
            return {
                'entailment_ratio': entailment_ratio,
                'entailment_count': predicted_entailments,
                'total_comparisons': len(reference_answers),
                'detailed_results': entailment_results,
            }
        except Exception as e:
            return {
                'entailment_ratio': 0.0,
                'entailment_count': 0,
                'total_comparisons': 0,
                'detailed_results': [],
                'error': str(e)
            }

    def _normalize_entailment_scores(self, model_output: Any) -> List[Dict[str, Any]]:
        """Normalize pipeline outputs into a flat list of label-score dicts"""
        try:
            if model_output is None:
                return []
            
            # Single input -> pipeline returns list of dicts
            if isinstance(model_output, list):
                if not model_output:
                    return []
                first = model_output[0]
                
                # Newer transformers: [ {'label': ..., 'score': ...}, ... ]
                if isinstance(first, dict) and 'label' in first:
                    return model_output
                # Older return_all_scores=True style: [ [ {...}, {...} ] ]
                if isinstance(first, list):
                    return first
            
            # Defensive: some pipelines might return dict directly
            if isinstance(model_output, dict) and 'label' in model_output:
                return [model_output]
            
            raise ValueError(f"Unexpected entailment output format: {type(model_output)}, value: {model_output}")
        except Exception as e:
            # Enhanced error logging
            import traceback
            print(f"❌ Entailment normalization error:")
            print(f"   Type: {type(model_output)}")
            print(f"   Value: {model_output}")
            print(f"   Exception: {e}")
            traceback.print_exc()
            raise

    # ========================================================================
    # BATCH EVALUATION
    # ========================================================================

    def evaluate_csv(self, csv_file: str, output_file: Optional[str] = None,
                    start_idx: int = 0, end_idx: Optional[int] = None) -> str:
        """
        Evaluate pre-generated answers from CSV file
        
        Args:
            csv_file: Path to CSV with generated answers
            output_file: Path to save evaluation results (auto-generated if None)
            start_idx: Start from this row
            end_idx: End at this row (None = process all)
            
        Returns:
            Path to evaluation results file
        """
        print(f"📂 Loading answers from: {csv_file}")
        df = pd.read_csv(csv_file)
        
        # Filter only successful generations
        df = df[df['status'] == 'success'].reset_index(drop=True)
        print(f"  Found {len(df)} successfully generated answers")
        
        # Subset
        end_idx = end_idx or len(df)
        df_subset = df.iloc[start_idx:end_idx].reset_index(drop=True)
        print(f"  Evaluating rows {start_idx} to {end_idx} ({len(df_subset)} questions)\n")
        
        # Prepare output file
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            input_name = Path(csv_file).stem
            output_file = self.results_dir / f"eval_{input_name}_{timestamp}.json"
        
        output_file = Path(output_file)
        
        # Check for existing results to resume
        results = []
        processed_count = 0
        
        if output_file.exists():
            print(f"📂 Found existing evaluation file: {output_file}")
            with open(output_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            processed_count = len(results)
            print(f"   Resuming from question {processed_count + 1}\n")
        
        # Evaluate each answer
        print("🚀 Starting evaluation\n")
        
        for idx in tqdm(range(processed_count, len(df_subset)), desc="Evaluating"):
            row = df_subset.iloc[idx]
            
            question = row['question']
            generated_answer = row['generated_answer']
            # Ensure reference_answers is a string before splitting
            ref_answers_raw = row['reference_answers']
            if isinstance(ref_answers_raw, str):
                reference_answers = [ans.strip() for ans in ref_answers_raw.split('|||') if ans.strip()]
            elif isinstance(ref_answers_raw, list):
                reference_answers = [str(ans).strip() for ans in ref_answers_raw if str(ans).strip()]
            else:
                reference_answers = [str(ref_answers_raw).strip()]
            
            question_id = row['question_id']
            
            try:
                # Evaluate with all three metrics
                llm_judge = self.evaluate_with_llm_judge(
                    question, generated_answer, reference_answers
                )
                
                similarity = self.evaluate_similarity(
                    generated_answer, reference_answers
                )
                
                entailment = self.evaluate_entailment(
                    generated_answer, reference_answers
                )
                
                # Compile result
                eval_result = {
                    'question_id': int(question_id),
                    'question': question,
                    'generated_answer': generated_answer,
                    'reference_answers': reference_answers,
                    'generation_time': float(row['generation_time']),
                    'llm_judge': llm_judge,
                    'similarity': similarity,
                    'entailment': entailment,
                    'evaluation_status': 'success',
                    'timestamp': datetime.now().isoformat()
                }
                
                results.append(eval_result)
                
            except Exception as e:
                print(f"\n❌ Error evaluating question {idx}: {str(e)}")
                results.append({
                    'question_id': int(question_id),
                    'question': question,
                    'evaluation_status': 'error',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
            
            # Save periodically
            if (idx + 1) % self.config['save_interval'] == 0 or (idx + 1) == len(df_subset):
                self._save_results(results, output_file)
        
        # Final save
        self._save_results(results, output_file)
        
        print(f"\n🎉 Evaluation complete!")
        print(f"   Results saved to: {output_file}")
        
        # Generate summary report
        self.generate_report(str(output_file))
        
        return str(output_file)
    
    def _save_results(self, results: List[Dict], filepath: Path):
        """Save results to JSON and CSV"""
        # Save full JSON
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Save CSV summary
        csv_path = filepath.with_suffix('.csv')
        summary_df = self._create_summary_dataframe(results)
        summary_df.to_csv(csv_path, index=False)
    
    def _create_summary_dataframe(self, results: List[Dict]) -> pd.DataFrame:
        """Create summary DataFrame from results"""
        summary_data = []
        
        for r in results:
            if r.get('evaluation_status') != 'success':
                continue
                
            summary_data.append({
                'question_id': r['question_id'],
                'question': r['question'][:100] + '...',
                'generation_time': r['generation_time'],
                'llm_correctness': r['llm_judge'].get('correctness_score', 0),
                'llm_completeness': r['llm_judge'].get('completeness_score', 0),
                'llm_overall': r['llm_judge'].get('overall_score', 0),
                'similarity_max': r['similarity'].get('max_similarity', 0),
                'similarity_mean': r['similarity'].get('mean_similarity', 0),
                'entailment_ratio': r['entailment'].get('entailment_ratio', 0),
            })
        
        return pd.DataFrame(summary_data)

    # ========================================================================
    # REPORTING
    # ========================================================================

    def generate_report(self, results_file: str) -> Dict[str, Any]:
        """Generate comprehensive evaluation report"""
        print(f"\n📈 Generating evaluation report...")
        
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        successful = [r for r in results if r.get('evaluation_status') == 'success']
        
        if not successful:
            print("❌ No successful evaluations found!")
            return {}
        
        # Calculate statistics
        report = {
            'total_questions': len(results),
            'successful_evaluations': len(successful),
            'failed_evaluations': len(results) - len(successful),
            'success_rate': len(successful) / len(results),
            'avg_generation_time': np.mean([r['generation_time'] for r in successful]),
            
            'llm_judge': {
                'avg_correctness': np.mean([r['llm_judge']['correctness_score'] for r in successful]),
                'avg_completeness': np.mean([r['llm_judge']['completeness_score'] for r in successful]),
                'avg_overall': np.mean([r['llm_judge']['overall_score'] for r in successful]),
                'std_correctness': np.std([r['llm_judge']['correctness_score'] for r in successful]),
                'std_completeness': np.std([r['llm_judge']['completeness_score'] for r in successful]),
                'std_overall': np.std([r['llm_judge']['overall_score'] for r in successful]),
            },
            
            'similarity': {
                'avg_max_similarity': np.mean([r['similarity']['max_similarity'] for r in successful]),
                'avg_mean_similarity': np.mean([r['similarity']['mean_similarity'] for r in successful]),
                'std_max_similarity': np.std([r['similarity']['max_similarity'] for r in successful]),
            },
            
            'entailment': {
                'avg_entailment_ratio': np.mean([r['entailment']['entailment_ratio'] for r in successful]),
                'std_entailment_ratio': np.std([r['entailment']['entailment_ratio'] for r in successful]),
            },
            
            'report_generated': datetime.now().isoformat()
        }
        
        # Save report
        report_file = Path(results_file).with_suffix('.report.json')
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        
        # Print report
        print("\n" + "="*80)
        print("EVALUATION REPORT")
        print("="*80)
        print(f"Total Questions: {report['total_questions']}")
        print(f"Successful: {report['successful_evaluations']} ({report['success_rate']:.1%})")
        print(f"Avg Generation Time: {report['avg_generation_time']:.2f}s")
        print(f"\nLLM Judge Scores (mean ± std):")
        print(f"  Correctness: {report['llm_judge']['avg_correctness']:.2f} ± {report['llm_judge']['std_correctness']:.2f}")
        print(f"  Completeness: {report['llm_judge']['avg_completeness']:.2f} ± {report['llm_judge']['std_completeness']:.2f}")
        print(f"  Overall: {report['llm_judge']['avg_overall']:.2f} ± {report['llm_judge']['std_overall']:.2f}")
        print(f"\nSimilarity Scores:")
        print(f"  Max Similarity: {report['similarity']['avg_max_similarity']:.3f} ± {report['similarity']['std_max_similarity']:.3f}")
        print(f"  Mean Similarity: {report['similarity']['avg_mean_similarity']:.3f}")
        print(f"\nEntailment:")
        print(f"  Entailment Ratio: {report['entailment']['avg_entailment_ratio']:.2%} ± {report['entailment']['std_entailment_ratio']:.2%}")
        print("="*80)
        print(f"\n📄 Report saved to: {report_file}")
        
        return report


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Evaluate pre-generated answers from CSV files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate all answers in a CSV file
  python evaluate_answers.py generated_answers/answers_split0_20241107_120000.csv
  
  # Evaluate specific range
  python evaluate_answers.py answers.csv --start 0 --end 1000
  
  # Generate report from existing evaluation results
  python evaluate_answers.py --report evaluation_results/eval_answers_split0_20241107_130000.json
  
  # Resume from previous evaluation
  python evaluate_answers.py answers.csv --resume evaluation_results/eval_answers_split0_20241107_130000.json
        """
    )
    
    parser.add_argument('csv_file', nargs='?', help='CSV file with generated answers')
    parser.add_argument('--start', type=int, default=0, help='Start index (default: 0)')
    parser.add_argument('--end', type=int, help='End index (default: all)')
    parser.add_argument('--output', type=str, help='Output file path')
    parser.add_argument('--report', type=str, help='Generate report from existing results file')
    parser.add_argument('--resume', type=str, help='Resume from existing evaluation file')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = AnswerEvaluator()
    
    # Report-only mode
    if args.report:
        evaluator.generate_report(args.report)
        return
    
    # Validate input
    if not args.csv_file:
        parser.print_help()
        return
    
    # Evaluate CSV
    evaluator.evaluate_csv(
        csv_file=args.csv_file,
        output_file=args.output or args.resume,
        start_idx=args.start,
        end_idx=args.end
    )


if __name__ == "__main__":
    main()
