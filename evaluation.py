"""
ELI5 Dataset Evaluation System
===============================

This script evaluates your multi-agent system's answers against the ELI5 dataset using:
1. LLM as a Judge (Gemini)
2. Sentence Transformer Similarity (Cosine Similarity)
3. Entailment Detection
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
from main import run_multi_agent_system, MODEL_CONFIGS

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
    "dataset_path": "data/eli5_dataset",  # Path where HuggingFace dataset will be cached
    "results_dir": "evaluation_results",  # Directory to store results
    "batch_size": 100,  # Process this many questions before saving
    "num_splits": 3,  # Split dataset into this many parts
    "similarity_model": "all-MiniLM-L6-v2",  # Sentence transformer model
    "entailment_model": "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",  # Entailment model (NLI fine-tuned)
    "max_retries": 3,  # Max retries for API calls
    "retry_delay": 2,  # Seconds to wait between retries
}

# ============================================================================
# SETUP MODELS AND DIRECTORIES
# ============================================================================

class EvaluationSystem:
    """Main evaluation system for ELI5 dataset"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize evaluation system with models and configurations"""
        self.config = {**EVALUATION_CONFIG, **(config or {})}
        
        # Create results directory
        self.results_dir = Path(self.config["results_dir"])
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize models
        print("🔧 Initializing evaluation models...")
        self._init_similarity_model()
        self._init_entailment_model()
        self._init_gemini()
        
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
        gemini_api_key = os.getenv('GEMINI_API_KEY')
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables!")
        
        # System instruction for the LLM judge
        system_instruction = """You are an expert evaluator for educational question-answering systems.
Your task is to evaluate generated answers against reference answers from the ELI5 dataset.

Evaluate on these criteria:
1. CORRECTNESS: Is the information factually accurate compared to the references?
2. COMPLETENESS: Does it cover the key points mentioned in the references?
3. OVERALL QUALITY: How well does the answer explain the concept overall?

Provide scores from 1-10 for each criterion, where:
- 1-3: Poor quality
- 4-6: Acceptable
- 7-8: Good
- 9-10: Excellent

Always be objective and consistent in your evaluations."""
        
        # Create model with structured output
        self.gemini_model = genai.Client()

# ============================================================================
# EVALUATION METRICS
# ============================================================================

    def evaluate_with_llm_judge(self, question: str, generated_answer: str, 
                                reference_answers: List[str]) -> Dict[str, Any]:
        """
        Evaluate answer quality using Gemini as LLM judge with structured output
        
        Args:
            question: The original question
            generated_answer: Answer from your multi-agent system
            reference_answers: List of reference answers from ELI5 dataset
            
        Returns:
            Dictionary with scores and reasoning
        """
        # Prepare reference answers (use top 3 if multiple)
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
                # Generate content with structured output
                response = self.gemini_model.models.generate_content(
                    model="gemini-2.5-flash",
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=LLMJudgeEvaluation
                    ),
                    contents=prompt
                )

                if response.text is None:
                    raise ValueError("Response text is None")
                
                result = LLMJudgeEvaluation.model_validate_json(response.text)

                return result.model_dump()
                
            except Exception as e:
                print(f"  ⚠️  LLM Judge attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.config['max_retries'] - 1:
                    time.sleep(self.config['retry_delay'])
        
        # If all retries failed, return error result
        return {
            'success': False,
            'correctness_score': 0,
            'completeness_score': 0,
            'overall_score': 0,
            'reasoning': "All retry attempts failed"
        }
    
    def evaluate_similarity(self, generated_answer: str, 
                           reference_answers: List[str]) -> Dict[str, Any]:
        """
        Evaluate semantic similarity using sentence transformers
        
        Args:
            generated_answer: Answer from your multi-agent system
            reference_answers: List of reference answers from ELI5 dataset
            
        Returns:
            Dictionary with similarity scores
        """
        try:
            # Encode generated answer
            gen_embedding = self.similarity_model.encode(
                generated_answer, 
                convert_to_tensor=True,
                device=self.device
            )
            
            # Encode reference answers
            ref_embeddings = self.similarity_model.encode(
                reference_answers,
                convert_to_tensor=True,
                device=self.device
            )
            
            # Calculate cosine similarities
            similarities = util.cos_sim(gen_embedding, ref_embeddings)[0]
            similarities_list = similarities.cpu().numpy().tolist()
            
            return {
                'max_similarity': float(max(similarities_list)),
                'mean_similarity': float(np.mean(similarities_list)),
                'min_similarity': float(min(similarities_list)),
                'all_similarities': similarities_list,
                'success': True
            }
        except Exception as e:
            print(f"  ⚠️  Similarity calculation failed: {str(e)}")
            return {
                'max_similarity': 0.0,
                'mean_similarity': 0.0,
                'min_similarity': 0.0,
                'all_similarities': [],
                'success': False,
                'error': str(e)
            }
    
    def evaluate_entailment(self, generated_answer: str, 
                           reference_answers: List[str]) -> Dict[str, Any]:
        """
        Evaluate entailment between generated and reference answers
        
        Args:
            generated_answer: Answer from your multi-agent system
            reference_answers: List of reference answers from ELI5 dataset
            
        Returns:
            Dictionary with entailment scores
        """
        try:
            entailment_results = []
            
            for ref_answer in reference_answers:
                # Check if generated answer entails reference (premise -> hypothesis)
                result = self.entailment_pipeline(
                    f"{generated_answer} [SEP] {ref_answer}",
                    truncation=True,
                    max_length=512
                )[0]
                
                entailment_results.append({
                    'label': result['label'],
                    'score': result['score']
                })
            
            # Calculate aggregate metrics
            entailment_count = sum(1 for r in entailment_results 
                                  if r['label'].lower() == 'entailment')
            
            return {
                'entailment_ratio': entailment_count / len(reference_answers) if reference_answers else 0,
                'entailment_count': entailment_count,
                'total_comparisons': len(reference_answers),
                'detailed_results': entailment_results,
                'success': True
            }
        except Exception as e:
            print(f"  ⚠️  Entailment evaluation failed: {str(e)}")
            return {
                'entailment_ratio': 0.0,
                'entailment_count': 0,
                'total_comparisons': 0,
                'detailed_results': [],
                'success': False,
                'error': str(e)
            }

# ============================================================================
# DATASET PROCESSING
# ============================================================================

    def load_eli5_dataset(self, split_index: Optional[int] = None) -> pd.DataFrame:
        """
        Load ELI5 dataset from HuggingFace
        
        Args:
            split_index: Which split to load (0, 1, or 2). None loads full dataset.
            
        Returns:
            DataFrame with questions and answers
        """
        from datasets import load_dataset
        
        print(f"📥 Loading ELI5 dataset...")
        dataset = load_dataset("sentence-transformers/eli5", split="train")
        
        df = pd.DataFrame({
            'query': dataset['question'],
            'answers': dataset['answer']
        })
        print(f"  Total rows in dataset: {len(df)}")
        
        if split_index is not None:
            # Split into parts
            num_splits = self.config['num_splits']
            split_size = len(df) // num_splits
            start_idx = split_index * split_size
            end_idx = start_idx + split_size if split_index < num_splits - 1 else len(df)
            
            df = df.iloc[start_idx:end_idx].reset_index(drop=True)
            print(f"  Using split {split_index + 1}/{num_splits}: rows {start_idx} to {end_idx}")
        
        return df
    
    def process_questions(self, df: pd.DataFrame, model_config_name: str = "config1",
                         start_idx: int = 0, end_idx: Optional[int] = None,
                         resume_from: Optional[str] = None) -> pd.DataFrame:
        """
        Process questions through multi-agent system and evaluate
        
        Args:
            df: DataFrame with ELI5 questions
            model_config_name: Which model config to use
            start_idx: Start processing from this index
            end_idx: End at this index (None = process all)
            resume_from: Path to existing results file to resume from
            
        Returns:
            DataFrame with results
        """
        if model_config_name not in MODEL_CONFIGS:
            raise ValueError(f"Invalid model config: {model_config_name}")
        
        model_config = MODEL_CONFIGS[model_config_name]
        end_idx = end_idx or len(df)
        
        print(f"\n🚀 Starting evaluation with {model_config_name}")
        print(f"   Processing rows {start_idx} to {end_idx}")
        print(f"   Model config: {model_config}\n")
        
        # Resume from existing results if provided
        results = []
        if resume_from and Path(resume_from).exists():
            print(f"📂 Resuming from: {resume_from}")
            with open(resume_from, 'r', encoding='utf-8') as f:
                results = json.load(f)
            print(f"   Loaded {len(results)} existing results")
            start_idx = len(results)
        
        # Create timestamped results file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"eval_{model_config_name}_split_{timestamp}.json"
        
        batch_results = []
        
        # Process each question
        for idx in tqdm(range(start_idx, end_idx), desc="Evaluating"):
            row = df.iloc[idx]
            question = row['query']
            reference_answers = row['answers'] if isinstance(row['answers'], list) else [row['answers']]
            
            try:
                print(f"\n{'='*80}")
                print(f"Question {idx + 1}/{end_idx}: {question[:100]}...")
                print(f"{'='*80}")
                
                # Generate answer using multi-agent system
                start_time = time.time()
                result = run_multi_agent_system(question, model_config)
                generation_time = time.time() - start_time
                
                generated_answer = result.get('final_answer', '')
                
                if not generated_answer or generated_answer == "No final answer generated":
                    print("  ⚠️  No answer generated, skipping evaluation")
                    eval_result = {
                        'question_id': idx,
                        'question': question,
                        'generated_answer': generated_answer,
                        'reference_answers': reference_answers,
                        'generation_time': generation_time,
                        'evaluation_status': 'failed_generation',
                        'error': 'No answer generated'
                    }
                else:
                    # Evaluate the answer
                    print("\n📊 Evaluating answer...")
                    
                    # 1. LLM as Judge
                    print("  1️⃣  LLM Judge (Gemini)...")
                    llm_judge = self.evaluate_with_llm_judge(
                        question, generated_answer, reference_answers
                    )
                    
                    # 2. Similarity
                    print("  2️⃣  Semantic Similarity...")
                    similarity = self.evaluate_similarity(
                        generated_answer, reference_answers
                    )
                    
                    # 3. Entailment
                    print("  3️⃣  Entailment Detection...")
                    entailment = self.evaluate_entailment(
                        generated_answer, reference_answers
                    )
                    
                    # Compile results
                    eval_result = {
                        'question_id': idx,
                        'question': question,
                        'generated_answer': generated_answer,
                        'reference_answers': reference_answers,
                        'generation_time': generation_time,
                        'model_config': model_config_name,
                        'evaluation_status': 'success',
                        'llm_judge': llm_judge,
                        'similarity': similarity,
                        'entailment': entailment,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    # Print summary
                    print(f"\n  ✅ Evaluation complete!")
                    print(f"     LLM Judge Overall: {llm_judge.get('overall_score', 'N/A')}/10")
                    print(f"     Max Similarity: {similarity.get('max_similarity', 0):.3f}")
                    print(f"     Entailment Ratio: {entailment.get('entailment_ratio', 0):.2%}")
                
                batch_results.append(eval_result)
                
                # Save batch results periodically
                if len(batch_results) >= self.config['batch_size']:
                    results.extend(batch_results)
                    self._save_results(results, results_file)
                    print(f"\n💾 Saved batch of {len(batch_results)} results")
                    batch_results = []
                
            except Exception as e:
                print(f"\n❌ Error processing question {idx}: {str(e)}")
                batch_results.append({
                    'question_id': idx,
                    'question': question,
                    'evaluation_status': 'error',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
        
        # Save final batch
        if batch_results:
            results.extend(batch_results)
            self._save_results(results, results_file)
        
        print(f"\n🎉 Evaluation complete! Results saved to: {results_file}")
        
        return pd.DataFrame(results)
    
    def _save_results(self, results: List[Dict], filepath: Path):
        """Save results to JSON file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Also save CSV summary
        csv_path = filepath.with_suffix('.csv')
        summary_df = self._create_summary_dataframe(results)
        summary_df.to_csv(csv_path, index=False)

# ============================================================================
# ANALYSIS AND REPORTING
# ============================================================================

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
    
    def generate_report(self, results_file: str) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report
        
        Args:
            results_file: Path to JSON results file
            
        Returns:
            Dictionary with aggregate statistics
        """
        print(f"\n📈 Generating evaluation report from: {results_file}")
        
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        # Filter successful evaluations
        successful = [r for r in results if r.get('evaluation_status') == 'success']
        
        if not successful:
            print("❌ No successful evaluations found!")
            return {}
        
        # Calculate aggregate statistics
        report = {
            'total_questions': len(results),
            'successful_evaluations': len(successful),
            'failed_evaluations': len(results) - len(successful),
            'success_rate': len(successful) / len(results),
            'avg_generation_time': np.mean([r['generation_time'] for r in successful]),
            
            # LLM Judge metrics (simplified to 3 main metrics)
            'llm_judge': {
                'avg_correctness': np.mean([r['llm_judge']['correctness_score'] for r in successful]),
                'avg_completeness': np.mean([r['llm_judge']['completeness_score'] for r in successful]),
                'avg_overall': np.mean([r['llm_judge']['overall_score'] for r in successful]),
            },
            
            # Similarity metrics
            'similarity': {
                'avg_max_similarity': np.mean([r['similarity']['max_similarity'] for r in successful]),
                'avg_mean_similarity': np.mean([r['similarity']['mean_similarity'] for r in successful]),
            },
            
            # Entailment metrics
            'entailment': {
                'avg_entailment_ratio': np.mean([r['entailment']['entailment_ratio'] for r in successful]),
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
        print(f"\nLLM Judge Scores (avg):")
        print(f"  Correctness: {report['llm_judge']['avg_correctness']:.2f}/10")
        print(f"  Completeness: {report['llm_judge']['avg_completeness']:.2f}/10")
        print(f"  Overall: {report['llm_judge']['avg_overall']:.2f}/10")
        print(f"\nSimilarity Scores:")
        print(f"  Max Similarity: {report['similarity']['avg_max_similarity']:.3f}")
        print(f"  Mean Similarity: {report['similarity']['avg_mean_similarity']:.3f}")
        print(f"\nEntailment:")
        print(f"  Entailment Ratio: {report['entailment']['avg_entailment_ratio']:.2%}")
        print("="*80)
        print(f"\n📄 Report saved to: {report_file}")
        
        return report


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Evaluate multi-agent system on ELI5 dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process first split (rows 0-108k) with config1
  python evaluation.py --split 0 --config config1
  
  # Process specific range with config2
  python evaluation.py --start 0 --end 1000 --config config2
  
  # Generate report from results file
  python evaluation.py --report evaluation_results/eval_config1_split_20251106_120000.json
  
  # Resume from previous run
  python evaluation.py --split 1 --config config1 --resume evaluation_results/eval_config1_split_20251106_120000.json
        """
    )
    
    parser.add_argument('--split', type=int, choices=[0, 1, 2],
                       help='Which split to process (0, 1, or 2)')
    parser.add_argument('--start', type=int, default=0,
                       help='Start index (default: 0)')
    parser.add_argument('--end', type=int,
                       help='End index (default: end of split/dataset)')
    parser.add_argument('--config', type=str, default='config1',
                       help='Model configuration to use (default: config1)')
    parser.add_argument('--report', type=str,
                       help='Generate report from results file')
    parser.add_argument('--resume', type=str,
                       help='Resume from existing results file')
    
    args = parser.parse_args()
    
    # Initialize evaluation system
    eval_system = EvaluationSystem()
    
    # Generate report mode
    if args.report:
        eval_system.generate_report(args.report)
        return
    
    # Load dataset
    df = eval_system.load_eli5_dataset(split_index=args.split)
    
    # Process questions
    results_df = eval_system.process_questions(
        df,
        model_config_name=args.config,
        start_idx=args.start,
        end_idx=args.end,
        resume_from=args.resume
    )
    
    print(f"\n✅ Processing complete! Processed {len(results_df)} questions.")


if __name__ == "__main__":
    main()
