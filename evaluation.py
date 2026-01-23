"""
Unified Evaluation Script
=========================

Metrics:
1. ROUGE (1, 2, L) 
2. Perplexity (GPT-2) 
3. Semantic Similarity (Sentence Transformers) 
4. Entailment (DeBERTa) 
5. LLM as a Judge (Llama 2 via Ollama)

Usage:
    python unified_evaluation.py --input generated_answers/answers.csv --output evaluation_results/
"""

import argparse
import ast
import json
import math
import os
import gc
import time
import requests
from pathlib import Path
from typing import Dict, List, Any, Optional, Sequence, Union
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

# Try importing ML libraries (handle absence for dry-run/setup)
try:
    from rouge_score import rouge_scorer
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    from sentence_transformers import SentenceTransformer, util
except ImportError:
    print("Warning: Some ML libraries not found. Ensure transformers, torch, rouge_score, sentence_transformers are installed.")

# Load environment variables
load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_CONFIG = {
    "perplexity_model": "gpt2",
    "similarity_model": "all-MiniLM-L6-v2",
    "entailment_model": "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
    "llm_model": "llama2:13b",
    "ollama_base_url": "http://localhost:11434",
    "llm_concurrency": 4,  # Parallel requests to Ollama
}

# ============================================================================
# DATA STRUCTURES
# ============================================================================

class LLMJudgeEvaluation(BaseModel):
    """Structured output schema for LLM judge evaluation"""
    correctness_score: int = Field(..., description="1-10 score for factual accuracy")
    completeness_score: int = Field(..., description="1-10 score for coverage of key points")
    overall_score: int = Field(..., description="1-10 overall quality score")
    reasoning: str = Field(..., description="Brief explanation of the scores")

# ============================================================================
# UTILS
# ============================================================================

def parse_reference_answers(raw_value: Any) -> List[str]:
    """Convert raw CSV cell into a list of strings."""
    if isinstance(raw_value, float) and math.isnan(raw_value):
        return []
    
    if isinstance(raw_value, list):
        return [str(item).strip() for item in raw_value if str(item).strip()]
        
    text = str(raw_value).strip()
    if not text:
        return []

    # Try JSON/Python literal
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
            if isinstance(parsed, str):
                return [parsed.strip()]
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except Exception:
            continue

    # Fallback separator
    if "|||" in text:
        return [s.strip() for s in text.split("|||") if s.strip()]
    
    return [text]

def clean_gpu_memory():
    """Clear GPU memory after a model is done."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()

# ============================================================================
# METRIC COMPUTATION (ISOLATED FUNCTIONS FOR SEQ PROCESSING)
# ============================================================================

def compute_rouge_column(df: pd.DataFrame) -> pd.DataFrame:
    """Compute ROUGE scores for the entire DataFrame."""
    print("running ROUGE evaluation...")
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    
    rouge_results = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="ROUGE"):
        generated = str(row.get("generated_answer", ""))
        references = parse_reference_answers(row.get("reference_answers", ""))
        
        metrics = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
        
        if generated and references:
            for reference in references:
                score = scorer.score(reference, generated)
                for key in metrics:
                    metrics[key] = max(metrics[key], score[key].fmeasure)
        
        rouge_results.append(metrics)
        
    result_df = pd.DataFrame(rouge_results)
    return pd.concat([df, result_df], axis=1)

def compute_perplexity_column(df: pd.DataFrame, model_name: str, device: str) -> pd.DataFrame:
    """Compute Perplexity. Loads and unloads model to save memory."""
    print(f"Loading PPL model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()
    
    ppl_scores = []
    max_length = model.config.max_position_embeddings
    
    # Helper to calculate PPL for one string
    def _calc_ppl(text):
        text = str(text).strip()
        if not text:
            return float("nan")
            
        encodings = tokenizer(text, return_tensors="pt")
        input_ids = encodings.input_ids.to(device)
        seq_len = input_ids.size(1)

        nlls = []
        prev_end = 0
        stride = 256
        
        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - begin_loc if begin_loc == 0 else end_loc - prev_end
            input_ids_slice = input_ids[:, begin_loc:end_loc]
            target_ids = input_ids_slice.clone()
            target_ids[:, :-trg_len] = -100 # Ignore context in loss

            with torch.no_grad():
                outputs = model(input_ids_slice, labels=target_ids)
                neg_log_likelihood = outputs.loss * trg_len

            nlls.append(neg_log_likelihood)
            prev_end = end_loc
            if end_loc == seq_len:
                break
        
        if not nlls:
            return float("nan")
            
        ppl = torch.exp(torch.stack(nlls).sum() / seq_len)
        return float(ppl.cpu().item())

    print("Running Perplexity evaluation...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="PPL"):
        ppl_scores.append(_calc_ppl(row.get("generated_answer", "")))
    
    # Unload
    del model
    del tokenizer
    clean_gpu_memory()
    print("PPL model unloaded.")
    
    df["perplexity"] = ppl_scores
    return df

def compute_similarity_column(df: pd.DataFrame, model_name: str, device: str) -> pd.DataFrame:
    """Compute Semantic Similarity."""
    print(f"Loading Similarity model: {model_name}...")
    model = SentenceTransformer(model_name)
    model.to(device)
    
    sim_results = []
    
    print("Running Similarity evaluation...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Similarity"):
        generated = str(row.get("generated_answer", ""))
        references = parse_reference_answers(row.get("reference_answers", ""))
        
        if not generated or not references:
            sim_results.append({
                "sim_max": 0.0, "sim_mean": 0.0, "sim_min": 0.0
            })
            continue

        gen_emb = model.encode(generated, convert_to_tensor=True, device=device)
        ref_embs = model.encode(references, convert_to_tensor=True, device=device)
        
        # Calculate cosine similarity
        similarities = util.cos_sim(gen_emb, ref_embs)[0].cpu().tolist()
        
        sim_results.append({
            "sim_max": max(similarities),
            "sim_mean": np.mean(similarities),
            "sim_min": min(similarities)
        })

    # Unload
    del model
    clean_gpu_memory()
    print("Similarity model unloaded.")
    
    result_df = pd.DataFrame(sim_results)
    return pd.concat([df, result_df], axis=1)

def compute_entailment_column(df: pd.DataFrame, model_name: str, device_id: int) -> pd.DataFrame:
    """Compute Entailment Ratio."""
    print(f"Loading Entailment model: {model_name}...")
    pipe = pipeline("text-classification", model=model_name, device=device_id, truncation=True, max_length=512)
    
    entail_results = []
    
    print("Running Entailment evaluation...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Entailment"):
        generated = str(row.get("generated_answer", ""))
        references = parse_reference_answers(row.get("reference_answers", ""))
        
        if not generated or not references:
            entail_results.append(0.0)
            continue
            
        entail_probs = []
        for ref in references:
            # Format: "premise [SEP] hypothesis" -> "generated [SEP] reference"
            # DeBERTa models often expect this format or just inputs
            inputs = f"{generated} [SEP] {ref}"
            output = pipe(inputs)
            # Output format is usually [{'label': 'entailment', 'score': 0.9}, ...]
            
            score = 0.0
            for item in output:
                if 'entail' in item['label'].lower():
                    score = item['score']
                    break
            entail_probs.append(score)
            
        entail_results.append(np.mean(entail_probs) if entail_probs else 0.0)

    # Unload
    del pipe
    clean_gpu_memory()
    print("Entailment model unloaded.")
    
    df["entailment_ratio"] = entail_results
    return df

# ============================================================================
# LLM JUDGE (OLLAMA)
# ============================================================================

def call_ollama_judge(question, generated, references, model, base_url):
    """Call Ollama API for a single row."""
    ref_text = "\\n".join([f"- {r}" for r in references[:3]])
    
    prompt = f"""[INST] You are an impartial judge. Evaluate the generated answer against the reference answers.

Question: {question}

Reference Answers:
{ref_text}

Generated Answer:
{generated}

Evaluate based on:
1. Correctness (1-10): Factual accuracy.
2. Completeness (1-10): Coverage of key points.
3. Overall (1-10): Overall quality.

Return ONLY a JSON object with this format, no other text:
{{
  "correctness_score": <int>,
  "completeness_score": <int>,
  "overall_score": <int>,
  "reasoning": "<string>"
}}
[/INST]
"""
    
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.1,
            "num_predict": 512
        }
    }
    
    try:
        response = requests.post(f"{base_url}/api/generate", json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        
        # Parse output
        content = result.get("response", "")
        # Try to clean markdown code blocks if present
        if "```" in content:
            content = content.split("```json")[-1].split("```")[0].strip()
        elif "```" in content: # Just code block
            content = content.split("```")[-1].split("```")[0].strip()
            
        return json.loads(content)
        
    except Exception as e:
        return {
            "correctness_score": 0,
            "completeness_score": 0,
            "overall_score": 0,
            "reasoning": f"Error: {str(e)}"
        }

def compute_llm_judge(df: pd.DataFrame, model_name: str, base_url: str, concurrency: int) -> pd.DataFrame:
    """Run LLM Judge in parallel using Ollama."""
    print(f"Running LLM Judge ({model_name}) with {concurrency} threads...")
    
    results = [None] * len(df)
    
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_to_idx = {}
        
        for idx, row in df.iterrows():
            generated = str(row.get("generated_answer", ""))
            references = parse_reference_answers(row.get("reference_answers", ""))
            question = str(row.get("question", ""))
            
            future = executor.submit(call_ollama_judge, question, generated, references, model_name, base_url)
            future_to_idx[future] = idx
            
        for future in tqdm(as_completed(future_to_idx), total=len(df), desc="LLM Judge"):
            idx = future_to_idx[future]
            try:
                res = future.result()
                results[idx] = res  # idx matches original df index
            except Exception as e:
                results[idx] = {"error": str(e)}

    # Convert list of dicts to DataFrame columns
    judge_df = pd.DataFrame.from_records(results)
    # Rename columns to avoid collision/clarity
    judge_df = judge_df.rename(columns={
        "correctness_score": "llm_correctness",
        "completeness_score": "llm_completeness",
        "overall_score": "llm_overall",
        "reasoning": "llm_reasoning"
    })
    
    # Handle missing/error columns if any
    for col in ["llm_correctness", "llm_completeness", "llm_overall"]:
        if col not in judge_df.columns:
            judge_df[col] = 0
            
    return pd.concat([df, judge_df], axis=1)

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Unified Evaluation Script")
    parser.add_argument("--input", required=True, help="Input CSV file")
    parser.add_argument("--output", required=True, help="Output directory or filename")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--max-rows", type=int, help="Limit number of rows for testing")
    parser.add_argument("--skip-llm", action="store_true", help="Skip the slow LLM judge step")
    parser.add_argument("--dry-run", action="store_true", help="Run without loading models (fast check)")
    args = parser.parse_args()
    
    # Setup Paths
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
        
    output_path = Path(args.output)
    if output_path.suffix == "":
        # It's a directory
        output_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_path / f"eval_unified_{input_path.stem}_{timestamp}.csv"
        summary_file = output_path / f"summary_{input_path.stem}_{timestamp}.json"
    else:
        # It's a file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_file = output_path
        summary_file = output_path.with_suffix(".json")

    # Load Data
    print(f"Reading {input_path}...")
    df = pd.read_csv(input_path)
    
    # Filter for success if column exists
    if "status" in df.columns:
        print(f"Filtering for status='success'. Original count: {len(df)}")
        df = df[df["status"] == "success"].reset_index(drop=True)
        print(f"New count: {len(df)}")
        
    if args.max_rows:
        df = df.head(args.max_rows)
        print(f"Capped to {len(df)} rows.")

    device = args.device
    print(f"Using device: {device}")

    # 1. ROUGE
    if not args.dry_run:
        df = compute_rouge_column(df)
    else:
        print("[Dry Run] Skipping ROUGE")

    # 2. Similarity
    if not args.dry_run:
        df = compute_similarity_column(df, DEFAULT_CONFIG["similarity_model"], device)
    else:
        print("[Dry Run] Skipping Similarity")

    # 3. Entailment
    if not args.dry_run:
        # Pipeline expects int device id (-1 for cpu)
        dev_id = 0 if device == "cuda" else -1
        df = compute_entailment_column(df, DEFAULT_CONFIG["entailment_model"], dev_id)
    else:
        print("[Dry Run] Skipping Entailment")

    # 4. Perplexity
    if not args.dry_run:
        df = compute_perplexity_column(df, DEFAULT_CONFIG["perplexity_model"], device)
    else:
        print("[Dry Run] Skipping Perplexity")

    # 5. LLM Judge
    if not args.skip_llm and not args.dry_run:
        df = compute_llm_judge(
            df, 
            DEFAULT_CONFIG["llm_model"], 
            DEFAULT_CONFIG["ollama_base_url"],
            DEFAULT_CONFIG["llm_concurrency"]
        )
    else:
        print("Skipping LLM Judge.")

    # Save Detailed Results
    df.to_csv(output_file, index=False)
    print(f"Saved detailed results to {output_file}")

    # Calculate and Save Summary
    summary = {
        "rows_processed": len(df),
        "timestamp": datetime.now().isoformat(),
        "metrics": {}
    }
    
    numeric_cols = [
        "rouge1", "rouge2", "rougeL", 
        "perplexity", 
        "sim_max", "sim_mean", 
        "entailment_ratio",
        "llm_correctness", "llm_completeness", "llm_overall"
    ]
    
    for col in numeric_cols:
        if col in df.columns:
            summary["metrics"][col] = {
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
                "max": float(df[col].max()),
                "min": float(df[col].min())
            }

    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_file}")
    
    print("\nEvaluation Summary:")
    print(json.dumps(summary["metrics"], indent=2))

if __name__ == "__main__":
    main()
