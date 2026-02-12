"""
RAGAS + Evaluation Metrics Script
==================================

Metrics:
  1. Factual Correctness   (RAGAS - LLM-based)
  2. BLEU Score             (RAGAS - traditional NLP)
  3. CHRF Score             (RAGAS - traditional NLP)
  4. ROUGE (1, 2, L)        (RAGAS - traditional NLP)
  5. Answer Accuracy        (RAGAS / NVIDIA - LLM-based)
  6. BERTScore (P/R/F1)     (HuggingFace Transformers - DeBERTa-based)
  7. Semantic Similarity    (SentenceTransformer)
  8. Perplexity             (GPT-2)

Usage:
    # Full CSV evaluation
    python ragas_evaluator.py --input generated_answers/baseline_llama3b_0_400.csv --output eval_ragas/

    # Limit rows
    python ragas_evaluator.py --input generated_answers/baseline_llama3b_0_400.csv --output eval_ragas/ --max-rows 50

    # Quick test on a single pair
    python ragas_evaluator.py --test --response "The Eiffel Tower is in Paris." --reference "The Eiffel Tower is located in Paris, France."
"""

import argparse
import ast
import asyncio
import gc
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# RAGAS imports
from ragas.metrics.collections import BleuScore, RougeScore
from ragas.metrics.collections import FactualCorrectness
from ragas.metrics.collections import SemanticSimilarity as RagasSemanticSimilarity
from ragas.metrics.collections import AnswerAccuracy

try:
    from ragas.metrics.collections import CHRFScore
except ImportError:
    from ragas.metrics.collections import ExactMatch as CHRFScore  # fallback
    print("Warning: CHRFScore not found, using fallback.")

from openai import AsyncOpenAI
from ragas.llms import llm_factory

# Similarity & Perplexity (same as evaluation.py)
from sentence_transformers import SentenceTransformer, util
from bert_score import score as bert_score
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_CONFIG = {
    "llm_model": "llama2:13b",
    "ollama_base_url": "http://localhost:11434/v1",
    "llm_adapter": "instructor",
    "similarity_model": "all-MiniLM-L6-v2",
    "bertscore_model": "microsoft/deberta-xlarge-mnli",
    "perplexity_model": "gpt2",
}

# ============================================================================
# UTILITIES
# ============================================================================


def parse_reference_answers(raw_value: Any) -> List[str]:
    """Convert raw CSV cell into a list of reference strings."""
    if isinstance(raw_value, float) and math.isnan(raw_value):
        return []

    if isinstance(raw_value, list):
        return [str(item).strip() for item in raw_value if str(item).strip()]

    text = str(raw_value).strip()
    if not text:
        return []

    # Try JSON / Python literal
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


def get_device() -> str:
    """Return best available device."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_llm(model_name: str, base_url: str, adapter: str):
    """Build a RAGAS LLM wrapper via OpenAI client → Ollama."""
    if adapter == "litellm":
        print(
            "Warning: 'litellm' adapter requires a LiteLLM client. "
            "For Ollama with OpenAI-compatible API, use 'instructor'."
        )
        adapter = "instructor"
    client = AsyncOpenAI(api_key="ollama", base_url=base_url)
    return llm_factory(
        model_name,
        client=client,
        provider="openai",
        adapter=adapter,
    )


def run_async(coro):
    """Run a coroutine in a fresh event loop (safe for scripts)."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        new_loop = asyncio.new_event_loop()
        try:
            return new_loop.run_until_complete(coro)
        finally:
            new_loop.close()
    return asyncio.run(coro)


# ============================================================================
# RAGAS METRIC FUNCTIONS
# ============================================================================


def compute_factual_correctness(df: pd.DataFrame, llm) -> pd.DataFrame:
    """Compute Factual Correctness (LLM-based, RAGAS)."""
    scorer = FactualCorrectness(llm=llm, atomicity="high", coverage="high")
    scores = []

    print("Running Factual Correctness…")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="FactualCorrectness"):
        generated = str(row.get("generated_answer", ""))
        references = parse_reference_answers(row.get("reference_answers", ""))

        if not generated or not references:
            scores.append(float("nan"))
            continue

        # Score against each reference, take max
        ref_scores = []
        for ref in references:
            try:
                s = run_async(scorer.ascore(response=generated, reference=ref))
                ref_scores.append(s)
            except Exception as e:
                print(f"  FactualCorrectness error: {e}")
                ref_scores.append(float("nan"))
        scores.append(max(ref_scores) if ref_scores else float("nan"))

    df["factual_correctness"] = scores
    return df


def compute_bleu(df: pd.DataFrame) -> pd.DataFrame:
    """Compute BLEU score (RAGAS)."""
    scorer = BleuScore()
    scores = []

    print("Running BLEU…")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="BLEU"):
        generated = str(row.get("generated_answer", ""))
        references = parse_reference_answers(row.get("reference_answers", ""))

        if not generated or not references:
            scores.append(float("nan"))
            continue

        ref_scores = []
        for ref in references:
            try:
                s = scorer.score(reference=ref, response=generated)
                ref_scores.append(s)
            except Exception as e:
                print(f"  BLEU error: {e}")
                ref_scores.append(float("nan"))
        scores.append(max(ref_scores) if ref_scores else float("nan"))

    df["bleu_score"] = scores
    return df


def compute_chrf(df: pd.DataFrame) -> pd.DataFrame:
    """Compute CHRF score (RAGAS)."""
    scorer = CHRFScore()
    scores = []

    print("Running CHRF…")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="CHRF"):
        generated = str(row.get("generated_answer", ""))
        references = parse_reference_answers(row.get("reference_answers", ""))

        if not generated or not references:
            scores.append(float("nan"))
            continue

        ref_scores = []
        for ref in references:
            try:
                s = scorer.score(reference=ref, response=generated)
                ref_scores.append(s)
            except Exception as e:
                print(f"  CHRF error: {e}")
                ref_scores.append(float("nan"))
        scores.append(max(ref_scores) if ref_scores else float("nan"))

    df["chrf_score"] = scores
    return df


def compute_rouge(df: pd.DataFrame) -> pd.DataFrame:
    """Compute ROUGE-1, ROUGE-2, ROUGE-L (RAGAS)."""
    rouge_types = ["rouge1", "rouge2", "rougeL"]
    scorers = {rt: RougeScore(rouge_type=rt) for rt in rouge_types}

    results = {rt: [] for rt in rouge_types}

    print("Running ROUGE…")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="ROUGE"):
        generated = str(row.get("generated_answer", ""))
        references = parse_reference_answers(row.get("reference_answers", ""))

        if not generated or not references:
            for rt in rouge_types:
                results[rt].append(float("nan"))
            continue

        for rt in rouge_types:
            ref_scores = []
            for ref in references:
                try:
                    s = scorers[rt].score(reference=ref, response=generated)
                    ref_scores.append(s)
                except Exception as e:
                    print(f"  ROUGE ({rt}) error: {e}")
                    ref_scores.append(float("nan"))
            results[rt].append(max(ref_scores) if ref_scores else float("nan"))

    for rt in rouge_types:
        df[rt] = results[rt]
    return df


def compute_answer_accuracy(df: pd.DataFrame, llm) -> pd.DataFrame:
    """Compute Answer Accuracy (NVIDIA / LLM-based, RAGAS)."""
    scorer = AnswerAccuracy(llm=llm)
    scores = []

    print("Running Answer Accuracy…")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="AnswerAccuracy"):
        generated = str(row.get("generated_answer", ""))
        references = parse_reference_answers(row.get("reference_answers", ""))
        question = str(row.get("question", ""))

        if not generated or not references:
            scores.append(float("nan"))
            continue

        ref_scores = []
        for ref in references:
            try:
                s = run_async(
                    scorer.ascore(user_input=question, response=generated, reference=ref)
                )
                ref_scores.append(s)
            except Exception as e:
                print(f"  AnswerAccuracy error: {e}")
                ref_scores.append(float("nan"))
        scores.append(max(ref_scores) if ref_scores else float("nan"))

    df["answer_accuracy"] = scores
    return df


def compute_similarity(df: pd.DataFrame, model_name: str, device: str) -> pd.DataFrame:
    """Compute Semantic Similarity (SentenceTransformer)."""
    print(f"Loading Similarity model: {model_name}…")
    model = SentenceTransformer(model_name)
    model.to(device)

    sim_results = []

    print("Running Similarity evaluation…")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Similarity"):
        generated = str(row.get("generated_answer", ""))
        references = parse_reference_answers(row.get("reference_answers", ""))

        if not generated or not references:
            sim_results.append({"sim_max": 0.0, "sim_mean": 0.0, "sim_min": 0.0})
            continue

        gen_emb = model.encode(generated, convert_to_tensor=True, device=device)
        ref_embs = model.encode(references, convert_to_tensor=True, device=device)

        similarities = util.cos_sim(gen_emb, ref_embs)[0].cpu().tolist()

        sim_results.append({
            "sim_max": max(similarities),
            "sim_mean": float(np.mean(similarities)),
            "sim_min": min(similarities),
        })

    del model
    clean_gpu_memory()
    print("Similarity model unloaded.")

    result_df = pd.DataFrame(sim_results)
    return pd.concat([df, result_df], axis=1)


def compute_bert_score(df: pd.DataFrame, model_name: str, device: str) -> pd.DataFrame:
    """Compute BERTScore (Precision/Recall/F1)."""
    print(f"Running BERTScore with model: {model_name}…")
    bert_results = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="BERTScore"):
        generated = str(row.get("generated_answer", ""))
        references = parse_reference_answers(row.get("reference_answers", ""))

        if not generated or not references:
            bert_results.append({"bert_p": float("nan"), "bert_r": float("nan"), "bert_f1": float("nan")})
            continue

        # Score against each reference, take max F1 (and its corresponding P/R)
        best = {"bert_p": float("nan"), "bert_r": float("nan"), "bert_f1": float("nan")}
        for ref in references:
            try:
                P, R, F1 = bert_score(
                    [generated],
                    [ref],
                    model_type=model_name,
                    lang="en",
                    device=device,
                )
                p = float(P[0].cpu().item())
                r = float(R[0].cpu().item())
                f1 = float(F1[0].cpu().item())
                if math.isnan(best["bert_f1"]) or f1 > best["bert_f1"]:
                    best = {"bert_p": p, "bert_r": r, "bert_f1": f1}
            except Exception as e:
                print(f"  BERTScore error: {e}")

        bert_results.append(best)

    result_df = pd.DataFrame(bert_results)
    return pd.concat([df, result_df], axis=1)


def compute_perplexity(df: pd.DataFrame, model_name: str, device: str) -> pd.DataFrame:
    """Compute Perplexity (GPT-2)."""
    print(f"Loading PPL model: {model_name}…")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()

    ppl_scores = []
    max_length = model.config.max_position_embeddings

    def _calc_ppl(text: str) -> float:
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
            target_ids[:, :-trg_len] = -100

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

    print("Running Perplexity evaluation…")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="PPL"):
        ppl_scores.append(_calc_ppl(row.get("generated_answer", "")))

    del model
    del tokenizer
    clean_gpu_memory()
    print("PPL model unloaded.")

    df["perplexity"] = ppl_scores
    return df


# ============================================================================
# SUMMARY
# ============================================================================


def generate_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute mean/std/max/min for all numeric metric columns."""
    metric_cols = [
        "factual_correctness", "bleu_score", "chrf_score",
        "rouge1", "rouge2", "rougeL",
        "answer_accuracy",
        "bert_p", "bert_r", "bert_f1",
        "sim_max", "sim_mean", "sim_min",
        "perplexity",
    ]
    summary = {}
    for col in metric_cols:
        if col in df.columns:
            vals = df[col].dropna()
            if len(vals) > 0:
                summary[col] = {
                    "mean": float(vals.mean()),
                    "std": float(vals.std()),
                    "max": float(vals.max()),
                    "min": float(vals.min()),
                    "count": int(len(vals)),
                }
    return summary


# ============================================================================
# TEST MODE
# ============================================================================


def run_test(response: str, reference: str, question: str, config: dict):
    """Run all metrics on a single response/reference pair and print."""
    print("=" * 60)
    print("TEST MODE")
    print("=" * 60)
    print(f"Response:  {response[:100]}{'…' if len(response) > 100 else ''}")
    print(f"Reference: {reference[:100]}{'…' if len(reference) > 100 else ''}")
    if question:
        print(f"Question:  {question[:100]}{'…' if len(question) > 100 else ''}")
    print("-" * 60)

    device = get_device()
    llm = build_llm(
        config["llm_model"],
        config["ollama_base_url"],
        config["llm_adapter"],
    )

    # Create a 1-row DF
    data = {
        "question": [question],
        "generated_answer": [response],
        "reference_answers": [reference],
    }
    df = pd.DataFrame(data)

    # RAGAS non-LLM metrics
    df = compute_bleu(df)
    df = compute_chrf(df)
    df = compute_rouge(df)

    # RAGAS LLM metrics
    df = compute_factual_correctness(df, llm)
    df = compute_answer_accuracy(df, llm)

    # Similarity & Perplexity
    df = compute_similarity(df, config["similarity_model"], device)
    df = compute_bert_score(df, config["bertscore_model"], device)
    df = compute_perplexity(df, config["perplexity_model"], device)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    metric_cols = [
        "bleu_score", "chrf_score", "rouge1", "rouge2", "rougeL",
        "factual_correctness", "answer_accuracy",
        "bert_p", "bert_r", "bert_f1",
        "sim_max", "sim_mean", "sim_min", "perplexity",
    ]
    for col in metric_cols:
        if col in df.columns:
            val = df[col].iloc[0]
            print(f"  {col:25s} = {val}")
    print("=" * 60)


# ============================================================================
# MAIN
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate generated answers with RAGAS + Sim + PPL metrics."
    )

    # CSV mode
    parser.add_argument("--input", type=str, help="Path to input CSV (generated_answers/)")
    parser.add_argument("--output", type=str, default="eval_ragas/", help="Output directory")
    parser.add_argument("--max-rows", type=int, default=None, help="Limit number of rows")
    parser.add_argument("--dry-run", action="store_true", help="Load data, skip metrics")

    # Test mode
    parser.add_argument("--test", action="store_true", help="Single-pair test mode")
    parser.add_argument("--response", type=str, help="Generated answer (test mode)")
    parser.add_argument("--reference", type=str, help="Reference answer (test mode)")
    parser.add_argument("--question", type=str, default="", help="Question (test mode)")

    # Config overrides
    parser.add_argument("--model", type=str, default=DEFAULT_CONFIG["llm_model"],
                        help="LLM model name for Ollama")
    parser.add_argument("--ollama-url", type=str, default=DEFAULT_CONFIG["ollama_base_url"],
                        help="Ollama base URL")
    parser.add_argument("--adapter", type=str, default=DEFAULT_CONFIG["llm_adapter"],
                        help="RAGAS LLM adapter (e.g., litellm, instructor)")
    parser.add_argument("--sim-model", type=str, default=DEFAULT_CONFIG["similarity_model"],
                        help="SentenceTransformer model for similarity")
    parser.add_argument("--bertscore-model", type=str, default=DEFAULT_CONFIG["bertscore_model"],
                        help="Model for BERTScore")
    parser.add_argument("--ppl-model", type=str, default=DEFAULT_CONFIG["perplexity_model"],
                        help="Model for perplexity")

    args = parser.parse_args()

    config = {
        "llm_model": args.model,
        "ollama_base_url": args.ollama_url,
        "llm_adapter": args.adapter,
        "similarity_model": args.sim_model,
        "bertscore_model": args.bertscore_model,
        "perplexity_model": args.ppl_model,
    }

    # ── Test mode ──
    if args.test:
        if not args.response or not args.reference:
            parser.error("--test requires --response and --reference")
        run_test(args.response, args.reference, args.question, config)
        return

    # ── CSV mode ──
    if not args.input:
        parser.error("--input is required (or use --test)")

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}")
        return

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from: {input_path}")
    df = pd.read_csv(input_path)
    print(f"  Total rows: {len(df)}")

    if args.max_rows:
        df = df.head(args.max_rows)
        print(f"  Limited to: {len(df)} rows")

    # Filter only successful rows
    if "status" in df.columns:
        before = len(df)
        df = df[df["status"] == "success"].reset_index(drop=True)
        print(f"  Filtered to successful: {len(df)} (dropped {before - len(df)})")

    if args.dry_run:
        print("\n[DRY RUN] Skipping metric computation.")
        stem = input_path.stem
        out_csv = output_dir / f"{stem}_ragas.csv"
        df.to_csv(out_csv, index=False)
        print(f"  Saved (no metrics): {out_csv}")
        return

    device = get_device()
    print(f"Device: {device}")

    t0 = time.time()

    # ── 1. Non-LLM RAGAS metrics (fast) ──
    print("\n" + "=" * 60)
    print("Phase 1: Non-LLM RAGAS metrics (BLEU, CHRF, ROUGE)")
    print("=" * 60)
    df = compute_bleu(df)
    df = compute_chrf(df)
    df = compute_rouge(df)

    # ── 2. LLM-based RAGAS metrics ──
    print("\n" + "=" * 60)
    print("Phase 2: LLM-based RAGAS metrics (FactualCorrectness, AnswerAccuracy)")
    print("=" * 60)
    llm = build_llm(
        config["llm_model"],
        config["ollama_base_url"],
        config["llm_adapter"],
    )
    df = compute_factual_correctness(df, llm)
    df = compute_answer_accuracy(df, llm)

    # ── 3. Similarity (SentenceTransformer) ──
    print("\n" + "=" * 60)
    print("Phase 3: Semantic Similarity (SentenceTransformer)")
    print("=" * 60)
    df = compute_similarity(df, config["similarity_model"], device)

    # ── 4. BERTScore ──
    print("\n" + "=" * 60)
    print("Phase 4: BERTScore")
    print("=" * 60)
    df = compute_bert_score(df, config["bertscore_model"], device)

    # ── 5. Perplexity (GPT-2) ──
    print("\n" + "=" * 60)
    print("Phase 5: Perplexity (GPT-2)")
    print("=" * 60)
    df = compute_perplexity(df, config["perplexity_model"], device)

    elapsed = time.time() - t0
    print(f"\nAll metrics computed in {elapsed:.1f}s")

    # ── Save results ──
    stem = input_path.stem
    out_csv = output_dir / f"{stem}_ragas.csv"
    df.to_csv(out_csv, index=False)
    print(f"Results saved: {out_csv}")

    # ── Summary ──
    summary = generate_summary(df)
    summary["_meta"] = {
        "input": str(input_path),
        "rows_evaluated": len(df),
        "elapsed_seconds": round(elapsed, 2),
        "timestamp": datetime.now().isoformat(),
        "config": config,
    }

    summary_path = output_dir / f"{stem}_ragas_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved: {summary_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for k, v in summary.items():
        if k.startswith("_"):
            continue
        print(f"  {k:25s}  mean={v['mean']:.4f}  std={v['std']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()