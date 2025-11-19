"""Compute perplexity and ROUGE scores for generated answers.

Usage:
    python compute_metrics.py \
        --input generated_answers/answers_0_1000.csv \
        --per-question-output evaluation_results/metrics_per_question.csv \
        --summary-output evaluation_results/metrics_summary.json

This script expects the CSV to contain at least the following columns:
    - question_id
    - generated_answer
    - reference_answers (single string with entries separated by '|||' or JSON list)
"""

from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import torch
from rouge_score import rouge_scorer
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_reference_answers(raw_value: str | float) -> List[str]:
    """Convert the raw CSV cell into a list of reference answers."""
    if isinstance(raw_value, float) and math.isnan(raw_value):
        return []

    if isinstance(raw_value, list):
        return [str(item).strip() for item in raw_value if str(item).strip()]

    text = str(raw_value).strip()
    if not text:
        return []

    # Try JSON or Python literal parsing first
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
        except Exception:
            continue
        if isinstance(parsed, str):
            return [parsed.strip()] if parsed.strip() else []
        if isinstance(parsed, Sequence):
            return [str(item).strip() for item in parsed if str(item).strip()]

    # Fallback to custom separator or single string
    if "|||" in text:
        return [segment.strip() for segment in text.split("|||") if segment.strip()]

    return [text]


def compute_perplexity(
    text: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    device: torch.device,
    stride: int = 256,
) -> float:
    """Compute autoregressive perplexity for a single string."""
    text = text.strip()
    if not text:
        return float("nan")

    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)
    seq_len = input_ids.size(1)

    max_length = model.config.max_position_embeddings
    nlls: List[torch.Tensor] = []
    prev_end = 0

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

    ppl = torch.exp(torch.stack(nlls).sum() / seq_len)
    return float(ppl.cpu().item())


def compute_rouge(
    prediction: str,
    references: Sequence[str],
    scorer: rouge_scorer.RougeScorer,
) -> Dict[str, float]:
    """Compute ROUGE scores, taking the best reference per metric."""
    metrics = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    if not references:
        return metrics

    prediction = prediction.strip()
    if not prediction:
        return metrics

    for reference in references:
        score = scorer.score(reference, prediction)
        for metric in metrics:
            metrics[metric] = max(metrics[metric], score[metric].fmeasure)

    return metrics


def summarize(values: Iterable[float]) -> Dict[str, float]:
    array = np.array([v for v in values if not np.isnan(v)], dtype=float)
    if array.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "std": float("nan")}
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "std": float(np.std(array)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute perplexity and ROUGE metrics.")
    parser.add_argument(
        "--input",
        default="generated_answers/answers_0_1000.csv",
        help="Path to the CSV containing generated answers",
    )
    parser.add_argument(
        "--per-question-output",
        default="evaluation_results/metrics_per_question.csv",
        help="Where to save the per-question metrics CSV",
    )
    parser.add_argument(
        "--summary-output",
        default="evaluation_results/metrics_summary.json",
        help="Where to save the aggregate metrics JSON",
    )
    parser.add_argument(
        "--perplexity-model",
        default="gpt2",
        help="Hugging Face model name to use for perplexity computation",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional limit on the number of rows to score (useful for smoke tests)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    per_question_output = Path(args.per_question_output)
    per_question_output.parent.mkdir(parents=True, exist_ok=True)

    summary_output = Path(args.summary_output)
    summary_output.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    if args.max_rows is not None:
        df = df.head(args.max_rows)

    print(f"Loaded {len(df)} rows from {input_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.perplexity_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.perplexity_model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    metrics_rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Scoring"):
        generated = str(row.get("generated_answer", ""))
        references = parse_reference_answers(row.get("reference_answers", ""))
        ppl = compute_perplexity(generated, tokenizer, model, device)
        rouge_scores = compute_rouge(generated, references, rouge)

        metrics_rows.append(
            {
                "question_id": row.get("question_id"),
                "perplexity": ppl,
                **rouge_scores,
            }
        )

    metrics_df = pd.DataFrame(metrics_rows)
    merged = df.merge(metrics_df, on="question_id", how="left")
    merged.to_csv(per_question_output, index=False)
    print(f"Per-question metrics saved to {per_question_output}")

    summary = {
        "perplexity": summarize(metrics_df["perplexity"].tolist()),
        "rouge1": summarize(metrics_df["rouge1"].tolist()),
        "rouge2": summarize(metrics_df["rouge2"].tolist()),
        "rougeL": summarize(metrics_df["rougeL"].tolist()),
        "rows_scored": len(metrics_df),
        "perplexity_model": args.perplexity_model,
    }

    with open(summary_output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Summary metrics saved to {summary_output}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
