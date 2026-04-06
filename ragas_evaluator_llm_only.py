"""
LLM-only evaluator for Answer Accuracy using vLLM.
"""

import argparse
import ast
import json
import math
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from tqdm import tqdm

from ragas.metrics.collections.answer_accuracy.util import (
    AnswerAccuracyInput,
    AnswerAccuracyJudge1Prompt,
    AnswerAccuracyJudge2Prompt,
)

DEFAULT_CONFIG = {
    "llm_model": "meta-llama/Llama-2-13b-chat-hf",
    "workers": 4,
    "sample_size": 500,
    "sample_seed": 42,
    "gpu_memory_utilization": 0.60,
    "tensor_parallel_size": 1,
    "max_model_len": 4096,
    "offline_batch_size": 64,
    "offline_temperature": 0.0,
    "offline_max_tokens": 64,
    "offline_max_num_seqs": 0,
}

FORCED_ATTENTION_BACKEND = "TRITON_ATTN"
INVALID_OUTPUT_RETRIES = 2
_RATING_FIELD_RE = re.compile(r'"rating"\s*:\s*([024])')


def parse_reference_answers(raw_value: Any) -> List[str]:
    """Convert raw CSV cell into a list of reference strings."""
    if isinstance(raw_value, float) and math.isnan(raw_value):
        return []

    if isinstance(raw_value, list):
        return [str(item).strip() for item in raw_value if str(item).strip()]

    text = str(raw_value).strip()
    if not text:
        return []

    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
            if isinstance(parsed, str):
                return [parsed.strip()]
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except Exception:
            continue

    if "|||" in text:
        return [s.strip() for s in text.split("|||") if s.strip()]

    return [text]


def pick_column(df: pd.DataFrame, candidates: List[str]) -> str | None:
    """Return the first matching column name from a candidate list."""
    for name in candidates:
        if name in df.columns:
            return name
    return None


def average_two_scores(score_1: float, score_2: float) -> float:
    """Average two scores while handling NaN values.

    If both scores are invalid, return 0.0 to keep aggregation stable.
    """
    if (not math.isnan(score_1)) and (not math.isnan(score_2)):
        return (score_1 + score_2) / 2.0
    if not math.isnan(score_1):
        return score_1
    if not math.isnan(score_2):
        return score_2
    return 0.0


def parse_rating_from_text(raw_text: str) -> float:
    """Extract rating (0/2/4) from model text and return as float."""
    text = (raw_text or "").strip()
    if not text:
        return float("nan")

    text = text.replace("```json", "").replace("```", "").strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and "rating" in parsed:
            rating = int(parsed["rating"])
            return float(rating) if rating in (0, 2, 4) else float("nan")
    except Exception:
        pass

    match = _RATING_FIELD_RE.search(text)
    if match:
        return float(match.group(1))

    standalone = re.search(r"\b([024])\b", text)
    if standalone:
        return float(standalone.group(1))

    return float("nan")


def build_offline_vllm_engine(args):
    """Create local vLLM engine + sampling params for offline scoring."""
    # Set backend env vars before importing vLLM so backend selection happens safely.
    os.environ["VLLM_ATTENTION_BACKEND"] = FORCED_ATTENTION_BACKEND
    os.environ["VLLM_DISABLE_FLASHINFER"] = "1"

    try:
        from vllm import LLM, SamplingParams
    except Exception as exc:
        raise RuntimeError(
            "Offline mode requires vLLM in the active Python environment."
        ) from exc

    print(f"VLLM_ATTENTION_BACKEND={FORCED_ATTENTION_BACKEND}")
    print("VLLM_DISABLE_FLASHINFER=1")

    max_num_seqs = args.offline_max_num_seqs
    if max_num_seqs <= 0:
        max_num_seqs = max(1, args.workers * max(1, args.offline_batch_size))

    print("Initializing offline vLLM engine...")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=max_num_seqs,
    )
    sampling_params = SamplingParams(
        temperature=args.offline_temperature,
        max_tokens=args.offline_max_tokens,
    )

    print(
        "Offline vLLM ready "
        f"(temp={args.offline_temperature}, max_tokens={args.offline_max_tokens}, "
        f"max_num_seqs={max_num_seqs})"
    )
    return llm, sampling_params


def compute_answer_accuracy_offline(
    df: pd.DataFrame,
    offline_llm,
    sampling_params,
    question_col: str,
    answer_col: str,
    reference_col: str,
    batch_size: int,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Compute Answer Accuracy using local offline vLLM only."""
    judge1 = AnswerAccuracyJudge1Prompt()
    judge2 = AnswerAccuracyJudge2Prompt()

    rows = list(df.to_dict(orient="records"))
    scores = [float("nan")] * len(rows)
    methods = ["failed"] * len(rows)
    references_by_row: Dict[int, List[str]] = {}

    counters = {
        "rows_total": len(rows),
        "missing_inputs": 0,
        "ragas_success": 0,
        "ragas_nan": 0,
        "exceptions": 0,
        "parse_failures": 0,
        "retry_attempts": 0,
    }

    prompt_records: List[Tuple[int, int, int, str]] = []

    for index, row in enumerate(rows):
        generated = str(row.get(answer_col, "")).strip()
        references = parse_reference_answers(row.get(reference_col, ""))
        question = str(row.get(question_col, "")).strip()

        if (not generated) or (not references) or (not question):
            methods[index] = "missing_input"
            counters["missing_inputs"] += 1
            continue

        references_by_row[index] = references

        for ref_index, ref in enumerate(references):
            prompt_1 = judge1.to_string(
                AnswerAccuracyInput(
                    query=question,
                    user_answer=generated,
                    reference_answer=ref,
                )
            )
            prompt_records.append((index, ref_index, 1, prompt_1))

            prompt_2 = judge2.to_string(
                AnswerAccuracyInput(
                    query=question,
                    user_answer=ref,
                    reference_answer=generated,
                )
            )
            prompt_records.append((index, ref_index, 2, prompt_2))

    rating_scores: Dict[Tuple[int, int, int], float] = {}
    safe_batch_size = max(1, batch_size)

    print("Running Answer Accuracy (offline vLLM)...")
    for start in tqdm(
        range(0, len(prompt_records), safe_batch_size),
        desc="AnswerAccuracyOffline",
    ):
        batch = prompt_records[start : start + safe_batch_size]
        pending = batch[:]
        max_attempts = 1 + INVALID_OUTPUT_RETRIES

        for attempt in range(max_attempts):
            prompts = [entry[3] for entry in pending]

            try:
                outputs = offline_llm.generate(prompts, sampling_params=sampling_params)
            except Exception as err:
                if attempt < max_attempts - 1:
                    counters["retry_attempts"] += len(pending)
                    continue

                counters["exceptions"] += len(pending)
                print(f"  Offline batch error: {err}")
                for row_idx, ref_idx, judge_id, _ in pending:
                    rating_scores[(row_idx, ref_idx, judge_id)] = float("nan")
                pending = []
                break

            still_invalid: List[Tuple[int, int, int, str]] = []
            for out_idx, entry in enumerate(pending):
                row_idx, ref_idx, judge_id, _ = entry

                raw_text = ""
                try:
                    if out_idx < len(outputs) and outputs[out_idx].outputs:
                        raw_text = str(outputs[out_idx].outputs[0].text).strip()
                except Exception:
                    pass

                rating = parse_rating_from_text(raw_text)
                if math.isnan(rating):
                    still_invalid.append(entry)
                else:
                    rating_scores[(row_idx, ref_idx, judge_id)] = rating / 4.0

            if not still_invalid:
                pending = []
                break

            if attempt < max_attempts - 1:
                counters["retry_attempts"] += len(still_invalid)
                pending = still_invalid
                continue

            counters["ragas_nan"] += len(still_invalid)
            counters["parse_failures"] += len(still_invalid)
            for row_idx, ref_idx, judge_id, _ in still_invalid:
                rating_scores[(row_idx, ref_idx, judge_id)] = float("nan")
            pending = []

    for row_idx, references in references_by_row.items():
        row_ref_scores: List[float] = []

        for ref_idx, _ in enumerate(references):
            judge_1 = rating_scores.get((row_idx, ref_idx, 1), float("nan"))
            judge_2 = rating_scores.get((row_idx, ref_idx, 2), float("nan"))
            ref_score = average_two_scores(judge_1, judge_2)
            row_ref_scores.append(ref_score)

        if row_ref_scores:
            scores[row_idx] = max(row_ref_scores)
        else:
            scores[row_idx] = 0.0

        methods[row_idx] = "offline_vllm"
        counters["ragas_success"] += 1

    df["answer_accuracy"] = scores
    df["answer_accuracy_method"] = methods
    return df, counters


def generate_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute mean/std/max/min for Answer Accuracy."""
    summary: Dict[str, Any] = {}

    if "answer_accuracy" in df.columns:
        vals = df["answer_accuracy"].dropna()
        if len(vals) > 0:
            summary["answer_accuracy"] = {
                "mean": float(vals.mean()),
                "std": float(vals.std()),
                "max": float(vals.max()),
                "min": float(vals.min()),
                "count": int(len(vals)),
            }

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate generated answers with offline local vLLM only."
    )

    parser.add_argument("--input", type=str, required=True, help="Path to input CSV")
    parser.add_argument("--output", type=str, default="outputs_final/", help="Output directory")
    parser.add_argument("--max-rows", type=int, default=None, help="Limit number of rows")
    parser.add_argument("--dry-run", action="store_true", help="Load data, skip metrics")

    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_CONFIG["llm_model"],
        help="Local model path/name for vLLM",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_CONFIG["workers"],
        help="Used for auto-sizing max_num_seqs",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_CONFIG["sample_size"],
        help="Number of random questions to evaluate per file",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=DEFAULT_CONFIG["sample_seed"],
        help="Random seed for reproducible sampling",
    )
    parser.add_argument(
        "--offline-batch-size",
        type=int,
        default=DEFAULT_CONFIG["offline_batch_size"],
        help="Prompt batch size for local vLLM",
    )
    parser.add_argument(
        "--offline-temperature",
        type=float,
        default=DEFAULT_CONFIG["offline_temperature"],
        help="Generation temperature for judge prompts",
    )
    parser.add_argument(
        "--offline-max-tokens",
        type=int,
        default=DEFAULT_CONFIG["offline_max_tokens"],
        help="Max tokens per judge response",
    )
    parser.add_argument(
        "--offline-max-num-seqs",
        type=int,
        default=DEFAULT_CONFIG["offline_max_num_seqs"],
        help="max_num_seqs for local vLLM (<=0 auto)",
    )

    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=DEFAULT_CONFIG["gpu_memory_utilization"],
        help="vLLM GPU memory utilization",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=DEFAULT_CONFIG["tensor_parallel_size"],
        help="vLLM tensor parallel size",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=DEFAULT_CONFIG["max_model_len"],
        help="vLLM max model length",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}")
        return

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from: {input_path}")
    df = pd.read_csv(input_path)
    print(f"  Total rows: {len(df)}")

    question_col = pick_column(df, ["question", "user_input", "query", "prompt"])
    answer_col = pick_column(df, ["generated_answer", "response", "answer", "model_answer", "output"])
    reference_col = pick_column(
        df,
        [
            "reference_answers",
            "reference_answer",
            "reference",
            "ground_truth",
            "gold_answer",
            "expected_answer",
        ],
    )

    if not question_col or not answer_col or not reference_col:
        print("Error: could not resolve required columns.")
        print(f"  Found columns: {list(df.columns)}")
        print("  Need question/user_input, generated_answer/response, and reference_answers/reference.")
        return

    print("  Column mapping:")
    print(f"    question  -> {question_col}")
    print(f"    response  -> {answer_col}")
    print(f"    reference -> {reference_col}")

    if args.max_rows:
        df = df.head(args.max_rows)
        print(f"  Limited to: {len(df)} rows")

    if "status" in df.columns:
        before = len(df)
        df = df[df["status"] == "success"].reset_index(drop=True)
        print(f"  Filtered to successful: {len(df)} (dropped {before - len(df)})")

    if args.sample_size <= 0:
        print("  Sampling disabled (--sample-size <= 0), using all rows.")
    elif len(df) > args.sample_size:
        df = df.sample(n=args.sample_size, random_state=args.sample_seed).reset_index(drop=True)
        print(f"  Random sample selected: {len(df)} rows (seed={args.sample_seed})")
    else:
        print(f"  Rows available ({len(df)}) <= sample size ({args.sample_size}), using all rows.")

    if args.dry_run:
        print("\n[DRY RUN] Skipping metric computation.")
        stem = input_path.stem
        out_csv = output_dir / f"{stem}_ragas_llm.csv"
        df.to_csv(out_csv, index=False)
        print(f"  Saved (no metrics): {out_csv}")
        return

    llm, sampling_params = build_offline_vllm_engine(args)

    t0 = time.time()
    print("\n" + "=" * 60)
    print("Phase 1: Answer Accuracy (offline custom dual-judge)")
    print("=" * 60)

    df, aa_counters = compute_answer_accuracy_offline(
        df,
        offline_llm=llm,
        sampling_params=sampling_params,
        question_col=question_col,
        answer_col=answer_col,
        reference_col=reference_col,
        batch_size=args.offline_batch_size,
    )

    elapsed = time.time() - t0
    print(f"\nAll LLM metrics computed in {elapsed:.1f}s")
    print("Answer Accuracy diagnostics:")
    print(f"  rows_total: {aa_counters['rows_total']}")
    print(f"  ragas_success: {aa_counters['ragas_success']}")
    print(f"  ragas_nan: {aa_counters['ragas_nan']}")
    print(f"  exceptions: {aa_counters['exceptions']}")
    print(f"  missing_inputs: {aa_counters['missing_inputs']}")
    print(f"  parse_failures: {aa_counters['parse_failures']}")
    print(f"  retry_attempts: {aa_counters['retry_attempts']}")

    stem = input_path.stem
    out_csv = output_dir / f"{stem}_ragas_llm.csv"
    df.to_csv(out_csv, index=False)
    print(f"Results saved: {out_csv}")

    summary = generate_summary(df)
    summary["_meta"] = {
        "input": str(input_path),
        "rows_evaluated": len(df),
        "elapsed_seconds": round(elapsed, 2),
        "timestamp": datetime.now().isoformat(),
        "config": {
            "inference_mode": "offline_only",
            "llm_model": args.model,
            "workers": args.workers,
            "sample_size": args.sample_size,
            "sample_seed": args.sample_seed,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "tensor_parallel_size": args.tensor_parallel_size,
            "max_model_len": args.max_model_len,
            "forced_attention_backend": FORCED_ATTENTION_BACKEND,
            "invalid_output_retries": INVALID_OUTPUT_RETRIES,
            "offline_batch_size": args.offline_batch_size,
            "offline_temperature": args.offline_temperature,
            "offline_max_tokens": args.offline_max_tokens,
            "offline_max_num_seqs": args.offline_max_num_seqs,
        },
        "answer_accuracy_diagnostics": aa_counters,
    }

    summary_path = output_dir / f"{stem}_ragas_llm_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
