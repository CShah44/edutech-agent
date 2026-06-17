"""
run_gptoss_judge_all.py
=======================
Runs GPT-OSS (NVIDIA NIM) LLM-as-judge evaluation across ALL 14 experiment
configs, sampling 50 rows each from outputs_llm_final/.

Features:
  - Auto-discovers all CSVs in outputs_llm_final/
  - Resumes: skips configs already evaluated in llm_metrics_gptoss/
  - Saves per-file CSV + JSON summary incrementally (never loses work)
  - Prints live progress & prints a comparison table at the end
  - Configurable via CLI or top-of-file constants

Usage:
    # Set API key first:
    export NVIDIA_API_KEY="nvapi-xxxxxxxxxxxx"
    # or add to .env:  NVIDIA_API_KEY=nvapi-xxxxxxxxxxxx

    python run_gptoss_judge_all.py

    # Dry-run (no API calls, just validate CSVs):
    python run_gptoss_judge_all.py --dry-run

    # Override sample size or model:
    python run_gptoss_judge_all.py --sample-size 50 --model nvidia/llama-3.3-70b-instruct

    # Re-evaluate even if output already exists:
    python run_gptoss_judge_all.py --force

Environment:
    NVIDIA_API_KEY  — required (set in .env or shell)
"""

# ============================================================================
# TOP-LEVEL CONFIGURATION (edit here or use CLI flags)
# ============================================================================

INPUT_DIR      = "outputs_llm_final"   # directory with ragas_llm CSVs
OUTPUT_DIR     = "llm_metrics_gptoss"  # output directory
# Judge model — meta/llama-3.3-70b-instruct is reliable on NVIDIA NIM
# Alternatives (if you want to try a different model):
#   openai/gpt-oss-120b                ← reasoning model, very slow & unreliable
#   nvidia/llama-3.1-nemotron-70b-instruct
#   nvidia/nemotron-3-super-120b-a12b
MODEL          = "meta/llama-3.3-70b-instruct"  # NVIDIA NIM — stable, fast, strong judge
BASE_URL       = "https://integrate.api.nvidia.com/v1"
SAMPLE_SIZE    = 50                    # rows per config
SAMPLE_SEED    = 42                    # for reproducibility
RATE_LIMIT_SEC = 1.5                   # 40 RPM limit → min 1.5s between calls
MAX_RETRIES    = 3
RETRY_DELAY    = 5                     # base seconds, doubles each retry
TEMPERATURE    = 0.1                   # low temperature for consistent scoring
MAX_TOKENS     = 1024                  # ample for JSON scores + reasoning
TOP_P          = 0.9
TIMEOUT_SEC    = 60

# ============================================================================
# IMPORTS
# ============================================================================

import argparse
import ast
import json
import math
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

try:
    from openai import OpenAI
    _have_openai = True
except ImportError:
    _have_openai = False

# ============================================================================
# JUDGE PROMPT  (ELI5-tuned, 4 dimensions)
# ============================================================================

JUDGE_PROMPT = """\
You are an expert evaluator assessing Explain-Like-I'm-Five (ELI5) answers.

Task: Evaluate the GENERATED ANSWER vs. the REFERENCE ANSWER(S).

=== Question ===
{question}

=== Reference Answer(s) ===
{references}

=== Generated Answer ===
{generated}

=== Evaluation Criteria (score 1-10 each) ===
1. correctness  — Factual accuracy; penalise hallucinations or wrong facts.
2. completeness — Coverage of key points in the reference.
3. eli5_quality — Simplicity of language; good analogies; clarity for a novice.
4. overall      — Holistic quality across all dimensions.

Return ONLY a JSON object in this exact format (no markdown, no extra text):
{{"correctness": <int>, "completeness": <int>, "eli5_quality": <int>, "overall": <int>, "reasoning": "<one-sentence rationale>"}}\
"""

# ============================================================================
# UTILITIES
# ============================================================================

def parse_reference_answers(raw: Any) -> List[str]:
    """Parse reference_answers cell into a list of strings."""
    if isinstance(raw, float) and math.isnan(raw):
        return []
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    text = str(raw).strip()
    if not text:
        return []
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
            if isinstance(parsed, str):
                return [parsed.strip()]
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            continue
    if "|||" in text:
        return [s.strip() for s in text.split("|||") if s.strip()]
    return [text]


def pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def clamp(val, lo=1, hi=10):
    try:
        return max(lo, min(hi, int(val)))
    except (TypeError, ValueError):
        return 0


# ============================================================================
# NVIDIA NIM CLIENT
# ============================================================================

class NIMJudge:
    """LLM-as-Judge backed by NVIDIA NIM (OpenAI-compatible endpoint)."""

    def __init__(self, model: str, base_url: str, api_key: str):
        if not _have_openai:
            raise ImportError("openai library required: pip install openai")
        if not api_key:
            raise ValueError(
                "NVIDIA_API_KEY is not set. "
                "Export it in your shell or add it to .env"
            )
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=TIMEOUT_SEC)

    # ------------------------------------------------------------------
    def _call(self, prompt: str) -> Dict[str, Any]:
        """Call the API with exponential-backoff retries."""
        for attempt in range(MAX_RETRIES):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an impartial evaluator. Return only JSON."},
                        {"role": "user",   "content": prompt},
                    ],
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                    top_p=TOP_P,
                    stream=False,
                )
                msg = resp.choices[0].message
                # gpt-oss-120b is a reasoning model — capture reasoning trace if present
                reasoning = getattr(msg, "reasoning_content", None)
                content = msg.content or ""

                # Empty response — reasoning models occasionally return nothing
                # (safety refusal or transient issue). Treat as retryable.
                if not content.strip():
                    raise ValueError(
                        "Empty response from model (no content). "
                        "Possible safety refusal or transient issue."
                    )

                result = self._parse(content)
                if reasoning:
                    result["_reasoning_trace"] = reasoning[:500]  # store snippet
                return result

            except Exception as exc:
                delay = RETRY_DELAY * (2 ** attempt)
                print(f"\n  ⚠  API error (attempt {attempt+1}/{MAX_RETRIES}): {exc}")
                if attempt < MAX_RETRIES - 1:
                    print(f"     Retrying in {delay}s …")
                    time.sleep(delay)
                else:
                    print(f"     Giving up. Logging as error row.")
                    return {
                        "correctness": 0, "completeness": 0,
                        "eli5_quality": 0, "overall": 0,
                        "reasoning": f"API Error: {str(exc)[:200]}",
                        "error": str(exc)[:200],
                    }

    # ------------------------------------------------------------------
    def _parse(self, raw: str) -> Dict[str, Any]:
        """Extract JSON from model output (handles markdown fences)."""
        text = raw.strip()
        # Strip ```json … ``` fences
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            parts = text.split("```")
            if len(parts) >= 3:
                text = parts[1].strip()

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = {}
            for key in ["correctness", "completeness", "eli5_quality", "overall"]:
                m = re.search(rf'"{key}"\s*:\s*(\d+)', text)
                parsed[key] = int(m.group(1)) if m else 0
            parsed["reasoning"] = "JSON parse failed"
            parsed["error"] = "json_parse_error"

        for key in ["correctness", "completeness", "eli5_quality", "overall"]:
            parsed.setdefault(key, 0)
            parsed[key] = clamp(parsed[key])
        parsed.setdefault("reasoning", "")
        return parsed

    # ------------------------------------------------------------------
    def judge(self, question: str, generated: str, references: List[str]) -> Dict[str, Any]:
        ref_text = "\n".join(f"- {r}" for r in references[:3])
        prompt = JUDGE_PROMPT.format(
            question=question[:1200],
            references=ref_text[:2500],
            generated=generated[:3500],
        )
        return self._call(prompt)


# ============================================================================
# PER-FILE EVALUATION
# ============================================================================

def evaluate_file(
    csv_path: Path,
    judge: NIMJudge,
    output_dir: Path,
    sample_size: int,
    seed: int,
    rate_limit: float,
    dry_run: bool,
) -> Optional[Dict]:
    """Evaluate one CSV. Returns summary dict or None on failure."""

    print(f"\n{'─'*60}")
    print(f"  📄  {csv_path.name}")
    print(f"{'─'*60}")

    # Load
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        print(f"  ✗  Failed to read CSV: {exc}")
        return None

    # Resolve columns
    q_col   = pick_col(df, ["question", "user_input", "query", "prompt"])
    gen_col = pick_col(df, ["generated_answer", "response", "answer", "model_answer"])
    ref_col = pick_col(df, ["reference_answers", "reference_answer", "reference", "ground_truth"])

    if not all([q_col, gen_col, ref_col]):
        print(f"  ✗  Missing required columns. Found: {df.columns.tolist()}")
        return None

    # Filter successful rows
    if "status" in df.columns:
        before = len(df)
        df = df[df["status"] == "success"].reset_index(drop=True)
        print(f"  Filtered to success: {len(df)}/{before} rows")

    if len(df) == 0:
        print("  ✗  No valid rows.")
        return None

    # Sample
    n = min(sample_size, len(df))
    sampled = df.sample(n=n, random_state=seed).reset_index(drop=True)
    print(f"  Sampled: {n} rows (seed={seed})")

    if dry_run:
        print("  [DRY RUN] Skipping API calls.")
        return None

    # Evaluate
    rows = []
    t0 = time.time()

    for i, row in enumerate(tqdm(sampled.itertuples(), total=n, desc="  Judging", ncols=70)):
        question   = str(getattr(row, q_col))
        generated  = str(getattr(row, gen_col))
        references = parse_reference_answers(getattr(row, ref_col))

        if not generated.strip() or not references:
            continue

        scores = judge.judge(question, generated, references)
        scores["question"]     = question[:120]
        scores["generated"]    = generated[:120]
        scores["question_id"]  = getattr(row, "question_id", i)
        rows.append(scores)

        if i < n - 1:
            time.sleep(rate_limit)

    elapsed = time.time() - t0
    print(f"  ✓  {len(rows)} evaluations in {elapsed:.1f}s")

    if not rows:
        return None

    # Summary stats
    metrics = ["correctness", "completeness", "eli5_quality", "overall"]
    summary: Dict[str, Any] = {"total_evaluated": len(rows), "errors": sum(1 for r in rows if "error" in r)}
    for m in metrics:
        vals = [r[m] for r in rows if m in r and r[m] > 0]
        if vals:
            summary[m] = {
                "mean":   round(float(np.mean(vals)),  3),
                "std":    round(float(np.std(vals)),   3),
                "median": round(float(np.median(vals)),3),
                "min":    int(np.min(vals)),
                "max":    int(np.max(vals)),
                "count":  len(vals),
            }
            print(f"    {m:15s}  mean={summary[m]['mean']:.2f}  std={summary[m]['std']:.2f}")

    summary["_meta"] = {
        "input_file": str(csv_path),
        "sample_size": len(rows),
        "sample_seed": seed,
        "model": judge.model,
        "base_url": BASE_URL,
        "elapsed_seconds": round(elapsed, 2),
        "timestamp": datetime.now().isoformat(),
    }

    # Save incrementally
    stem = csv_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    df_out = pd.DataFrame(rows)
    df_out.to_csv(output_dir / f"{stem}_gptoss_judge.csv", index=False)

    with open(output_dir / f"{stem}_gptoss_judge_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"  Saved → {output_dir}/{stem}_gptoss_judge_summary.json")
    return summary


# ============================================================================
# COMPARISON TABLE
# ============================================================================

METRIC_LABELS = ["correctness", "completeness", "eli5_quality", "overall"]


def print_comparison(summaries: Dict[str, dict]) -> None:
    if not summaries:
        return

    w = max(len(k) for k in summaries) + 2
    header = f"{'Config':<{w}}" + "".join(f"{m[:8]:>12}" for m in METRIC_LABELS)
    print("\n" + "=" * (w + 12 * len(METRIC_LABELS)))
    print("  GPT-OSS LLM-as-Judge — Final Comparison Table")
    print("=" * (w + 12 * len(METRIC_LABELS)))
    print(header)
    print("-" * (w + 12 * len(METRIC_LABELS)))

    for name, s in sorted(summaries.items()):
        row = f"{name:<{w}}"
        for m in METRIC_LABELS:
            if m in s:
                row += f"{s[m]['mean']:>9.2f}±{s[m]['std']:.2f}"
            else:
                row += f"{'N/A':>12}"
        print(row)

    print("=" * (w + 12 * len(METRIC_LABELS)))

    # Baseline vs. arch comparison
    print("\n  Δ Multi-Agent vs. Baseline (mean):")
    for m in METRIC_LABELS:
        b_vals = [s[m]["mean"] for n, s in summaries.items() if "baseline" in n and m in s]
        a_vals = [s[m]["mean"] for n, s in summaries.items() if "arch" in n and m in s]
        if b_vals and a_vals:
            b_avg = np.mean(b_vals)
            a_avg = np.mean(a_vals)
            diff  = a_avg - b_avg
            pct   = diff / b_avg * 100 if b_avg else float("inf")
            sign  = "▲" if diff > 0 else "▼"
            print(f"    {m:15s}  baseline={b_avg:.2f}  arch={a_avg:.2f}  "
                  f"{sign} {diff:+.2f} ({pct:+.1f}%)")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run GPT-OSS (NVIDIA NIM) LLM-as-judge on all experiment configs."
    )
    parser.add_argument("--input-dir",   default=INPUT_DIR,   help="Directory with CSV files")
    parser.add_argument("--output",      default=OUTPUT_DIR,  help="Output directory")
    parser.add_argument("--model",       default=MODEL,       help="NVIDIA NIM model name")
    parser.add_argument("--base-url",    default=BASE_URL,    help="API base URL")
    parser.add_argument("--api-key",     default=None,        help="NVIDIA API key (or set NVIDIA_API_KEY)")
    parser.add_argument("--sample-size", type=int, default=SAMPLE_SIZE, help="Rows to sample per config")
    parser.add_argument("--sample-seed", type=int, default=SAMPLE_SEED, help="Random seed")
    parser.add_argument("--rate-limit",  type=float, default=RATE_LIMIT_SEC, help="Seconds between API calls")
    parser.add_argument("--dry-run",     action="store_true", help="Validate CSVs without API calls")
    parser.add_argument("--force",       action="store_true", help="Re-evaluate even if output exists")
    parser.add_argument("--files",       nargs="*",           help="Limit to specific CSV files (basename or path)")
    args = parser.parse_args()

    # Resolve API key
    api_key = args.api_key or os.environ.get("NVIDIA_API_KEY", "")

    # Discover CSVs
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: input directory not found: {input_dir}")
        sys.exit(1)

    if args.files:
        all_csvs = []
        for f in args.files:
            p = Path(f)
            if not p.exists():
                p = input_dir / f
            if p.exists():
                all_csvs.append(p)
            else:
                print(f"Warning: file not found: {f}")
    else:
        all_csvs = sorted(input_dir.glob("*_ragas_llm.csv"))

    if not all_csvs:
        print(f"No CSV files found in {input_dir}/")
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Skip already-evaluated configs (resume support)
    to_process = []
    skipped     = []
    for csv_path in all_csvs:
        out_summary = output_dir / f"{csv_path.stem}_gptoss_judge_summary.json"
        if out_summary.exists() and not args.force:
            skipped.append(csv_path)
        else:
            to_process.append(csv_path)

    print("=" * 60)
    print("  GPT-OSS LLM-as-Judge — Batch Evaluation")
    print("=" * 60)
    print(f"  Model        : {args.model}")
    print(f"  Sample size  : {args.sample_size} per config")
    print(f"  Seed         : {args.sample_seed}")
    print(f"  Input dir    : {input_dir}/")
    print(f"  Output dir   : {output_dir}/")
    print(f"  Total configs: {len(all_csvs)}")
    print(f"  To evaluate  : {len(to_process)}")
    print(f"  Skipped      : {len(skipped)} (already done; use --force to redo)")
    if args.dry_run:
        print("  Mode         : DRY RUN (no API calls)")
    print("=" * 60)

    if not to_process:
        print("\nNothing to evaluate. Use --force to re-run.")
    else:
        # Initialize judge
        if not args.dry_run:
            try:
                judge = NIMJudge(model=args.model, base_url=args.base_url, api_key=api_key)
            except (ValueError, ImportError) as exc:
                print(f"\n✗  Could not initialize judge: {exc}")
                sys.exit(1)
        else:
            judge = None  # not used in dry-run

        # Evaluate each file
        new_summaries: Dict[str, dict] = {}
        for csv_path in to_process:
            result = evaluate_file(
                csv_path=csv_path,
                judge=judge,
                output_dir=output_dir,
                sample_size=args.sample_size,
                seed=args.sample_seed,
                rate_limit=args.rate_limit,
                dry_run=args.dry_run,
            )
            if result:
                new_summaries[csv_path.stem] = result

    # Load ALL summaries (new + previously saved) for comparison
    all_summaries: Dict[str, dict] = {}
    for json_file in sorted(output_dir.glob("*_gptoss_judge_summary.json")):
        with open(json_file) as fh:
            all_summaries[json_file.stem.replace("_gptoss_judge_summary", "")] = json.load(fh)

    # Print comparison table
    if all_summaries:
        print_comparison(all_summaries)

        # Save master comparison JSON
        master = {
            "num_configs": len(all_summaries),
            "model": args.model,
            "sample_size": args.sample_size,
            "timestamp": datetime.now().isoformat(),
            "configs": {
                name: {m: s[m] for m in METRIC_LABELS if m in s}
                for name, s in all_summaries.items()
            },
        }
        master_path = output_dir / "gptoss_all_configs_comparison.json"
        with open(master_path, "w") as fh:
            json.dump(master, fh, indent=2)
        print(f"\n  Master comparison saved → {master_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
