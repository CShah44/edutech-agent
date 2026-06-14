"""
GPT-OSS LLM Judge for ELI5 Research
====================================

Evaluates generated answers using NVIDIA's GPT-OSS models (via build.nvidia.com).
Samples N rows per file and judges on Accuracy, Completeness, and Overall quality.

Usage:
    # Single file evaluation
    python evaluate_with_gptoss_judge.py --input generated_answers/baseline_llama3b_0_400.csv

    # With specific model and sample size
    python evaluate_with_gptoss_judge.py --input answers.csv --model nvidia/gpt-oss-120b --sample-size 50

Environment:
    NVIDIA_API_KEY or .env file with NVIDIA_API_KEY
"""

import argparse
import ast
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List
from datetime import datetime

import pandas as pd
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Try to import OpenAI
try:
    from openai import OpenAI
    have_openai = True
except ImportError:
    have_openai = False
    print("Warning: openai not installed. Install with: pip install openai")

# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_CONFIG = {
    "model": "nvidia/gpt-oss-120b",  # Default to the 120B version
    "base_url": "https://integrate.api.nvidia.com/v1",
    "sample_size": 50,
    "sample_seed": 42,
    "max_retries": 3,
    "retry_delay": 5,  # Base retry delay in seconds
    "temperature": 0.1,
    "max_tokens": 512,
    "top_p": 0.9,
    "rate_limit_delay": 0.5,  # Delay between API calls in seconds
    "timeout": 60,
}

# ============================================================================
# JUDGE PROMPT
# ============================================================================

JUDGE_PROMPT = """You are an expert evaluator assessing the quality of Explain-Like-I'm-Five (ELI5) answers.

Task: Evaluate the following generated answer against the reference answer(s).

Question: {question}

Reference Answer(s):
{references}

Generated Answer:
{generated}

Evaluate on these dimensions (1-10 scale each):
1. correctness (1-10): Factual accuracy. Does it contain factual errors?
2. completeness (1-10): Coverage of key points from the reference.
3. eli5_quality (1-10): How well it explains like a 5-year-old (simple language, analogies, clarity).
4. overall (1-10): Overall quality considering all factors.

Return ONLY a JSON object with this exact format (no markdown, no extra text):
{{"correctness": <int>, "completeness": <int>, "eli5_quality": <int>, "overall": <int>, "reasoning": "<brief explanation>"}}"""

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


# ============================================================================
# NVIDIA API CLIENT
# ============================================================================

class GPTOSSJudge:
    """GPT-OSS Judge using NVIDIA API (build.nvidia.com)."""

    def __init__(
        self,
        model: str = DEFAULT_CONFIG["model"],
        base_url: str = DEFAULT_CONFIG["base_url"],
        api_key: str | None = None,
        temperature: float = DEFAULT_CONFIG["temperature"],
        max_tokens: int = DEFAULT_CONFIG["max_tokens"],
        top_p: float = DEFAULT_CONFIG["top_p"],
        timeout: int = DEFAULT_CONFIG["timeout"],
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p

        # API key from parameter, env, or error
        if api_key is None:
            api_key = os.environ.get("NVIDIA_API_KEY")

        if not api_key:
            raise ValueError(
                "NVIDIA API key required. Set NVIDIA_API_KEY env var or pass --api-key."
            )

        if not have_openai:
            raise ImportError("openai library required. Install: pip install openai")

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )

    def _call_api(self, prompt: str, max_retries: int = DEFAULT_CONFIG["max_retries"]) -> Dict[str, Any]:
        """Call NVIDIA API with retries."""
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an impartial evaluator."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                )

                content = response.choices[0].message.content
                return self._parse_json(content)

            except Exception as e:
                delay = DEFAULT_CONFIG["retry_delay"] * (2 ** attempt)
                print(f"  API error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    print(f"  Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    print(f"  Failed after {max_retries} attempts.")
                    return {
                        "correctness": 0,
                        "completeness": 0,
                        "eli5_quality": 0,
                        "overall": 0,
                        "reasoning": f"API Error: {str(e)[:200]}",
                        "error": str(e)[:200],
                    }

    def _parse_json(self, content: str) -> Dict[str, Any]:
        """Parse JSON from model response."""
        content = content.strip()

        # Remove markdown code blocks if present
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            parts = content.split("```")
            if len(parts) >= 3:
                content = parts[1].strip()

        try:
            parsed = json.loads(content)
            # Ensure all keys exist
            for key in ["correctness", "completeness", "eli5_quality", "overall"]:
                if key not in parsed:
                    parsed[key] = 0
            if "reasoning" not in parsed:
                parsed["reasoning"] = "No reasoning provided."
            return parsed

        except (json.JSONDecodeError, ValueError) as e:
            # Try to extract using regex
            import re
            scores = {}
            for key in ["correctness", "completeness", "eli5_quality", "overall"]:
                pattern = rf'"{key}"\s*:\s*(\d+)'
                match = re.search(pattern, content)
                scores[key] = int(match.group(1)) if match else 0

            scores["reasoning"] = f"JSON parse failed: {str(e)[:100]}"
            scores["error"] = str(e)[:100]
            return scores

    def evaluate_single(
        self, question: str, generated: str, references: List[str]
    ) -> Dict[str, Any]:
        """Evaluate a single question-answer pair."""
        ref_text = "\n".join([f"- {r}" for r in references[:3]])
        prompt = JUDGE_PROMPT.format(
            question=question[:1000],  # Truncate very long questions
            references=ref_text[:2000],  # Truncate references
            generated=generated[:3000],  # Truncate generated text
        )

        result = self._call_api(prompt)

        # Ensure scores are within 1-10 range
        for key in ["correctness", "completeness", "eli5_quality", "overall"]:
            if key in result:
                try:
                    val = int(result[key])
                    result[key] = max(1, min(10, val))
                except (ValueError, TypeError):
                    result[key] = 0

        return result


# ============================================================================
# SAMPLING & OUTPUT
# ============================================================================

def sample_rows(df: pd.DataFrame, sample_size: int, seed: int) -> pd.DataFrame:
    """Sample rows from DataFrame with reproducible seed."""
    if len(df) <= sample_size:
        return df.copy()
    return df.sample(n=sample_size, random_state=seed).reset_index(drop=True)


def save_results(results: List[Dict], summary: Dict, output_dir: Path, input_stem: str):
    """Save detailed results and summary."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save detailed CSV
    df_out = pd.DataFrame(results)
    csv_path = output_dir / f"{input_stem}_gptoss_judge.csv"
    df_out.to_csv(csv_path, index=False)
    print(f"  Detailed results: {csv_path}")

    # Save JSON summary
    json_path = output_dir / f"{input_stem}_gptoss_judge_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary: {json_path}")


def generate_summary(results: List[Dict]) -> Dict[str, Any]:
    """Generate summary statistics from judge results."""
    metrics = ["correctness", "completeness", "eli5_quality", "overall"]
    summary = {
        "total_evaluated": len(results),
        "errors": sum(1 for r in results if "error" in r.get("raw", {})),
    }

    for metric in metrics:
        scores = [r[metric] for r in results if metric in r]
        if scores:
            summary[metric] = {
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "min": float(np.min(scores)),
                "max": float(np.max(scores)),
                "median": float(np.median(scores)),
            }

    return summary


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate generated answers with GPT-OSS (NVIDIA API) judge."
    )
    parser.add_argument("--input", type=str, required=True, help="Path to input CSV")
    parser.add_argument("--output", type=str, default="llm_metrics_gptoss/", help="Output directory")
    parser.add_argument("--model", type=str, default=DEFAULT_CONFIG["model"], help="GPT-OSS model name")
    parser.add_argument("--base-url", type=str, default=DEFAULT_CONFIG["base_url"], help="NVIDIA API base URL")
    parser.add_argument("--api-key", type=str, default=None, help="NVIDIA API key (or set NVIDIA_API_KEY)")
    parser.add_argument("--sample-size", type=int, default=DEFAULT_CONFIG["sample_size"], help="Number of rows to sample")
    parser.add_argument("--sample-seed", type=int, default=DEFAULT_CONFIG["sample_seed"], help="Random seed for sampling")
    parser.add_argument("--dry-run", action="store_true", help="Validate setup without API calls")
    parser.add_argument("--max-rows", type=int, default=None, help="Limit rows before sampling")
    args = parser.parse_args()

    # Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return

    # Resolve columns
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows from {input_path}")

    question_col = pick_column(df, ["question", "user_input", "query", "prompt"])
    answer_col = pick_column(df, ["generated_answer", "response", "answer", "model_answer", "output"])
    reference_col = pick_column(df, ["reference_answers", "reference_answer", "reference", "ground_truth"])

    if not all([question_col, answer_col, reference_col]):
        print(f"Error: Missing required columns. Found: {list(df.columns)}")
        return

    # Filter for success if status column exists
    if "status" in df.columns:
        before = len(df)
        df = df[df["status"] == "success"]
        print(f"  Filtered to success: {len(df)} (dropped {before - len(df)})")

    # Limit rows if specified
    if args.max_rows:
        df = df.head(args.max_rows)
        print(f"  Limited to: {len(df)} rows")

    # Sample
    sampled = sample_rows(df, args.sample_size, args.sample_seed)
    print(f"\nSampled {len(sampled)} rows (seed={args.sample_seed})")

    # Dry run
    if args.dry_run:
        print("\n[DRY RUN] No API calls made.")
        print("First 3 rows to be evaluated:")
        for _, row in sampled.head(3).iterrows():
            print(f"  Q: {str(row[question_col])[:80]}...")
        return

    # Initialize judge
    try:
        judge = GPTOSSJudge(
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
        )
    except (ValueError, ImportError) as e:
        print(f"Error initializing judge: {e}")
        return

    print(f"\nEvaluating with GPT-OSS Judge (model: {args.model})")
    print(f"Base URL: {args.base_url}")
    print("-" * 60)

    # Evaluate each sample
    results = []
    t_start = time.time()

    for i, (_, row) in enumerate(tqdm(sampled.iterrows(), total=len(sampled), desc="Judging")):
        generated = str(row[answer_col])
        question = str(row[question_col])
        references = parse_reference_answers(row[reference_col])

        if not generated or not references:
            print(f"  Row {i}: Missing data, skipping")
            continue

        result = judge.evaluate_single(question, generated, references)
        result["question"] = question
        result["generated_answer"] = generated[:100]  # Truncate for storage
        result["question_id"] = row.get("question_id", i)
        result["_model"] = args.model
        result["_base_url"] = args.base_url
        results.append(result)

        # Rate limiting
        if i < len(sampled) - 1:
            time.sleep(DEFAULT_CONFIG["rate_limit_delay"])

    elapsed = time.time() - t_start
    print(f"\nCompleted {len(results)} evaluations in {elapsed:.1f}s")

    # Generate summary
    summary = generate_summary(results)
    summary["_meta"] = {
        "input": str(input_path),
        "sample_size": len(sampled),
        "sample_seed": args.sample_seed,
        "model": args.model,
        "base_url": args.base_url,
        "elapsed_seconds": round(elapsed, 2),
        "timestamp": datetime.now().isoformat(),
    }

    # Print summary
    print("\n" + "=" * 60)
    print("GPT-OSS JUDGE SUMMARY")
    print("=" * 60)
    for metric in ["correctness", "completeness", "eli5_quality", "overall"]:
        if metric in summary:
            s = summary[metric]
            print(f"  {metric:20s}  mean={s['mean']:.2f}  std={s['std']:.2f}  "
                  f"min={s['min']:.0f}  max={s['max']:.0f}  median={s['median']:.2f}")
    print("=" * 60)

    # Save
    output_dir = Path(args.output)
    save_results(results, summary, output_dir, input_path.stem)


if __name__ == "__main__":
    main()
