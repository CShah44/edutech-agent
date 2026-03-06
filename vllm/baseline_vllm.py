"""
Generate baseline answers using a vLLM model (no tools/agents)
===============================================================

This script keeps the same baseline behavior as baseline_llama.py:
- Load ELI5 dataset from pickle cache
- Process a question index range
- Generate one answer per question with the same ELI5-style prompt
- Save results to CSV with the same schema

Performance change only:
- Uses vLLM offline batch inference (`LLM.generate`) for higher throughput.

Usage:
    python baseline_llama_vllm.py --start 0 --end 1000 --output baseline_answers/llama3b_0_1000.csv
"""

import argparse
import csv
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm
from vllm import LLM, SamplingParams


MODEL_NAME_ALIASES = {
    "llama3.2:1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama3.2:3b": "meta-llama/Llama-3.2-3B-Instruct",
    "mistral:7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "qwen2.5:3b": "Qwen/Qwen2.5-3B-Instruct",
    "qwen2.5:7b": "Qwen/Qwen2.5-7B-Instruct",
    "gemma2:2b": "google/gemma-2-2b-it",
    "gemma2:9b": "google/gemma-2-9b-it",
}


def load_eli5_dataset(cache_file: str = "eli5_dataset_cache.pkl"):
    """Load the ELI5 dataset from cached pickle file."""
    cache_path = Path(cache_file)

    if not cache_path.exists():
        raise FileNotFoundError(
            f"Dataset cache not found at {cache_path}. "
            "Please run 'python load_dataset.py' first to create the cache."
        )

    print(f"📂 Loading dataset from cache: {cache_path}")
    with open(cache_path, "rb") as file_handle:
        dataframe = pickle.load(file_handle)

    return dataframe


def format_reference_answers(answers: list) -> str:
    """Format reference answers as pipe-separated string."""
    if not answers:
        return ""

    if isinstance(answers, list):
        if all(isinstance(answer, dict) and "answer" in answer for answer in answers):
            return "|||".join(
                [answer["answer"].strip() for answer in answers if answer.get("answer")]
            )
        return "|||".join([str(answer).strip() for answer in answers if str(answer).strip()])

    return str(answers)


def build_prompt(question: str) -> str:
    """Build ELI5-aligned prompt to match multi-agent system behavior."""
    return f"""You are an ELI5 (Explain Like I'm 5) expert. Answer the following question in simple language a 5-year-old would understand.

Your rules:
- Use simple everyday words
- Use fun comparisons (like toys, games, animals, things kids know)
- Make it flow like a story (not just a list)
- Let the explanation be as long as needed to cover the topic naturally
- Do NOT mention "ELI5" explicitly in your answer
- Focus on clarity and engagement over brevity

Question: {question}

Provide a clear and engaging answer:"""


def resolve_vllm_model_source(model_arg: str, local_model_dir: str = "") -> str:
    """Resolve model source for vLLM from direct ID/path or Ollama-style alias."""
    model_arg_stripped = model_arg.strip()

    direct_path = Path(model_arg_stripped)
    if direct_path.exists():
        return str(direct_path)

    if local_model_dir:
        alias_key = model_arg_stripped.lower()
        mapped_model_name = MODEL_NAME_ALIASES.get(alias_key, model_arg_stripped)
        local_candidate = Path(local_model_dir) / mapped_model_name
        if local_candidate.exists():
            return str(local_candidate)

    return MODEL_NAME_ALIASES.get(model_arg_stripped.lower(), model_arg_stripped)


def generate_batch_answers(
    batch_indices: List[int],
    batch_rows: List[Dict[str, Any]],
    llm: LLM,
    sampling_params: SamplingParams,
) -> List[Dict[str, Any]]:
    """Generate answers for one batch and return result rows."""
    prompts = [build_prompt(row["query"]) for row in batch_rows]
    batch_start = time.time()

    try:
        outputs = llm.generate(prompts, sampling_params=sampling_params)
        batch_elapsed = time.time() - batch_start
        avg_generation_time = batch_elapsed / max(1, len(batch_rows))

        results: List[Dict[str, Any]] = []
        for local_idx, output in enumerate(outputs):
            question_idx = batch_indices[local_idx]
            row = batch_rows[local_idx]
            reference_answers_str = format_reference_answers(row.get("answers", []))

            generated_answer = ""
            if output.outputs:
                generated_answer = output.outputs[0].text.strip()

            results.append(
                {
                    "question_id": question_idx,
                    "question": row["query"],
                    "generated_answer": generated_answer,
                    "reference_answers": reference_answers_str,
                    "generation_time": avg_generation_time,
                    "status": "success",
                    "timestamp": datetime.now().isoformat(),
                    "error": "",
                }
            )

        return results

    except Exception as exception:
        batch_elapsed = time.time() - batch_start
        avg_generation_time = batch_elapsed / max(1, len(batch_rows))
        error_message = str(exception)

        return [
            {
                "question_id": batch_indices[local_idx],
                "question": batch_rows[local_idx]["query"],
                "generated_answer": "",
                "reference_answers": format_reference_answers(
                    batch_rows[local_idx].get("answers", [])
                ),
                "generation_time": avg_generation_time,
                "status": "error",
                "timestamp": datetime.now().isoformat(),
                "error": error_message,
            }
            for local_idx in range(len(batch_rows))
        ]


def main():
    parser = argparse.ArgumentParser(
        description="Generate baseline answers using vLLM (batch offline inference)"
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Starting question index (inclusive)",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=3000,
        help="Ending question index (exclusive)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="vllm/baseline_answers/llama3b_0_3000.csv",
        help="Output CSV file path",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="vLLM model identifier or Ollama-style alias (e.g., llama3.2:3b)",
    )
    parser.add_argument(
        "--local-model-dir",
        type=str,
        default="",
        help="Optional base directory containing local model folders",
    )
    parser.add_argument(
        "--cache-file",
        type=str,
        default="eli5_dataset_cache.pkl",
        help="Path to cached ELI5 dataset pickle file",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Compatibility knob from baseline script; used to scale max_num_seqs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Number of prompts sent per vLLM generate call (default: 20)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.4,
        help="Sampling temperature (default: 0.4)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=700,
        help="Maximum generated tokens per answer (default: 700)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.60,
        help="vLLM GPU memory utilization target (default: 0.60)",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size for multi-GPU inference (default: 1)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="Maximum model context length (default: 4096)",
    )

    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("🤖 Baseline Llama Answer Generator (vLLM Batch)")
    print("=" * 80)
    resolved_model = resolve_vllm_model_source(
        model_arg=args.model,
        local_model_dir=args.local_model_dir,
    )

    print(f"Model input: {args.model}")
    print(f"Model resolved: {resolved_model}")
    print(f"Question range: {args.start} to {args.end}")
    print(f"Batch size: {args.batch_size}")
    print(f"Workers compatibility value: {args.workers}")
    print(f"Output file: {args.output}")
    print()

    print("Loading ELI5 dataset from cache...")
    dataframe = load_eli5_dataset(args.cache_file)
    print(f"✅ Loaded {len(dataframe)} questions from cache")
    print()

    print("Initializing vLLM engine...")
    max_num_seqs = max(1, args.workers * args.batch_size)
    llm = LLM(
        model=resolved_model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=max_num_seqs,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    print(
        "vLLM initialized "
        f"(temperature: {args.temperature}, max_tokens: {args.max_tokens}, max_num_seqs: {max_num_seqs})"
    )
    print()

    fieldnames = [
        "question_id",
        "question",
        "generated_answer",
        "reference_answers",
        "generation_time",
        "status",
        "timestamp",
        "error",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    range_end = min(args.end, len(dataframe))
    total_questions = max(0, range_end - args.start)
    print(f"Generating answers for questions {args.start} to {range_end}...")

    with tqdm(total=total_questions, desc="Processing") as progress_bar:
        for batch_start in range(args.start, range_end, args.batch_size):
            batch_end = min(batch_start + args.batch_size, range_end)

            batch_indices = list(range(batch_start, batch_end))
            batch_rows = [dataframe.iloc[index].to_dict() for index in batch_indices]

            batch_results = generate_batch_answers(
                batch_indices=batch_indices,
                batch_rows=batch_rows,
                llm=llm,
                sampling_params=sampling_params,
            )

            with open(output_path, "a", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                for result_row in sorted(batch_results, key=lambda row: row["question_id"]):
                    writer.writerow(result_row)

            progress_bar.update(len(batch_indices))

    print()
    print("=" * 80)
    print("✅ Generation complete!")
    print(f"📁 Results saved to: {args.output}")
    print("=" * 80)


if __name__ == "__main__":
    main()
