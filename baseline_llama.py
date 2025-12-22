"""
Generate baseline answers using Llama 3B model (no tools/agents)
================================================================

This script uses the base Llama 3B model from Ollama to generate answers
for the ELI5 dataset without any tools or agent framework.

Usage:
    python baseline_llama.py --start 0 --end 1000 --output baseline_answers/llama3b_0_1000.csv
"""

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from langchain_ollama import ChatOllama
from tqdm import tqdm
import pickle


def load_eli5_dataset(cache_file: str = "eli5_dataset_cache.pkl"):
    """Load the ELI5 dataset from cached pickle file."""
    cache_path = Path(cache_file)
    
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Dataset cache not found at {cache_path}. "
            "Please run 'python load_dataset.py' first to create the cache."
        )
    
    print(f"📂 Loading dataset from cache: {cache_path}")
    with open(cache_path, 'rb') as f:
        df = pickle.load(f)
    
    return df


def generate_answer(question: str, llm: ChatOllama) -> Dict[str, Any]:
    """Generate answer using base Llama model with ELI5-style prompt."""
    start_time = time.time()
    
    try:
        # ELI5-aligned prompt to match multi-agent system behavior
        prompt = f"""You are an ELI5 (Explain Like I'm 5) expert. Answer the following question in simple language a 5-year-old would understand.

Your rules:
- Use simple everyday words
- Use fun comparisons (like toys, games, animals, things kids know)
- Make it flow like a story (not just a list)
- Let the explanation be as long as needed to cover the topic naturally
- Do NOT mention "ELI5" explicitly in your answer
- Focus on clarity and engagement over brevity

Question: {question}

Provide a clear and engaging answer:"""
        
        response = llm.invoke(prompt)
        answer = response.content.strip()
        
        generation_time = time.time() - start_time
        
        return {
            "generated_answer": answer,
            "generation_time": generation_time,
            "status": "success",
            "error": ""
        }
    
    except Exception as e:
        generation_time = time.time() - start_time
        return {
            "generated_answer": "",
            "generation_time": generation_time,
            "status": "error",
            "error": str(e)
        }


def format_reference_answers(answers: list) -> str:
    """Format reference answers as pipe-separated string."""
    if not answers:
        return ""
    
    # Handle different formats
    if isinstance(answers, list):
        if all(isinstance(a, dict) and 'answer' in a for a in answers):
            # Format: [{'answer': 'text'}, ...]
            return '|||'.join([a['answer'].strip() for a in answers if a.get('answer')])
        else:
            # Format: ['text1', 'text2', ...]
            return '|||'.join([str(a).strip() for a in answers if str(a).strip()])
    
    return str(answers)


def process_single_question(idx: int, question_data: dict, llm: ChatOllama) -> Dict[str, Any]:
    """Process a single question and return result row."""
    question = question_data['query']
    reference_answers = question_data.get('answers', [])
    reference_answers_str = format_reference_answers(reference_answers)
    
    result = generate_answer(question, llm)
    
    return {
        'question_id': idx,
        'question': question,
        'generated_answer': result['generated_answer'],
        'reference_answers': reference_answers_str,
        'generation_time': result['generation_time'],
        'status': result['status'],
        'timestamp': datetime.now().isoformat(),
        'error': result['error']
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate baseline answers using Llama 3B model"
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Starting question index (inclusive)"
    )
    parser.add_argument(
        "--end",
        type=int,
        default=1000,
        help="Ending question index (exclusive)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="baseline_answers/llama3b_0_1000.csv",
        help="Output CSV file path"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama3.2:3b",
        help="Ollama model name to use"
    )
    parser.add_argument(
        "--cache-file",
        type=str,
        default="eli5_dataset_cache.pkl",
        help="Path to cached ELI5 dataset pickle file"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of concurrent workers for GPU utilization (default: 4)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Number of results to buffer before writing to CSV (default: 20)"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("🤖 Baseline Llama Answer Generator (Concurrent)")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Question range: {args.start} to {args.end}")
    print(f"Workers: {args.workers} (concurrent GPU utilization)")
    print(f"Batch size: {args.batch_size}")
    print(f"Output file: {args.output}")
    print()
    
    # Load dataset from pickle cache
    print("Loading ELI5 dataset from cache...")
    df = load_eli5_dataset(args.cache_file)
    print(f"✅ Loaded {len(df)} questions from cache")
    print()
    
    # Initialize LLM instances for concurrent workers
    print(f"Initializing {args.workers} LLM instances for concurrent processing...")
    llms = []
    for i in range(args.workers):
        llm = ChatOllama(
            model=args.model,
            temperature=0.4,
        )
        llms.append(llm)
    print(f"Model initialized (temperature: 0.4, {args.workers} workers)")
    print()
    
    # Prepare CSV file
    fieldnames = [
        'question_id',
        'question',
        'generated_answer',
        'reference_answers',
        'generation_time',
        'status',
        'timestamp',
        'error'
    ]
    
    # Write CSV header
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    
    # Process questions concurrently
    print(f"Generating answers for questions {args.start} to {args.end}...")
    
    csv_lock = Lock()
    results_buffer: List[Dict[str, Any]] = []
    completed = 0
    total_questions = min(args.end, len(df)) - args.start
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Submit all tasks
        future_to_idx = {}
        for idx in range(args.start, min(args.end, len(df))):
            question_data = df.iloc[idx].to_dict()
            # Round-robin LLM assignment
            llm = llms[idx % args.workers]
            future = executor.submit(process_single_question, idx, question_data, llm)
            future_to_idx[future] = idx
        
        # Process completed tasks with progress bar
        with tqdm(total=total_questions, desc="Processing") as pbar:
            for future in as_completed(future_to_idx):
                try:
                    row = future.result()
                    results_buffer.append(row)
                    completed += 1
                    
                    # Write batch to CSV when buffer is full
                    if len(results_buffer) >= args.batch_size:
                        with csv_lock:
                            with open(output_path, 'a', newline='', encoding='utf-8') as csvfile:
                                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                                for result_row in sorted(results_buffer, key=lambda x: x['question_id']):
                                    writer.writerow(result_row)
                            results_buffer.clear()
                    
                except Exception as e:
                    print(f"\n❌ Error processing question: {str(e)}")
                
                pbar.update(1)
        
        # Write remaining results
        if results_buffer:
            with csv_lock:
                with open(output_path, 'a', newline='', encoding='utf-8') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    for result_row in sorted(results_buffer, key=lambda x: x['question_id']):
                        writer.writerow(result_row)
    
    print()
    print("=" * 80)
    print("✅ Generation complete!")
    print(f"📁 Results saved to: {args.output}")
    print("=" * 80)


if __name__ == "__main__":
    main()
