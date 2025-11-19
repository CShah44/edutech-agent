"""
Direct question processor - processes each question in-memory without subprocess overhead.
Saves after every question for resumability. Delays after every 10 questions to rest Ollama.
"""

import time
from pathlib import Path
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import multiprocessing as mp

# Import from simple_agent
from simple_agent import answer_question, load_eli5_dataset

# Configuration
DELAY_AFTER_N_QUESTIONS = 10  # Rest after this many questions
DELAY_DURATION = 10  # Seconds to wait after N questions
TOTAL_QUESTIONS = 1000  # Total questions to process
OUTPUT_FILE = "generated_answers/answers_0_1000.csv"
TIMEOUT_SECONDS = 120  # 2 minutes timeout per question
RETRY_DELAY = 10  # Delay before retry on timeout


def _question_worker(question, result_queue):
    """Worker executed in a separate process to avoid thread deadlocks."""
    try:
        # Ensure fresh caches inside the worker process
        from simple_agent import answer_question, clear_all_caches

        clear_all_caches()
        result = answer_question(question)
        result_queue.put({"result": result})
    except Exception as exc:  # pragma: no cover - worker exceptions bubbled up
        result_queue.put({"error": str(exc)})


def _cleanup_queue(queue):
    """Best-effort queue cleanup to avoid resource leaks."""
    try:
        queue.close()
    except Exception:
        pass
    try:
        queue.join_thread()
    except Exception:
        pass

def process_single_question(question_data, question_idx):
    """Process a single question and return result with timeout protection."""
    from simple_agent import clear_all_caches

    question = question_data['query']
    reference_answers = question_data['answers']
    if not isinstance(reference_answers, list):
        reference_answers = [reference_answers]

    ctx = mp.get_context("spawn")
    max_retries = 2

    for attempt in range(max_retries):
        print(f"   ▶️ Attempt {attempt + 1}/{max_retries} for question {question_idx}")

        clear_all_caches()
        result_queue = ctx.Queue()
        worker = ctx.Process(target=_question_worker, args=(question, result_queue))

        start_time = time.time()
        worker.start()
        worker.join(TIMEOUT_SECONDS)

        if worker.is_alive():
            print(f"\n⏱️ Timeout on attempt {attempt + 1}/{max_retries} - terminating worker process...")
            worker.terminate()
            worker.join()

            clear_all_caches()
            _cleanup_queue(result_queue)

            if attempt < max_retries - 1:
                print(f"   Resting {RETRY_DELAY}s before retry with fresh state...")
                time.sleep(RETRY_DELAY)
                continue
            else:
                _cleanup_queue(result_queue)
                return {
                    'question_id': question_idx,
                    'question': question,
                    'generated_answer': '',
                    'reference_answers': '|||'.join(reference_answers),
                    'generation_time': TIMEOUT_SECONDS,
                    'status': 'error',
                    'error': f'Timeout after {max_retries} attempts',
                    'timestamp': datetime.now().isoformat()
                }

        worker.join()

        try:
            payload = result_queue.get_nowait()
        except Exception:
            payload = {}
        finally:
            _cleanup_queue(result_queue)

        generation_time = time.time() - start_time

        if not payload:
            print(f"\n⚠️ Worker exited without result on attempt {attempt + 1}" )
            clear_all_caches()
            if attempt < max_retries - 1:
                print(f"   Resting {RETRY_DELAY}s before retry...")
                time.sleep(RETRY_DELAY)
                continue
            return {
                'question_id': question_idx,
                'question': question,
                'generated_answer': '',
                'reference_answers': '|||'.join(reference_answers),
                'generation_time': generation_time,
                'status': 'error',
                'error': 'Worker exited without result',
                'timestamp': datetime.now().isoformat()
            }

        if 'error' in payload:
            print(f"\n❌ Worker raised error: {payload['error']}")
            clear_all_caches()
            if attempt < max_retries - 1:
                print(f"   Resting {RETRY_DELAY}s before retry...")
                time.sleep(RETRY_DELAY)
                continue
            return {
                'question_id': question_idx,
                'question': question,
                'generated_answer': '',
                'reference_answers': '|||'.join(reference_answers),
                'generation_time': generation_time,
                'status': 'error',
                'error': payload['error'],
                'timestamp': datetime.now().isoformat()
            }

        result = payload['result']

        return {
            'question_id': question_idx,
            'question': question,
            'generated_answer': result.get('final_answer', 'No answer generated'),
            'reference_answers': '|||'.join(reference_answers),
            'generation_time': generation_time,
            'status': 'success',
            'timestamp': datetime.now().isoformat()
        }

    # Fallback, though control flow should return earlier
    return {
        'question_id': question_idx,
        'question': question,
        'generated_answer': '',
        'reference_answers': '|||'.join(reference_answers),
        'generation_time': TIMEOUT_SECONDS,
        'status': 'error',
        'error': 'Unhandled retry termination',
        'timestamp': datetime.now().isoformat()
    }

def main():
    print("="*60)
    print("🤖 Direct Question Processor (In-Memory)")
    print("="*60)
    print(f"Total questions: {TOTAL_QUESTIONS}")
    print(f"Delay every: {DELAY_AFTER_N_QUESTIONS} questions")
    print(f"Delay duration: {DELAY_DURATION} seconds")
    print(f"Output file: {OUTPUT_FILE}")
    print("="*60)
    
    # Ensure output directory exists
    Path("generated_answers").mkdir(exist_ok=True)
    output_path = Path(OUTPUT_FILE)
    
    # Load dataset
    print("\n📥 Loading dataset...")
    df = load_eli5_dataset()
    
    # Check for existing progress
    start_from = 0
    all_results = []
    
    if output_path.exists():
        try:
            existing_df = pd.read_csv(output_path)
            all_results = existing_df.to_dict('records')
            start_from = len(all_results)
            print(f"\n📂 Resuming from question {start_from + 1}/{TOTAL_QUESTIONS}")
        except:
            print(f"\n📂 Starting fresh")
    
    # Process questions
    failed_count = 0
    
    for idx in tqdm(range(start_from, TOTAL_QUESTIONS), desc="Processing questions", initial=start_from, total=TOTAL_QUESTIONS):
        question_data = df.iloc[idx]
        
        result = process_single_question(question_data, idx)
        all_results.append(result)
        
        if result['status'] == 'error':
            failed_count += 1
        
        # Save after every question
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(output_path, index=False)
        
        # Rest after every N questions
        questions_processed = (idx - start_from + 1)
        if questions_processed % DELAY_AFTER_N_QUESTIONS == 0 and idx < TOTAL_QUESTIONS - 1:
            print(f"\n😴 Resting for {DELAY_DURATION} seconds... ({idx + 1}/{TOTAL_QUESTIONS} done)")
            time.sleep(DELAY_DURATION)
    
    # Final summary
    print("\n" + "="*60)
    print("🎉 ALL QUESTIONS PROCESSED!")
    print("="*60)
    print(f"Total: {TOTAL_QUESTIONS}")
    print(f"Success: {TOTAL_QUESTIONS - failed_count}")
    print(f"Failed: {failed_count}")
    print(f"Results: {OUTPUT_FILE}")
    print("="*60)

if __name__ == "__main__":
    main()
