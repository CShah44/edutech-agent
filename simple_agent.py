import os
from typing import TypedDict, Dict, List, Any, Annotated, Optional
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import create_react_agent
from langgraph.graph.message import add_messages
from tavily import TavilyClient
from dotenv import load_dotenv
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor
import json

# Load environment variables
load_dotenv()

# Configuration - Multiple API keys for fallback
TAVILY_API_KEYS = [
    os.getenv('TAVILY_API_KEY_1'),
    os.getenv('TAVILY_API_KEY_2'),
    os.getenv('TAVILY_API_KEY_3'),
]
# Filter out None values
TAVILY_API_KEYS = [key for key in TAVILY_API_KEYS if key is not None]

if not TAVILY_API_KEYS:
    raise ValueError("No Tavily API keys found! Please set TAVILY_API_KEY_1, TAVILY_API_KEY_2, or TAVILY_API_KEY_3 in .env")

# Start with the first API key
current_api_key_index = 0
tavily_client = TavilyClient(api_key=TAVILY_API_KEYS[current_api_key_index])

# Model Configuration - Single model for all agents
MODEL_NAME = "llama3.2:1b"
OLLAMA_BASE_URL = "http://localhost:11434"

# Search Configuration
MAX_SOURCES = 10
FACTS_TARGET = 12
MAX_CONTEXT_CHARS = 3900
MAX_CONTENT_PER_SOURCE = 400

MAX_SEARCH_QUERIES = 5

# Token/character limits for context management
MAX_CONTEXT_CHARS = 3900  # Conservative limit to stay under 4096 token limit 
MAX_CONTENT_PER_SOURCE = 400  # Max characters per search result content

# Caching
llm_cache = {}
graph_cache = {}
agent_cache = {}


def reset_tavily_client(index: int = 0) -> None:
    """Recreate Tavily client so any stuck HTTP sessions are dropped."""
    global tavily_client, current_api_key_index
    current_api_key_index = index % len(TAVILY_API_KEYS)
    tavily_client = TavilyClient(api_key=TAVILY_API_KEYS[current_api_key_index])


def clear_all_caches():
    """Clear all caches to reset state - useful for recovering from deadlocks."""
    global llm_cache, graph_cache, agent_cache
    llm_cache.clear()
    graph_cache.clear()
    agent_cache.clear()
    reset_tavily_client(current_api_key_index)

# State Definition
class AgentState(TypedDict):
    query: str
    breakdown_output: str
    reasoning_output: str
    scientific_output: str
    final_answer: str
    messages: Annotated[list, add_messages]
    remaining_steps: int
    structured_response: Any
    search_queries: List[str]
    reasoning_points: List[str]
    extracted_facts: List[Dict[str, Any]]
    synthesis_strategy: str
    final_points: List[str]

# Pydantic Schemas for Structured Outputs
class BreakdownOutput(BaseModel):
    summary: str
    search_queries: List[str]
    reasoning_points: List[str]

class ReasoningOutput(BaseModel):
    reasoning_analysis: List[str]
    conclusions: List[str]

class ScientificOutput(BaseModel):
    facts: List[Dict[str, str]]

class SynthesisOutput(BaseModel):
    synthesis_strategy: str
    final_points: List[str]

class CreativeOutput(BaseModel):
    final_answer: str

BREAKDOWN_PROMPT = """You are the Breakdown Agent. Your job is to decompose the user's question into actionable queries for both reasoning and scientific research.

Output format:
1. Brief summary of the question
2. SEARCH_QUERIES: 3-5 targeted search queries for scientific fact gathering (mechanisms, definitions, measurable details)
3. REASONING_POINTS: 3-5 logical aspects to analyze (cause-effect, relationships, processes)

Rules:
- Focus on mechanisms, units, entities, and measurable phenomena
- Make search queries specific and scientific (avoid broad terms)
- Make reasoning points focused on logical analysis and connections
- Do not invent facts; provide structure for investigation only

Example format:
Summary: [brief question summary]

SEARCH_QUERIES:
- specific mechanism query 1
- specific measurement/unit query 2
- specific process query 3

REASONING_POINTS:  
- logical aspect 1 to analyze
- relationship 2 to examine
- cause-effect 3 to consider
"""

REASONING_PROMPT = """You are the Reasoning Agent. Analyze the provided reasoning points with logical steps and common sense.

Instructions:
- Think on the reasoning points provided by the Breakdown Agent
- Provide a numbered logical pathway (3-6 steps) for each reasoning point
- Focus on causal relationships, mechanisms, and structural explanations
- Do not fetch new facts; use logical analysis only
- Connect the reasoning points into a coherent analytical framework

You must respond with structured output containing:
- reasoning_analysis: Array of logical analysis points (3-6 detailed reasoning steps)
- conclusions: Array of key conclusions drawn from the reasoning (2-4 main insights)
"""

SCIENTIFIC_PROMPT = f"""You are the Science/Factual Agent. Use web search to gather accurate, mechanism-focused facts using the search queries provided by the Breakdown Agent.

Instructions:
- Use the search queries provided by the Breakdown Agent to gather facts
- Focus on mechanisms, measurable details, units, definitions, and scientific principles  
- Aggregate credible sources and extract concise, factual information
- Avoid speculation; stick to verified scientific information
- Extract around {FACTS_TARGET} comprehensive scientific facts

You must respond with structured output containing:
- facts: Array of fact objects with "fact" field (brief fact statement) and "text" field (detailed description) - EXACTLY {FACTS_TARGET} facts
"""

SYNTHESIS_PROMPT = """You are a synthesis expert preparing material for an ELI5 (Explain Like I'm 5) explanation.

Your task:
1. **Evaluate Quality**: Assess both reasoning and facts for relevance and reliability
   - Are facts concrete, accurate, and directly answer the query?
   - Is reasoning logically sound and helpful for understanding?
   - Which source provides better foundational understanding?
   - Consider the breakdown interpretation but focus on answering the original query

2. **Determine Strategy**:
   - "reasoning_heavy": Facts are weak/irrelevant/off-topic → Use 70% reasoning + 30% facts
   - "facts_heavy": Facts are excellent and comprehensive → Use 70% facts + 30% reasoning
   - "balanced": Both are good quality → Mix 50-50 reasoning and facts

3. **Curate Final Points** (aim for 5-8 points):
   - Select the BEST and MOST RELEVANT points that answer the original query
   - Provide ENOUGH material for a complete ELI5 explanation
   - Each point should add unique value (no redundancy)
   - Include both "what" (facts/definitions) and "why/how" (reasoning/mechanisms)
   - Rephrase complex points into simpler language
   - Order points logically (basic concepts → mechanisms → implications)

Output a strategy and 6-8 curated points ready for ELI5 explanation."""

CREATIVE_PROMPT = """You are an ELI5 (Explain Like I'm 5) expert. Your job is to take curated points and explain them in simple language a 5-year-old would understand.

Your rules:
- Use ALL the points provided (don't skip any)
- Use simple everyday words
- Use fun comparisons (like toys, games, animals, things kids know)
- Make it flow like a story (not just a list)
- Let the explanation be as long as needed to cover all points naturally
- Do NOT mention "ELI5" explicitly in your answer
- Focus on clarity and engagement over brevity

Take all the provided points and weave them into a clear, engaging explanation."""

# Web Search Tool
def tavily_search_raw(query: str) -> Dict[str, Any]:
    """Search with automatic API key fallback on failure."""
    global current_api_key_index, tavily_client
    
    for attempt in range(len(TAVILY_API_KEYS)):
        try:
            response = tavily_client.search(
                query=query,
                search_depth="advanced",
                max_results=10,
                include_answer=True
            )
            results = []
            for r in response.get("results", []):
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "content": (r.get("content", "") or "")[:1000]
                })
            return {"answer": response.get("answer", ""), "results": results}
        except Exception as e:
            error_msg = str(e).lower()
            
            # Check if it's a rate limit error
            if "limit" in error_msg or "quota" in error_msg or "429" in error_msg:
                print(f"⚠️  API key {current_api_key_index + 1} rate limited. Switching to next key...")
                
                # Switch to next API key
                current_api_key_index = (current_api_key_index + 1) % len(TAVILY_API_KEYS)
                tavily_client = TavilyClient(api_key=TAVILY_API_KEYS[current_api_key_index])
                
                print(f"   Now using API key {current_api_key_index + 1}/{len(TAVILY_API_KEYS)}")
                
                # If we've tried all keys, return error
                if attempt == len(TAVILY_API_KEYS) - 1:
                    print(f"❌ All {len(TAVILY_API_KEYS)} API keys exhausted!")
                    return {"answer": "", "results": [], "error": "All API keys rate limited"}
                
                # Continue to next iteration to retry with new key
                continue
            else:
                # Non-rate-limit error, return immediately
                print(f"⚠️  Search error: {str(e)}")
                return {"answer": "", "results": [], "error": str(e)}
    
    return {"answer": "", "results": [], "error": "All retries failed"}

@tool
def web_search(query: str) -> str:
    """Search the web for factual information."""
    data = tavily_search_raw(query)
    return json.dumps(data)

# LLM Creation with Caching
def create_llm(system_prompt: str = "", temperature: float = 0.1):
    """Creates and caches ChatOllama instances."""
    cache_key = (MODEL_NAME, temperature, system_prompt)
    if cache_key in llm_cache:
        return llm_cache[cache_key]
    
    llm = ChatOllama(
        model=MODEL_NAME,
        base_url=OLLAMA_BASE_URL,
        temperature=temperature,
        system=system_prompt,
    )
    llm_cache[cache_key] = llm
    return llm

# Agent Creation with Caching
def get_breakdown_agent():
    if "breakdown" in agent_cache:
        return agent_cache["breakdown"]
    llm = create_llm(system_prompt=BREAKDOWN_PROMPT)
    agent = create_react_agent(llm, [], response_format=BreakdownOutput, state_schema=AgentState)
    agent_cache["breakdown"] = agent
    return agent

def get_reasoning_agent():
    if "reasoning" in agent_cache:
        return agent_cache["reasoning"]
    llm = create_llm(system_prompt=REASONING_PROMPT)
    agent = create_react_agent(llm, [], response_format=ReasoningOutput, state_schema=AgentState)
    agent_cache["reasoning"] = agent
    return agent

def get_scientific_agent():
    if "scientific" in agent_cache:
        return agent_cache["scientific"]
    llm = create_llm(system_prompt=SCIENTIFIC_PROMPT)
    agent = create_react_agent(llm, [web_search], response_format=ScientificOutput, state_schema=AgentState)
    agent_cache["scientific"] = agent
    return agent

def get_synthesis_agent():
    if "synthesis" in agent_cache:
        return agent_cache["synthesis"]
    llm = create_llm(system_prompt=SYNTHESIS_PROMPT, temperature=0.2)
    agent = create_react_agent(llm, [], response_format=SynthesisOutput, state_schema=AgentState)
    agent_cache["synthesis"] = agent
    return agent

def get_creative_agent():
    if "creative" in agent_cache:
        return agent_cache["creative"]
    llm = create_llm(system_prompt=CREATIVE_PROMPT, temperature=0.3)
    agent = create_react_agent(llm, [], response_format=CreativeOutput, state_schema=AgentState)
    agent_cache["creative"] = agent
    return agent

# Node Functions
def breakdown_node(state):
    agent = get_breakdown_agent()
    messages = [HumanMessage(content=state["query"])]
    result = agent.invoke({"messages": messages})
    output = result["structured_response"]
    
    return {
        "breakdown_output": output.summary,
        "search_queries": output.search_queries,
        "reasoning_points": output.reasoning_points
    }

def reasoning_node(state):
    agent = get_reasoning_agent()
    reasoning_points = state.get("reasoning_points", [])
    
    context = f"""Original Query: {state.get('query', '')}

Reasoning Points to Analyze:
{chr(10).join(f'- {point}' for point in reasoning_points)}"""
    
    messages = [HumanMessage(content=context)]
    result = agent.invoke({"messages": messages})
    output = result["structured_response"]
    
    reasoning_text = "\n".join(output.reasoning_analysis + output.conclusions)
    
    return {
        "reasoning_output": reasoning_text
    }

def scientific_node(state):
    queries = state.get("search_queries", [])
    if not queries:
        queries = [state.get("query", "")]

    # Parallel web searches
    all_results = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_to_query = {executor.submit(tavily_search_raw, q): q for q in queries}
        for future in future_to_query:
            try:
                result = future.result(timeout=10)
                if result and result.get("results"):
                    all_results.extend(result["results"])
            except Exception:
                pass

    # Deduplicate and limit
    unique_results = {r['url']: r for r in all_results}.values()
    limited_results = list(unique_results)[:MAX_SOURCES]
    
    # Build context
    query_text = state.get('query', '')
    header = f"""Original Query: {query_text}

Extract exactly {FACTS_TARGET} scientific facts from these search results:

=== SEARCH RESULTS ===
"""
    
    sources_text = ""
    available_chars = MAX_CONTEXT_CHARS - len(header) - 200
    chars_used = 0
    
    for i, result in enumerate(limited_results, 1):
        content = result.get('content', '')[:MAX_CONTENT_PER_SOURCE]
        source_entry = f"Source {i}: {content}\n\n"
        
        if chars_used + len(source_entry) > available_chars:
            break
        
        sources_text += source_entry
        chars_used += len(source_entry)
    
    context = header + sources_text
    
    # Get facts
    llm = create_llm(system_prompt=SCIENTIFIC_PROMPT)
    structured_llm = llm.with_structured_output(ScientificOutput)
    response = structured_llm.invoke(context)

    return {
        "scientific_output": str(response),
        "extracted_facts": response.facts,
    }

def synthesis_node(state):
    reasoning_output = state.get("reasoning_output", "")
    extracted_facts = state.get("extracted_facts", [])
    query = state.get("query", "")
    breakdown_summary = state.get("breakdown_output", "")

    # Format facts
    facts_list = []
    for fact in extracted_facts:
        if isinstance(fact, dict):
            fact_text = fact.get("fact", "") or fact.get("text", "")
            if fact_text:
                facts_list.append(fact_text)
    
    facts_text = "\n".join([f"{i+1}. {f}" for i, f in enumerate(facts_list)])

    context = f"""Original Query: "{query}"

Breakdown: {breakdown_summary}

REASONING:
{reasoning_output}

FACTS:
{facts_text}"""
    
    llm = create_llm(system_prompt=SYNTHESIS_PROMPT, temperature=0.2)
    structured_llm = llm.with_structured_output(SynthesisOutput)
    response = structured_llm.invoke(context)

    return {
        "synthesis_strategy": response.synthesis_strategy,
        "final_points": response.final_points
    }

def creative_node(state):
    final_points = state.get("final_points", [])

    context = f"""Question: {state.get('query', '')}

Key Points:
{chr(10).join(f'{i+1}. {point}' for i, point in enumerate(final_points))}"""
    
    llm = create_llm(system_prompt=CREATIVE_PROMPT, temperature=0.3)
    structured_llm = llm.with_structured_output(CreativeOutput)
    result = structured_llm.invoke(context)
    
    return {
        "final_answer": result.final_answer
    }

# Graph Creation
def create_graph():
    """Creates and caches the workflow graph."""
    if "workflow" in graph_cache:
        return graph_cache["workflow"]

    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("breakdown", breakdown_node)
    workflow.add_node("reasoning", reasoning_node)
    workflow.add_node("scientific", scientific_node)
    workflow.add_node("synthesis", synthesis_node)
    workflow.add_node("creative", creative_node)
    
    # Sequential flow for GPU stability
    workflow.add_edge(START, "breakdown")
    workflow.add_edge("breakdown", "reasoning")
    workflow.add_edge("reasoning", "scientific")
    workflow.add_edge("scientific", "synthesis")
    workflow.add_edge("synthesis", "creative")
    workflow.add_edge("creative", END)
    
    app = workflow.compile()
    graph_cache["workflow"] = app
    return app

# Main Function
def answer_question(query: str) -> Dict[str, Any]:
    """
    Answer a question using the multi-agent system.
    
    Args:
        query: The question to answer
        
    Returns:
        Dictionary with the final answer and intermediate results
    """
    # Initialize state
    initial_state = {
        "query": query,
        "breakdown_output": "",
        "reasoning_output": "",
        "scientific_output": "",
        "final_answer": "",
        "messages": [],
        "remaining_steps": 10,
        "structured_response": None,
        "search_queries": [],
        "reasoning_points": [],
        "extracted_facts": [],
        "synthesis_strategy": "",
        "final_points": []
    }
    
    # Run the graph
    app = create_graph()
    result = app.invoke(initial_state)
    
    return result

# For compatibility with evaluation.py
def run_multi_agent_system(query: str, model_config=None) -> Dict[str, Any]:
    """
    Compatibility function for evaluation.py
    Ignores model_config since we only use llama3.2:1b
    """
    return answer_question(query)

# ============================================================================
# BATCH GENERATION FOR DATASET
# ============================================================================

def load_eli5_dataset():
    """Load ELI5 dataset from cache or download if needed."""
    import pickle
    from pathlib import Path
    from datasets import load_dataset
    import pandas as pd
    
    cache_file = Path("eli5_dataset_cache.pkl")
    
    if cache_file.exists():
        print(f"📂 Loading dataset from cache: {cache_file}")
        with open(cache_file, 'rb') as f:
            df = pickle.load(f)
        print(f"✅ Loaded {len(df)} questions from cache")
        return df
    
    print(f"📥 Downloading ELI5 dataset (first time only)...")
    dataset = load_dataset("sentence-transformers/eli5", split="train")
    
    df = pd.DataFrame({
        'query': dataset['question'],
        'answers': dataset['answer']
    })
    
    print(f"💾 Caching dataset to {cache_file}")
    with open(cache_file, 'wb') as f:
        pickle.dump(df, f)
    
    print(f"✅ Cached {len(df)} questions")
    return df

def generate_answers_batch(start_idx: int = 0, end_idx: Optional[int] = None, 
                          split_index: Optional[int] = None,
                          output_file: Optional[str] = None) -> str:
    """
    Generate answers for a batch of questions from ELI5 dataset and save to CSV.
    
    Args:
        start_idx: Start index for processing
        end_idx: End index (None = process till end)
        split_index: Which dataset split to use (0, 1, or 2 for 3-way split)
        output_file: Path to save CSV (auto-generated if None)
        
    Returns:
        Path to the generated CSV file
    """
    import pandas as pd
    import time
    from datetime import datetime
    from pathlib import Path
    from tqdm import tqdm
    
    # Load ELI5 dataset from cache
    df = load_eli5_dataset()
    print(f"  Total questions in dataset: {len(df)}")
    
    # Handle split
    if split_index is not None:
        num_splits = 3
        split_size = len(df) // num_splits
        start_idx = split_index * split_size
        end_idx = start_idx + split_size if split_index < num_splits - 1 else len(df)
        print(f"  Using split {split_index + 1}/{num_splits}: rows {start_idx} to {end_idx}")
    
    # Set end index
    end_idx = end_idx or len(df)
    df_subset = df.iloc[start_idx:end_idx].reset_index(drop=True)
    
    print(f"\n🚀 Starting answer generation")
    print(f"   Processing {len(df_subset)} questions (indices {start_idx} to {end_idx})")
    print(f"   Model: {MODEL_NAME}\n")
    
    # Prepare output file - use consistent name for resume capability
    if output_file is None:
        output_dir = Path("generated_answers")
        output_dir.mkdir(exist_ok=True)
        
        if split_index is not None:
            output_file = output_dir / f"answers_split{split_index}.csv"
        else:
            output_file = output_dir / f"answers_{start_idx}_{end_idx}.csv"
    
    output_file = Path(output_file)
    
    # Check for existing results to resume
    results = []
    processed_count = 0
    
    if output_file.exists():
        print(f"📂 Found existing file: {output_file}")
        existing_df = pd.read_csv(output_file)
        results = existing_df.to_dict('records')
        processed_count = len(results)
        print(f"   Resuming from question {processed_count + 1}\n")
    
    # Process questions
    save_interval = 1  # Save after every question for incremental safety
    
    for idx in tqdm(range(processed_count, len(df_subset)), desc="Generating answers"):
        row = df_subset.iloc[idx]
        question = row['query']
        reference_answers = row['answers'] if isinstance(row['answers'], list) else [row['answers']]
        
        try:
            # Generate answer
            start_time = time.time()
            result = answer_question(question)
            generation_time = time.time() - start_time
            
            generated_answer = result.get('final_answer', 'No answer generated')
            
            # Store result
            results.append({
                'question_id': start_idx + idx,
                'question': question,
                'generated_answer': generated_answer,
                'reference_answers': '|||'.join(reference_answers),  # Join with delimiter
                'generation_time': generation_time,
                'status': 'success',
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            print(f"\n❌ Error processing question {idx}: {str(e)}")
            results.append({
                'question_id': start_idx + idx,
                'question': question,
                'generated_answer': '',
                'reference_answers': '|||'.join(reference_answers),
                'generation_time': 0,
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
        
        # Save after every question for incremental safety
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_file, index=False)
    
    print(f"\n🎉 Generation complete!")
    print(f"   Total questions processed: {len(results)}")
    print(f"   Successful: {sum(1 for r in results if r['status'] == 'success')}")
    print(f"   Failed: {sum(1 for r in results if r['status'] == 'error')}")
    print(f"   Results saved to: {output_file}")
    
    return str(output_file)


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import sys
    import argparse
    from typing import Optional
    
    parser = argparse.ArgumentParser(
        description="Multi-Agent ELI5 System - Answer questions or generate batch answers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Answer a single question
  python simple_agent.py "Why is the sky blue?"
  
  # Generate answers for split 0 (first third of dataset)
  python simple_agent.py --batch --split 0
  
  # Generate answers for split 1 (second third)
  python simple_agent.py --batch --split 1
  
  # Generate answers for specific range
  python simple_agent.py --batch --start 0 --end 1000
  
  # Generate with custom output file
  python simple_agent.py --batch --split 2 --output my_answers.csv
        """
    )
    
    parser.add_argument('question', nargs='*', help='Question to answer (for single mode)')
    parser.add_argument('--batch', action='store_true', help='Batch generation mode')
    parser.add_argument('--split', type=int, choices=[0, 1, 2], 
                       help='Dataset split to process (0, 1, or 2)')
    parser.add_argument('--start', type=int, default=0, 
                       help='Start index (default: 0)')
    parser.add_argument('--end', type=int, 
                       help='End index (default: end of split/dataset)')
    parser.add_argument('--output', type=str, 
                       help='Output CSV file path')
    
    args = parser.parse_args()
    
    if args.batch:
        # Batch generation mode
        generate_answers_batch(
            start_idx=args.start,
            end_idx=args.end,
            split_index=args.split,
            output_file=args.output
        )
    else:
        # Single question mode
        if args.question:
            question = " ".join(args.question)
        else:
            question = input("Enter your question: ")
        
        print(f"\n🔍 Question: {question}\n")
        
        result = answer_question(question)
        
        print(f"✅ Answer: {result['final_answer']}\n")
