import os
import pickle
import time
from pathlib import Path
from typing import TypedDict, Dict, List, Any, Annotated, Optional, Literal
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import create_react_agent
from langgraph.graph.message import add_messages
import wikipediaapi
from datasets import load_dataset
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor
import json
import numpy as np

# Load environment variables
load_dotenv()

# Model Configuration - Single model for all agents
MODEL_NAME = "llama3.2:1b"
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_NUM_THREADS = 8  # CPU threads for Ollama
OLLAMA_REQUEST_TIMEOUT = 120  # Timeout for LLM requests

# RAG Configuration
RAG_DATASET_NAME = "open-thoughts/OpenThoughts-114k"
RAG_CACHE_DIR = "./rag_cache"

# Search Configuration
MAX_SOURCES = 5 # Reduced since we have high quality RAG/Wiki
FACTS_TARGET = 12
MAX_CONTEXT_CHARS = 3900
MAX_CONTENT_PER_SOURCE = 1000

# Performance Tracking
ENABLE_TIMING = True  # Set to False to disable timing logs

def log_time(message: str, start_time: float):
    """Log execution time if timing is enabled."""
    if ENABLE_TIMING:
        elapsed = time.time() - start_time
        print(f"⏱️  {message}: {elapsed:.2f}s")

# Caching
llm_cache = {}
graph_cache = {}
agent_cache = {}
rag_cache = {"dataset": None, "bm25": None, "corpus": None}

def clear_all_caches():
    """Clear all caches to reset state."""
    global llm_cache, graph_cache, agent_cache, rag_cache
    llm_cache.clear()
    graph_cache.clear()
    agent_cache.clear()
    rag_cache = {"dataset": None, "bm25": None, "corpus": None}

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
    synthesis_strategy: Literal["reasoning_heavy", "facts_heavy", "balanced"]
    final_points: List[str]

class CreativeOutput(BaseModel):
    final_answer: str

BREAKDOWN_PROMPT = """You are the Breakdown Agent. Your job is to decompose the user's question into actionable queries for both reasoning and scientific research.

Output format:
1. Brief summary of the question
2. SEARCH_QUERIES: 3-5 targeted search queries for scientific fact gathering (mechanisms, definitions, measurable details). Identify if they are better for "WIKIPEDIA" (definitions, basic facts) or "RAG" (complex reasoning, specific scientific cases).
3. REASONING_POINTS: 3-5 logical aspects to analyze (cause-effect, relationships, processes)

Rules:
- Focus on mechanisms, units, entities, and measurable phenomena
- Make search queries specific and scientific (avoid broad terms)
- Make reasoning points focused on logical analysis and connections
- Do not invent facts; provide structure for investigation only

Example format:
Summary: [brief question summary]

SEARCH_QUERIES:
- [WIKIPEDIA] mechanism of action
- [RAG] examples of similar chemical reactions
- [WIKIPEDIA] definition of thermodynamics

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

SCIENTIFIC_PROMPT = f"""You are the Science/Factual Agent. Your goal is to gather accurate facts using Wikipedia and a specialized OpenThoughts RAG dataset.

Instructions:
- Analyze the search queries provided.
- Use `wikipedia_search` for definitions, general scientific concepts, and established facts.
- Use `rag_search` for complex reasoning traces, specific examples, and deeper scientific relationships found in the dataset.
- Aggregate credible information from these tools.
- Extract around {FACTS_TARGET} comprehensive scientific facts.

You must respond with structured output containing:
- facts: Array of fact objects with "fact" field (brief fact statement) and "text" field (detailed description) - EXACTLY {FACTS_TARGET} facts
"""

SYNTHESIS_PROMPT = """You are a synthesis expert preparing material for an ELI5 (Explain Like I'm 5) explanation.

Your task:
1. **Evaluate Quality**: Assess both reasoning and facts for relevance and reliability
   - Are facts concrete, accurate, and directly answer the query?
   - Is reasoning logically sound and helpful for understanding?
   - Which source provides better foundational understanding?

2. **Determine Strategy** (MUST be one of these exact values):
   - "reasoning_heavy": Facts are weak/irrelevant -> Use 70% reasoning + 30% facts
   - "facts_heavy": Facts are excellent and comprehensive -> Use 70% facts + 30% reasoning  
   - "balanced": Both are good quality -> Mix 50-50 reasoning and facts

3. **Curate Final Points** (aim for 4-6 points, NOT more):
   - Select ONLY the MOST RELEVANT points that answer the original query
   - Each point should be concise (1-2 sentences max)
   - No redundancy - each point must add unique value
   - Order points logically (basic concepts -> mechanisms -> implications)

Output a strategy (exactly one of: reasoning_heavy, facts_heavy, balanced) and 4-6 curated points."""

CREATIVE_PROMPT = """You are an ELI5 (Explain Like I'm 5) expert. Your job is to take curated points and explain them in simple language a 5-year-old would understand.

Your rules:
- Use ALL the points provided (don't skip any)
- Use simple everyday words that a 5-year-old would understand
- Use fun comparisons (like toys, games, animals, things kids know)
- Keep it SHORT and CONCISE - aim for 3-5 sentences per point
- Total answer should be 400-600 characters MAX
- Do NOT mention "ELI5" explicitly in your answer
- **OUTPUT FORMAT**: You must return a JSON object with a single field "final_answer".
- **IMPORTANT**: The ENTIRE story/explanation must go into the "final_answer" string. Do not put a summary there. Put the full text there.

Write a brief, engaging explanation covering all points concisely."""

# ============================================================================
# TOOLS
# ============================================================================

from sentence_transformers import SentenceTransformer, util

def get_rag_resources():
    """Lazy load RAG resources (Dataset, BM25, Semantic Model) with disk caching."""
    global rag_cache
    if rag_cache["dataset"] is not None and rag_cache["bm25"] is not None and rag_cache.get("model") is not None:
        return rag_cache["dataset"], rag_cache["bm25"], rag_cache["corpus"], rag_cache["model"], rag_cache["embeddings"]

    cache_dir = Path(RAG_CACHE_DIR)
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / "rag_resources_hybrid.pkl"

    if cache_file.exists():
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
                rag_cache["dataset"] = data["dataset"]
                rag_cache["bm25"] = data["bm25"]
                rag_cache["corpus"] = data["corpus"]
                rag_cache["embeddings"] = data["embeddings"]
                rag_cache["model"] = SentenceTransformer('all-MiniLM-L6-v2') 
            return rag_cache["dataset"], rag_cache["bm25"], rag_cache["corpus"], rag_cache["model"], rag_cache["embeddings"]
        except Exception as e:
            print(f"⚠️ Cache load failed ({e}), rebuilding...")

    try:
        ds = load_dataset(RAG_DATASET_NAME, "metadata", split="train")
        
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Prepare corpus
        corpus = []
        for item in ds:
            text_parts = []
            if item.get('problem'): text_parts.append(str(item['problem']))
            if item.get('deepseek_reasoning'): text_parts.append(str(item['deepseek_reasoning']))
            solution = item.get('ground_truth_solution') or item.get('deepseek_solution')
            if solution: text_parts.append(str(solution))
            corpus.append(" ".join(text_parts))
            
        # BM25
        tokenized_corpus = [doc.split(" ") for doc in corpus]
        bm25 = BM25Okapi(tokenized_corpus)
        
        # Semantic Embeddings
        corpus_embeddings = model.encode(corpus, batch_size=32, show_progress_bar=True, convert_to_tensor=True)
        
        rag_cache["dataset"] = ds
        rag_cache["bm25"] = bm25
        rag_cache["corpus"] = corpus
        rag_cache["model"] = model
        rag_cache["embeddings"] = corpus_embeddings
        
        # Save to disk (Exclude model object, it's not pickleable easily across versions, save embeddings)
        with open(cache_file, 'wb') as f:
            pickle.dump({
                "dataset": ds,
                "bm25": bm25,
                "corpus": corpus,
                "embeddings": corpus_embeddings.cpu().numpy() # Save as numpy to be safe
            }, f)
            
        return ds, bm25, corpus, model, corpus_embeddings
        
    except Exception as e:
        print(f"❌ Error loading RAG resources: {e}")
        return None, None, None, None, None

@tool
def rag_search(query: str) -> str:
    """
    Search the OpenThoughts dataset using Hybrid Search (BM25 + Semantic).
    Finds relevant reasoning traces and scientific info.
    """
    ds, bm25, corpus, model, embeddings = get_rag_resources()
    if not ds or not bm25:
        return "RAG system unavailable."

    # 1. BM25 Search (Keyword match)
    tokenized_query = query.split(" ")
    bm25_scores = bm25.get_scores(tokenized_query)
    
    # 2. Semantic Search
    query_embedding = model.encode(query, convert_to_tensor=True)
    # Ensure embeddings are tensor on the SAME DEVICE as query embedding
    if not isinstance(embeddings, type(query_embedding)):
        import torch
        embeddings = torch.tensor(embeddings)
    # Move embeddings to same device as query (fixes CUDA/CPU mismatch)
    embeddings = embeddings.to(query_embedding.device)
        
    semantic_scores = util.cos_sim(query_embedding, embeddings)[0]
    
    # 3. Hybrid Merge
    top_k_bm25 = np.argsort(bm25_scores)[-5:][::-1]
    top_k_semantic = np.argsort(semantic_scores.cpu().numpy())[-5:][::-1]
    
    combined_indices = list(set(top_k_bm25) | set(top_k_semantic))
    
    results = []
    for idx in combined_indices:
        item = ds[int(idx)]
        
        final_solution = item.get('ground_truth_solution')
        if not final_solution:
            final_solution = item.get('deepseek_solution')
            
        content = ""
        if item.get('problem'): content += f"Problem: {item['problem']}\n"
        if item.get('deepseek_reasoning'): content += f"Reasoning: {item['deepseek_reasoning'][:1500]}...\n"
        if final_solution: content += f"Solution: {final_solution[:500]}...\n"
            
        results.append(content)
    
    return "\n---\n".join(results[:5])

def batch_rag_search(queries: List[str]) -> Dict[str, str]:
    """
    Optimized batch RAG search that encodes all queries at once.
    Returns a dictionary mapping query -> results.
    """
    ds, bm25, corpus, model, embeddings = get_rag_resources()
    if not ds or not bm25:
        return {q: "RAG system unavailable." for q in queries}
    
    # Batch encode all queries at once (MAJOR OPTIMIZATION)
    query_embeddings = model.encode(queries, convert_to_tensor=True, batch_size=len(queries))
    
    # Ensure embeddings compatibility
    if not isinstance(embeddings, type(query_embeddings)):
        import torch
        embeddings = torch.tensor(embeddings)
    embeddings = embeddings.to(query_embeddings.device)
    
    results_dict = {}
    
    for idx, query in enumerate(queries):
        # 1. BM25 Search
        tokenized_query = query.split(" ")
        bm25_scores = bm25.get_scores(tokenized_query)
        
        # 2. Semantic Search (use pre-computed batch embedding)
        query_embedding = query_embeddings[idx]
        semantic_scores = util.cos_sim(query_embedding, embeddings)[0]
        
        # 3. Hybrid Merge
        top_k_bm25 = np.argsort(bm25_scores)[-5:][::-1]
        top_k_semantic = np.argsort(semantic_scores.cpu().numpy())[-5:][::-1]
        
        combined_indices = list(set(top_k_bm25) | set(top_k_semantic))
        
        results = []
        for doc_idx in combined_indices:
            item = ds[int(doc_idx)]
            
            final_solution = item.get('ground_truth_solution')
            if not final_solution:
                final_solution = item.get('deepseek_solution')
                
            content = ""
            if item.get('problem'): content += f"Problem: {item['problem']}\n"
            if item.get('deepseek_reasoning'): content += f"Reasoning: {item['deepseek_reasoning'][:1500]}...\n"
            if final_solution: content += f"Solution: {final_solution[:500]}...\n"
                
            results.append(content)
        
        results_dict[query] = "\n---\n".join(results[:5])
    
    return results_dict

# Update Prompt with Chain-of-Thought
SCIENTIFIC_PROMPT = f"""You are the Science/Factual Agent. Your goal is to gather accurate facts using Wikipedia and a specialized OpenThoughts RAG dataset.

Instructions:
1. **Analyze** the search queries provided -> THINK step-by-step about which tool fits best.
2. **Wikipedia**: Use for definitions, general scientific concepts, and established facts.
3. **RAG (Hybrid Search)**: Use for complex reasoning traces, specific examples, and deeper logical connections.
4. **Cite Sources**: When extracting facts, you MUST explicitly state the source type (e.g., "[Wikipedia] The mitochondria..." or "[RAG] In a similar problem...").

Goal: Extract exactly {FACTS_TARGET} comprehensive scientific facts.

You must respond with structured output containing:
- facts: Array of fact objects with "fact" field (brief fact statement with citation) and "text" field (detailed description)
"""

@tool
def wikipedia_search(query: str) -> str:
    """
    Search Wikipedia for a summary of the query.
    Useful for definitions, basic facts, and general knowledge.
    """
    wiki_wiki = wikipediaapi.Wikipedia('EduTech-Agent/1.0', 'en')
    page = wiki_wiki.page(query)
    
    if page.exists():
        return f"Wait, I found a page for '{query}'.\nSummary: {page.summary[:2000]}"
    else:
        return f"Page '{query}' not found on Wikipedia."



# LLM Creation with Caching
def create_llm(system_prompt: str = "", temperature: float = 0.1):
    """Creates and caches ChatOllama instances with optimized settings."""
    cache_key = (MODEL_NAME, temperature, system_prompt)
    if cache_key in llm_cache:
        return llm_cache[cache_key]
    
    llm = ChatOllama(
        model=MODEL_NAME,
        base_url=OLLAMA_BASE_URL,
        temperature=temperature,
        system=system_prompt,
        num_thread=OLLAMA_NUM_THREADS,
        timeout=OLLAMA_REQUEST_TIMEOUT,
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

def creative_node(state):
    final_points = state.get("final_points", [])
    
    context = f"""Question: {state.get('query', '')}

Key Points to Explain:
{chr(10).join(f'{i+1}. {point}' for i, point in enumerate(final_points))}

Please write the ELI5 explanation now based on these points.
Remember: JSON output only. The full story goes in 'final_answer'."""
    
    # Use structured output directly for stricter enforcement
    llm = create_llm(system_prompt=CREATIVE_PROMPT, temperature=0.5)
    structured_llm = llm.with_structured_output(CreativeOutput)
    
    try:
        response = structured_llm.invoke(context)
        final_answer = response.final_answer
    except Exception as e:
        final_answer = "Sorry, I couldn't generate an explanation at this time."

    return {
        "final_answer": final_answer
    }

# Node Functions
def breakdown_node(state):
    start = time.time()
    agent = get_breakdown_agent()
    messages = [HumanMessage(content=state["query"])]
    result = agent.invoke({"messages": messages})
    output = result["structured_response"]
    log_time("Breakdown node", start)
    
    return {
        "breakdown_output": output.summary,
        "search_queries": output.search_queries,
        "reasoning_points": output.reasoning_points
    }

def reasoning_node(state):
    reasoning_points = state.get("reasoning_points", [])
    
    agent = get_reasoning_agent()
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
    search_queries = state.get("search_queries", [])
    if not search_queries:
        search_queries = [state.get("query", "")]
    
    # ============================================================
    # MANDATORY: Always search the knowledge bases first
    # OPTIMIZED: Batch RAG encoding + parallel Wikipedia searches
    # ============================================================
    
    all_retrieved_context = []
    
    # Prepare search tasks
    rag_queries = search_queries[:3]  # Limit to 3 queries
    wiki_queries = search_queries[:2]  # Limit to 2 queries
    
    # OPTIMIZED: Batch RAG search (encode all queries at once - MAJOR SPEEDUP)
    try:
        rag_results = batch_rag_search(rag_queries)
        for query, result in rag_results.items():
            if result and "unavailable" not in result.lower():
                all_retrieved_context.append(f"[RAG Results for '{query}']:\n{result}")
    except Exception as e:
        print(f"   ⚠️ Batch RAG search failed: {e}")
        # Fallback to individual searches if batch fails
        for query in rag_queries:
            try:
                result = rag_search.invoke(query)
                if result and "unavailable" not in result.lower():
                    all_retrieved_context.append(f"[RAG Results for '{query}']:\n{result}")
            except Exception as e2:
                print(f"   ⚠️ RAG search failed for '{query}': {e2}")
    
    # Wikipedia searches in parallel
    def search_wiki(query):
        try:
            result = wikipedia_search.invoke(query)
            if result and "not found" not in result.lower():
                return (query, result)
        except Exception as e:
            print(f"   ⚠️ Wikipedia search failed for '{query}': {e}")
        return None
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        wiki_futures = [executor.submit(search_wiki, q) for q in wiki_queries]
        
        for future in wiki_futures:
            result = future.result()
            if result:
                query, content = result
                all_retrieved_context.append(f"[Wikipedia for '{query}']:\n{content}")
    
    # Combine all retrieved context
    combined_context = "\n\n---\n\n".join(all_retrieved_context)
    
    # ============================================================
    # Now use LLM to extract facts FROM the retrieved context
    # ============================================================
    
    extraction_prompt = f"""You are a fact extraction expert. Your job is to extract scientific facts from the provided knowledge base context.

RETRIEVED KNOWLEDGE BASE CONTEXT:
{combined_context[:8000]}

ORIGINAL QUESTION: {state.get('query', '')}

Instructions:
- Extract {FACTS_TARGET} facts ONLY from the context above
- Each fact must be grounded in the retrieved context
- Include citations like [RAG] or [Wikipedia] for each fact
- If context is insufficient, extract what you can and note the gap

Respond with structured output containing:
- facts: Array of fact objects with "fact" (brief statement with citation) and "text" (detailed description)
"""
    
    llm = create_llm(system_prompt="You are a fact extraction expert.", temperature=0.1)
    structured_llm = llm.with_structured_output(ScientificOutput)
    
    try:
        output = structured_llm.invoke(extraction_prompt)
    except Exception as e:
        output = ScientificOutput(facts=[{"fact": "Extraction failed", "text": str(e)}])

    return {
        "scientific_output": str(output),
        "extracted_facts": output.facts,
    }

def parallel_analysis_node(state):
    """
    Execute reasoning and scientific analysis in parallel for 2x speedup.
    """
    start = time.time()
    with ThreadPoolExecutor(max_workers=2) as executor:
        # Submit both nodes concurrently
        reasoning_future = executor.submit(reasoning_node, state)
        scientific_future = executor.submit(scientific_node, state)
        
        # Wait for both to complete
        reasoning_result = reasoning_future.result()
        scientific_result = scientific_future.result()
    
    log_time("Parallel analysis (reasoning + scientific)", start)
    
    # Merge results
    merged_state = {}
    merged_state.update(reasoning_result)
    merged_state.update(scientific_result)
    
    return merged_state

def synthesis_node(state):
    reasoning_output = state.get("reasoning_output", "")
    extracted_facts = state.get("extracted_facts", [])
    query = state.get("query", "")
    breakdown_summary = state.get("breakdown_output", "")
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


# Graph Creation
def create_graph():
    """Creates and caches the workflow graph with parallel execution."""
    if "workflow" in graph_cache:
        return graph_cache["workflow"]

    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("breakdown", breakdown_node)
    workflow.add_node("parallel_analysis", parallel_analysis_node)  # Parallel reasoning + scientific
    workflow.add_node("synthesis", synthesis_node)
    workflow.add_node("creative", creative_node)
    
    # Optimized flow: breakdown -> parallel analysis -> synthesis -> creative
    workflow.add_edge(START, "breakdown")
    workflow.add_edge("breakdown", "parallel_analysis")
    workflow.add_edge("parallel_analysis", "synthesis")
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
        with open(cache_file, 'rb') as f:
            df = pickle.load(f)
        return df
    dataset = load_dataset("sentence-transformers/eli5", split="train")
    
    df = pd.DataFrame({
        'query': dataset['question'],
        'answers': dataset['answer']
    })
    with open(cache_file, 'wb') as f:
        pickle.dump(df, f)
    return df

def generate_answers_batch(start_idx: int = 0, end_idx: Optional[int] = None, 
                          split_index: Optional[int] = None,
                          output_file: Optional[str] = None) -> str:
    """
    Generate answers for a batch of questions from ELI5 dataset and save to CSV.
    Uses PARALLEL PROCESSING (3 workers) to maximize Ollama throughput.
    """
    import pandas as pd
    import time
    from datetime import datetime
    from pathlib import Path
    from tqdm import tqdm
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    # Load ELI5 dataset from cache
    df = load_eli5_dataset()
    
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
    
    print(f"\n🚀 Starting PARALLEL answer generation")
    print(f"   Processing {len(df_subset)} questions (indices {start_idx} to {end_idx})")
    print(f"   Model: {MODEL_NAME}")
    print(f"   Concurrency: 3 workers\n")
    
    # Prepare output file
    if output_file is None:
        output_dir = Path("generated_answers")
        output_dir.mkdir(exist_ok=True)
        if split_index is not None:
            output_file = output_dir / f"answers_split{split_index}.csv"
        else:
            output_file = output_dir / f"answers_{start_idx}_{end_idx}.csv"
    
    output_file = Path(output_file)
    
    # Thread-safe data collection
    results = []
    processed_count = 0
    file_lock = threading.Lock()
    
    # Resume capability
    if output_file.exists():
        print(f"📂 Found existing file: {output_file}")
        try:
            existing_df = pd.read_csv(output_file)
            results = existing_df.to_dict('records')
            processed_count = len(results)
            print(f"   Resuming from {processed_count} existing records...")
        except Exception as e:
            print(f"   ⚠️ Could not read existing file: {e}. Starting fresh.")
    
    # Helper to check if a question index is already done
    # For simplicity in this resume logic, we assume sequential processing was intended
    # so we skip the first 'processed_count' items.
    questions_to_process = []
    
    for idx in range(processed_count, len(df_subset)):
        row = df_subset.iloc[idx]
        questions_to_process.append({
            'local_idx': idx,
            'global_id': start_idx + idx,
            'question': row['query'],
            'reference_answers': row['answers'] if isinstance(row['answers'], list) else [row['answers']]
        })

    if not questions_to_process:
        print("   All questions already processed!")
        return str(output_file)

    def process_single_question(item):
        """Worker function for a single question."""
        q_text = item['question']
        ref_answers = item['reference_answers']
        
        start_time = time.time()
        try:
            # Run the agent
            result = answer_question(q_text)
            gen_time = time.time() - start_time
            gen_answer = result.get('final_answer', 'No answer generated')
            status = 'success'
            error_msg = ''
        except Exception as e:
            gen_time = 0
            gen_answer = ''
            status = 'error'
            error_msg = str(e)
            # print(f"\n❌ Error on QID {item['global_id']}: {e}") # Reduce noise in parallel

        # Return structured result
        return {
            'question_id': item['global_id'],
            'question': q_text,
            'generated_answer': gen_answer,
            'reference_answers': '|||'.join(ref_answers),
            'generation_time': gen_time,
            'status': status,
            'error': error_msg,
            'timestamp': datetime.now().isoformat()
        }

    # Run with ThreadPoolExecutor
    print(f"   Spawning 3 worker threads...")
    with ThreadPoolExecutor(max_workers=3) as executor:
        # Submit all tasks
        futures = [executor.submit(process_single_question, item) for item in questions_to_process]
        
        # Process as they complete
        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating"):
            res_data = future.result()
            
            # Critical Section: Write to file
            with file_lock:
                results.append(res_data)
                # Incremental save (overwriting file with updated full list)
                pd.DataFrame(results).to_csv(output_file, index=False)
                
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
        
        result = answer_question(question)
        
        print(f"✅ Answer: {result['final_answer']}\n")
