"""
Multi-Agent ELI5 System - vLLM Optimised
=========================================

Identical workflow to arch_1/simple_agent.py:
  breakdown -> parallel_analysis (reasoning + scientific) -> synthesis -> creative

Performance changes only:
  - ChatOllama replaced by vLLM offline engine (LLM.chat + StructuredOutputsParams decoding)
  - Batch mode uses STAGED BATCHING:
      All N questions pass through each stage as a single batched vLLM call.
      This turns N*5 sequential LLM calls into 5 large batch calls.
      Retrieval (RAG + Wikipedia) is parallelised with ThreadPoolExecutor.

Usage (single question):
    python vllm/simple_agent_vllm.py "Why is the sky blue?"

Usage (batch):
    python vllm/simple_agent_vllm.py --batch --start 0 --end 3000
    python vllm/simple_agent_vllm.py --batch --split 0 --chunk-size 100
"""

import os, json, pickle, time, threading, csv, gc, atexit
from datetime import datetime
from pathlib import Path
from typing import TypedDict, Dict, List, Any, Annotated, Optional, Literal
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm
from pydantic import BaseModel
from dotenv import load_dotenv

from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

import wikipediaapi
from datasets import load_dataset
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages

load_dotenv()

# ============================================================================
# CONFIGURATION  (mirrors simple_agent.py)
# ============================================================================

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

MODEL_NAME_ALIASES: Dict[str, str] = {
    "llama3.2:1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama3.2:3b": "meta-llama/Llama-3.2-3B-Instruct",
    "mistral:7b":  "mistralai/Mistral-7B-Instruct-v0.3",
    "qwen2.5:3b":  "Qwen/Qwen2.5-3B-Instruct",
    "qwen2.5:7b":  "Qwen/Qwen2.5-7B-Instruct",
    "gemma2:2b":   "google/gemma-2-2b-it",
    "gemma2:9b":   "google/gemma-2-9b-it",
}

GPU_MEMORY_UTILIZATION = 0.85
MAX_MODEL_LEN          = 4096
TENSOR_PARALLEL_SIZE   = 1
MAX_NUM_SEQS           = 256

RAG_DATASET_NAME      = "open-thoughts/OpenThoughts-114k"
RAG_CACHE_DIR         = "./rag_cache"
FACTS_TARGET          = 12
ENABLE_TIMING         = True


def log_time(msg: str, t0: float) -> None:
    if ENABLE_TIMING:
        print(f"⏱️  {msg}: {time.time() - t0:.2f}s")


def resolve_model(name: str) -> str:
    s = name.strip()
    if Path(s).exists():
        return s
    return MODEL_NAME_ALIASES.get(s.lower(), s)


# ============================================================================
# CACHES
# ============================================================================

_vllm_engine: Optional[LLM]    = None
graph_cache:  Dict[str, Any]   = {}
rag_cache:    Dict[str, Any]   = {"dataset": None, "bm25": None, "corpus": None}


def clear_all_caches() -> None:
    global graph_cache, rag_cache
    shutdown_vllm_engine()
    graph_cache.clear()
    rag_cache = {"dataset": None, "bm25": None, "corpus": None}


def get_vllm_engine() -> LLM:
    global _vllm_engine
    if _vllm_engine is None:
        resolved = resolve_model(MODEL_NAME)
        print(f"🚀 Initialising vLLM engine: {resolved}")
        _vllm_engine = LLM(
            model=resolved,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            max_model_len=MAX_MODEL_LEN,
            tensor_parallel_size=TENSOR_PARALLEL_SIZE,
            max_num_seqs=MAX_NUM_SEQS,
        )
    return _vllm_engine


def shutdown_vllm_engine() -> None:
    """Best-effort cleanup for vLLM and torch distributed/CUDA resources."""
    global _vllm_engine

    engine = _vllm_engine
    _vllm_engine = None
    if engine is None:
        return

    # Try public/known shutdown hooks if present.
    try:
        llm_engine = getattr(engine, "llm_engine", None)
        if llm_engine is not None and hasattr(llm_engine, "shutdown"):
            llm_engine.shutdown()
    except Exception:
        pass

    try:
        if hasattr(engine, "shutdown"):
            engine.shutdown()
    except Exception:
        pass

    try:
        import torch

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:
        pass

    del engine
    gc.collect()


atexit.register(shutdown_vllm_engine)


# ============================================================================
# STATE & SCHEMAS  (identical to simple_agent.py)
# ============================================================================

class AgentState(TypedDict):
    query:               str
    breakdown_output:    str
    reasoning_output:    str
    scientific_output:   str
    final_answer:        str
    messages:            Annotated[list, add_messages]
    remaining_steps:     int
    structured_response: Any
    search_queries:      List[str]
    reasoning_points:    List[str]
    extracted_facts:     List[Dict[str, Any]]
    synthesis_strategy:  str
    final_points:        List[str]


class BreakdownOutput(BaseModel):
    summary:          str
    search_queries:   List[str]
    reasoning_points: List[str]


class ReasoningOutput(BaseModel):
    reasoning_analysis: List[str]
    conclusions:        List[str]


class ScientificOutput(BaseModel):
    facts: List[Dict[str, str]]


class SynthesisOutput(BaseModel):
    synthesis_strategy: Literal["reasoning_heavy", "facts_heavy", "balanced"]
    final_points:       List[str]


class CreativeOutput(BaseModel):
    final_answer: str


# ============================================================================
# PROMPTS  (identical to simple_agent.py)
# ============================================================================

BREAKDOWN_PROMPT = """You are the Breakdown Agent. Your job is to decompose the user's question into actionable queries for both reasoning and scientific research.

Output format:
1. Brief summary of the question
2. SEARCH_QUERIES: 3-5 targeted search queries for scientific fact gathering (mechanisms, definitions, measurable details). Identify if they are better for "WIKIPEDIA" (definitions, basic facts) or "RAG" (complex reasoning, specific scientific cases).
3. REASONING_POINTS: 3-5 logical aspects to analyze (cause-effect, relationships, processes)

Rules:
- Focus on mechanisms, units, entities, and measurable phenomena
- Make search queries specific and scientific
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

SCIENTIFIC_EXTRACTION_PROMPT = f"""You are a fact extraction expert. Your job is to extract scientific facts from the provided knowledge base context.

Instructions:
- Extract {FACTS_TARGET} facts ONLY from the context above
- Each fact must be grounded in the retrieved context
- Include citations like [RAG] or [Wikipedia] for each fact
- If context is insufficient, extract what you can and note the gap

Respond with structured output containing:
- facts: Array of fact objects with "fact" (brief statement with citation) and "text" (detailed description)
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
# vLLM GENERATION HELPER
# ============================================================================

def vllm_generate_structured(
    conversations: List[List[dict]],
    pydantic_model: type,
    temperature: float = 0.1,
    max_tokens: int    = 800,
) -> List[Optional[Any]]:
    """
    Batch structured generation via vLLM.chat() + StructuredOutputsParams decoding.
    Accepts a list of conversations; returns parsed Pydantic objects (None on failure).
    """
    engine   = get_vllm_engine()
    schema   = pydantic_model.model_json_schema()
    sampling = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        structured_outputs=StructuredOutputsParams(json=schema),
    )
    outputs  = engine.chat(conversations, sampling_params=sampling)

    results: List[Optional[Any]] = []
    for out in outputs:
        raw = out.outputs[0].text.strip()
        obj = None
        try:
            obj = pydantic_model.model_validate_json(raw)
        except Exception:
            try:
                s, e = raw.find("{"), raw.rfind("}") + 1
                if s != -1 and e > s:
                    obj = pydantic_model.model_validate_json(raw[s:e])
            except Exception:
                obj = None
        results.append(obj)
    return results


# ============================================================================
# RAG & WIKIPEDIA  (identical logic to simple_agent.py)
# ============================================================================

def get_rag_resources():
    global rag_cache
    if (rag_cache["dataset"] is not None
            and rag_cache["bm25"] is not None
            and rag_cache.get("model") is not None):
        return (rag_cache["dataset"], rag_cache["bm25"], rag_cache["corpus"],
                rag_cache["model"], rag_cache["embeddings"])

    cache_dir  = Path(RAG_CACHE_DIR)
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / "rag_resources_hybrid.pkl"

    if cache_file.exists():
        try:
            with open(cache_file, "rb") as fh:
                data = pickle.load(fh)
            rag_cache.update({"dataset": data["dataset"], "bm25": data["bm25"],
                              "corpus": data["corpus"], "embeddings": data["embeddings"],
                              "model": SentenceTransformer("all-MiniLM-L6-v2")})
            return (rag_cache["dataset"], rag_cache["bm25"], rag_cache["corpus"],
                    rag_cache["model"], rag_cache["embeddings"])
        except Exception as exc:
            print(f"⚠️ RAG cache load failed ({exc}), rebuilding…")

    try:
        ds    = load_dataset(RAG_DATASET_NAME, "metadata", split="train")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        corpus = []
        for item in ds:
            parts = []
            if item.get("problem"):            parts.append(str(item["problem"]))
            if item.get("deepseek_reasoning"): parts.append(str(item["deepseek_reasoning"]))
            sol = item.get("ground_truth_solution") or item.get("deepseek_solution")
            if sol: parts.append(str(sol))
            corpus.append(" ".join(parts))
        bm25       = BM25Okapi([doc.split() for doc in corpus])
        embeddings = model.encode(corpus, batch_size=32, show_progress_bar=True, convert_to_tensor=True)
        rag_cache.update({"dataset": ds, "bm25": bm25, "corpus": corpus,
                          "model": model, "embeddings": embeddings})
        with open(cache_file, "wb") as fh:
            pickle.dump({"dataset": ds, "bm25": bm25, "corpus": corpus,
                         "embeddings": embeddings.cpu().numpy()}, fh)
        return ds, bm25, corpus, model, embeddings
    except Exception as exc:
        print(f"❌ Error loading RAG resources: {exc}")
        return None, None, None, None, None


@tool
def rag_search(query: str) -> str:
    """Search the OpenThoughts dataset using Hybrid Search (BM25 + Semantic)."""
    ds, bm25, corpus, model, embeddings = get_rag_resources()
    if not ds or not bm25:
        return "RAG system unavailable."
    import torch
    bm25_scores     = bm25.get_scores(query.split())
    query_emb       = model.encode(query, convert_to_tensor=True)
    if not isinstance(embeddings, torch.Tensor):
        embeddings  = torch.tensor(embeddings)
    embeddings      = embeddings.to(query_emb.device)
    sem_scores      = util.cos_sim(query_emb, embeddings)[0]
    top_bm25        = np.argsort(bm25_scores)[-5:][::-1]
    top_sem         = np.argsort(sem_scores.cpu().numpy())[-5:][::-1]
    combined        = list(set(top_bm25) | set(top_sem))
    results         = []
    for idx in combined:
        item    = ds[int(idx)]
        sol     = item.get("ground_truth_solution") or item.get("deepseek_solution")
        content = ""
        if item.get("problem"):            content += f"Problem: {item['problem']}\n"
        if item.get("deepseek_reasoning"): content += f"Reasoning: {item['deepseek_reasoning'][:1500]}…\n"
        if sol:                            content += f"Solution: {sol[:500]}…\n"
        results.append(content)
    return "\n---\n".join(results[:5])


def batch_rag_search(queries: List[str]) -> Dict[str, str]:
    """Optimised batch RAG search — one forward pass for all queries."""
    ds, bm25, corpus, model, embeddings = get_rag_resources()
    if not ds or not bm25:
        return {q: "RAG system unavailable." for q in queries}
    import torch
    q_embs = model.encode(queries, convert_to_tensor=True, batch_size=len(queries))
    if not isinstance(embeddings, torch.Tensor):
        embeddings = torch.tensor(embeddings)
    embeddings = embeddings.to(q_embs.device)
    out: Dict[str, str] = {}
    for i, query in enumerate(queries):
        bm25_scores = bm25.get_scores(query.split())
        sem_scores  = util.cos_sim(q_embs[i], embeddings)[0]
        combined    = list(set(np.argsort(bm25_scores)[-5:][::-1]) |
                          set(np.argsort(sem_scores.cpu().numpy())[-5:][::-1]))
        parts = []
        for idx in combined:
            item = ds[int(idx)]
            sol  = item.get("ground_truth_solution") or item.get("deepseek_solution")
            c    = ""
            if item.get("problem"):            c += f"Problem: {item['problem']}\n"
            if item.get("deepseek_reasoning"): c += f"Reasoning: {item['deepseek_reasoning'][:1500]}…\n"
            if sol:                            c += f"Solution: {sol[:500]}…\n"
            parts.append(c)
        out[query] = "\n---\n".join(parts[:5])
    return out


def _do_wiki_search(query: str) -> Optional[tuple]:
    try:
        wiki = wikipediaapi.Wikipedia("EduTech-Agent/1.0", "en")
        page = wiki.page(query)
        if page.exists():
            return query, f"Summary: {page.summary[:2000]}"
    except Exception as exc:
        print(f"   ⚠️ Wikipedia failed for '{query}': {exc}")
    return None


@tool
def wikipedia_search(query: str) -> str:
    """Search Wikipedia for a summary."""
    r = _do_wiki_search(query)
    return f"Found '{query}'.\n{r[1]}" if r else f"Page '{query}' not found."


def _build_retrieval_context(search_queries: List[str]) -> str:
    parts: List[str] = []
    try:
        rag_res = batch_rag_search(search_queries[:3])
        for q, res in rag_res.items():
            if res and "unavailable" not in res.lower():
                parts.append(f"[RAG for '{q}']:\n{res}")
    except Exception as exc:
        print(f"   ⚠️ RAG failed: {exc}")

    with ThreadPoolExecutor(max_workers=2) as ex:
        for fut in as_completed([ex.submit(_do_wiki_search, q) for q in search_queries[:2]]):
            r = fut.result()
            if r:
                parts.append(f"[Wikipedia for '{r[0]}']:\n{r[1]}")
    return "\n\n---\n\n".join(parts)


# ============================================================================
# NODE FUNCTIONS  (same logic, vLLM backend)
# ============================================================================

def breakdown_node(state: Dict) -> Dict:
    t0 = time.time()
    convs = [[{"role": "system", "content": BREAKDOWN_PROMPT},
              {"role": "user",   "content": state["query"]}]]
    out = vllm_generate_structured(convs, BreakdownOutput, temperature=0.1, max_tokens=600)[0]
    if out is None:
        out = BreakdownOutput(summary=state["query"],
                              search_queries=[state["query"]],
                              reasoning_points=[state["query"]])
    log_time("Breakdown node", t0)
    return {"breakdown_output": out.summary,
            "search_queries": out.search_queries,
            "reasoning_points": out.reasoning_points}


def reasoning_node(state: Dict) -> Dict:
    ctx   = (f"Original Query: {state.get('query', '')}\n\n"
             "Reasoning Points:\n"
             + "\n".join(f"- {p}" for p in state.get("reasoning_points", [])))
    convs = [[{"role": "system", "content": REASONING_PROMPT},
              {"role": "user",   "content": ctx}]]
    out   = vllm_generate_structured(convs, ReasoningOutput, temperature=0.1, max_tokens=700)[0]
    if out is None:
        out = ReasoningOutput(reasoning_analysis=[], conclusions=[])
    return {"reasoning_output": "\n".join(out.reasoning_analysis + out.conclusions)}


def scientific_node(state: Dict) -> Dict:
    queries = state.get("search_queries") or [state.get("query", "")]
    context = _build_retrieval_context(queries)
    prompt  = (f"RETRIEVED CONTEXT:\n{context[:8000]}\n\n"
               f"ORIGINAL QUESTION: {state.get('query', '')}\n\n"
               "Extract facts grounded in the context.")
    convs   = [[{"role": "system", "content": SCIENTIFIC_EXTRACTION_PROMPT},
                {"role": "user",   "content": prompt}]]
    out     = vllm_generate_structured(convs, ScientificOutput, temperature=0.1, max_tokens=1200)[0]
    if out is None:
        out = ScientificOutput(facts=[{"fact": "Extraction failed", "text": ""}])
    return {"scientific_output": str(out), "extracted_facts": out.facts}


def parallel_analysis_node(state: Dict) -> Dict:
    """
    Retrieval runs in a background thread; once done, reasoning + extraction
    are submitted as a single 2-item batch to vLLM.
    """
    t0             = time.time()
    queries        = state.get("search_queries") or [state.get("query", "")]
    retrieval_buf: Dict[str, str] = {}

    def do_retrieval():
        retrieval_buf["ctx"] = _build_retrieval_context(queries)

    th = threading.Thread(target=do_retrieval)
    th.start()

    r_ctx   = (f"Original Query: {state.get('query', '')}\n\n"
               "Reasoning Points:\n"
               + "\n".join(f"- {p}" for p in state.get("reasoning_points", [])))
    r_conv  = [{"role": "system", "content": REASONING_PROMPT},
               {"role": "user",   "content": r_ctx}]

    th.join()
    ctx    = retrieval_buf.get("ctx", "")
    e_ctx  = (f"RETRIEVED CONTEXT:\n{ctx[:8000]}\n\n"
              f"ORIGINAL QUESTION: {state.get('query', '')}\n\n"
              "Extract facts grounded in the context.")
    e_conv = [{"role": "system", "content": SCIENTIFIC_EXTRACTION_PROMPT},
              {"role": "user",   "content": e_ctx}]

    # Single 2-prompt batch → both generated in one engine pass, each with its own schema
    raw      = get_vllm_engine().chat(
                 [r_conv, e_conv],
                 sampling_params=[
                     SamplingParams(temperature=0.1, max_tokens=1200,
                         structured_outputs=StructuredOutputsParams(json=ReasoningOutput.model_json_schema())),
                     SamplingParams(temperature=0.1, max_tokens=1200,
                         structured_outputs=StructuredOutputsParams(json=ScientificOutput.model_json_schema())),
                 ])

    def _parse(text: str, model: type):
        try: return model.model_validate_json(text)
        except Exception:
            try:
                s, e2 = text.find("{"), text.rfind("}") + 1
                return model.model_validate_json(text[s:e2])
            except Exception:
                return None

    r_out = _parse(raw[0].outputs[0].text.strip(), ReasoningOutput) \
            or ReasoningOutput(reasoning_analysis=[], conclusions=[])
    e_out = _parse(raw[1].outputs[0].text.strip(), ScientificOutput) \
            or ScientificOutput(facts=[{"fact": "Extraction failed", "text": ""}])

    log_time("Parallel analysis (reasoning + scientific)", t0)
    return {"reasoning_output":  "\n".join(r_out.reasoning_analysis + r_out.conclusions),
            "scientific_output": str(e_out),
            "extracted_facts":   e_out.facts}


def synthesis_node(state: Dict) -> Dict:
    facts_text = "\n".join(
        f"{i+1}. {f.get('fact') or f.get('text', '')}"
        for i, f in enumerate(state.get("extracted_facts", []))
        if isinstance(f, dict)
    )
    ctx   = (f'Original Query: "{state.get("query", "")}"\n\n'
             f'Breakdown: {state.get("breakdown_output", "")}\n\n'
             f'REASONING:\n{state.get("reasoning_output", "")}\n\n'
             f'FACTS:\n{facts_text}')
    convs = [[{"role": "system", "content": SYNTHESIS_PROMPT},
              {"role": "user",   "content": ctx}]]
    out   = vllm_generate_structured(convs, SynthesisOutput, temperature=0.2, max_tokens=600)[0]
    if out is None:
        out = SynthesisOutput(synthesis_strategy="balanced", final_points=[])
    return {"synthesis_strategy": out.synthesis_strategy, "final_points": out.final_points}


def creative_node(state: Dict) -> Dict:
    pts   = state.get("final_points", [])
    ctx   = (f"Question: {state.get('query', '')}\n\n"
             "Key Points to Explain:\n"
             + "\n".join(f"{i+1}. {p}" for i, p in enumerate(pts))
             + "\n\nJSON output only. Full story in 'final_answer'.")
    convs = [[{"role": "system", "content": CREATIVE_PROMPT},
              {"role": "user",   "content": ctx}]]
    out   = vllm_generate_structured(convs, CreativeOutput, temperature=0.5, max_tokens=700)[0]
    return {"final_answer": out.final_answer if out else "Sorry, I couldn't generate an explanation."}


# ============================================================================
# GRAPH  (identical structure to simple_agent.py)
# ============================================================================

def create_graph():
    if "workflow" in graph_cache:
        return graph_cache["workflow"]
    wf = StateGraph(AgentState)
    wf.add_node("breakdown",         breakdown_node)
    wf.add_node("parallel_analysis", parallel_analysis_node)
    wf.add_node("synthesis",         synthesis_node)
    wf.add_node("creative",          creative_node)
    wf.add_edge(START,               "breakdown")
    wf.add_edge("breakdown",         "parallel_analysis")
    wf.add_edge("parallel_analysis", "synthesis")
    wf.add_edge("synthesis",         "creative")
    wf.add_edge("creative",           END)
    app = wf.compile()
    graph_cache["workflow"] = app
    return app


def answer_question(query: str) -> Dict[str, Any]:
    """Answer a single question using the full multi-agent graph."""
    state: AgentState = {
        "query": query, "breakdown_output": "", "reasoning_output": "",
        "scientific_output": "", "final_answer": "", "messages": [],
        "remaining_steps": 10, "structured_response": None,
        "search_queries": [], "reasoning_points": [],
        "extracted_facts": [], "synthesis_strategy": "", "final_points": [],
    }
    return create_graph().invoke(state)


def run_multi_agent_system(query: str, model_config=None) -> Dict[str, Any]:
    """Compatibility shim for evaluation.py."""
    return answer_question(query)


# ============================================================================
# STAGED BATCH GENERATION  (KEY vLLM optimisation)
# ============================================================================
#
#  Original: N questions × 5 sequential LLM calls = N×5 calls total
#  Staged:   5 stages × 1 batched vLLM call (N prompts each) = 5 calls total
#
#  Stage 1: breakdown       →  batch N prompts
#  Stage 2: retrieval       →  parallel ThreadPool (no LLM)
#           reasoning       →  ⎤ combined as one 2N-prompt
#           extraction      →  ⎦ batch vLLM call
#  Stage 3: synthesis       →  batch N prompts
#  Stage 4: creative        →  batch N prompts
# ============================================================================

def _merge(base: Dict, upd: Dict) -> Dict:
    m = dict(base); m.update(upd); return m


def _staged_breakdown(states: List[Dict]) -> List[Dict]:
    convs = [[{"role": "system", "content": BREAKDOWN_PROMPT},
              {"role": "user",   "content": s["query"]}] for s in states]
    outs  = vllm_generate_structured(convs, BreakdownOutput, temperature=0.1, max_tokens=600)
    result = []
    for s, o in zip(states, outs):
        if o is None:
            o = BreakdownOutput(summary=s["query"],
                                search_queries=[s["query"]],
                                reasoning_points=[s["query"]])
        result.append({"breakdown_output": o.summary,
                       "search_queries": o.search_queries,
                       "reasoning_points": o.reasoning_points})
    return result


def _staged_parallel_analysis(states: List[Dict]) -> List[Dict]:
    per_rag  = [s.get("search_queries", [s["query"]])[:3] for s in states]
    per_wiki = [s.get("search_queries", [s["query"]])[:2] for s in states]

    # Deduplicated batch RAG
    all_rag_q = list({q for qs in per_rag  for q in qs})
    all_wik_q = list({q for qs in per_wiki for q in qs})

    print(f"   🔍 Batch RAG: {len(all_rag_q)} unique queries for {len(states)} questions…")
    try:
        rag_res = batch_rag_search(all_rag_q)
    except Exception as exc:
        print(f"   ⚠️ Batch RAG failed: {exc}"); rag_res = {}

    print(f"   🌐 Wikipedia: {len(all_wik_q)} unique queries…")
    wiki_res: Dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        for fut in as_completed([ex.submit(_do_wiki_search, q) for q in all_wik_q]):
            r = fut.result()
            if r: wiki_res[r[0]] = r[1]

    # Per-question context
    contexts = []
    for rag_qs, wik_qs in zip(per_rag, per_wiki):
        parts = []
        for q in rag_qs:
            if q in rag_res and "unavailable" not in rag_res[q].lower():
                parts.append(f"[RAG for '{q}']:\n{rag_res[q]}")
        for q in wik_qs:
            if q in wiki_res:
                parts.append(f"[Wikipedia for '{q}']:\n{wiki_res[q]}")
        contexts.append("\n\n---\n\n".join(parts))

    # Build interleaved [R0,E0,R1,E1,...] batch
    combined_convs: List[List[dict]] = []
    for s, ctx in zip(states, contexts):
        r_ctx = (f"Original Query: {s.get('query', '')}\n\n"
                 "Reasoning Points:\n"
                 + "\n".join(f"- {p}" for p in s.get("reasoning_points", [])))
        e_ctx = (f"RETRIEVED CONTEXT:\n{ctx[:8000]}\n\n"
                 f"ORIGINAL QUESTION: {s.get('query', '')}\n\n"
                 "Extract facts grounded in the context.")
        combined_convs.append([{"role": "system", "content": REASONING_PROMPT},
                                {"role": "user",   "content": r_ctx}])
        combined_convs.append([{"role": "system", "content": SCIENTIFIC_EXTRACTION_PROMPT},
                                {"role": "user",   "content": e_ctx}])

    print(f"   🤖 vLLM batch: {len(combined_convs)} prompts (reasoning + extraction)…")
    # Build per-request SamplingParams list (even=reasoning schema, odd=extraction schema)
    r_schema = ReasoningOutput.model_json_schema()
    e_schema = ScientificOutput.model_json_schema()
    sampling_list = []
    for i in range(len(states)):
        sampling_list.append(SamplingParams(temperature=0.1, max_tokens=1200,
            structured_outputs=StructuredOutputsParams(json=r_schema)))
        sampling_list.append(SamplingParams(temperature=0.1, max_tokens=1200,
            structured_outputs=StructuredOutputsParams(json=e_schema)))
    raw = get_vllm_engine().chat(combined_convs, sampling_params=sampling_list)

    def _parse(text: str, model: type):
        try: return model.model_validate_json(text)
        except Exception:
            try:
                s2, e2 = text.find("{"), text.rfind("}") + 1
                return model.model_validate_json(text[s2:e2])
            except Exception: return None

    updates = []
    for i in range(len(states)):
        r_out = _parse(raw[i*2  ].outputs[0].text.strip(), ReasoningOutput) \
                or ReasoningOutput(reasoning_analysis=[], conclusions=[])
        e_out = _parse(raw[i*2+1].outputs[0].text.strip(), ScientificOutput) \
                or ScientificOutput(facts=[{"fact": "Extraction failed", "text": ""}])
        updates.append({
            "reasoning_output":  "\n".join(r_out.reasoning_analysis + r_out.conclusions),
            "scientific_output": str(e_out),
            "extracted_facts":   e_out.facts,
        })
    return updates


def _staged_synthesis(states: List[Dict]) -> List[Dict]:
    convs = []
    for s in states:
        facts_text = "\n".join(
            f"{i+1}. {f.get('fact') or f.get('text','')}"
            for i, f in enumerate(s.get("extracted_facts", [])) if isinstance(f, dict))
        ctx = (f'Original Query: "{s.get("query","")}" \n\n'
               f'Breakdown: {s.get("breakdown_output","")}\n\n'
               f'REASONING:\n{s.get("reasoning_output","")}\n\n'
               f'FACTS:\n{facts_text}')
        convs.append([{"role": "system", "content": SYNTHESIS_PROMPT},
                      {"role": "user",   "content": ctx}])
    outs = vllm_generate_structured(convs, SynthesisOutput, temperature=0.2, max_tokens=600)
    return [{"synthesis_strategy": (o or SynthesisOutput(synthesis_strategy="balanced", final_points=[])).synthesis_strategy,
             "final_points":       (o or SynthesisOutput(synthesis_strategy="balanced", final_points=[])).final_points}
            for o in outs]


def _staged_creative(states: List[Dict]) -> List[Dict]:
    convs = []
    for s in states:
        pts = s.get("final_points", [])
        ctx = (f"Question: {s.get('query','')}\n\n"
               "Key Points:\n"
               + "\n".join(f"{i+1}. {p}" for i, p in enumerate(pts))
               + "\n\nJSON output only. Full story in 'final_answer'.")
        convs.append([{"role": "system", "content": CREATIVE_PROMPT},
                      {"role": "user",   "content": ctx}])
    outs = vllm_generate_structured(convs, CreativeOutput, temperature=0.5, max_tokens=700)
    return [{"final_answer": (o.final_answer if o else "Sorry, I couldn't generate an explanation.")}
            for o in outs]


def staged_batch_process(questions_data: List[Dict]) -> List[Dict]:
    """Process N questions through all 4 pipeline stages using staged batching."""
    states = [{"query": item["question"]} for item in questions_data]

    print(f"\n📋 Stage 1/4 — Breakdown  ({len(states)} questions)…")
    t0 = time.time()
    for i, upd in enumerate(_staged_breakdown(states)):
        states[i] = _merge(states[i], upd)
    log_time("Stage 1 breakdown", t0)

    print(f"\n🔬 Stage 2/4 — Parallel analysis  ({len(states)} questions)…")
    t0 = time.time()
    for i, upd in enumerate(_staged_parallel_analysis(states)):
        states[i] = _merge(states[i], upd)
    log_time("Stage 2 parallel analysis", t0)

    print(f"\n🔗 Stage 3/4 — Synthesis  ({len(states)} questions)…")
    t0 = time.time()
    for i, upd in enumerate(_staged_synthesis(states)):
        states[i] = _merge(states[i], upd)
    log_time("Stage 3 synthesis", t0)

    print(f"\n✍️  Stage 4/4 — Creative  ({len(states)} questions)…")
    t0 = time.time()
    for i, upd in enumerate(_staged_creative(states)):
        states[i] = _merge(states[i], upd)
    log_time("Stage 4 creative", t0)

    results = []
    for item, state in zip(questions_data, states):
        ref = item.get("reference_answers", [])
        results.append({
            "question_id":      item["global_id"],
            "question":          item["question"],
            "generated_answer":  state.get("final_answer", ""),
            "reference_answers": "|||".join(ref) if isinstance(ref, list) else str(ref),
            "generation_time":   0.0,
            "status":            "success" if state.get("final_answer") else "error",
            "timestamp":         datetime.now().isoformat(),
            "error":             "",
        })
    return results


# ============================================================================
# DATASET LOADING  (identical to simple_agent.py)
# ============================================================================

def load_eli5_dataset(cache_file: str = "eli5_dataset_cache.pkl"):
    import pandas as pd
    p = Path(cache_file)
    if p.exists():
        with open(p, "rb") as fh: return pickle.load(fh)
    ds = load_dataset("sentence-transformers/eli5", split="train")
    df = pd.DataFrame({"query": ds["question"], "answers": ds["answer"]})
    with open(p, "wb") as fh: pickle.dump(df, fh)
    return df


# ============================================================================
# GENERATE ANSWERS BATCH  (same interface as simple_agent.py)
# ============================================================================

def generate_answers_batch(
    start_idx:   int           = 0,
    end_idx:     Optional[int] = None,
    split_index: Optional[int] = None,
    output_file: Optional[str] = None,
    chunk_size:  int           = 50,
) -> str:
    import pandas as pd
    df = load_eli5_dataset()

    if split_index is not None:
        split_size = len(df) // 3
        start_idx  = split_index * split_size
        end_idx    = (start_idx + split_size) if split_index < 2 else len(df)
        print(f"  Split {split_index+1}/3: rows {start_idx}→{end_idx}")

    end_idx = end_idx or len(df)
    df_sub  = df.iloc[start_idx:end_idx].reset_index(drop=True)
    print(f"\n🚀 Staged-batch agent generation (vLLM)  |  {len(df_sub)} questions  |  chunk={chunk_size}")

    if output_file is None:
        out_dir = Path("vllm/agent_answers"); out_dir.mkdir(parents=True, exist_ok=True)
        tag = f"split{split_index}" if split_index is not None else f"{start_idx}_{end_idx}"
        output_file = str(out_dir / f"agent_vllm_{tag}.csv")

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    existing_ids: set = set()
    all_results:  List[Dict] = []
    fieldnames = ["question_id","question","generated_answer","reference_answers",
                  "generation_time","status","timestamp","error"]

    if output_path.exists():
        try:
            ex_df        = pd.read_csv(output_path)
            all_results  = ex_df.to_dict("records")
            existing_ids = {r["question_id"] for r in all_results}
            print(f"   Resuming — {len(existing_ids)} already done.")
        except Exception as exc:
            print(f"   ⚠️ Can't read existing file ({exc}). Starting fresh.")

    if not output_path.exists():
        with open(output_path, "w", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=fieldnames).writeheader()

    work = []
    for li in range(len(df_sub)):
        gid = start_idx + li
        if gid in existing_ids: continue
        row = df_sub.iloc[li]
        work.append({"global_id": gid, "question": row["query"],
                     "reference_answers": row["answers"] if isinstance(row["answers"], list)
                                          else [row["answers"]]})

    if not work:
        print("   All questions already processed!"); return str(output_path)

    with tqdm(total=len(work), desc="Staged batches") as pbar:
        for ci in range(0, len(work), chunk_size):
            chunk   = work[ci: ci + chunk_size]
            res     = staged_batch_process(chunk)
            all_results.extend(res)
            with open(output_path, "a", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=fieldnames)
                for row in sorted(res, key=lambda r: r["question_id"]):
                    writer.writerow(row)
            pbar.update(len(chunk))

    ok = sum(1 for r in all_results if r["status"] == "success")
    print(f"\n🎉 Done!  {ok}/{len(all_results)} successful  →  {output_path}")
    return str(output_path)


# ============================================================================
# CLI  (identical interface to simple_agent.py + chunk-size flag)
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Multi-Agent ELI5 System — vLLM Optimised",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python vllm/simple_agent_vllm.py "Why is the sky blue?"
  python vllm/simple_agent_vllm.py --batch --split 0
  python vllm/simple_agent_vllm.py --batch --start 0 --end 3000
  python vllm/simple_agent_vllm.py --batch --start 0 --end 3000 --chunk-size 100
  python vllm/simple_agent_vllm.py --batch --start 0 --end 3000 --model llama3.2:1b
        """)

    parser.add_argument("question",     nargs="*")
    parser.add_argument("--batch",      action="store_true")
    parser.add_argument("--split",      type=int, choices=[0,1,2])
    parser.add_argument("--start",      type=int, default=0)
    parser.add_argument("--end",        type=int)
    parser.add_argument("--output",     type=str)
    parser.add_argument("--chunk-size", type=int, default=50,
                        help="Questions per staged-batch chunk (default 50)")
    parser.add_argument("--model",      type=str, default=MODEL_NAME,
                        help="vLLM model id or Ollama-style alias")
    args = parser.parse_args()

    if args.model != MODEL_NAME:
        MODEL_NAME = resolve_model(args.model)

    try:
        if args.batch:
            generate_answers_batch(start_idx=args.start, end_idx=args.end,
                                   split_index=args.split, output_file=args.output,
                                   chunk_size=args.chunk_size)
        else:
            q      = " ".join(args.question) if args.question else input("Enter your question: ")
            result = answer_question(q)
            print(f"\n✅ Answer: {result['final_answer']}\n")
    finally:
        shutdown_vllm_engine()