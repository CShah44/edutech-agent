import os
import json
import time
from datetime import datetime
from typing import TypedDict, Dict, List, Any, Annotated
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import create_react_agent
from langgraph.graph.message import add_messages
from tavily import TavilyClient
from dotenv import load_dotenv
import gspread
from google.oauth2.service_account import Credentials
import re
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor

# Load environment variables
load_dotenv()

tavily_api_key = os.getenv('TAVILY_API_KEY') 

# Configuration
tavily_client = TavilyClient(api_key=tavily_api_key)

# Google Sheets Configuration
GOOGLE_SHEETS_CONFIG = {
    "spreadsheet_name": "Edutech-Agent",  # Change this to your sheet name
    "questions_column": "A",  # Column containing questions
    "credentials_file": "credentials.json",  # Google Service Account credentials file
    "worksheet_name": "Sheet3"  # Name of the worksheet
}

# Retrieval and synthesis tuning
MAX_SEARCH_QUERIES = 5
MAX_SOURCES = 10
FACTS_TARGET_COUNT = 12
# CREATIVE_SENTENCE_TARGET removed - let the LLM decide naturally

# Model configurations for different agents
MODEL_CONFIGS = {
    "config1": {
        "reasoning_model": "mistral:7b",
        "scientific_model": "mistral:7b", 
        "creative_model": "mistral:7b"
    },
    "config2": {
        "reasoning_model": "llama3.2:latest",
        "scientific_model": "llama3.2:latest",
        "creative_model": "llama3.2:latest"
    },
    "config3": {
        "reasoning_model": "mistral:7b",
        "scientific_model": "llama3.2:latest",
        "creative_model": "llama3.2:latest"
    },
    "config4": {
        "reasoning_model": "mistral:7b",
        "scientific_model": "llama3.2:latest",
        "creative_model": "mistral:7b"
    },
    "config5": {
        "reasoning_model": "llama3.2:1b",
        "scientific_model": "llama3.2:1b",
        "creative_model": "llama3.2:1b"
    }
    # Add more configurations as needed
}

# System prompts
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
- Extract around {FACTS_TARGET_COUNT} comprehensive scientific facts

You must respond with structured output containing:
- facts: Array of fact objects with "fact" field (brief fact statement) and "text" field (detailed description) - EXACTLY {FACTS_TARGET_COUNT} facts
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

# State definition
class AgentState(TypedDict):
    query: str
    breakdown_output: str
    reasoning_output: str
    scientific_output: str
    final_answer: str
    messages: Annotated[list, add_messages]
    remaining_steps: int
    model_config: Dict[str, str]
    # Required for structured outputs
    structured_response: Any
    search_queries: List[str]  # From breakdown agent
    reasoning_points: List[str]  # From breakdown agent
    extracted_facts: List[Dict[str, Any]]  # From scientific agent
    # Synthesis node outputs
    synthesis_strategy: str  # "reasoning_heavy", "facts_heavy", or "balanced"
    final_points: List[str]  # Curated points from synthesis node

# Web search tool
def tavily_search_raw(query: str) -> Dict[str, Any]:
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
        return {"answer": "", "results": [], "error": str(e)}

@tool
def web_search(query: str) -> str:
    """Search the web for factual information using Tavily. Returns JSON string."""
    data = tavily_search_raw(query)
    try:
        return json.dumps(data)
    except Exception:
        return json.dumps({"answer": "", "results": [], "error": "serialization-failed"})

# Create LLMs
def create_llm(model="mistral:7b", temperature=0.1, system=""):
    return ChatOllama(
        model=model,
        reasoning=False, # only for thinking models TODO
        base_url="http://localhost:11434",
        temperature=temperature,
        system=system,
    )

class breakdown_structure(BaseModel):
    summary: str
    search_queries: List[str]
    reasoning_points: List[str]

class reasoning_structure(BaseModel):
    reasoning_analysis: List[str]  # List of logical analysis points
    conclusions: List[str]  # Key conclusions from reasoning

class scientific_structure(BaseModel):
    facts: List[Dict[str, str]]  # List of {"fact": "brief fact", "text": "detailed description"} objects

class synthesis_structure(BaseModel):
    synthesis_strategy: str  # "reasoning_heavy", "facts_heavy", or "balanced"
    final_points: List[str]  # Curated 3-5 points for creative agent

class creative_structure(BaseModel):
    final_answer: str

# Create agents with configurable models
def create_breakdown_agent(model_config):
    model = model_config.get("reasoning_model", "llama3.2:1b")
    llm = create_llm(model, system=BREAKDOWN_PROMPT)
    return create_react_agent(llm, [], response_format=breakdown_structure, state_schema=AgentState)

def create_reasoning_agent(model_config):
    model = model_config.get("reasoning_model", "llama3.2:1b")
    llm = create_llm(model, system=REASONING_PROMPT)
    return create_react_agent(llm, [], response_format=reasoning_structure, state_schema=AgentState)

def create_scientific_agent(model_config):
    model = model_config.get("scientific_model", "llama3.2:1b")
    llm = create_llm(model, system=SCIENTIFIC_PROMPT)
    tools = [web_search]
    return create_react_agent(llm, tools, response_format=scientific_structure, state_schema=AgentState)

def create_creative_agent(model_config):
    model = model_config.get("creative_model", "llama3.2:1b")
    # No system prompt - we'll provide everything in the context
    llm = create_llm(model, temperature=0.3)

    return create_react_agent(llm, [], response_format=creative_structure, state_schema=AgentState)

# Agent wrapper functions
def breakdown_node(state):
    agent = create_breakdown_agent(state["model_config"])
    messages = [HumanMessage(content=state["query"])]
    result = agent.invoke({"messages": messages})
    
    # Get structured output directly from structured_response
    structured_output: breakdown_structure = result["structured_response"]
    
    return {
        "breakdown_output": structured_output.summary,
        "search_queries": structured_output.search_queries,
        "reasoning_points": structured_output.reasoning_points
    }

def reasoning_node(state):
    agent = create_reasoning_agent(state["model_config"])
    
    # Get reasoning points from breakdown
    reasoning_points = state.get("reasoning_points", [])
    
    # Create context with the original query and reasoning points
    context = f"""Original Query: {state.get('query', '')}

Reasoning Points to Analyze:
{chr(10).join(f'- {point}' for point in reasoning_points)}

Provide logical analysis for each point."""
    
    messages = [HumanMessage(content=context)]
    result = agent.invoke({"messages": messages})
    
    # Get structured output directly from structured_response
    structured_output: reasoning_structure = result["structured_response"]
    
    # Convert structured output to readable format for downstream agents
    reasoning_text = "\n".join(structured_output.reasoning_analysis + structured_output.conclusions)
    
    return {
        "reasoning_output": reasoning_text,
        "structured_reasoning": structured_output
    }

def scientific_node(state):
    # 1. Get search queries from the breakdown agent
    queries = state.get("search_queries", [])
    if not queries:
        queries = [state.get("query", "")]

    # 2. Perform all web searches in parallel for maximum efficiency
    all_results = []
    with ThreadPoolExecutor() as executor:
        future_to_query = {executor.submit(tavily_search_raw, q): q for q in queries}
        for future in future_to_query:
            try:
                result = future.result()
                if result and result.get("results"):
                    all_results.extend(result["results"])
            except Exception as e:
                print(f"Error during parallel search for query '{future_to_query[future]}': {e}")

    # 3. Create a single, comprehensive context for the LLM
    # Deduplicate results by URL to avoid redundant information
    unique_results = {r['url']: r for r in all_results}.values()
    
    context_for_llm = f"""Original Query: {state.get('query', '')}

Here are the pre-fetched web search results. Your task is to analyze all of them and extract exactly {FACTS_TARGET_COUNT} distinct, comprehensive scientific facts based on the original query. Focus on mechanisms, definitions, and measurable details.

=== AGGREGATED SEARCH RESULTS ===
"""
    for i, result in enumerate(unique_results, 1):
        context_for_llm += f"Source {i} (URL: {result['url']}):\n{result['content']}\n\n"

    # 4. Call the LLM once to extract all facts from the aggregated content
    scientific_llm = create_llm(state["model_config"].get("scientific_model", "mistral:7b"), system=SCIENTIFIC_PROMPT)
    structured_llm = scientific_llm.with_structured_output(scientific_structure)
    
    response = structured_llm.invoke(context_for_llm)

    return {
        "scientific_output": str(response),
        "extracted_facts": response.facts,
    }

def synthesis_node(state):
    """
    Quality gate that analyzes reasoning and facts to create prioritized points.
    Decides which source is more reliable and creates curated list for creative agent.
    """
    reasoning_output = state.get("reasoning_output", "")
    extracted_facts = state.get("extracted_facts", [])
    query = state.get("query", "")
    breakdown_summary = state.get("breakdown_output", "")

    if not reasoning_output and not extracted_facts:
        return {"synthesis_strategy": "no_input", "final_points": []}

    # Format facts for analysis
    facts_list = []
    for fact in extracted_facts:
        if isinstance(fact, dict):
            fact_text = fact.get("fact", "") or fact.get("text", "")
            if fact_text:
                facts_list.append(fact_text)
    
    facts_text = "\n".join([f"{i+1}. {f}" for i, f in enumerate(facts_list)])

    # ONLY provide data in context - instructions are in system prompt
    context = f"""Original Query: "{query}"

Breakdown Summary: {breakdown_summary}

LOGICAL REASONING:
{reasoning_output}

SCIENTIFIC FACTS:
{facts_text}"""
    
    synthesis_llm = create_llm(state["model_config"].get("reasoning_model", "mistral:7b"), temperature=0.2, system=SYNTHESIS_PROMPT)
    structured_llm = synthesis_llm.with_structured_output(synthesis_structure)
    
    response = structured_llm.invoke(context)
    
    print(f"\n🎯 SYNTHESIS STRATEGY: {response.synthesis_strategy}")
    print(f"   Curated Points: {len(response.final_points)}")
    
    if len(response.final_points) < 4:
        print(f"   ⚠️  WARNING: Only {len(response.final_points)} points provided - may not be enough for comprehensive ELI5!")

    return {
        "synthesis_strategy": response.synthesis_strategy,
        "final_points": response.final_points
    }

def creative_node(state):
    # Get curated points from synthesis node
    final_points = state.get("final_points", [])

    # ONLY provide data in context - instructions are in system prompt
    context = f"""Question: {state.get('query', '')}

Key Points:
{chr(10).join(f'{i+1}. {point}' for i, point in enumerate(final_points))}"""
    
    creative_llm = create_llm(state["model_config"].get("creative_model", "mistral:7b"), temperature=0.3, system=CREATIVE_PROMPT)
    structured_llm = creative_llm.with_structured_output(creative_structure)
    
    result = structured_llm.invoke(context)
    
    return {
        "final_answer": result.final_answer
    }

def print_agent_state(final_state):
    """Print comprehensive agent state information"""
    print(f"\n{'='*80}")
    print("MULTI-AGENT SYSTEM - FINAL STATE")
    print(f"{'='*80}")
    
    # Original query
    print(f"🔍 ORIGINAL QUERY: {final_state.get('query', 'N/A')}")
    print(f"🔧 MODEL CONFIG: {final_state.get('model_config', {})}")
    
    # Breakdown Agent Results
    print(f"\n🧩 BREAKDOWN AGENT:")
    print(f"   Summary: {final_state.get('breakdown_output', 'N/A')}")
    search_queries = final_state.get('search_queries', [])
    print(f"   Search Queries ({len(search_queries)}):")
    for i, query in enumerate(search_queries, 1):
        print(f"      {i}. {query}")
    reasoning_points = final_state.get('reasoning_points', [])
    print(f"   Reasoning Points ({len(reasoning_points)}):")
    for i, point in enumerate(reasoning_points, 1):
        print(f"      {i}. {point}")
    
    # Reasoning Agent Results
    print(f"\n🧠 REASONING AGENT:")
    reasoning_output = final_state.get('reasoning_output', 'N/A')
    if reasoning_output and reasoning_output != 'N/A':
        print(f"   Analysis: {reasoning_output[:200]}{'...' if len(reasoning_output) > 200 else ''}")
    else:
        print(f"   Analysis: {reasoning_output}")
    
    # Scientific Agent Results
    print(f"\n🔬 SCIENTIFIC AGENT:")

    print("Scientific Output: ", final_state.get('scientific_output', 'N/A'))
    
    facts = final_state.get('extracted_facts', [])
    print(f"   Facts Extracted ({len(facts)}):")
    for i, fact in enumerate(facts[:5], 1):  # Show first 5 facts
        if isinstance(fact, dict):
            # Try to get 'fact' field first, then 'text' field
            fact_text = fact.get('fact', '') or fact.get('text', '')
            if fact_text:
                fact_display = fact_text[:100] + ('...' if len(fact_text) > 100 else '')
                print(f"      {i}. {fact_display}")
        else:
            print(f"      {i}. {str(fact)[:100]}{'...' if len(str(fact)) > 100 else ''}")
    if len(facts) > 5:
        print(f"      ... and {len(facts) - 5} more facts")
    
    # Synthesis Results (Quality Gate)
    synthesis_strategy = final_state.get('synthesis_strategy', '')
    final_points = final_state.get('final_points', [])
    if synthesis_strategy or final_points:
        print(f"\n⚡ SYNTHESIS (Quality Gate):")
        print(f"   Strategy: {synthesis_strategy}")
        print(f"   Final Points for ELI5 ({len(final_points)}):")
        for i, point in enumerate(final_points, 1):
            print(f"      {i}. {point}")
    
    # Creative Agent Results
    print(f"\n🎨 CREATIVE AGENT:")
    final_answer = final_state.get('final_answer', 'N/A')
    print(f"   Final Answer: {final_answer}")
    
    # Statistics
    print(f"\n📊 STATISTICS:")
    print(f"   Search Queries Generated: {len(search_queries)}")
    print(f"   Reasoning Points: {len(reasoning_points)}")
    print(f"   Scientific Facts: {len(facts)}")
    print(f"   Synthesis Points: {len(final_points)}")
    print(f"   Synthesis Strategy: {synthesis_strategy if synthesis_strategy else 'N/A'}")
    print(f"   Final Answer Length: {len(final_answer) if final_answer != 'N/A' else 0} characters")
    
    print(f"{'='*80}")

# Create and run the graph
def create_graph():
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("breakdown", breakdown_node)
    workflow.add_node("reasoning", reasoning_node)
    workflow.add_node("scientific", scientific_node)  
    workflow.add_node("synthesis", synthesis_node)
    workflow.add_node("creative", creative_node)
    
    # Flow: start -> breakdown -> (reasoning, scientific) -> synthesis -> creative
    # synthesis acts as quality gate that analyzes both reasoning and facts
    workflow.add_edge(START, "breakdown")
    workflow.add_edge("breakdown", "reasoning")
    workflow.add_edge("breakdown", "scientific")
    workflow.add_edge("reasoning", "synthesis")
    workflow.add_edge("scientific", "synthesis")
    workflow.add_edge("synthesis", "creative")
    workflow.add_edge("creative", END)
    
    return workflow.compile()

def visualize_graph_only():
    """Generate and save graph visualizations without running the workflow"""
    print("🎨 Creating workflow graph visualization...")
    
    app = create_graph()
    
    try:
        # Try PNG format (requires pygraphviz)
        try:
            graph_png = app.get_graph().draw_png()
            with open("edutech_workflow.png", "wb") as f:
                f.write(graph_png)
            print("✅ PNG graph saved as edutech_workflow.png")
        except ImportError:
            print("💡 Install pygraphviz for PNG: pip install pygraphviz")
        except Exception as e:
            print(f"⚠️ PNG generation failed: {e}")
        
        # Try Mermaid format
        try:
            mermaid_code = app.get_graph().draw_mermaid()
            with open("edutech_workflow.mmd", "w") as f:
                f.write(mermaid_code)
            print("✅ Mermaid code saved as edutech_workflow.mmd")
            print("💡 Visualize at: https://mermaid.live/")
            
            # Print the mermaid code for quick viewing
            print("\n📊 MERMAID CODE:")
            print("-" * 40)
            print(mermaid_code)
            print("-" * 40)
        except Exception as e:
            print(f"⚠️ Mermaid generation failed: {e}")
        
        # ASCII representation
        try:
            ascii_viz = app.get_graph().draw_ascii()
            print("\n📈 ASCII WORKFLOW:")
            print(ascii_viz)
        except Exception as e:
            print(f"⚠️ ASCII generation failed: {e}")
            
    except Exception as e:
        print(f"❌ Graph visualization failed: {e}")
        
    print("\n🏗️ WORKFLOW STRUCTURE:")
    print("START → breakdown → (reasoning + scientific) → creative → END")
    print("- Breakdown: Decomposes question into search queries + reasoning points")
    print("- Reasoning: Logical analysis of reasoning points from breakdown") 
    print("- Scientific: Web search using search queries from breakdown + fact extraction")
    print("- Creative: Synthesizes reasoning analysis + scientific facts into ELI5 answer")

# Google Sheets Helper Functions
def setup_google_sheets():
    """Setup Google Sheets connection"""
    try:
        # Define the scope
        scope = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive"
        ]
        
        # Load credentials
        credentials = Credentials.from_service_account_file(
            GOOGLE_SHEETS_CONFIG["credentials_file"], 
            scopes=scope
        )
        
        # Initialize the client
        client = gspread.authorize(credentials)
        
        return client
    except Exception as e:
        print(f"Error setting up Google Sheets: {e}")
        return None

def get_questions_from_sheet(client):
    """Get questions from Google Sheets"""
    try:
        # Open the spreadsheet
        spreadsheet = client.open(GOOGLE_SHEETS_CONFIG["spreadsheet_name"])
        worksheet = spreadsheet.worksheet(GOOGLE_SHEETS_CONFIG["worksheet_name"])

        # Get all questions from the specified column (skip header)
        questions = worksheet.col_values(1)[1:]  # Assuming column A, skip header
        
        # Filter out empty questions
        questions = [q.strip() for q in questions if q.strip()]
        
        return questions, worksheet
    except Exception as e:
        print(f"Error reading questions from sheet: {e}")
        return [], None

def get_column_letter(col_num):
    """Convert column number to letter (e.g., 1->A, 2->B, 27->AA)"""
    result = ""
    while col_num > 0:
        col_num -= 1
        result = chr(col_num % 26 + ord('A')) + result
        col_num //= 26
    return result

def find_or_create_answer_column(worksheet, config_name):
    """Find existing column for this config or create a new one"""
    try:
        # Get all values from the first row (headers)
        headers = worksheet.row_values(1)
        
        # Look for existing column with this config name
        for i, header in enumerate(headers):
            if header == config_name:
                return i + 1  # gspread uses 1-based indexing
        
        # If not found, create new column
        next_col = len(headers) + 1
        col_letter = get_column_letter(next_col)
        
        # Add header for new column using proper format
        worksheet.update(values=[[config_name]], range_name=f"{col_letter}1", value_input_option='RAW')
        
        return next_col
    except Exception as e:
        print(f"Error finding/creating answer column: {e}")
        return None

def save_answers_to_sheet(worksheet, answers, config_name):
    """Save answers to the appropriate column in Google Sheets"""
    try:
        # Find or create the column for this configuration
        col_num = find_or_create_answer_column(worksheet, config_name)
        if not col_num:
            return False
        
        col_letter = get_column_letter(col_num)
        
        # Prepare the data as a list of lists (batch update)
        values_to_update = [[answer] for answer in answers]
        
        # Define the range for updating (start from row 2, skip header)
        start_cell = f"{col_letter}2"
        end_row = len(answers) + 1
        end_cell = f"{col_letter}{end_row}"
        range_name = f"{start_cell}:{end_cell}"
        
        # Update all cells at once using batch update
        worksheet.update(values=values_to_update, range_name=range_name, value_input_option='RAW')
        
        print(f"✅ Answers saved to column {col_letter} ({config_name})")
        return True
    except Exception as e:
        print(f"Error saving answers to sheet: {e}")
        return False

def save_agent_state_to_excel(question, state):
    """Save comprehensive agent state to excel file"""
    try:
        # Setup Google Sheets client
        client = setup_google_sheets()
        if not client:
            print("❌ Could not setup Google Sheets for detailed state saving")
            return
        
        # Open spreadsheet - try to get the detailed sheet or create it
        try:
            spreadsheet = client.open(GOOGLE_SHEETS_CONFIG["spreadsheet_name"])
            try:
                sheet = spreadsheet.worksheet("Detailed_States")
            except:
                # Create detailed states worksheet if it doesn't exist
                sheet = spreadsheet.add_worksheet(title="Detailed_States", rows="1000", cols="10")
                # Add headers
                headers = ["Timestamp", "Question", "Search Queries", "Facts Count", "Reasoning Points", 
                          "Final Answer", "Search Results", "Extracted Facts", "Reasoning Analysis", "Agent Config"]
                sheet.insert_row(headers, 1)
        except Exception as e:
            print(f"❌ Could not access spreadsheet for detailed state saving: {e}")
            return
        
        # Prepare comprehensive data
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Extract data from state
        search_queries = " | ".join(state.get("search_queries", [])) if state.get("search_queries") else "None"
        
        # Extract facts
        facts_data = state.get("extracted_facts", [])
        if isinstance(facts_data, list) and facts_data:
            facts_count = len(facts_data)
            facts_list = []
            for fact in facts_data:
                if isinstance(fact, dict):
                    fact_text = fact.get("fact", "") or fact.get("text", "")
                    if fact_text:
                        facts_list.append(fact_text)
                else:
                    facts_list.append(str(fact))
            facts_text = " | ".join([f"{i+1}. {fact}" for i, fact in enumerate(facts_list)])
        else:
            facts_count = 0
            facts_text = "No facts extracted"
        
        # Extract reasoning
        reasoning_output = state.get("reasoning_output", "")
        structured_reasoning = state.get("structured_reasoning", {})
        if reasoning_output and reasoning_output.strip():
            reasoning_text = reasoning_output[:500] + ("..." if len(reasoning_output) > 500 else "")
        elif isinstance(structured_reasoning, dict):
            analysis = structured_reasoning.get("reasoning_analysis", [])
            conclusions = structured_reasoning.get("conclusions", [])
            all_points = analysis + conclusions
            reasoning_text = " | ".join([f"{i+1}. {point}" for i, point in enumerate(all_points)])
        else:
            reasoning_text = "No reasoning analysis"
        
        # Final answer
        final_answer = state.get("final_answer", "")
        if not final_answer or final_answer.strip() == "":
            final_answer = "No final answer generated"
        
        # Configuration info
        config_info = f"Facts Target: {FACTS_TARGET_COUNT}, Creative Target: {CREATIVE_SENTENCE_TARGET}"
        
        # Truncate long texts for Excel cells
        def truncate_text(text, max_length=1000):
            return text[:max_length] + "..." if len(text) > max_length else text
        
        new_row = [
            timestamp,
            question,
            search_queries,
            facts_count,
            len(state.get("reasoning_points", [])),
            truncate_text(final_answer),
            truncate_text(facts_text),
            truncate_text(reasoning_text),
            config_info
        ]
        
        # Append row
        sheet.append_row(new_row)
        
        print(f"✅ Detailed agent state saved to Excel at {timestamp}")
        
    except Exception as e:
        print(f"❌ Failed to save detailed state to Excel: {e}")
        print("Continuing without detailed state saving...")

def process_questions_from_sheets(config_name="config1"):
    """Process all questions from Google Sheets and save answers"""
    print(f"\n{'='*80}")
    print(f"PROCESSING QUESTIONS FROM GOOGLE SHEETS")
    print(f"Configuration: {config_name}")
    print(f"{'='*80}")
    
    # Setup Google Sheets
    client = setup_google_sheets()
    if not client:
        print("❌ Failed to setup Google Sheets connection")
        return

    # Get questions
    questions, worksheet = get_questions_from_sheet(client)
    if not questions:
        print("❌ No questions found in the sheet")
        return
    
    print(f"Found {len(questions)} questions to process")
    
    # Get model configuration
    if config_name not in MODEL_CONFIGS:
        print(f"❌ Configuration '{config_name}' not found")
        return
    
    model_config = MODEL_CONFIGS[config_name]
    print(f"Using models: {model_config}")
    
    # Process each question
    answers = []
    for i, question in enumerate(questions, 1):
        print(f"\n--- Processing Question {i}/{len(questions)} ---")
        print(f"Q: {question}")
        
        try:
            # Initialize state - exactly same as single mode
            initial_state = {
                "query": question,
                "breakdown_output": "",
                "reasoning_output": "",
                "scientific_output": "",
                "final_answer": "",
                "messages": [],
                "model_config": model_config,
                "structured_response": None,
                "search_queries": [],
                "reasoning_points": [],
                "extracted_facts": [],
                "synthesis_strategy": "",
                "final_points": []
            }
            
            # Create and run graph - exactly same as single mode
            app = create_graph()
            result = app.invoke(initial_state)
            
            # Extract answer safely
            answer = result.get("final_answer", "")
            if not answer or answer.strip() == "":
                answer = "No answer generated"
            
            answers.append(answer)
            print(f"A: {answer}")
            
            # Save detailed state to Excel for this question (optional, continue if fails)
            try:
                save_agent_state_to_excel(question, result)
            except Exception as excel_error:
                print(f"⚠️ Failed to save detailed state for question {i}: {excel_error}")
                print("Continuing with next question...")
            
            # Add small delay between questions to avoid overwhelming the system
            if i < len(questions):  # Don't delay after the last question
                time.sleep(2)
            
        except Exception as e:
            error_msg = f"Error processing question {i}: {str(e)}"
            print(f"❌ {error_msg}")
            import traceback
            print(f"Full error trace: {traceback.format_exc()}")
            answers.append(f"ERROR: {error_msg}")
    
    # Save answers to sheet
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    column_name = f"{config_name}_{timestamp}"
    
    success = save_answers_to_sheet(worksheet, answers, column_name)
    if success:
        print(f"\n✅ All answers saved to Google Sheets in column '{column_name}'")
    else:
        print(f"\n❌ Failed to save answers to Google Sheets")
    
    return answers

def run_multi_agent_system(query, model_config=None):
    if model_config is None:
        model_config = MODEL_CONFIGS["config1"]
    
    # Initialize state
    initial_state = {
        "query": query,
        "breakdown_output": "",
        "reasoning_output": "",
        "scientific_output": "",
        "final_answer": "",
        "messages": [],
        "model_config": model_config,
        "structured_response": None,
        "search_queries": [],
        "reasoning_points": [],
        "extracted_facts": [],
        "synthesis_strategy": "",
        "final_points": []
    }
    
    # Create and run graph
    app = create_graph()
    result = app.invoke(initial_state)

    # Print comprehensive state information
    print_agent_state(result)
    
    return result

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == "sheets":
            # Google Sheets mode
            config_name = sys.argv[2] if len(sys.argv) > 2 else "config1"
            
            print("Available configurations:")
            for config in MODEL_CONFIGS.keys():
                print(f"  - {config}: {MODEL_CONFIGS[config]}")
            print(f"\nUsing configuration: {config_name}")
            
            process_questions_from_sheets(config_name)

        elif mode == "single":
            # Single question mode
            config_name = sys.argv[2] if len(sys.argv) > 2 else "config1"
            question = " ".join(sys.argv[3:]) if len(sys.argv) > 3 else input("Enter your question: ")
            
            if config_name not in MODEL_CONFIGS:
                print(f"❌ Configuration '{config_name}' not found")
                print("Available configurations:", list(MODEL_CONFIGS.keys()))
                exit(1)
            
            model_config = MODEL_CONFIGS[config_name]
            run_multi_agent_system(question, model_config)
            
        elif mode == "graph":
            # Generate workflow graph visualization only
            visualize_graph_only()
            
        else:
            print("Usage:")
            print("  python main.py sheets [config_name]           # Process questions from Google Sheets")
            print("  python main.py single [config_name] [question] # Single question mode")
            print("  python main.py graph                          # Generate workflow graph visualization")
            print("\nAvailable configurations:")
            for config in MODEL_CONFIGS.keys():
                print(f"  - {config}: {MODEL_CONFIGS[config]}")
    else:
        # Interactive mode - single question
        print("Available configurations:")
        for config in MODEL_CONFIGS.keys():
            print(f"  - {config}: {MODEL_CONFIGS[config]}")
        
        config_name = input(f"Enter configuration name (default: config1): ").strip() or "config1"
        
        if config_name not in MODEL_CONFIGS:
            print(f"❌ Configuration '{config_name}' not found. Using config1.")
            config_name = "config1"
        
        question = input("Enter your question: ")
        model_config = MODEL_CONFIGS[config_name]
        run_multi_agent_system(question, model_config)