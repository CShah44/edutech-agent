import os
import json
from datetime import datetime
from typing import TypedDict, Dict, List, Any
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import create_react_agent
from tavily import TavilyClient
from dotenv import load_dotenv
import gspread
from google.oauth2.service_account import Credentials
import re
from pydantic import BaseModel

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
CREATIVE_SENTENCE_TARGET = 10

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
    #best
    "config3": {
        "reasoning_model": "mistral:7b",
        "scientific_model": "llama3.2:latest",
        "creative_model": "llama3.2:latest"
    },
    "config4": {
        "reasoning_model": "mistral:7b",
        "scientific_model": "llama3.2:latest",
        "creative_model": "mistral:7b"
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
- facts: Array of fact objects with "text" field containing mechanism-focused scientific facts (EXACTLY {FACTS_TARGET_COUNT} facts)
"""

CREATIVE_PROMPT = f"""You are the Creative Agent. Compose the final answer ONLY from provided extracted facts and reasoning analysis.

Instructions:
- Explain like explaining to a 5 year old; use around {CREATIVE_SENTENCE_TARGET} sentences for depth
- Do not add new facts. Use only the given facts and reasoning.
- Include detailed mechanisms, processes, and cause-effect relationships
- Add connecting words for smooth flow between concepts
- Make each sentence informative and educational
- Do not include any citations, sources or references in the final answer.

You must respond with structured output containing:
- final_answer: String containing the complete ELI5 explanation (around {CREATIVE_SENTENCE_TARGET} sentences with detailed explanations)
"""

# Create LLMs

# State definition
class AgentState(TypedDict):
    query: str
    breakdown_output: str
    reasoning_output: str
    scientific_output: str
    final_answer: str
    messages: list
    remaining_steps: int
    model_config: Dict[str, str]
    # Required for structured outputs
    structured_response: Any
    search_queries: List[str]  # From breakdown agent
    reasoning_points: List[str]  # From breakdown agent
    sources: List[Dict[str, Any]]  # [{id,title,url,content}]
    extracted_facts: List[Dict[str, Any]]  # [{text}]
    citations: List[Dict[str, Any]]  # mirror of sources or citation map
    selected_sentences: List[str]

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
        reasoning=False, # only for thinking models
        base_url="http://localhost:11434",
        temperature=temperature,
        system=system
    )

class breakdown_structure(BaseModel):
    summary: str
    search_queries: List[str]
    reasoning_points: List[str]

class reasoning_structure(BaseModel):
    reasoning_analysis: List[str]  # List of logical analysis points
    conclusions: List[str]  # Key conclusions from reasoning

class scientific_structure(BaseModel):
    facts: List[Dict[str, str]]  # List of {"text": "fact content"} objects

class creative_structure(BaseModel):
    final_answer: str

# Create agents with configurable models
def create_breakdown_agent(model_config):
    model = model_config.get("reasoning_model", "mistral:7b")
    llm = create_llm(model, system=BREAKDOWN_PROMPT)
    return create_react_agent(llm, [], response_format=breakdown_structure, state_schema=AgentState)

def create_reasoning_agent(model_config):
    model = model_config.get("reasoning_model", "mistral:7b")
    llm = create_llm(model, system=REASONING_PROMPT)
    return create_react_agent(llm, [], response_format=reasoning_structure, state_schema=AgentState)

def create_scientific_agent(model_config):
    model = model_config.get("scientific_model", "mistral:7b")
    llm = create_llm(model, system=SCIENTIFIC_PROMPT)
    tools = [web_search]
    return create_react_agent(llm, tools, response_format=scientific_structure, state_schema=AgentState)

def create_creative_agent(model_config):
    model = model_config.get("creative_model", "mistral:7b")
    llm = create_llm(model, system=CREATIVE_PROMPT)
    return create_react_agent(llm, [], response_format=creative_structure, state_schema=AgentState)

# Agent wrapper functions
def breakdown_node(state):
    agent = create_breakdown_agent(state["model_config"])
    messages = [HumanMessage(content=state["query"])]
    result = agent.invoke({"messages": messages})
    
    # Get structured output directly from structured_response
    structured_output = result["structured_response"]
    
    return {
        "breakdown_output": structured_output.summary,
        "search_queries": structured_output.search_queries,
        "reasoning_points": structured_output.reasoning_points
    }

# Removed old parsing functions - now using structured outputs directly

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
    structured_output = result["structured_response"]
    
    # Convert structured output to readable format for downstream agents
    reasoning_text = "\n".join(structured_output.reasoning_analysis + structured_output.conclusions)
    
    return {
        "reasoning_output": reasoning_text,
        "structured_reasoning": structured_output
    }

def scientific_node(state):
    agent = create_scientific_agent(state["model_config"])
    
    # Get search queries from breakdown agent
    queries = state.get("search_queries", [])
    
    # Fallback to original query if no search queries provided
    if not queries:
        queries = [state.get("query", "")]
    
    # Create context for the scientific agent
    context = f"""Original Query: {state.get('query', '')}

Search Queries to Research:
{chr(10).join(f'- {query}' for query in queries)}

Use web search to gather scientific facts and return them in the structured format."""
    
    messages = [HumanMessage(content=context)]
    result = agent.invoke({"messages": messages})
    
    # Get structured output directly from structured_response
    structured_output = result["structured_response"]
    
    return {
        "scientific_output": str(structured_output),
        "extracted_facts": structured_output.facts,
    }

def creative_node(state):
    agent = create_creative_agent(state["model_config"])
    
    # Get inputs from both agents
    reasoning_output = state.get("reasoning_output", "")
    facts = state.get("extracted_facts", [])
    
    # Extract fact text for synthesis
    fact_texts = []
    for f in facts:
        if isinstance(f, dict) and "text" in f:
            text = f.get("text", "").strip()
            if text:
                fact_texts.append(text)
    
    # Limit facts to target count
    selected_facts = fact_texts[:CREATIVE_SENTENCE_TARGET]
    
    # Create comprehensive context for Creative Agent
    context = f"""User Query: {state.get('query', '')}

Reasoning Analysis:
{reasoning_output}

Scientific Facts:
{chr(10).join(f'- {fact}' for fact in selected_facts)}

Create a final answer that synthesizes the reasoning and facts. Explain like to a 5 year old."""
    
    messages = [HumanMessage(content=context)]
    result = agent.invoke({"messages": messages})
    
    # Get structured output directly from structured_response
    structured_output = result["structured_response"]
    
    return {
        "final_answer": structured_output.final_answer, 
        "selected_sentences": selected_facts,
        "reasoning_used": reasoning_output,
        "structured_creative": structured_output
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
        if isinstance(fact, dict) and 'text' in fact:
            fact_text = fact['text'][:100] + ('...' if len(fact['text']) > 100 else '')
            print(f"      {i}. {fact_text}")
    if len(facts) > 5:
        print(f"      ... and {len(facts) - 5} more facts")
    
    # Creative Agent Results
    print(f"\n🎨 CREATIVE AGENT:")
    final_answer = final_state.get('final_answer', 'N/A')
    print(f"   Final Answer: {final_answer}")
    
    # Statistics
    print(f"\n📊 STATISTICS:")
    print(f"   Search Queries Generated: {len(search_queries)}")
    print(f"   Reasoning Points: {len(reasoning_points)}")
    print(f"   Scientific Facts: {len(facts)}")
    print(f"   Final Answer Length: {len(final_answer) if final_answer != 'N/A' else 0} characters")
    
    print(f"{'='*80}")

# Create and run the graph
def create_graph():
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("breakdown", breakdown_node)
    workflow.add_node("reasoning", reasoning_node)
    workflow.add_node("scientific", scientific_node)  
    workflow.add_node("creative", creative_node)
    
    # Flow: start -> breakdown -> (reasoning, scientific) -> creative
    workflow.add_edge(START, "breakdown")
    workflow.add_edge("breakdown", "reasoning")
    workflow.add_edge("breakdown", "scientific")
    workflow.add_edge("reasoning", "creative")
    workflow.add_edge("scientific", "creative")
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
        search_results_text = str(state.get("search_results", ""))[:500] + "..." if len(str(state.get("search_results", ""))) > 500 else str(state.get("search_results", ""))
        
        # Extract facts
        facts_data = state.get("extracted_facts", {})
        if isinstance(facts_data, dict) and "facts" in facts_data:
            facts_list = facts_data["facts"]
            facts_count = len(facts_list)
            facts_text = " | ".join([f"{i+1}. {fact}" for i, fact in enumerate(facts_list)])
        else:
            facts_count = 0
            facts_text = "No facts extracted"
        
        # Extract reasoning
        reasoning_data = state.get("reasoning_analysis", {})
        if isinstance(reasoning_data, dict) and "reasoning_points" in reasoning_data:
            reasoning_points = reasoning_data["reasoning_points"]
            reasoning_text = " | ".join([f"{i+1}. {point}" for i, point in enumerate(reasoning_points)])
        else:
            reasoning_text = "No reasoning analysis"
        
        # Final answer
        final_answer_data = state.get("final_answer", {})
        if isinstance(final_answer_data, dict) and "final_answer" in final_answer_data:
            final_answer = final_answer_data["final_answer"]
        else:
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
            len(reasoning_data.get("reasoning_points", [])) if isinstance(reasoning_data, dict) else 0,
            truncate_text(final_answer),
            truncate_text(search_results_text),
            truncate_text(facts_text),
            truncate_text(reasoning_text),
            config_info
        ]
        
        # Append row
        sheet.append_row(new_row)
        
        print(f"✅ Detailed agent state saved to Excel at {timestamp}")
        print(f"   📊 Facts extracted: {facts_count}")
        print(f"   🧠 Reasoning points: {len(reasoning_data.get('reasoning_points', [])) if isinstance(reasoning_data, dict) else 0}")
        print(f"   📝 Final answer length: {len(final_answer)} characters")
        
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
            # Initialize state
            initial_state = {
                "query": question,
                "breakdown_output": "",
                "reasoning_output": "",
                "scientific_output": "",
                "final_answer": "",
                "messages": [],
                "model_config": model_config,
                "remaining_steps": 0,
                "structured_response": None,
                "search_queries": [],
                "reasoning_points": [],
                "sources": [],
                "extracted_facts": [],
                "citations": [],
                "selected_sentences": []
            }
            
            # Create and run graph
            app = create_graph()
            result = app.invoke(initial_state)
            
            answer = result["final_answer"]
            answers.append(answer)
            
            print(f"A: {answer}")
            
            # Save detailed state to Excel for this question
            save_agent_state_to_excel(question, result)
            
        except Exception as e:
            error_msg = f"Error processing question: {str(e)}"
            print(f"❌ {error_msg}")
            answers.append(error_msg)
    
    # Save answers to sheet
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    column_name = f"{config_name}_{timestamp}"
    
    success = save_answers_to_sheet(worksheet, answers, column_name)
    if success:
        print(f"\n✅ All answers saved to Google Sheets in column '{column_name}'")
    else:
        print(f"\n❌ Failed to save answers to Google Sheets")
    
    return answers

def test_google_sheets_writing(config_name="test_config"):
    """Test function to verify Google Sheets writing functionality without processing questions"""
    print(f"\n{'='*80}")
    print(f"TESTING GOOGLE SHEETS WRITING FUNCTIONALITY")
    print(f"Configuration: {config_name}")
    print(f"{'='*80}")
    
    # Setup Google Sheets
    client = setup_google_sheets()
    if not client:
        print("❌ Failed to setup Google Sheets connection")
        return False
    
    try:
        # Open the spreadsheet
        spreadsheet = client.open(GOOGLE_SHEETS_CONFIG["spreadsheet_name"])
        worksheet = spreadsheet.worksheet(GOOGLE_SHEETS_CONFIG["worksheet_name"])
        
        print("✅ Successfully connected to Google Sheets")
        
        # Create test data
        test_answers = [
            "This is test answer 1 for Google Sheets integration.",
            "This is test answer 2 to verify batch writing works correctly.",
            "Test answer 3: The system can handle multiple responses at once.",
            "Final test answer 4: All systems operational!"
        ]
        
        print(f"📝 Created {len(test_answers)} test answers")
        
        # Generate timestamp for unique column name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        column_name = f"{config_name}_{timestamp}"
        
        print(f"🔧 Testing column creation and data writing for: {column_name}")
        
        # Test the save function
        success = save_answers_to_sheet(worksheet, test_answers, column_name)
        
        if success:
            print(f"\n✅ TEST PASSED: Successfully wrote test data to Google Sheets!")
            print(f"   Column: {column_name}")
            print(f"   Rows written: {len(test_answers)}")
            
            # Try to read back the data to verify
            try:
                # Find the column number
                headers = worksheet.row_values(1)
                col_num = None
                for i, header in enumerate(headers):
                    if header == column_name:
                        col_num = i + 1
                        break
                
                if col_num:
                    col_letter = get_column_letter(col_num)
                    # Read back the written data
                    written_data = worksheet.col_values(col_num)[1:]  # Skip header
                    
                    print(f"🔍 Verification: Read back {len(written_data)} entries from column {col_letter}")
                    
                    # Compare with what we wrote
                    if len(written_data) == len(test_answers):
                        print("✅ Data length matches!")
                        for i, (original, read_back) in enumerate(zip(test_answers, written_data)):
                            if original.strip() == read_back.strip():
                                print(f"   ✅ Row {i+2}: Data matches")
                            else:
                                print(f"   ⚠️  Row {i+2}: Data mismatch")
                                print(f"      Original: {original}")
                                print(f"      Read back: {read_back}")
                    else:
                        print(f"⚠️  Data length mismatch: wrote {len(test_answers)}, read {len(written_data)}")
                else:
                    print("⚠️  Could not find the created column for verification")
                    
            except Exception as e:
                print(f"⚠️  Verification failed: {e}")
            
            return True
        else:
            print(f"\n❌ TEST FAILED: Could not write test data to Google Sheets")
            return False
            
    except Exception as e:
        print(f"❌ TEST FAILED: Error during testing: {e}")
        return False

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
        "remaining_steps": 0,
        "structured_response": None,
        "search_queries": [],
        "reasoning_points": [],
        "sources": [],
        "extracted_facts": [],
        "citations": [],
        "selected_sentences": []
    }
    
    # Create and run graph
    app = create_graph()
    result = app.invoke(initial_state)

    # Print comprehensive state information
    print_agent_state(result)
    
    # Optional graph visualization (silent)
    try:
        # Save graph files without printing details
        try:
            graph_png = app.get_graph().draw_png()
            with open("workflow_graph.png", "wb") as f:
                f.write(graph_png)
        except:
            pass
        
        try:
            mermaid_code = app.get_graph().draw_mermaid()
            with open("workflow_graph.mmd", "w") as f:
                f.write(mermaid_code)
        except:
            pass
            
    except:
        pass

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
            
        elif mode == "test":
            # Test Google Sheets writing functionality
            config_name = sys.argv[2] if len(sys.argv) > 2 else "test_config"
            print(f"🧪 Testing Google Sheets writing functionality with config: {config_name}")
            test_google_sheets_writing(config_name)
            
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
            print("  python main.py test [config_name]             # Test Google Sheets writing functionality") 
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