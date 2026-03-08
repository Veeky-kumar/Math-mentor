import json
from typing import Dict, TypedDict, List
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from config.settings import GROQ_API_KEY
from rag.retriever import retrieve_context
from tools.sympy_tools import evaluate_expression

# The state dictionary for the LangGraph
class MathWorkflowState(TypedDict):
    original_text: str
    parsed_problem: dict
    topic: str
    retrieved_context: list
    solver_output: str
    is_verified: bool
    verification_feedback: str
    verification_attempts: int
    final_explanation: str
    hitl_required: bool

# Initialize ChatGroq LLM
llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile", groq_api_key=GROQ_API_KEY)
llm_json = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile", groq_api_key=GROQ_API_KEY).bind(response_format={"type": "json_object"})

def parser_agent(state: MathWorkflowState) -> MathWorkflowState:
    '''Cleans OCR/ASR output and converts math question to structured JSON.'''
    prompt = f"""
    You are a math problem parser. You will receive raw, messy text from OCR or audio transcripts.
    Clean the text and structure it into a JSON object with:
    - 'question': the clean mathematical question
    - 'given_variables': any variables provided
    - 'goal': what needs to be found
    If the text is entirely unintelligible, return {{"ambiguous": true}}.
    Return ONLY a valid JSON object.
    
    Raw Text: {state['original_text']}
    """
    
    response = llm_json.invoke(prompt)
    try:
        parsed = json.loads(response.content)
    except:
        parsed = {"question": state['original_text'], "error": "failed to parse json"}
        
    state["parsed_problem"] = parsed
    
    # Check if HITL needed
    if parsed.get("ambiguous", False) or "error" in parsed:
        state["hitl_required"] = True
    else:
        state["hitl_required"] = False
        
    return state

def router_agent(state: MathWorkflowState) -> MathWorkflowState:
    '''Classifies the math topic: algebra / probability / calculus / linear algebra'''
    # Assume hitl is handled in UI, if workflow resumes, continue or we skip if hitl req
    prompt = f"""
    Classify the following math problem into exactly ONE of these categories:
    algebra, probability, calculus, linear algebra, general
    
    Problem: {state['parsed_problem'].get('question', '')}
    Return ONLY the category word.
    """
    response = llm.invoke(prompt)
    state["topic"] = response.content.strip().lower()
    return state

def retrieval_step(state: MathWorkflowState) -> MathWorkflowState:
    '''Retrieve context based on parsed question'''
    query = state['parsed_problem'].get('question', '')
    context = retrieve_context(query)
    state["retrieved_context"] = context
    return state

def solver_agent(state: MathWorkflowState) -> MathWorkflowState:
    '''Uses RAG retrieved math formulas and Python tools to solve'''
    question = state['parsed_problem'].get('question', '')
    context_str = "\\n".join([f"- {c['content']} (Topic: {c['topic']})" for c in state.get('retrieved_context', [])])
    
    feedback = state.get("verification_feedback", "")
    feedback_prompt = f"Previous attempts had an error: {feedback}\\nPlease fix it." if feedback else ""
    
    prompt = f"""
    You are an expert Math Solver. Solve the problem step-by-step.
    
    Problem: {question}
    
    Relevant Formulas / Math Rules:
    {context_str}
    
    {feedback_prompt}
    
    Show your work clearly. If an equation needs evaluation, show the algebraic steps.
    """
    response = llm.invoke(prompt)
    state["solver_output"] = response.content
    return state

def verifier_agent(state: MathWorkflowState) -> MathWorkflowState:
    '''Checks correctness, constraints, calculation errors'''
    prompt = f"""
    You are a math verification agent. Check the proposed solution for:
    - calculation errors
    - edge cases neglected
    - logical correctness
    
    Original Problem: {state['parsed_problem'].get('question', '')}
    Proposed Solution: {state['solver_output']}
    
    If it is 100% correct and robust, reply EXACTLY with valid JSON: {{"is_correct": true, "feedback": "Looks good"}}
    If there is an error, reply EXACTLY with valid JSON describing the error: {{"is_correct": false, "feedback": "<detailed error>"}}
    """
    response = llm_json.invoke(prompt)
    try:
        result = json.loads(response.content)
        state["is_verified"] = result.get("is_correct", False)
        state["verification_feedback"] = result.get("feedback", "")
    except:
        state["is_verified"] = False
        state["verification_feedback"] = "Could not verify properly."
        
    state["verification_attempts"] = state.get("verification_attempts", 0) + 1
    
    if not state["is_verified"] and state["verification_attempts"] >= 3:
        # Prevent infinite loop
        state["hitl_required"] = True
        
    return state

def explainer_agent(state: MathWorkflowState) -> MathWorkflowState:
    '''Produces student-friendly step-by-step explanation'''
    prompt = f"""
    You are a friendly Math Mentor. Based on the verified solution, explain it to a high school student in a clear, encouraging, step-by-step manner. Use markdown for math formatting (e.g., $x^2$).
    
    Problem: {state['parsed_problem'].get('question', '')}
    Solution: {state['solver_output']}
    """
    response = llm.invoke(prompt)
    state["final_explanation"] = response.content
    return state

# Define conditional edges
def route_after_parser(state: MathWorkflowState) -> str:
    if state.get("hitl_required"):
        return END
    return "router"

def route_after_verifier(state: MathWorkflowState) -> str:
    if state.get("is_verified"):
        return "explainer"
    if state.get("hitl_required", False):
        return END
    # loop back to solver to fix error
    return "solver"

# Build Graph
builder = StateGraph(MathWorkflowState)

# Add Nodes
builder.add_node("parser", parser_agent)
builder.add_node("router", router_agent)
builder.add_node("retriever", retrieval_step)
builder.add_node("solver", solver_agent)
builder.add_node("verifier", verifier_agent)
builder.add_node("explainer", explainer_agent)

# Add Edges
builder.add_edge(START, "parser")
builder.add_conditional_edges("parser", route_after_parser, {"router": "router", END: END})
builder.add_edge("router", "retriever")
builder.add_edge("retriever", "solver")
builder.add_edge("solver", "verifier")
builder.add_conditional_edges("verifier", route_after_verifier, {"explainer": "explainer", "solver": "solver", END: END})
builder.add_edge("explainer", END)

workflow = builder.compile()

def run_math_workflow(initial_text: str):
    initial_state = {
        "original_text": initial_text,
        "parsed_problem": {},
        "topic": "",
        "retrieved_context": [],
        "solver_output": "",
        "is_verified": False,
        "verification_feedback": "",
        "verification_attempts": 0,
        "final_explanation": "",
        "hitl_required": False
    }
    
    return workflow.invoke(initial_state)
