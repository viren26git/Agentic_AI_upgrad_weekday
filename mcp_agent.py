from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END
from openai import AzureOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-05-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

# ------------------------
# MCP STATE (STRUCTURED)
# ------------------------

class MCPState(TypedDict, total=False):
    task: str
    user: str
    constraints: List[str]
    plan: str
    execution: str
    history: List[str]

# ------------------------
# AGENT 1: PLANNER
# ------------------------

def planner(state: MCPState):
    prompt = f"""
Task: {state['task']}
Constraints: {state['constraints']}

Create a structured plan.
"""

    res = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[{"role": "user", "content": prompt}]
    )

    history = state.get("history", [])
    history.append("Planner executed")

    return {
        "plan": res.choices[0].message.content,
        "history": history
    }

# ------------------------
# AGENT 2: EXECUTOR
# ------------------------

def executor(state: MCPState):
    prompt = f"""
Execute this plan:
{state['plan']}
"""

    res = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[{"role": "user", "content": prompt}]
    )

    history = state.get("history", [])
    history.append("Executor executed")

    return {
        "execution": res.choices[0].message.content,
        "history": history
    }

# ------------------------
# FINAL
# ------------------------

def final(state: MCPState):
    return {
        "result": f"""
User: {state['user']}
Task: {state['task']}

Plan:
{state['plan']}

Execution:
{state['execution']}

History:
{state['history']}
"""
    }

# ------------------------
# GRAPH
# ------------------------

builder = StateGraph(MCPState)

builder.add_node("planner", planner)
builder.add_node("executor", executor)
builder.add_node("final", final)

builder.set_entry_point("planner")

builder.add_edge("planner", "executor")
builder.add_edge("executor", "final")
builder.add_edge("final", END)

graph = builder.compile()

# ------------------------
# RUN
# ------------------------

state = {
    "task": "Build secure login system",
    "user": "admin",
    "constraints": ["secure", "fast"],
    "history": []
}

result = graph.invoke(state)

print(result["result"])
