from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from openai import AzureOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-02-01",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

# ------------------------
# MCP STATE
# ------------------------

class MCPState(TypedDict, total=False):
    task: str
    priority: str
    constraints: List[str]
    decision: str
    output: str

# ------------------------
# ANALYZER (reads MCP)
# ------------------------

def analyzer(state: MCPState):
    prompt = f"""
Task: {state['task']}
Priority: {state['priority']}
Constraints: {state['constraints']}

Decide approach.
"""

    res = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[{"role": "user", "content": prompt}]
    )

    return {"decision": res.choices[0].message.content}

# ------------------------
# EXECUTION BASED ON MCP
# ------------------------

def executor(state: MCPState):

    if "secure" in state["constraints"]:
        approach = "Using secure architecture"
    else:
        approach = "Using basic architecture"

    return {
        "output": f"""
Decision:
{state['decision']}

Execution:
{approach}
"""
    }

# ------------------------
# GRAPH
# ------------------------

builder = StateGraph(MCPState)

builder.add_node("analyzer", analyzer)
builder.add_node("executor", executor)

builder.set_entry_point("analyzer")

builder.add_edge("analyzer", "executor")
builder.add_edge("executor", END)

graph = builder.compile()

# ------------------------
# RUN
# ------------------------

state = {
    "task": "Deploy payment system",
    "priority": "high",
    "constraints": ["secure", "scalable"]
}

result = graph.invoke(state)

print(result["output"])
