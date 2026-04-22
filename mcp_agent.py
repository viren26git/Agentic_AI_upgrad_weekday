from typing import TypedDict, List, Dict
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
# MCP STATE (STRUCTURED)
# ------------------------

class MCPState(TypedDict, total=False):
    task: str
    user: str
    constraints: List[str]
    plan: str
    execution: str
    history: List[str]
    result: str

# ------------------------
# AGENT 1: PLANNER
# ------------------------

def planner(state: MCPState):
    print(" PLanner running !!!")

    task= state.get("task","")
    constraints= state.get("constraints",[])

    prompt= f"Task: {task} \n Constraints: {constraints}\n Create a plan"

    try:

        res = client.chat.completions.create(
            model=DEPLOYMENT,
            messages=[{"role": "user", "content": prompt}]
        )

        plan = res.choices[0].message.content
    
    except Exception as e:
        plan = f"Planner error :{str(e)}"

    history = state.get("history", [])
    history.append("Planner executed")

    return {
        "plan": plan,
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

    print(" Final node is running !!")

    result = f"""

    User: {state.get('user')}
    Task: {state.get('task')}

    Plan:
    {state.get('plan')}

    Execution:
    {state.get('execution')}

    History:
    {state.get('history')}
    """
    
    return {"result" : result}

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

if __name__ == "__main__":

    state = {
        "task": "Build secure login system",
        "user": "admin",
        "constraints": ["secure", "fast"],
        "history": []
    }

    result = graph.invoke(state)
    print("\n Final Output:\n")
    print(result["result"])
