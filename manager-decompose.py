from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from openai import AzureOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-05-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    timeout=10
)

DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

class State(TypedDict, total=False):
    input: str
    subtasks: List[str]
    result1: str
    result2: str
    final: str

# ------------------------
# MANAGER (DECOMPOSE)
# ------------------------

def manager(state):
    prompt = f"Break into 2 subtasks: {state['input']}"

    res = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[{"role": "user", "content": prompt}]
    )

    tasks = res.choices[0].message.content.split("\n")

    return {"subtasks": tasks[:2]}

# ------------------------
# WORKERS (PARALLEL)
# ------------------------

def worker1(state):
    task = state["subtasks"][0]
    return {"result1": f"Worker1 done: {task}"}

def worker2(state):
    task = state["subtasks"][1]
    return {"result2": f"Worker2 done: {task}"}

# ------------------------
# FINAL
# ------------------------

def final(state):
    return {
        "final": f"""
{subtasks if (subtasks := state.get('subtasks')) else ""}
Result1: {state['result1']}
Result2: {state['result2']}
"""
    }

# ------------------------
# GRAPH
# ------------------------

builder = StateGraph(State)

builder.add_node("manager", manager)
builder.add_node("worker1", worker1)
builder.add_node("worker2", worker2)
builder.add_node("final", final)

builder.set_entry_point("manager")

# parallel fan-out
builder.add_edge("manager", "worker1")
builder.add_edge("manager", "worker2")

# merge
builder.add_edge("worker1", "final")
builder.add_edge("worker2", "final")

builder.add_edge("final", END)

graph = builder.compile()

state = {"input": "Build an e-commerce system"}
result = graph.invoke(state)

print(result["final"])
