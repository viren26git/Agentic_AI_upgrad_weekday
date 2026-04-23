from typing import TypedDict
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
    summary: str
    risks: str
    final: str

# ------------------------
# WORKERS
# ------------------------

def summary_agent(state):
    prompt = f"Summarize: {state['input']}"
    res = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[{"role": "user", "content": prompt}]
    )
    return {"summary": res.choices[0].message.content}

def risk_agent(state):
    prompt = f"List risks: {state['input']}"
    res = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[{"role": "user", "content": prompt}]
    )
    return {"risks": res.choices[0].message.content}

# ------------------------
# MANAGER
# ------------------------

def manager(state):
    prompt = f"""
Combine:
Summary: {state['summary']}
Risks: {state['risks']}
"""

    res = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[{"role": "user", "content": prompt}]
    )

    return {"final": res.choices[0].message.content}

# ------------------------
# GRAPH
# ------------------------

builder = StateGraph(State)

builder.add_node("summary", summary_agent)
builder.add_node("risk", risk_agent)
builder.add_node("manager", manager)

builder.set_entry_point("summary")

builder.add_edge("summary", "risk")
builder.add_edge("risk", "manager")
builder.add_edge("manager", END)

graph = builder.compile()

# RUN
state = {"input": "Deploy payment system"}
result = graph.invoke(state)

print(result["final"])
