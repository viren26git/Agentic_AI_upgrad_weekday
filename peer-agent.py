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
    draft: str
    feedback: str
    refined: str

# ------------------------
# AGENT A (CREATOR)
# ------------------------

def agent_a(state):
    prompt = f"Create solution: {state['input']}"

    res = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[{"role": "user", "content": prompt}]
    )

    return {"draft": res.choices[0].message.content}

# ------------------------
# AGENT B (REVIEWER)
# ------------------------

def agent_b(state):
    prompt = f"Review this:\n{state['draft']}"

    res = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[{"role": "user", "content": prompt}]
    )

    return {"feedback": res.choices[0].message.content}

# ------------------------
# AGENT A REFINES
# ------------------------

def refine(state):
    prompt = f"""
Improve based on feedback:
{state['draft']}
Feedback: {state['feedback']}
"""

    res = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[{"role": "user", "content": prompt}]
    )

    return {"refined": res.choices[0].message.content}

# ------------------------
# GRAPH
# ------------------------

builder = StateGraph(State)

builder.add_node("agent_a", agent_a)
builder.add_node("agent_b", agent_b)
builder.add_node("refine", refine)

builder.set_entry_point("agent_a")

builder.add_edge("agent_a", "agent_b")
builder.add_edge("agent_b", "refine")
builder.add_edge("refine", END)

graph = builder.compile()

state = {"input": "Design login system"}
result = graph.invoke(state)

print(result["refined"])
