from typing import TypedDict
from langgraph.graph import StateGraph, END
from openai import AzureOpenAI
from dotenv import load_dotenv
import os
import time

load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-02-01",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

class State(TypedDict, total=False):
    input: str
    agent1: str
    agent2: str
    logs: list
    final: str

# ------------------------
# MONITOR WRAPPER
# ------------------------

def monitor(name, func):
    def wrapper(state):
        start = time.time()
        result = func(state)
        end = time.time()

        log = {
            "agent": name,
            "time": round(end - start, 3),
            "output_keys": list(result.keys())
        }

        logs = state.get("logs", [])
        logs.append(log)

        return {**result, "logs": logs}

    return wrapper

# ------------------------
# AGENTS
# ------------------------

def agent1(state):
    prompt = f"Explain: {state['input']}"
    res = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[{"role": "user", "content": prompt}]
    )
    return {"agent1": res.choices[0].message.content}

def agent2(state):
    prompt = f"Give examples: {state['input']}"
    res = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[{"role": "user", "content": prompt}]
    )
    return {"agent2": res.choices[0].message.content}

# ------------------------
# FINAL
# ------------------------

def final(state):
    return {
        "final": f"""
Output1: {state['agent1']}
Output2: {state['agent2']}

Logs:
{state['logs']}
"""
    }

# ------------------------
# GRAPH
# ------------------------

builder = StateGraph(State)

builder.add_node("agent1", monitor("Agent1", agent1))
builder.add_node("agent2", monitor("Agent2", agent2))
builder.add_node("final", final)

builder.set_entry_point("agent1")

builder.add_edge("agent1", "agent2")
builder.add_edge("agent2", "final")
builder.add_edge("final", END)

graph = builder.compile()

state = {"input": "AI in healthcare"}
state = graph.invoke(state)

print(state["final"])
