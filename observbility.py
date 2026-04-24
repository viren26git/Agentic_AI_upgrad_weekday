from typing import TypedDict, List, Annotated
from langgraph.graph import StateGraph, END
import operator
import time
import random

# ------------------------
# STATE (MERGE SAFE)
# ------------------------

class State(TypedDict, total=False):
    input: str
    selected_agent: str
    output: str

    # observability
    logs: Annotated[List[str], operator.add]
    trace: Annotated[List[str], operator.add]
    metrics: Annotated[List[dict], operator.add]

# ------------------------
# OBSERVABILITY HELPERS
# ------------------------

def log(step):
    msg = f"[LOG] {step}"
    print(msg)
    return [msg]

def trace(step):
    msg = f"[TRACE] {step}"
    print(msg)
    return [msg]

def metric(agent_name, start_time):
    duration = round(time.time() - start_time, 3)
    return [{
        "agent": agent_name,
        "latency": duration
    }]

# ------------------------
# ROUTER
# ------------------------

def router(state: State):
    t = time.time()

    logs = log("Router deciding agent")
    traces = trace("Router → Agent Pool")

    agents = ["agent1", "agent2", "agent3"]
    selected = random.choice(agents)

    print(f"Selected: {selected}")

    return {
        "selected_agent": selected,
        "logs": logs,
        "trace": traces,
        "metrics": metric("router", t)
    }

# ------------------------
# AGENT POOL
# ------------------------

def agent1(state: State):
    if state["selected_agent"] != "agent1":
        return {}

    t = time.time()

    logs = log("Agent1 processing")
    traces = trace("Router → Agent1")

    time.sleep(1)

    return {
        "output": "Agent1 result",
        "logs": logs,
        "trace": traces,
        "metrics": metric("agent1", t)
    }

def agent2(state: State):
    if state["selected_agent"] != "agent2":
        return {}

    t = time.time()

    logs = log("Agent2 processing")
    traces = trace("Router → Agent2")

    time.sleep(2)

    return {
        "output": "Agent2 result",
        "logs": logs,
        "trace": traces,
        "metrics": metric("agent2", t)
    }

def agent3(state: State):
    if state["selected_agent"] != "agent3":
        return {}

    t = time.time()

    logs = log("Agent3 processing")
    traces = trace("Router → Agent3")

    time.sleep(1.5)

    return {
        "output": "Agent3 result",
        "logs": logs,
        "trace": traces,
        "metrics": metric("agent3", t)
    }

# ------------------------
# FINAL NODE
# ------------------------

def final(state: State):
    logs = log("Final aggregation")
    traces = trace("Agent → Final")

    return {
        "logs": logs,
        "trace": traces,
        "output": state.get("output", "No output")
    }

# ------------------------
# GRAPH BUILD
# ------------------------

builder = StateGraph(State)

builder.add_node("router", router)
builder.add_node("agent1", agent1)
builder.add_node("agent2", agent2)
builder.add_node("agent3", agent3)
builder.add_node("final", final)

builder.set_entry_point("router")

# fan-out (pool)
builder.add_edge("router", "agent1")
builder.add_edge("router", "agent2")
builder.add_edge("router", "agent3")

# merge
builder.add_edge("agent1", "final")
builder.add_edge("agent2", "final")
builder.add_edge("agent3", "final")

builder.add_edge("final", END)

graph = builder.compile()

# ------------------------
# RUN
# ------------------------

if __name__ == "__main__":

    state = {
        "input": "Process request"
    }

    result = graph.invoke(state)

    print("\n--- FINAL OUTPUT ---")
    print(result["output"])

    print("\n--- TRACE ---")
    for t in result["trace"]:
        print(t)

    print("\n--- LOGS ---")
    for l in result["logs"]:
        print(l)

    print("\n--- METRICS ---")
    for m in result["metrics"]:
        print(m)
