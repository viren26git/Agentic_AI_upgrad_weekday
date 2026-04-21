from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from graphviz import Digraph
import datetime

# ------------------------
# STATE (IMPORTANT)
# ------------------------

class State(TypedDict, total=False):
    input: str
    history: List[str]
    output: str

# ------------------------
# LOGGING
# ------------------------

def log(step, state):
    print(f"[{datetime.datetime.now()}] Step: {step} | State: {state}")

# ------------------------
# MEMORY NODE
# ------------------------

def memory_node(state: State):
    log("Memory Node", state)

    history = state.get("history", [])
    history = history + [state.get("input", "")]   # safe append

    print("Memory:", history)

    return {
        "history": history
    }

# ------------------------
# PROCESS NODE
# ------------------------

def process(state: State):
    log("Process Node", state)

    user_input = state.get("input", "")

    return {
        "output": "Processed: " + user_input
    }

# ------------------------
# FINAL NODE
# ------------------------

def final(state: State):
    log("Final Node", state)

    print("Tracing: Completed flow")

    return {
        "output": state.get("output", "")
    }

# ------------------------
# BUILD GRAPH
# ------------------------

builder = StateGraph(State)

builder.add_node("memory", memory_node)
builder.add_node("process", process)
builder.add_node("final", final)

builder.set_entry_point("memory")

builder.add_edge("memory", "process")
builder.add_edge("process", "final")
builder.add_edge("final", END)

graph = builder.compile()

# ------------------------
# RUN WITH MEMORY
# ------------------------

if __name__ == "__main__":

    state = {
        "input": "Hello",
        "history": []
    }

    result = graph.invoke(state)
    print("Output:", result["output"])

    # second run (memory persists)
    state["input"] = "Next step"

    result = graph.invoke(state)
    print("Output:", result["output"])

# ------------------------
# VISUALIZATION
# ------------------------

def visualize():
    dot = Digraph()

    dot.node("Memory")
    dot.node("Process")
    dot.node("Final")

    dot.edge("Memory", "Process")
    dot.edge("Process", "Final")

    dot.render("memory_agent_graph", format="png", cleanup=True)
    print("\nGraph saved as memory_agent_graph.png")

visualize()
