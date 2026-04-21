from typing import TypedDict
from langgraph.graph import StateGraph, END
from graphviz import Digraph
import datetime

# ------------------------
# STATE (IMPORTANT)
# ------------------------

class State(TypedDict, total=False):
    input: str
    next: str
    result: str
    tool: str
    output: str

# ------------------------
# LOGGING
# ------------------------

def log(step, state):
    print(f"[{datetime.datetime.now()}] Step: {step} | State: {state}")

# ------------------------
# ROUTER
# ------------------------

def router(state: State):
    log("Router", state)

    user_input = state.get("input", "")

    if any(char.isdigit() for char in user_input):
        return {"next": "calculator"}
    else:
        return {"next": "search"}

# ------------------------
# TOOLS
# ------------------------

def calculator(state: State):
    log("Calculator Tool", state)

    user_input = state.get("input", "")

    try:
        result = str(eval(user_input))
    except:
        result = "Error in calculation"

    return {
        "result": result,
        "tool": "calculator"
    }


def search(state: State):
    log("Search Tool", state)

    user_input = state.get("input", "")

    return {
        "result": "Search result for: " + user_input,
        "tool": "search"
    }

# ------------------------
# FINAL
# ------------------------

def final(state: State):
    log("Final", state)

    print(f"\nTool used: {state.get('tool')}")

    return {
        "output": state.get("result", "No result")
    }

# ------------------------
# GRAPH BUILD
# ------------------------

builder = StateGraph(State)

builder.add_node("router", router)
builder.add_node("calculator", calculator)
builder.add_node("search", search)
builder.add_node("final", final)

builder.set_entry_point("router")

builder.add_conditional_edges(
    "router",
    lambda state: state["next"],
    {
        "calculator": "calculator",
        "search": "search"
    }
)

builder.add_edge("calculator", "final")
builder.add_edge("search", "final")
builder.add_edge("final", END)

graph = builder.compile()

# ------------------------
# VISUALIZATION
# ------------------------

def visualize():
    dot = Digraph()

    dot.node("Input")
    dot.node("Router")
    dot.node("Calculator")
    dot.node("Search")
    dot.node("Final")

    dot.edge("Input", "Router")
    dot.edge("Router", "Calculator")
    dot.edge("Router", "Search")
    dot.edge("Calculator", "Final")
    dot.edge("Search", "Final")

    dot.render("tool_agent_graph", format="png", cleanup=True)
    print("\nGraph saved as tool_agent_graph.png")

# ------------------------
# RUN
# ------------------------

if __name__ == "__main__":
    visualize()

    while True:
        user_input = input("\nAsk: ")

        if user_input.lower() == "exit":
            break

        # IMPORTANT: pass normal dict (NOT custom class)
        result = graph.invoke({"input": user_input})

        print("Output:", result["output"])

