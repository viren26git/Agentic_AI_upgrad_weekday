from langgraph.graph import StateGraph
from graphviz import Digraph

# -------------------------------
# STATE
# -------------------------------
class State(dict):
    pass

# -------------------------------
# NODE 1: ANALYZE
# -------------------------------
def analyze(state):
    user_input = state["input"]

    if "delete" in user_input:
        state["risk"] = "high"
    else:
        state["risk"] = "low"

    print("Analyze: risk =", state["risk"])
    return state

# -------------------------------
# NODE 2: HUMAN APPROVAL
# -------------------------------
def approval(state):
    if state["risk"] == "high":
        print("Approval Required ❗")
        state["approved"] = False   # simulate rejection
    else:
        state["approved"] = True

    return state

# -------------------------------
# NODE 3: RETRY LOGIC
# -------------------------------
def retry_node(state):
    print("Retrying...")
    for i in range(2):
        print(f"Attempt {i+1}")

    state["result"] = "Operation failed after retries"
    return state

# -------------------------------
# NODE 4: FINAL
# -------------------------------
def final_node(state):
    if state.get("approved"):
        output = "Operation Successful"
    else:
        output = state.get("result", "Rejected")

    print("Final:", output)
    return {"output": output}

# -------------------------------
# BUILD GRAPH
# -------------------------------
builder = StateGraph(State)

builder.add_node("analyze", analyze)
builder.add_node("approval", approval)
builder.add_node("retry", retry_node)
builder.add_node("final", final_node)

builder.set_entry_point("analyze")

# Flow
builder.add_edge("analyze", "approval")

def route(state):
    if not state.get("approved"):
        return "retry"
    return "final"

builder.add_conditional_edges("approval", route)

builder.add_edge("retry", "final")

graph = builder.compile()

# -------------------------------
# RUN
# -------------------------------
if __name__ == "__main__":
    result = graph.invoke({"input": "delete all data"})
    print("\nFinal Output:", result["output"])

# -------------------------------
# VISUALIZE GRAPH
# -------------------------------
dot = Digraph()

dot.edge("analyze", "approval")
dot.edge("approval", "retry")
dot.edge("approval", "final")
dot.edge("retry", "final")

dot.render("advanced_graph", format="png", cleanup=True)
