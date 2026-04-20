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
    print("Analyze: Starting parallel tasks")
    return state

# -------------------------------
# NODE 2A: TASK 1
# -------------------------------
def task1(state):
    print("Task1: Processing data")
    state["task1_result"] = "Data processed"
    return state

# -------------------------------
# NODE 2B: TASK 2
# -------------------------------
def task2(state):
    print("Task2: Generating report")
    state["task2_result"] = "Report generated"
    return state

# -------------------------------
# NODE 3: FINAL (MERGE)
# -------------------------------
def final(state):
    result = f"{state.get('task1_result')} + {state.get('task2_result')}"
    print("Final:", result)
    return {"output": result}

# -------------------------------
# BUILD GRAPH
# -------------------------------
builder = StateGraph(State)

builder.add_node("analyze", analyze)
builder.add_node("task1", task1)
builder.add_node("task2", task2)
builder.add_node("final", final)

builder.set_entry_point("analyze")

# Parallel edges
builder.add_edge("analyze", "task1")
builder.add_edge("analyze", "task2")

# Merge
builder.add_edge("task1", "final")
builder.add_edge("task2", "final")

graph = builder.compile()

# -------------------------------
# RUN
# -------------------------------
if __name__ == "__main__":
    result = graph.invoke({"input": "run parallel"})
    print("\nFinal Output:", result["output"])

# -------------------------------
# VISUALIZE
# -------------------------------
dot = Digraph()
dot.edge("analyze", "task1")
dot.edge("analyze", "task2")
dot.edge("task1", "final")
dot.edge("task2", "final")
dot.render("parallel_graph", format="png", cleanup=True)
