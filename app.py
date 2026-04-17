from langgraph.graph import StateGraph
from dotenv import load_dotenv
import os
from openai import AzureOpenAI
from graphviz import Digraph

# -------------------------------
# LOAD ENV
# -------------------------------
load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-02-15-preview"
)

DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

# -------------------------------
# STATE (Shared Memory)
# -------------------------------
class State(dict):
    pass

# -------------------------------
# NODE 1: ANALYZE (LLM + Routing)
# -------------------------------
def analyze(state):
    user_input = state["input"]

    response = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {"role": "system", "content": "Classify issue as delay, payment, or other"},
            {"role": "user", "content": user_input}
        ]
    )

    result = response.choices[0].message.content.lower()

    if "delay" in result:
        state["route"] = "delay"
        state["risk"] = "low"
    elif "payment" in result:
        state["route"] = "payment"
        state["risk"] = "medium"
    else:
        state["route"] = "general"
        state["risk"] = "high"

    print(f"Node1 (Analyze): route={state['route']}, risk={state['risk']}")
    return state

# -------------------------------
# NODE 2A: DELAY HANDLER
# -------------------------------
def delay_node(state):
    state["result"] = "Handled delay: compensation initiated"
    return state

# -------------------------------
# NODE 2B: PAYMENT HANDLER
# -------------------------------
def payment_node(state):
    state["result"] = "Handled payment: issue resolved"
    return state

# -------------------------------
# NODE 2C: GENERAL (HIGH RISK)
# -------------------------------
def general_node(state):
    state["result"] = "General issue detected"
    return state

# -------------------------------
# NODE 3: HUMAN APPROVAL
# -------------------------------
def human_approval(state):
    if state["risk"] == "high":
        print("⚠️ High Risk Detected - Approval Required")
        state["approved"] = False  # simulate rejection
    else:
        state["approved"] = True

    return state

# -------------------------------
# NODE 4: RETRY LOGIC
# -------------------------------
def retry_node(state):
    print("Retrying operation...")
    for i in range(2):
        print(f"Attempt {i+1}")

    state["result"] = "Fallback after retries"
    return state

# -------------------------------
# NODE 5: FINAL RESPONSE
# -------------------------------
def final_node(state):
    output = f"Issue handled via {state.get('route')} | Result: {state.get('result')}"
    print("Node Final:", output)
    return {"output": output}

# -------------------------------
# BUILD GRAPH
# -------------------------------
builder = StateGraph(State)

builder.add_node("analyze", analyze)
builder.add_node("delay", delay_node)
builder.add_node("payment", payment_node)
builder.add_node("general", general_node)
builder.add_node("approval", human_approval)
builder.add_node("retry", retry_node)
builder.add_node("final", final_node)

# Entry point
builder.set_entry_point("analyze")

# -------------------------------
# ROUTING AFTER ANALYZE
# -------------------------------
def route_after_analyze(state):
    return state["route"]

builder.add_conditional_edges("analyze", route_after_analyze)

# Connect handlers to approval
builder.add_edge("delay", "approval")
builder.add_edge("payment", "approval")
builder.add_edge("general", "approval")

# -------------------------------
# APPROVAL ROUTING
# -------------------------------
def route_after_approval(state):
    if not state.get("approved"):
        return "retry"
    return "final"

builder.add_conditional_edges("approval", route_after_approval)

# Retry goes to final
builder.add_edge("retry", "final")

# Compile graph
graph = builder.compile()

# -------------------------------
# RUN GRAPH
# -------------------------------
if __name__ == "__main__":
    user_input = input("Enter your issue: ")

    result = graph.invoke({"input": user_input})

    print("\n✅ Final Output:", result["output"])

# -------------------------------
# GRAPH VISUALIZATION
# -------------------------------
def visualize():
    dot = Digraph()

    dot.edge("analyze", "delay")
    dot.edge("analyze", "payment")
    dot.edge("analyze", "general")

    dot.edge("delay", "approval")
    dot.edge("payment", "approval")
    dot.edge("general", "approval")

    dot.edge("approval", "final")
    dot.edge("approval", "retry")

    dot.edge("retry", "final")

    dot.render("workflow", format="png", cleanup=True)

visualize()
