from langgraph.graph import StateGraph
from dotenv import load_dotenv
import os
from openai import AzureOpenAI

# Load environment variables
load_dotenv()

# Azure OpenAI client (UPDATED)
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-02-01"
)

DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

# -------------------------------
# STATE (Graph Memory)
# -------------------------------
from typing import TypedDict

class State(TypedDict, total=False):
    input: str
    issue: str
    action: str
    output: str

# -------------------------------
# NODE 1: ANALYZE USER INPUT
# -------------------------------
def analyze(state):
    user_input = state.get("input", "")

    response = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {"role": "system", "content": "Classify issue as delay, payment or other"},
            {"role": "user", "content": user_input}
        ]
    )

    result = response.choices[0].message.content.lower()

    if "delay" in result:
        issue = "delay"
    elif "payment" in result:
        issue = "payment"
    else:
        issue = "other"

    print("Node1 (Analyze):", issue)

    return {**state, "issue": issue}

# -------------------------------
# NODE 2A: DELAY
# -------------------------------
def handle_delay(state):
    print("Node2A: Delay handler")
    return {**state, "action": "Provide compensation for delay"}

# -------------------------------
# NODE 2B: PAYMENT
# -------------------------------
def handle_payment(state):
    print("Node2B: Payment handler")
    return {**state, "action": "Resolve payment issue"}

# -------------------------------
# NODE 3: FINAL RESPONSE
# -------------------------------
def final_response(state):
    output = f"Issue: {state.get('issue')} | Action: {state.get('action')}"
    print("Node3 (Final):", output)
    return {**state, "output": output}

# -------------------------------
# BUILD GRAPH
# -------------------------------
builder = StateGraph(State)

builder.add_node("analyze", analyze)
builder.add_node("delay", handle_delay)
builder.add_node("payment", handle_payment)
builder.add_node("final", final_response)

# Entry
builder.set_entry_point("analyze")

# Conditional branching
def route(state):
    issue = state.get("issue")

    if issue == "delay":
        return "delay"
    elif issue == "payment":
        return "payment"
    else:
        return "final"

builder.add_conditional_edges("analyze", route)

# Connect nodes
builder.add_edge("delay", "final")
builder.add_edge("payment", "final")

# Compile graph
graph = builder.compile()

# -------------------------------
# RUN
# -------------------------------
result = graph.invoke({"input": "My order is delayed"})

print("\nFinal Output:", result["output"])


from graphviz import Digraph

def visualize():
    dot = Digraph()

    # Nodes
    dot.node("analyze")
    dot.node("delay")
    dot.node("payment")
    dot.node("final")

    # Edges
    dot.edge("analyze", "delay", label="delay")
    dot.edge("analyze", "payment", label="payment")
    dot.edge("delay", "final")
    dot.edge("payment", "final")

    dot.render("graph", format="png", cleanup=True)

visualize()
