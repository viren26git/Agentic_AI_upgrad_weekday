from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from openai import AzureOpenAI
from dotenv import load_dotenv
from graphviz import Digraph
import os
import datetime

# ------------------------
# LOAD ENV
# ------------------------

load_dotenv()

# ------------------------
# AZURE CLIENT
# ------------------------

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-05-01-preview",   # IMPORTANT
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

# ------------------------
# STATE
# ------------------------

class State(TypedDict, total=False):
    input: str
    history: List[str]
    analysis: str
    report: str
    validation: str
    output: str

# ------------------------
# LOGGING
# ------------------------

def log(step, state):
    print(f"[{datetime.datetime.now()}] {step} | State keys: {list(state.keys())}")

# ------------------------
# MEMORY NODE
# ------------------------

def memory(state: State):
    log("Memory", state)

    history = state.get("history", [])
    history = history + [state.get("input", "")]

    print("Memory:", history)

    return {"history": history}

# ------------------------
# ANALYZE NODE
# ------------------------

def analyze(state: State):
    log("Analyze", state)

    prompt = f"Analyze this requirement:\n{state.get('input','')}"

    try:
        response = client.chat.completions.create(
            model=DEPLOYMENT,
            messages=[{"role": "user", "content": prompt}]
        )

        return {"analysis": response.choices[0].message.content}

    except Exception as e:
        print("ERROR in Analyze:", e)
        return {"analysis": "Error during analysis"}

# ------------------------
# REPORT NODE
# ------------------------

def generate_report(state: State):
    log("Report", state)

    prompt = f"Generate a structured report based on:\n{state.get('analysis','')}"

    try:
        response = client.chat.completions.create(
            model=DEPLOYMENT,
            messages=[{"role": "user", "content": prompt}]
        )

        return {"report": response.choices[0].message.content}

    except Exception as e:
        print("ERROR in Report:", e)
        return {"report": "Error generating report"}

# ------------------------
# VALIDATION NODE
# ------------------------

def validate(state: State):
    log("Validate", state)

    prompt = f"Validate this report and suggest improvements:\n{state.get('report','')}"

    try:
        response = client.chat.completions.create(
            model=DEPLOYMENT,
            messages=[{"role": "user", "content": prompt}]
        )

        return {"validation": response.choices[0].message.content}

    except Exception as e:
        print("ERROR in Validation:", e)
        return {"validation": "Error in validation"}

# ------------------------
# FINAL NODE
# ------------------------

def final(state: State):
    log("Final", state)

    final_output = f"""
=== ANALYSIS ===
{state.get('analysis','')}

=== REPORT ===
{state.get('report','')}

=== VALIDATION ===
{state.get('validation','')}
"""

    return {"output": final_output}

# ------------------------
# BUILD GRAPH
# ------------------------

builder = StateGraph(State)

builder.add_node("memory", memory)
builder.add_node("analyze", analyze)
builder.add_node("report", generate_report)
builder.add_node("validate", validate)
builder.add_node("final", final)

builder.set_entry_point("memory")

builder.add_edge("memory", "analyze")
builder.add_edge("analyze", "report")
builder.add_edge("report", "validate")
builder.add_edge("validate", "final")
builder.add_edge("final", END)

graph = builder.compile()

# ------------------------
# VISUALIZATION
# ------------------------

def visualize():
    dot = Digraph(comment="Enterprise Agent Flow")

    dot.node("Input", "User Input")
    dot.node("Memory", "Memory")
    dot.node("Analyze", "Analyze")
    dot.node("Report", "Report")
    dot.node("Validate", "Validate")
    dot.node("Final", "Final")

    dot.edge("Input", "Memory")
    dot.edge("Memory", "Analyze")
    dot.edge("Analyze", "Report")
    dot.edge("Report", "Validate")
    dot.edge("Validate", "Final")

    dot.render("enterprise_agent_graph", format="png", cleanup=True)
    print("\nGraph saved as enterprise_agent_graph.png")

# ------------------------
# RUN
# ------------------------

if __name__ == "__main__":

    visualize()  # generate graph

    state = {
        "input": "Build a secure login system",
        "history": []
    }

    state = graph.invoke(state)

    print("\nFINAL OUTPUT:\n")
    print(state["output"])

