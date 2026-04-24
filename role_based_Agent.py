from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from openai import AzureOpenAI
from dotenv import load_dotenv
import os
import time

# ------------------------
# LOAD ENV
# ------------------------

load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-05-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    timeout=10
)

DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

# ------------------------
# STATE
# ------------------------

class State(TypedDict, total=False):
    input: str
    roles: List[str]
    engineer: str
    security: str
    business: str
    final: str
    logs: List[str]

# ------------------------
# LOGGING
# ------------------------

def log(step, state):
    msg = f"[{time.strftime('%H:%M:%S')}] {step}"
    print(msg)

    logs = state.get("logs", [])
    logs.append(msg)

    return logs

# ------------------------
# MANAGER (ROUTER)
# ------------------------

def manager(state: State):
    logs = log("Manager: Routing task", state)

    query = state["input"].lower()

    roles = []

    if "security" in query or "secure" in query:
        roles.append("security")

    if "cost" in query or "business" in query:
        roles.append("business")

    # engineer always included
    roles.append("engineer")

    return {
        "roles": roles,
        "logs": logs
    }

# ------------------------
# ENGINEER AGENT
# ------------------------

def engineer_agent(state: State):
    logs = log("Engineer Agent Running", state)

    if "engineer" not in state.get("roles", []):
        return {"engineer": "Skipped", "logs": logs}

    prompt = f"As a software engineer, design: {state['input']}"

    try:
        res = client.chat.completions.create(
            model=DEPLOYMENT,
            messages=[{"role": "user", "content": prompt}]
        )
        output = res.choices[0].message.content
    except Exception as e:
        output = f"Error: {str(e)}"

    return {"engineer": output, "logs": logs}

# ------------------------
# SECURITY AGENT
# ------------------------

def security_agent(state: State):
    logs = log("Security Agent Running", state)

    if "security" not in state.get("roles", []):
        return {"security": "Skipped", "logs": logs}

    prompt = f"As a security expert, analyze risks: {state['input']}"

    try:
        res = client.chat.completions.create(
            model=DEPLOYMENT,
            messages=[{"role": "user", "content": prompt}]
        )
        output = res.choices[0].message.content
    except Exception as e:
        output = f"Error: {str(e)}"

    return {"security": output, "logs": logs}

# ------------------------
# BUSINESS AGENT
# ------------------------

def business_agent(state: State):
    logs = log("Business Agent Running", state)

    if "business" not in state.get("roles", []):
        return {"business": "Skipped", "logs": logs}

    prompt = f"As a business analyst, evaluate cost/impact: {state['input']}"

    try:
        res = client.chat.completions.create(
            model=DEPLOYMENT,
            messages=[{"role": "user", "content": prompt}]
        )
        output = res.choices[0].message.content
    except Exception as e:
        output = f"Error: {str(e)}"

    return {"business": output, "logs": logs}

# ------------------------
# FINAL AGGREGATOR
# ------------------------

def final(state: State):
    logs = log("Final Aggregation", state)

    return {
        "final": f"""
=== ENGINEER ===
{state.get('engineer')}

=== SECURITY ===
{state.get('security')}

=== BUSINESS ===
{state.get('business')}

=== LOGS ===
{state.get('logs')}
""",
        "logs": logs
    }

# ------------------------
# GRAPH (AGENT FARM)
# ------------------------

builder = StateGraph(State)

builder.add_node("manager", manager)
builder.add_node("engineer", engineer_agent)
builder.add_node("security", security_agent)
builder.add_node("business", business_agent)
builder.add_node("final", final)

builder.set_entry_point("manager")

# parallel execution (agent farm)
builder.add_edge("manager", "engineer")
builder.add_edge("manager", "security")
builder.add_edge("manager", "business")

# aggregation
builder.add_edge("engineer", "final")
builder.add_edge("security", "final")
builder.add_edge("business", "final")

builder.add_edge("final", END)

graph = builder.compile()

# ------------------------
# RUN
# ------------------------

if __name__ == "__main__":

    state = {
        "input": "Build a secure and cost-efficient payment system",
        "logs": []
    }

    result = graph.invoke(state)

    print("\n FINAL OUTPUT:\n")
    print(result["final"])
