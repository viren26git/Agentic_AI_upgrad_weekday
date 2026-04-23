from typing import TypedDict
from langgraph.graph import StateGraph, END
from openai import AzureOpenAI
from dotenv import load_dotenv
import os
import requests
import sqlite3

# ------------------------
# LOAD ENV
# ------------------------

load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-02-01",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

# ------------------------
# STATE
# ------------------------

class State(TypedDict, total=False):
    input: str
    tool: str
    tool_output: str
    final: str

# ------------------------
# TOOL 1: CALCULATOR (SAFE)
# ------------------------

def calculator_tool(query):
    try:
        allowed_chars = "0123456789+-*/(). "
        if not all(c in allowed_chars for c in query):
            return "Invalid math expression"
        return str(eval(query))
    except:
        return "Calculation error"

# ------------------------
# TOOL 2: WEATHER API (REAL)
# ------------------------

def weather_tool(query):
    try:
        city = query.lower().replace("weather in", "").strip()

        url = f"http://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q={city}"
        res = requests.get(url)

        if res.status_code != 200:
            return f"API Error: {res.text}"

        data = res.json()

        temp = data["current"]["temp_c"]
        condition = data["current"]["condition"]["text"]

        return f"{city.title()} Weather: {temp}°C, {condition}"

    except Exception as e:
        return f"Weather API error: {str(e)}"

# ------------------------
# TOOL 3: DATABASE (SQLite)
# ------------------------

def db_tool(query):
    try:
        conn = sqlite3.connect("company.db")
        cursor = conn.cursor()

        # Example safe queries only
        if "employees" in query.lower():
            cursor.execute("SELECT * FROM employees")
        else:
            return "Only 'employees' table allowed"

        rows = cursor.fetchall()
        conn.close()

        return str(rows)

    except Exception as e:
        return f"DB error: {str(e)}"

# ------------------------
# PLANNER (LLM decides tool)
# ------------------------

def planner(state: State):
    query = state["input"]

    prompt = f"""
Choose the correct tool:
- calculator → math expressions
- weather → weather queries
- database → company/employee data
- search → general queries

Query: {query}

Reply ONLY tool name.
"""

    res = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[{"role": "user", "content": prompt}]
    )

    tool = res.choices[0].message.content.strip().lower()

    return {"tool": tool}

# ------------------------
# TOOL EXECUTOR
# ------------------------

def tool_executor(state: State):
    tool = state["tool"]
    query = state["input"]

    if "calculator" in tool:
        output = calculator_tool(query)

    elif "weather" in tool:
        output = weather_tool(query)

    elif "database" in tool:
        output = db_tool(query)

    else:
        output = f"Search result (simulated): {query}"

    return {"tool_output": output}

# ------------------------
# FINAL RESPONSE
# ------------------------

def final(state: State):
    prompt = f"""
User Query: {state['input']}
Tool Used: {state['tool']}
Tool Output: {state['tool_output']}

Generate a helpful final answer.
"""

    res = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[{"role": "user", "content": prompt}]
    )

    return {"final": res.choices[0].message.content}

# ------------------------
# GRAPH
# ------------------------

builder = StateGraph(State)

builder.add_node("planner", planner)
builder.add_node("tool_executor", tool_executor)
builder.add_node("final", final)

builder.set_entry_point("planner")

builder.add_edge("planner", "tool_executor")
builder.add_edge("tool_executor", "final")
builder.add_edge("final", END)

graph = builder.compile()

# ------------------------
# RUN
# ------------------------

if __name__ == "__main__":

    print("Tool Agent Started (type 'exit' to quit)\n")

    while True:
        user_input = input("Ask: ")

        if user_input.lower() == "exit":
            break

        state = {"input": user_input}
        state = graph.invoke(state)

        print("\n--- RESULT ---")
        print("Tool Used:", state["tool"])
        print("Answer:", state["final"])
        print("----------------\n")
