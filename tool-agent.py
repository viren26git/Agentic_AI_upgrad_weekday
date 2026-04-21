import os
import json
from dotenv import load_dotenv
from openai import AzureOpenAI
from graphviz import Digraph

load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-02-01",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

# ------------------------
# TOOLS
# ------------------------

def calculator(expression):
    try:
        return str(eval(expression))
    except:
        return "Error in calculation"

def weather(city):
    return f"{city}: 30°C, Sunny"

# ------------------------
# TOOL DEFINITIONS (for LLM)
# ------------------------

tools = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Perform math calculations",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string"}
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "weather",
            "description": "Get weather of a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"}
                },
                "required": ["city"]
            }
        }
    }
]

# ------------------------
# AGENT LOOP
# ------------------------

def run_agent(user_input):
    response = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        messages=[{"role": "user", "content": user_input}],
        tools=tools
    )

    msg = response.choices[0].message

    # If tool is called
    if msg.tool_calls:
        for call in msg.tool_calls:
            name = call.function.name
            args = json.loads(call.function.arguments)

            if name == "calculator":
                result = calculator(args["expression"])
            elif name == "weather":
                result = weather(args["city"])
            else:
                result = "Unknown tool"

            return f"[Tool Used: {name}] → {result}"

    return msg.content

# ------------------------
# GRAPH VISUALIZATION
# ------------------------

def draw_graph():
    dot = Digraph()

    dot.node("User Input")
    dot.node("LLM निर्णय")
    dot.node("Tool Call")
    dot.node("Final Output")

    dot.edge("User Input", "LLM निर्णय")
    dot.edge("LLM निर्णय", "Tool Call", label="if needed")
    dot.edge("LLM निर्णय", "Final Output", label="direct")
    dot.edge("Tool Call", "Final Output")

    dot.render("agent_graph", format="png", cleanup=True)
    print("Graph saved as agent_graph.png")

# ------------------------
# RUN
# ------------------------

if __name__ == "__main__":
    draw_graph()

    while True:
        user_input = input("Ask: ")
        if user_input.lower() == "exit":
            break

        output = run_agent(user_input)
        print("Output:", output)
