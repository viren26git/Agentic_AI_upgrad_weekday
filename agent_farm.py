from typing import TypedDict
from langgraph.graph import StateGraph, END
from openai import AzureOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-02-01",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

class State(TypedDict, total=False):
    input: str
    summary: str
    keywords: str
    sentiment: str
    final: str

# ------------------------
# PARALLEL AGENTS
# ------------------------

def summary_agent(state):
    prompt = f"Summarize: {state['input']}"
    res = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[{"role": "user", "content": prompt}]
    )
    return {"summary": res.choices[0].message.content}

def keyword_agent(state):
    prompt = f"Extract keywords: {state['input']}"
    res = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[{"role": "user", "content": prompt}]
    )
    return {"keywords": res.choices[0].message.content}

def sentiment_agent(state):
    prompt = f"Sentiment (positive/negative/neutral): {state['input']}"
    res = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[{"role": "user", "content": prompt}]
    )
    return {"sentiment": res.choices[0].message.content}

# ------------------------
# MERGE
# ------------------------

def final(state):
    return {
        "final": f"""
Summary: {state['summary']}
Keywords: {state['keywords']}
Sentiment: {state['sentiment']}
"""
    }

# ------------------------
# GRAPH (PARALLEL)
# ------------------------

builder = StateGraph(State)

builder.add_node("summary", summary_agent)
builder.add_node("keywords", keyword_agent)
builder.add_node("sentiment", sentiment_agent)
builder.add_node("final", final)

builder.set_entry_point("summary")

# parallel fan-out
builder.add_edge("summary", "keywords")
builder.add_edge("summary", "sentiment")

# fan-in
builder.add_edge("keywords", "final")
builder.add_edge("sentiment", "final")

builder.add_edge("final", END)

graph = builder.compile()

# RUN
state = {"input": "AI is transforming healthcare rapidly"}
state = graph.invoke(state)

print(state["final"])


