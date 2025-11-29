import asyncio
import os

import httpx
import streamlit as st

from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent


client = MultiServerMCPClient({
    "tools": {
        "command": "python3",
        "args": ["mcp_server.py"],
        "transport": "stdio"
    }
})

tools = asyncio.run(client.get_tools())

openai_key = os.getenv("OPENAI_API_KEY")
http_client = httpx.Client(verify=False)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(
    model="openai/gpt-4.1", 
    api_key=OPENAI_API_KEY, 
    base_url="https://models.github.ai/inference",
    http_client=http_client,
    temperature=0
)
agent = create_agent(llm, tools)

st.title("AI Agent (MCP Version)")
task = st.text_input("Assign me a task")

if task:
    response = asyncio.run(agent.ainvoke({"messages": task}))
    st.write(response)
    final_output = response["messages"][-1].content
    st.write(final_output)

