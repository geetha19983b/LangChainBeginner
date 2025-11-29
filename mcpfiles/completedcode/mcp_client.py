import asyncio
import os
import httpx
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from dotenv import load_dotenv
load_dotenv()

# Initialize agent only once
@st.cache_resource
def get_agent():
    # Use stdio transport
    client = MultiServerMCPClient({
        "tools": {
            "command": "python",
            "args": [r"C:\Users\Geetha\Codes\udemy\langchaindemo\mcpfiles\completedcode\mcp_server.py"],
            "transport": "stdio"
        }
    })
    
    # Get tools - must be run in async context
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    tools = loop.run_until_complete(client.get_tools())
    
    # Setup LLM
    openai_key = os.getenv("OPENAI_API_KEY")
    http_client = httpx.Client(verify=False)

    
    llm=ChatOpenAI(model="openai/gpt-4.1", api_key=openai_key, base_url="https://models.github.ai/inference",
    http_client=http_client)
    
    agent = create_agent(llm, tools)
    return agent, loop

# Streamlit UI
st.title("AI Agent (MCP Version)")

# Get agent
agent, loop = get_agent()

task = st.text_input("Assign me a task")

if task:
    with st.spinner("Processing..."):
        # Run agent
        response = loop.run_until_complete(agent.ainvoke({"messages": task}))
        
        # Display response
        final_output = response["messages"][-1].content
        st.success("Response:")
        st.write(final_output)