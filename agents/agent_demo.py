import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper, DuckDuckGoSearchAPIWrapper
#from langgraph.prebuilt import create_react_agent
from langchain.agents import create_agent
from dotenv import load_dotenv
import httpx

# ------------------------------
# 1. LLM setup
# ------------------------------
load_dotenv()
http_client = httpx.Client(verify=False)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(
    model="openai/gpt-4.1", 
    api_key=OPENAI_API_KEY, 
    base_url="https://models.github.ai/inference",
    http_client=http_client,
    temperature=0
)

# ------------------------------
# 2. Tools (Wikipedia + DuckDuckGo)
# ------------------------------
wikipedia_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
ddg_tool = DuckDuckGoSearchRun(api_wrapper=DuckDuckGoSearchAPIWrapper())

tools = [wikipedia_tool, ddg_tool]

# ------------------------------
# 3. Create Agent with LangGraph
# ------------------------------
agent_executor = create_agent(llm, tools)

# ------------------------------
# 4. Streamlit UI
# ------------------------------
st.title("🤖 AI Agent (ReAct style)")

task = st.text_input("Assign me a task", placeholder="e.g., What is the capital of France?")

if task:
    with st.spinner("Agent is thinking..."):
        try:
            # Invoke the agent
            result = agent_executor.invoke({"messages": [("user", task)]})
            
            # Extract the final answer
            final_message = result["messages"][-1]
            
            st.success("✅ Task completed!")
            st.write("**Answer:**")
            st.write(final_message.content)
            
            # Optional: Show the reasoning steps
            with st.expander("🔍 See Agent's Reasoning Steps"):
                for msg in result["messages"]:
                    st.write(f"**{msg.type}:** {msg.content}")
                    
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.exception(e)