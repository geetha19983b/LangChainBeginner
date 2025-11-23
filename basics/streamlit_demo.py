import os
from langchain_openai import ChatOpenAI
import streamlit as st
from langchain_core.globals import set_debug

set_debug(True)

from dotenv import load_dotenv
import httpx

load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")
http_client = httpx.Client(verify=False)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm=ChatOpenAI(model="openai/gpt-4.1", api_key=OPENAI_API_KEY, base_url="https://models.github.ai/inference",
    http_client=http_client)

st.title("Ask Anything")

question = st.text_input("Enter the question:")

if question:
    response = llm.invoke(question)
    st.write(response.content)