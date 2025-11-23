import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

import httpx

load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")
http_client = httpx.Client(verify=False)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm=ChatOpenAI(model="openai/gpt-4.1", api_key=OPENAI_API_KEY, base_url="https://models.github.ai/inference",
    http_client=http_client)

prompt_template = ChatPromptTemplate.from_messages(
[
    ("system","You are a GenAI Trainer.Answer any questions "
              "related to the GenAI how to embark on carrer in GenAI"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
]
)

chain = prompt_template | llm

history_for_chain = StreamlitChatMessageHistory()

chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id : history_for_chain,
    input_messages_key="input",
    history_messages_key="chat_history"
)

st.title("GenAI Trainer")

input = st.text_input("Enter the question:")

if input:
    response = chain_with_history.invoke({"input":input},
                                         {"configurable":{"session_id":"abc123"}})
    st.write(response.content)

st.write("HISTORY")
st.write(history_for_chain)