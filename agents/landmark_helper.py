import os
import base64
import streamlit as st

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_agent
from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper, DuckDuckGoSearchAPIWrapper
from langchain_core.globals import set_debug
from dotenv import load_dotenv
import httpx

set_debug(True)


# ------------------------------
# 1. Helper to encode image
# ------------------------------
def encode_image(image_file):
    return base64.b64encode(image_file.read()).decode()


# ------------------------------
# 2. LLM setup (gpt-4o for vision + tools)
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
# 3. Vision prompt (identify landmark)
# ------------------------------
vision_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that can identify a landmark."),
        (
            "human",
            [
                {"type": "text", "text": "return the landmark name"},
                {
                    "type": "image_url",
                    "image_url": {
                        # NOTE: `image` will be passed at runtime via chain.invoke({"image": ...})
                        "url": "data:image/jpeg;base64,{image}",
                        "detail": "low",
                    },
                },
            ],
        ),
    ]
)

vision_chain = vision_prompt | llm


# ------------------------------
# 4. Tools (Wikipedia + DuckDuckGo)
# ------------------------------
wikipedia_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
ddg_tool = DuckDuckGoSearchRun(api_wrapper=DuckDuckGoSearchAPIWrapper())

tools = [wikipedia_tool, ddg_tool]


# ------------------------------
# 5. ReAct-style agent (new v1 API)
# ------------------------------
react_system_prompt = """
You are a ReAct-style AI agent.

Follow this loop carefully:
1. THOUGHT: Think step by step about what to do next.
2. ACTION: When needed, call one of the tools (wikipedia, ddg-search).
3. OBSERVATION: Read the tool result and decide the next step.

Repeat THOUGHT → ACTION → OBSERVATION
until you are ready to give the final answer.

When you are confident, stop using tools and respond with a clear, concise final answer to the user.
"""

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=react_system_prompt
)


# ------------------------------
# 6. Streamlit UI
# ------------------------------
st.title("Landmark Helper (Vision + ReAct Agent)")

uploaded_file = st.file_uploader("Upload your image", type=["jpg", "png"])
question = st.text_input("Enter a question about the landmark")

task = None

# First: use vision chain to get landmark name
if uploaded_file and question:
    image_b64 = encode_image(uploaded_file)
    vision_response = vision_chain.invoke({"image": image_b64})
    landmark_name = vision_response.content
    task = question + " " + landmark_name

# Then: send combined task to tools agent
if task:
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": task + " without explanation",
                }
            ]
        }
    )
    final_msg = result["messages"][-1]
    st.write(final_msg.content)