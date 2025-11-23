import streamlit as st
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

from dotenv import load_dotenv
import httpx

load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")
http_client = httpx.Client(verify=False)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm=ChatOpenAI(model="openai/gpt-4.1", api_key=OPENAI_API_KEY, 
               base_url="https://models.github.ai/inference",
               http_client=http_client)
prompt_template = PromptTemplate(
    input_variables=["city","month","language","budget"],
    template="""Welcome to the {city} travel guide!
    If you're visiting in {month}, here's what you can do:
    1. Must-visit attractions.
    2. Local cuisine you must try.
    3. Useful phrases in {language}.
    4. Tips for traveling on a {budget} budget.
    Enjoy your trip!
    """
)

st.title("Travel Guide")

city = st.text_input("Enter the city:")
month = st.text_input("Enter the month of travel")
language = st.text_input("Enter the language:")
budget = st.selectbox("Travel Budget",["Low","Medium","High"])


if city and month and language and budget:
    response = llm.invoke(prompt_template.format(city=city,
                                                 month=month,
                                                 language=language,
                                                 budget=budget
                                                 ))
    st.write(response.content)