#from langchain_community.chat_models import ChatOllama
from langchain_ollama import ChatOllama

llm=ChatOllama(model="gemma:3b")

question = input("Enter the question")
response = llm.invoke(question)
print(response.content)