import os
import httpx
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from dotenv import load_dotenv

load_dotenv()
http_client = httpx.Client(verify=False)


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm=OpenAIEmbeddings(api_key=OPENAI_API_KEY,model="openai/text-embedding-3-small",
    base_url="https://models.github.ai/inference",
    http_client=http_client)


document = TextLoader("job_listings.txt").load()
text_splitter= RecursiveCharacterTextSplitter(chunk_size=200,
                                              chunk_overlap=10)
chunks=text_splitter.split_documents(document)
db=Chroma.from_documents(chunks,llm)
retriever = db.as_retriever()

text = input("Enter the query")

docs = retriever.invoke(text)

for doc in docs:
    print(doc.page_content)