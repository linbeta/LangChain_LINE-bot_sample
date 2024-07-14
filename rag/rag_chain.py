from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant
from qdrant_client.http import models
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.pydantic_v1 import BaseModel
from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")

# Use OpenAI ChatGPT
embeddings_model = OpenAIEmbeddings(
    api_key=OPENAI_API_KEY,
    model="text-embedding-3-large"
)

client = QdrantClient(
	url=QDRANT_URL, 
    api_key=QDRANT_API_KEY)
collection_name="LangChain-bot"
qdrant = Qdrant(client, collection_name, embeddings_model)

retriever = qdrant.as_retriever(search_kwargs={"k": 3})

model = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model="gpt-4o",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

prompt = ChatPromptTemplate.from_template("""請回答依照 context 裡的資訊來回答問題:
<context>
{context}
</context>
Question: {input}""")


document_chain = create_stuff_documents_chain(model, prompt)

retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Add typing for input
class Question(BaseModel):
    input: str


rag_chain = retrieval_chain.with_types(input_type=Question)

