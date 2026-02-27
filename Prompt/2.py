from langchain_groq import ChatGroq
# from langchain_core.messages import SystemMessage, HumanMessage , AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore")
import os

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

model = ChatGroq(model="llama-3.1-8b-instant")

prompts = ChatPromptTemplate([
    ("system", "You are a pirate"),
    ("human", "{message}"),
])

chain = prompts | model | StrOutputParser()
result = chain.invoke({"message": "What is one piece?"})
print(result)
