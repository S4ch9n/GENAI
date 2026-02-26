import warnings
warnings.filterwarnings("ignore")

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGroq(model="llama-3.1-8b-instant")

prompt = PromptTemplate(
  input_variables=["country"],
  template ="what is the capital of {country}"
)
chain = prompt | llm | StrOutputParser()
result = chain.invoke({"country": "India"})
print(result)  # directly get string! no .content needed!