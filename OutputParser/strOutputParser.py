from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore")
import os

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))


model = ChatGroq(model="llama-3.1-8b-instant")

#1st template
template1 = PromptTemplate(
  template="Write a detailed report on this {topic}"
)

#2nd template
template2 = PromptTemplate(
  template= "Write 5 line summary on this {text}"
)
chain = template1 | model | StrOutputParser() | template2 | model | StrOutputParser()

result = chain.invoke({"topic" : "Generative AI"})

print(result)