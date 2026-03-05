#simple chain example

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore")
import os

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

model = ChatGroq(model="llama-3.1-8b-instant")


propmt = PromptTemplate(
  template="Make summary of this {topic} in 5 lines ",
  input_variables=['topic']
)

parser = StrOutputParser()

chain = propmt | model | parser

result = chain.invoke({"topic": "Naruto"}) 
print(result)
