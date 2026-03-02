from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import JsonOutputParser
import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

model = ChatGroq(model="llama-3.1-8b-instant")

template = PromptTemplate(
  template='Based on this character , tell me the name of anime {query}\n {format_instruction}',
  partial_variables={'format_instruction' : JsonOutputParser().get_format_instructions()}
)

chain = template | model |JsonOutputParser()

result = chain.invoke({"query" : "Goku" })
print(result)