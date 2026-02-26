import warnings
warnings.filterwarnings("ignore")

from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGroq(model="llama-3.1-8b-instant")
prompt = PromptTemplate(
  input_variables =[ "topic"],
  template = "summarize this {topic}",
)
prompt_value = prompt.format(topic = "Generative AI")
result = llm.invoke(prompt_value)
print(result.content)