import warnings
warnings.filterwarnings("ignore")

from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGroq(model="llama-3.1-8b-instant")

# Prompt Template
prompt = PromptTemplate(
    input_variables=["topic"],
    template="Explain {topic} in very simple words."
)

formatted_prompt = prompt.format(topic="Machine Learning")

result = llm.invoke(formatted_prompt)

print("Prompt:", formatted_prompt)
print("Response:", result.content)