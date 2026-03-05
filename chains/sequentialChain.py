from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore")
import os

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

model = ChatGroq(model="llama-3.1-8b-instant")

prompt1 = PromptTemplate(
  template= "Write on this{topic}",
  input_variables=["topic"]
)
prompt2 = PromptTemplate(
  template="Make a summary of this {text}",
  input_variables=["text"]
)
parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({"topic" : "Future of genAi"})

print(result)

chain.get_graph().print_ascii()

# {"topic": "Future of GenAI"}
#         ↓
# prompt1 → fills {topic} → "Write on this Future of GenAI"
#         ↓
# model → generates detailed text
#         ↓
# parser → converts to plain string
#         ↓
# prompt2 → fills {text} with previous output → "Make a summary of this..."
#         ↓
# model → generates summary
#         ↓
# parser → converts to plain string
#         ↓
# Final Output!
