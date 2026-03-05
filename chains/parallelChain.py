from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel 
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore")
import os

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

model1 = ChatGroq(model="llama-3.1-8b-instant")
model2 = ChatGroq(model="llama-3.3-70b-versatile")

prompt1 = PromptTemplate(
  template= "Write on this{topic}",
  input_variables=["topic"]
)
prompt2 = PromptTemplate(
  template="Make a summary of this {topic}",  
  input_variables=["topic"]  
)

prompt3 = PromptTemplate(
  template="Merge the provided {topic} and the {summary} and tell me your review about that",
  input_variables=["topic", "summary"]  
)

parser = StrOutputParser()

paraller_chain = RunnableParallel({
  'topic' : prompt1 | model1 | parser,
  'summary' : prompt2 | model2 | parser
})

merge_chain = prompt3 | model1 | parser

chain = paraller_chain | merge_chain

result = chain.invoke({"topic" : "One punch man"})
print(result)