from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage , AIMessage
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore")
import os

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

model = ChatGroq(model="llama-3.1-8b-instant")

result1 = model.invoke(
  [
  SystemMessage(content="You are IIT teacher"),
  HumanMessage(content="What is the best way to learn DSA , and master it in 3 months ? ")
  ]
)

print(result1.content)

result2 = model.invoke([
    SystemMessage(content="You are IIT teacher"),
  HumanMessage(content="What is the best way to learn DSA , and master it in 3 months ? "),
  AIMessage(content=result1.content),
  HumanMessage(content = "What is the best way to learn time complexity and understand it deeply and solve the problem , it is hard to grasp it"),
])
print(result2.content)


result3 = model.invoke([
  SystemMessage(content="You are IIT teacher"),
  HumanMessage(content="What is the best way to learn DSA , and master it in 3 months ? "),
  AIMessage(content=result1.content),
  HumanMessage(content = "What is the best way to learn time complexity and understand it deeply and solve the problem , it is hard to grasp it"),
  AIMessage(content=result2.content),
  HumanMessage(content="Thanx for the help")
  ])
print(result3.content)
