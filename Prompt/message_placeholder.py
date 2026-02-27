from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

model = ChatGroq(model="llama-3.1-8b-instant")

chat_template = ChatPromptTemplate([
    ("system", "You are a helpful instructor"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

chat_history = [
    HumanMessage(content="My name is Sachin"),
    AIMessage(content="Hello Sachin! Nice to meet you!"),
]

prompt = chat_template.invoke({
    "chat_history": chat_history,
    "question": "What is my name?"
})

result = model.invoke(prompt)
print(result.content)