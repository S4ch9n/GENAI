import warnings
warnings.filterwarnings("ignore")

import os
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage,SystemMessage
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

model = ChatGroq(model="llama-3.1-8b-instant")

chat_history = [
    SystemMessage(content="You are a helpful assistant. You have access to the full conversation history. Always refer to previous messages when answering.")
]
while True:
    user_input = input("You: ")
    if user_input == 'exit':
        break
    
    chat_history.append(HumanMessage(content=user_input))  # store as HumanMessage
    result = model.invoke(chat_history)  # pass full history!
    chat_history.append(AIMessage(content=result.content))  # store as AIMessage
    
    print("AI:", result.content)

print(chat_history)