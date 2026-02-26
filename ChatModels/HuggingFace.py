from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv
import os
import warnings
warnings.filterwarnings('ignore')


load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

llm = HuggingFaceEndpoint(
  repo_id = "mistralai/Mistral-7B-Instruct-v0.2" , #which model we want to use
  task = "text-generation" #which task we want to perform with this model
)

#ChatHuggingFace is just a wrapper that converts plain LLM into a Chat Model format!
model = ChatHuggingFace(llm = llm)

result = model.invoke("what is capital of india ?")
print(result.content)