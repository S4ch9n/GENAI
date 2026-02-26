from langchain_huggingface import HuggingFaceEmbeddings 
import os
import warnings
warnings.filterwarnings("ignore")
# from dotenv import load_dotenv

# load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
result = embedding.embed_query("Delhi is the capital of India ") #can generate the embedding of single document(.embed_query)
print(str(result))



documents = ["Birds can fly",
             "CPU is the brain of computer",
             "Paris is the capital of France"]
result2 = embedding.embed_documents(documents) #this function can generate the embedding of multiple documents(.embed_documents)
print(str(result2))
