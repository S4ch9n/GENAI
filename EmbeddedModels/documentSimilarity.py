from langchain_huggingface import HuggingFaceEmbeddings
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

documents = [
    "Machine learning is a field of artificial intelligence that allows computers to learn from data and improve performance without explicit programming.",

    "Deep learning is a subset of machine learning that uses neural networks with many layers to model complex patterns in data.",

    "Natural language processing enables machines to understand, interpret, and generate human language.",

    "Large language models are trained on massive text datasets to generate human-like responses.",

    "Vector databases store embeddings and allow fast similarity search for semantic retrieval.",

    "LangChain is a framework that helps developers build applications powered by large language models.",

    "Embeddings convert text into numerical vectors so that semantic meaning can be captured mathematically.",

    "Retrieval Augmented Generation combines information retrieval and text generation to improve answer accuracy.",

    "Artificial intelligence is widely used in healthcare, finance, education, and autonomous systems.",

    "Chatbots use natural language processing and machine learning to interact with users conversationally."
]

query = 'Tell me about artificial intelligence'

doc_embeding = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query) 
scores = cosine_similarity([query_embedding], doc_embeding)

index, score = sorted(list(enumerate(scores[0])), key=lambda x: x[1])[-1]

print(query)
print(documents[index])
print("similarity score is:", score)