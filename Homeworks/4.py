import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

# Load embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Q2 documents
documents = [
    'Bugs introduced by the intern had to be squashed by the lead developer.',
    'Bugs found by the quality assurance engineer were difficult to debug.',
    'Bugs are common throughout the warm summer months, according to the entomologist.',
    'Bugs, in particular spiders, are extensively studied by arachnologists.'
]
query = "Who is responsible for a coding project and fixing others' mistakes?"

# Get embeddings
embeddings = np.array(embedding_model.embed_documents(documents))
query_embedding = embedding_model.embed_query(query)

# Q1 - efficient L2 distance
# L2 distance is the straight line distance between two vectors - like measuring distance between two points!
l2_dist_manual_improved = np.zeros([4,4])
for i in range(embeddings.shape[0]):
    for j in range(embeddings.shape[0]):
        if i == j:
            l2_dist_manual_improved[i][j] = 0
        elif j < i:
            l2_dist_manual_improved[i][j] = l2_dist_manual_improved[j][i]
        else:
            l2_dist_manual_improved[i][j] = np.sqrt(np.sum((embeddings[i] - embeddings[j])**2))

print("Q1 - L2 Distance Matrix:")
print(l2_dist_manual_improved)

# Q2 - similarity search
scores = cosine_similarity([query_embedding], embeddings)
index, score = sorted(list(enumerate(scores[0])), key=lambda x: x[1])[-1]
print("\nQ2 - Most similar document:", documents[index])
print("Similarity score:", score)