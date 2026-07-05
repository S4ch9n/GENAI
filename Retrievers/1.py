from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

#load document
loader = PyPDFLoader("daa.pdf")

documents = loader.load()
#split into shunks
splitter = RecursiveCharacterTextSplitter(
  chunk_size = 500,
  chunk_overlap = 50
)

chunks = splitter.split_documents(documents)


#create embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


#store in FAISS vector store
db = FAISS.from_documents(chunks,embeddings)

print("vector store created")
print(f"tota chunks created {len(chunks)}")

# retirivers similarity search 
# retriever = db.as_retriever(
    # search_type="similarity",
    # search_kwargs={"k": 3}
# )
# 
# docs = retriever.invoke("What is algorithm?")
# 
# for doc in docs:
    # print(doc.page_content)
    # print("-" * 40)


# retirivers similarity search with score
# docs = db.similarity_search_with_score(
    # "What is Algorithm",
    # k=4
# )
# docs = db.similarity_search_with_score(
    # "What is Algorithm",
    # k=4
# )
# 
# for doc , score in docs: 
  # print("score : ", score)
  # print(doc.page_content)
  # print("-" * 50)



#retirivers mmr
retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k":4,
        "fetch_k":20,
        "lambda_mult":0.5
    }
)

docs = retriever.invoke("What is Algorithm?")

for i, doc in enumerate(docs, start=1):
    print(f"Chunk {i}")
    print(doc.page_content)
    print("-"*50)



