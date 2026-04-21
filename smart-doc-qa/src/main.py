from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load PDF
loader = PyPDFLoader("../data/daa.pdf")
pages = loader.load()
print(f"Total pages: {len(pages)}")

# Chunk
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(pages)
print(f"Total chunks: {len(chunks)}")

# Embeddings + Vector store
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(chunks, embeddings)
print("Vector store created")

# Retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# LLM + Chain
llm = ChatGroq(model_name="llama-3.1-8b-instant")

prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the following context:
{context}

Question: {question}
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Test
# Interactive Q&A
print("\n🤖 Smart Document Q&A Ready!")
print("Type 'quit' to exit\n")

while True:
    question = input("Your question: ")
    if question.lower() == 'quit':
        break
    answer = rag_chain.invoke(question)
    print(f"\nAnswer: {answer}\n")