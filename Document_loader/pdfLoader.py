from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('daa.pdf')  # load pdf file

docs = loader.load()  # convert pdf content into list of LangChain document objects

print(docs)           # prints all pages
print(len(docs))      # prints total number of pages
print(docs[0].page_content)  # prints text content of first page
print(docs[0].metadata)      # prints metadata like page number, source file