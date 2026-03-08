from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path='books',
    glob='*.pdf',  # load all PDF files from the folder
    loader_cls=PyPDFLoader  # use PyPDFLoader to read each PDF
)

# load() loads all documents into memory at once - good for small files
# docs = loader.load()
# print(len(docs))              # total number of pages loaded from all PDFs
# print(docs[0].page_content)   # text content of first page
# print(docs[100].page_content) # text content of 100th page
# print(docs[0].metadata)       # info about first page - which file, page number etc

# lazy_load() loads documents one by one - memory efficient for large files
docs = loader.lazy_load()

for document in docs:  # iterate through each document one at a time
    print(document.metadata)  # print metadata of each page