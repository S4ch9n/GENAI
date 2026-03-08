from langchain_community.document_loaders import TextLoader

loader = TextLoader('1.txt')

docs = loader.load() #convert the txt file to llm document object

print(docs)

print(docs[0].page_content)
print(docs[0].metadata)