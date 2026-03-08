from langchain_community.document_loaders.csv_loader import CSVLoader

loader = CSVLoader(file_path='students.csv')

docs = loader.load()

print(len(docs))
print(docs[0])