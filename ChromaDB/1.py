import chromadb
client = chromadb.Client() 


collection = client.create_collection(name = 'vehicles') 

print("collection Created : " , collection.name)

#add data to collection
collection.add(
    documents=["Car is fast", 
               "Bike is cheap"],
    ids=["1", "2"]
)


#query the collection
result = collection.query(
    query_texts=["fast vehicle"],
    n_results=1,
    include=["documents", "distances", "embeddings"]
)

print(result)