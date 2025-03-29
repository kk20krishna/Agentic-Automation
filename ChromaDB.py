import chromadb
import os

# Use a relative path to store data in a directory named "chroma_db"
chroma_dir = "chroma_db"
chroma_client = chromadb.PersistentClient(path=chroma_dir)

# switch `create_collection` to `get_or_create_collection` to avoid creating a new collection every time
collection = chroma_client.get_or_create_collection(name="my_collection")

# switch `add` to `upsert` to avoid adding the same documents every time
collection.upsert(
    documents=[
        "This is a document about pineapple",
        "This is a document about oranges",
        "This is a document about bananas",
        "This is a document about apples",
        "This is a document about grapes",
        "This is a document about strawberries",
        "This is a document about blueberries",
        "This is a document about Kerala"
    ],
    ids=["id1", "id2", "id3", "id4", "id5", "id6", "id7", "id8"],
    metadatas=[
        {"source": "doc1"},
        {"source": "doc2"},
        {"source": "doc3"},
        {"source": "doc4"},
        {"source": "doc5"},
        {"source": "doc6"},
        {"source": "doc7"},
        {"source": "doc8"}
    ],
)

results = collection.query(
    query_texts=["This is a query document about kerala"], # Chroma will embed this for you
    n_results=2 # how many results to return
)

print("Results:")
if results['documents']:
    if results['documents'][0]:
        for doc in results['documents'][0]:
            print(doc)
    else:
        print("No documents found in the first result set.")
else:
    print("No results found for the query.")

# Print the full results object for debugging
print("\nFull Results Object (for debugging):")
print(results)
