import chromadb
import os
from dotenv import load_dotenv

load_dotenv()

chroma_client = chromadb.CloudClient(
    api_key=os.getenv("CHROMA_API_KEY"),
    tenant=os.getenv("CHROMA_TENANT"),
    database=os.getenv("CHROMA_DATABASE"),
)

collection = chroma_client.get_or_create_collection(name="documents")

document = [
    {"id": "doc1", "text": "Hello, how are you"},
    {"id": "doc2", "text": "How are you Today"},
    {"id": "doc3", "text": "Goodby see you later!"},
]

collection.add(
    ids=["doc1", "doc2", "doc3"],
    documents=["Hello, how are you", "How are you Today", "Goodbye see you later!"],
)


result = collection.query(query_texts=["Hello, World"], n_results=2)

print(result)
