from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import numpy as np
load_dotenv()

embedding = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")

def basic_embeddings():
    #single text
    text = "What is Machine Learning?"
    single_embeddings = embedding.embed_query(text)

    print(f"Vector dimension: {len(single_embeddings)}")
    print(f"First 5 values: {single_embeddings[:5]}")
    print(f"vector form : {np.linalg.norm(single_embeddings):.4f}")


def batch_embeddings():

    text = [
        "What is Machine Learning?",
        "Explain the concept of overfitting in ML.",
        "How does neural network works?"
    ]

    batch_embedding = embedding.embed_documents(text)
    for i, emb in enumerate(batch_embedding):
        print(f"Text {i+1} - Vector dimensions: {len(emb)}")
        print(f"Text {i+1} - First 5 values: {emb[:5]}")
        print(f"Text {i+1} - Vector norm: {np.linalg.norm(emb):.4f}")

if __name__ == "__main__":
    # basic_embeddings()
    batch_embeddings()
