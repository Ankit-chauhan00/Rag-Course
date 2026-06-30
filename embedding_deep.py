import tempfile

import numpy as np
from dotenv import load_dotenv
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

embedding = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")


def basic_embeddings():
    # single text
    text = "What is Machine Learning?"
    single_embeddings = embedding.embed_query(text)

    print(f"Vector dimension: {len(single_embeddings)}")
    print(f"First 5 values: {single_embeddings[:5]}")
    print(f"vector form : {np.linalg.norm(single_embeddings):.4f}")


def batch_embeddings():

    text = [
        "What is Machine Learning?",
        "Explain the concept of overfitting in ML.",
        "How does neural network works?",
    ]

    batch_embedding = embedding.embed_documents(text)
    for i, emb in enumerate(batch_embedding):
        print(f"Text {i + 1} - Vector dimensions: {len(emb)}")
        print(f"Text {i + 1} - First 5 values: {emb[:5]}")
        print(f"Text {i + 1} - Vector norm: {np.linalg.norm(emb):.4f}")


"""
result shows that  which is the highest score is the nearest to the semantic meaning of the query
query = "What programming languages exist?"


0.7304: Python is a programming language
0.6464: JavaScript is used for web development
0.6016: Deep learning uses neural networks
0.5798: Machine learning enables AI applications
0.5762: Cats are popular pets

"""


def similarity_search():

    docs = [
        "Python is a programming language",
        "JavaScript is used for web development",
        "Machine learning enables AI applications",
        "Deep learning uses neural networks",
        "Cats are popular pets",
    ]

    query = "What programming languages exist?"

    embedded_docs = embedding.embed_documents(docs)
    embedded_query = embedding.embed_query(query)

    def cosine_similarity(vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    similarities = [
        cosine_similarity(embedded_query, doc_vec) for doc_vec in embedded_docs
    ]

    ranked_docs = sorted(zip(docs, similarities), key=lambda x: x[1], reverse=True)

    print(f"Query: {query}\n")
    print("Ranked by Similarity:")

    for doc, score in ranked_docs:
        print(f"{score:.4f}: {doc}")


# Caching ---
def embedding_caching():

    with tempfile.TemporaryDirectory() as tempdir:
        store = LocalFileStore(root_path=tempdir)

        cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
            underlying_embeddings=GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-2"
            ),
            document_embedding_cache=store,
            namespace="exercise",
        )

        text = "What is Reinforcement Learning?"

        # first call - hits api
        print("First call (API):")
        vector1 = cached_embeddings.embed_documents([text])
        print(f" Embedded {len(vector1)} Documnet")

        # Second call - from cache
        print("\nSecond call (Cache)")
        vectors2 = cached_embeddings.embed_documents([text])
        print(f" Ebedded {len(vectors2)} documents")

        # verify same result
        print(f"\nSame vectors: {np.allclose(vector1[0], vectors2[0])}")




if __name__ == "__main__":
    # basic_embeddings()
    # batch_embeddings()
    # similarity_search()
    embedding_caching()
