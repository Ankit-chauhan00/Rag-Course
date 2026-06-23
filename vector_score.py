from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import (GoogleGenerativeAIEmbeddings)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
import tempfile

load_dotenv()

embedding_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")


# Sample documents
SAMPLE_DOCS = [
    Document(
        page_content="LangChain is a framework for developing applications powered by language models.",
        metadata={"source": "langchain_docs", "topic": "overview"},
    ),
    Document(
        page_content="LangGraph is a library for building stateful, multi-actor applications with LLMs.",
        metadata={"source": "langgraph_docs", "topic": "overview"},
    ),
    Document(
        page_content="Vector stores are databases optimized for storing and searching embeddings.",
        metadata={"source": "vector_guide", "topic": "database"},
    ),
    Document(
        page_content="RAG combines retrieval with generation for more accurate LLM responses.",
        metadata={"source": "rag_guide", "topic": "architecture"},
    ),
    Document(
        page_content="Embeddings convert text into numerical vectors for semantic similarity.",
        metadata={"source": "embeddings_guide", "topic": "fundamentals"},
    ),
    Document(
        page_content="Chroma is an open-source embedding database for AI applications.",
        metadata={"source": "chroma_docs", "topic": "database"},
    ),
    Document(
        page_content="FAISS is a library for efficient similarity search developed by Facebook.",
        metadata={"source": "faiss_docs", "topic": "database"},
    ),
    Document(
        page_content="Pinecone is a managed vector database service for production workloads.",
        metadata={"source": "pinecone_docs", "topic": "database"},
    ),
]


def chroma_basics ():
    with tempfile.TemporaryDirectory() as tempdir:

        # create a vector store from document
        vectorstore = Chroma.from_documents(
            documents=SAMPLE_DOCS, embedding=embedding_model, persist_directory=tempdir
        )

        print(f"Vector store created {vectorstore._collection.count()} documnent are persisted.")

        # perform similarity search 

        query = "What is LangChain?"
        results = vectorstore.similarity_search_with_score(query, k=2)

        for doc, score in results:
            print(f"Score: {score}")
            print(doc.page_content)
            print("-" * 50)

if __name__ == "__main__":
    chroma_basics()