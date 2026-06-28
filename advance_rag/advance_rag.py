from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.retrievers import (
    ContextualCompressionRetriever,
    ParentDocumentRetriever,
)
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_classic.storage import InMemoryStore
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# Sample knowledge base for demos
TECH_DOCS = [
    Document(
        page_content="Python is a high-level programming language known for its simplicity and readability. It supports multiple programming paradigms including procedural, object-oriented, and functional programming. Python is widely used in web development, data science, artificial intelligence, and automation.",
        metadata={
            "topic": "programming",
            "language": "python",
            "difficulty": "beginner",
        },
    ),
    Document(
        page_content="JavaScript is the language of the web. It runs in browsers and on servers with Node.js. Modern frameworks like React, Vue, and Angular make building interactive web applications efficient. JavaScript supports asynchronous programming with Promises and async/await.",
        metadata={
            "topic": "programming",
            "language": "javascript",
            "difficulty": "intermediate",
        },
    ),
    Document(
        page_content="Machine learning is a subset of AI that enables systems to learn from data. Supervised learning uses labeled data, while unsupervised learning finds patterns in unlabeled data. Popular ML frameworks include TensorFlow, PyTorch, and scikit-learn.",
        metadata={
            "topic": "ai",
            "subtopic": "machine_learning",
            "difficulty": "advanced",
        },
    ),
    Document(
        page_content="LangChain is a framework for building LLM applications. It provides tools for prompts, chains, agents, and memory. LangChain supports multiple LLM providers including OpenAI, Anthropic, and local models.",
        metadata={
            "topic": "ai",
            "subtopic": "llm_frameworks",
            "difficulty": "intermediate",
        },
    ),
    Document(
        page_content="LangGraph is a library for building stateful, multi-actor applications with LLMs. Key features include state management, cycles and loops, human-in-the-loop workflows, and persistence. LangGraph extends LangChain for complex agent architectures.",
        metadata={
            "topic": "ai",
            "subtopic": "llm_frameworks",
            "difficulty": "advanced",
        },
    ),
    Document(
        page_content="Docker is a platform for containerizing applications. Containers package code and dependencies together for consistent deployment. Docker Compose orchestrates multi-container applications. Kubernetes scales Docker containers in production.",
        metadata={
            "topic": "devops",
            "subtopic": "containers",
            "difficulty": "intermediate",
        },
    ),
    Document(
        page_content="PostgreSQL is an advanced open-source relational database. It supports JSON data types, full-text search, and extensions like pgvector for vector similarity search. PostgreSQL is ACID compliant and highly extensible.",
        metadata={
            "topic": "database",
            "type": "relational",
            "difficulty": "intermediate",
        },
    ),
    Document(
        page_content="Vector databases like Pinecone, Chroma, and Qdrant are optimized for storing and searching embeddings. They enable semantic similarity search for RAG applications. Most support metadata filtering and hybrid search combining keywords with vectors.",
        metadata={"topic": "database", "type": "vector", "difficulty": "intermediate"},
    ),
]


def demo_parent_document():

    print("=" * 60)
    print("PARENT DOCUMENT RETRIVER")
    print("Small search for precise search, large chunks for context")
    print("=" * 60)

    # Long document to demonstrate parent/child splitting

    long_doc = Document(
        page_content="""

# Complete Guide to Building AI Agents

## Chapter 1: Introduction to AI Agents

AI agents are autonomous systems that can perceive their environment, make decisions, and take actions to achieve goals. Unlike simple chatbots, agents can use tools, maintain state, and execute multi-step plans.

The key components of an AI agent include:
- A language model for reasoning
- Tools for interacting with external systems
- Memory for maintaining context
- A planning mechanism for complex tasks

## Chapter 2: Agent Frameworks

Several frameworks exist for building AI agents:

LangChain provides the foundational abstractions for chains and simple agents. It excels at straightforward tool-calling patterns and integrates with many LLM providers.

LangGraph extends LangChain for complex, stateful agents. It introduces graph-based state management, enabling cycles, human-in-the-loop workflows, and persistent execution.

CrewAI focuses on multi-agent collaboration, allowing teams of specialized agents to work together on complex tasks.

## Chapter 3: Production Considerations

Deploying agents to production requires careful attention to:
- Error handling and fallbacks
- Token usage optimization
- Observability and tracing
- Security and access control
- State persistence and recovery

LangSmith provides observability for LangChain/LangGraph applications, offering tracing, evaluation, and monitoring capabilities.
       
""",
        metadata={"source": "ai_agent_guide.md"},
    )

    # splitters

    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)

    # Storage
    vectorstore = Chroma(
        collection_name="parent_child_demo",
        embedding_function=GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-2"
        ),
    )

    store = InMemoryStore()

    # create retriver
    retriver = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )

    # add documents
    retriver.add_documents([long_doc])

    queries = [
        "What is LangGraph used for?",
        "Which framework supports human in the loop?",
        "What production concerns should I consider?",
        "Which framework is best for multiple agents?",
        "What does LangSmith provide?",
    ]

    for query in queries:
        print("=" * 60)
        print(f"Query: {query}")

        # Child retrieval
        child = vectorstore.similarity_search(query, k=1)[0]

        print("\n--- Child Chunk ---")
        print(f"Length: {len(child.page_content)}")
        print(child.page_content)

        # Parent retrieval
        parent = retriver.invoke(query)[0]

        print("\n--- Parent Chunk ---")
        print(f"Length: {len(parent.page_content)}")
        print(parent.page_content[:500])
        print("\n")


def create_base_vector():
    """Create a basic vector store from demos."""
    return Chroma.from_documents(
        documents=TECH_DOCS,
        embedding=GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2"),
    )


def demo_contextual_compression():
    """Contextual Compression extracts only revelant parts."""

    print("=" * 60)
    print("CONTEXTUAL COMPRESSION RETRIVAL")
    print("Extracts only query-relevant content from documnets")
    print("=" * 60)

    vectorstore = create_base_vector()
    llm = GoogleGenerativeAI(model="gemini-2.5-flash")

    # create compressor
    compressor = LLMChainExtractor.from_llm(llm)

    # wrap retriver with compressor
    compression_retrival = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
    )

    query = "What frameworks exists for building LLm applicatrions?"

    print(f"\nQuery: {query}")


    # without compression
    base_docs = vectorstore.as_retriever(search_kwargs={"k": 2}).invoke(query)
    print("\n--- Without Compression (full chunks) --- ")
    for doc in base_docs:
        print(f"Length: {len(doc.page_content)} chars")
        print(f"Content: {doc.page_content[:150]}...\n")




    # with compression

    compressed_docs = compression_retrival.invoke(query)
    print("\n--- WITH Compression (relevant only) ---")
    for doc in compressed_docs:
        print(f"Length: {len(doc.page_content)} Chars")
        print(f"Content: {doc.page_content}\n")




if __name__ == "__main__":
    # demo_parent_document()
    demo_contextual_compression()