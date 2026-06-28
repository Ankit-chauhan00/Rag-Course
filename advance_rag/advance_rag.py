from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import InMemoryStore
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()


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


if __name__ == "__main__":
    demo_parent_document()
