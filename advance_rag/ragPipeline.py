"""
Building Rag Pipeline
Complete retrieval-augmented generation implemented

"""

import tempfile

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# Sample knowledge base
KNOWLEDGE_BASE = """# LangChain Framework

LangChain is a framework for developing applications powered by language models. It was created by Harrison Chase in October 2022.

## Core Components

1. **Models**: LangChain supports various LLM providers including OpenAI, Anthropic, and local models.

2. **Prompts**: Templates for structuring inputs to language models.

3. **Chains**: Sequences of calls to models and other components.

4. **Agents**: Systems that use LLMs to determine which actions to take.

5. **Memory**: Components for persisting state between chain/agent calls.

## LangGraph

LangGraph is a library for building stateful, multi-actor applications. Key features:
- State management
- Cycles and loops
- Human-in-the-loop
- Persistence

## Pricing

LangChain itself is open source and free. LangSmith (the observability platform) has a free tier and paid plans starting at $39/month.

## Getting Started

Install with: pip install langchain langchain-openai
Create your first chain in under 10 lines of code.
"""

embedding_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")


def create_kb():
    """Create a vector store from knowledge base"""

    # Split the knowledebase in chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    doc = Document(
        page_content=KNOWLEDGE_BASE, metadata={"source": "langchain_knowledge_base.md"}
    )

    chunks = splitter.split_documents([doc])

    # create vector store form chunks

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=tempfile.mkdtemp(),
    )

    return vector_store


def demo_basic_rag():
    vector_store = create_kb()
    retriver = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 2}
    )

    """ Rag Prompt Trmplate """
    prompt = ChatPromptTemplate.from_template(
        """
Answer the question only on the basis of the following context:

{context}

Question: {question}

Answer:

MAke sure to answer in consise manner, and if you don't know the answer, just say "I dont know"
"""
    )

    def format_docs(docs) -> str:
        return "\n\n".join([doc.page_content for doc in docs])

    # Rag Chain
    rag_chain = (
        {"context": retriver | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # test the Rag Chain
    #test

    questions = [
        "What is LangChain?",
        "who created LangChain?",
        "What is LangGraph used for?",
    ]


    print("Basic Rag Demo: \n")
    for q in questions:
        answer = rag_chain.invoke(q)
        print(f"Q: {q}")
        print(f"A: {answer}\n")


def demo_rag_with_source():


    vector_store = create_kb()
    retreiver = vector_store.as_retriever(
        search_type = "similarity", search_kwargs={"k": 2}
    )


    # Rag prompt templelate with source
    prompt =  ChatPromptTemplate.from_template(
        """
Answer the question based on the below context. Include which source you used.

Context:
{context}


Question: {question}

Answer (include sources):
"""
    )

    def format_docs_with_sources(docs):
        formatted = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get('source', 'unknown')
            formatted.append(f"[{i+1}] {source}: \n{doc.page_content}")
        return "\n\n".join(formatted)

    rag_chain = (
        {
            "context": retreiver | format_docs_with_sources,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    print( "RAG with sources:\n")
    answer = rag_chain.invoke("What are the core components of Langchain ?")
    print("Q: What are the core components?\n")
    print(f"A: {answer}")

    
if __name__ == "__main__":
    demo_rag_with_source()