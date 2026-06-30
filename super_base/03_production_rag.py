import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector

load_dotenv()


# configration
@dataclass
class Config:
    # Database - use pooler Url in production
    database_url: str = os.getenv("SUPERBASE_DATABASE_URL")

    collection_name: str = "ankit_production_documents"

    # Model settings

    embedding_model: str = "models/gemini-embedding-2"
    chat_model: str = "gemini-2.5-flash"

    # Search settings
    default_k: int = 5
    min_similarity: float = 0.5


class RAGservice:
    """
    Production redy rag service with PGvector (PostgreSQL + Vector)
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self._vectorstore = None
        self._chain = None

    @property
    def vectorstore(self) -> PGVector:
        """Lazy initialization of vectorstore"""
        if self._vectorstore is None:
            embeddings = GoogleGenerativeAIEmbeddings(model=self.config.embedding_model)
            self._vectorstore = PGVector(
                embeddings=embeddings,
                collection_name=self.config.collection_name,
                connection=self.config.database_url,
                use_jsonb=True,
            )
        return self._vectorstore

    @property
    def chain(self):
        "Lazy initialization of Rag Chain"
        if self._chain is None:
            self._chain = self._create_chain()
        return self._chain

    def _create_chain(self):
        """Create Chain RAG"""

        retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.config.default_k}
        )

        llm = ChatGoogleGenerativeAI(model=self.config.chat_model)

        prompt = ChatPromptTemplate.from_template(
            """
You are a helpful assistant. Answer the question based on the provide context only

Context: 
{context}

Question: {question}

Answer concisely and accurately. If the context dosen't contain the revelant information
just say "I don't have enough information to answer that question."
"""
        )

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        return (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

    def add_documents(self, documents: list[Document]) -> list[str]:
        """Add Document to vector Store"""
        return self.vectorstore.add_documents(documents)

    def search(
        self, query: str, k: Optional[int] = None, filter_dict: Optional[dict] = None
    ) -> list[tuple[Document, float]]:
        """Search with optional filtering"""
        search_kwargs = {"k": k or self.config.default_k}
        if filter_dict:
            search_kwargs["filter"] = filter_dict

        return self.vectorstore.similarity_search_with_score(query, **search_kwargs)

    def ask(self, question: str) -> str:
        """Ask a question using RAG"""
        return self.chain.invoke(question)

    def ask_with_source(self, question: str) -> dict:
        """Ask a question and return sources"""

        # Get relevant documnets
        docs_with_scores = self.search(question)

        # Generate answer
        answer = self.ask(question)

        return {
            "answer": answer,
            "sources": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "similarity": score,
                }
                for doc, score in docs_with_scores
            ],
        }


def main():
    # Add some sample Documents with metadata
    print("\n Adding Sample document...")

    sample_docs = [
        Document(
            page_content="""
PGVector is a PostgreSQL extension for storing and searching vector embeddings.
It enables semantic similarity search directly inside PostgreSQL databases.
PGVector is widely used in Retrieval-Augmented Generation (RAG) systems because it combines relational data with AI embeddings.
""",
            metadata={
                "topic": "pgvector",
                "difficulty": "beginner",
                "category": "database",
                "source": "rag-course",
            },
        ),
        Document(
            page_content="""
HNSW (Hierarchical Navigable Small World) is an approximate nearest neighbor indexing algorithm.
It organizes vectors into multiple graph layers for extremely fast similarity search.
Important parameters include M (number of connections) and ef_search (search accuracy).
""",
            metadata={
                "topic": "hnsw",
                "difficulty": "intermediate",
                "category": "vector-index",
                "source": "rag-course",
            },
        ),
        Document(
            page_content="""
Chunking divides large documents into smaller pieces before embedding.
Common chunking strategies include recursive chunking, semantic chunking,
sentence chunking, and parent-child chunking.
Proper chunking significantly improves retrieval quality.
""",
            metadata={
                "topic": "chunking",
                "difficulty": "beginner",
                "category": "retrieval",
                "source": "rag-course",
            },
        ),
        Document(
            page_content="""
LangGraph is a framework for building stateful AI agents.
It represents workflows as graphs where nodes perform actions
and edges define transitions.
LangGraph supports memory, human-in-the-loop workflows,
parallel execution, and durable execution.
""",
            metadata={
                "topic": "langgraph",
                "difficulty": "intermediate",
                "category": "framework",
                "source": "rag-course",
            },
        ),
        Document(
            page_content="""
Hybrid search combines semantic vector search with keyword-based search.
This approach improves retrieval accuracy by matching both meaning and exact keywords.
Hybrid search is commonly implemented using BM25 together with vector similarity.
""",
            metadata={
                "topic": "hybrid-search",
                "difficulty": "intermediate",
                "category": "retrieval",
                "source": "rag-course",
            },
        ),
        Document(
            page_content="""
RAG cost optimization techniques include embedding dimension reduction,
vector quantization, embedding caching, batch embedding requests,
response caching, choosing the right embedding model,
and storing only high-quality chunks.
These techniques reduce latency and API costs.
""",
            metadata={
                "topic": "rag-cost",
                "difficulty": "advanced",
                "category": "optimization",
                "source": "rag-course",
            },
        ),
        Document(
            page_content="""
Scaling Retrieval-Augmented Generation involves vertical scaling
using larger machines and horizontal scaling using multiple distributed workers.
Large production systems often distribute vector databases,
retrievers, embedding services, and LLM inference independently.
""",
            metadata={
                "topic": "rag-scaling",
                "difficulty": "advanced",
                "category": "deployment",
                "source": "rag-course",
            },
        ),
        Document(
            page_content="""
The Multi Query Retriever generates several variations of the user's query
using an LLM.
Documents are retrieved for each variation and merged together,
improving recall for semantic search.
""",
            metadata={
                "topic": "multi-query",
                "difficulty": "intermediate",
                "category": "retrieval",
                "source": "rag-course",
            },
        ),
        Document(
            page_content="""
The Parent Document Retriever stores small chunks for embedding
while returning the larger parent document during retrieval.
This provides precise search together with sufficient context
for the language model.
""",
            metadata={
                "topic": "parent-document",
                "difficulty": "intermediate",
                "category": "retrieval",
                "source": "rag-course",
            },
        ),
        Document(
            page_content="""
Contextual compression retrieves documents first and then compresses them
using another model.
Only the information relevant to the user's query is returned,
reducing context length and improving answer quality.
""",
            metadata={
                "topic": "compression",
                "difficulty": "advanced",
                "category": "optimization",
                "source": "rag-course",
            },
        ),
        Document(
            page_content="""
LangSmith is an observability platform for LLM applications.
It helps developers monitor traces,
measure latency,
debug chains,
evaluate responses,
and compare prompts and model performance.
""",
            metadata={
                "topic": "langsmith",
                "difficulty": "beginner",
                "category": "monitoring",
                "source": "rag-course",
            },
        ),
        Document(
            page_content="""
Reranking improves retrieval quality by reordering retrieved documents.
A cross-encoder or reranking model scores candidate documents
based on the user query and returns the most relevant ones.
""",
            metadata={
                "topic": "reranking",
                "difficulty": "advanced",
                "category": "retrieval",
                "source": "rag-course",
            },
        ),
        Document(
            page_content="""
Query rewriting expands or reformulates the user's query
to improve retrieval.
Techniques include synonym expansion,
keyword extraction,
HyDE,
step-back prompting,
and LLM-generated alternative queries.
""",
            metadata={
                "topic": "query-rewriting",
                "difficulty": "advanced",
                "category": "retrieval",
                "source": "rag-course",
            },
        ),
        Document(
            page_content="""
Guardrails reduce hallucinations in LLM applications.
Common techniques include input validation,
output validation,
retrieval grounding,
citation verification,
confidence scoring,
fact checking,
and policy enforcement before returning responses.
""",
            metadata={
                "topic": "guardrails",
                "difficulty": "advanced",
                "category": "safety",
                "source": "rag-course",
            },
        ),
    ]


    services = RAGservice()

    ids = services.add_documents(sample_docs)
    print(f"Added {ids} documents")

    print("\nTesting Search...")

    print("=" * 50)

    test_queries = [
        "What is PGVector?",
        "Explain HNSW indexing",
        "How does chunking improve RAG?",
        "What is LangGraph used for?",
        "How does hybrid search work?",
        "Ways to reduce RAG costs",
        "How can I scale a RAG application?",
        "Explain Multi Query Retriever",
        "What is Parent Document Retriever?",
        "How does contextual compression work?",
        "What is LangSmith?",
        "Explain reranking",
        "How does query rewriting improve retrieval?",
        "How can I reduce hallucinations in LLMs?",
    ]

    for q in test_queries:
        print("=" * 60)
        print("Question:", q)
        print()

        answer = services.ask(q)


        print(answer)


if __name__ == "__main__":
    main()