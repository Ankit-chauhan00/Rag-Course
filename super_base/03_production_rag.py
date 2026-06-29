from dotenv import load_dotenv
from langchain_postgres import PGVector
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
import os

load_dotenv()


# configration
@dataclass
class Config:

    # Database - use pooler Url in production
    database_url: str = os.getenv("SUPERBASE_DATABASE_URL")

    collection_name: str = "ankit_production_documents"

    # Model settings

    embedding_model: str = "models/gemini-embedding-2"
    chat_model : str = "gemini-2.5-flash"

    # Search settings
    default_k: int = 5
    min_similarity: float  = 0.5



class RAGservice:
    """
    Production redy rag service with PGvector (PostgreSQL + Vector)
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self._vectorstore = None
        self._chain = None

    @property
    def Vectorstore(self)-> PGVector:
        """Lazy initialization of vectorstore"""
        if self._vectorstore is None:
            embeddings = GoogleGenerativeAIEmbeddings(model=self.config.embedding_model)
            self._vectorstore = PGVector(
                embeddings=embeddings,
                collection_name=self.config.collection_name,
                connection=self.config.database_url,
                use_jsonb=True
            )
        return self._vectorstore

    @property
    def chain()
        