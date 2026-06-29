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

