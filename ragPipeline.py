from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import (GoogleGenerativeAIEmbeddings)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from pydantic import BaseModel, Field

load_dotenv()