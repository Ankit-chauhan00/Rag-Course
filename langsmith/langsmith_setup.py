"""
LangSmith Setup and Observablity
Production monitoring for Langchain/Langraph
"""

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import traceable
from langsmith.run_trees import RunTree
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# enable Tracing environment variables
os.environ["LANGSMITH_TRACING"] = "true"


@traceable(name="basic_chaining")
def demo_basic_traning():
    """Basic Langsmmith Tracing"""

    prompt = ChatPromptTemplate.from_template(
        "Explain {topic} in one sentense"
    )

    chain = prompt | llm | StrOutputParser()

    print("Basic Tracing Demo:\n")
    print("Running chain with Langsmith tracing enabled")

    result  = chain.invoke({"topic": "Machine Learning"})

    print(f"Result : {result}")
    print("\nCheck Langshith Dashboard for trace Details.")