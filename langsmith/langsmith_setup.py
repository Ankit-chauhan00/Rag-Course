"""
LangSmith Setup and Observablity
Production monitoring for Langchain/Langraph
"""

import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from langsmith import traceable

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# enable Tracing environment variables
os.environ["LANGSMITH_TRACING"] = "true"


@traceable(name="basic_chaining")
def demo_basic_traning():
    """Basic Langsmmith Tracing"""

    prompt = ChatPromptTemplate.from_template("Explain {topic} in one sentense")

    chain = prompt | llm | StrOutputParser()

    print("Basic Tracing Demo:\n")
    print("Running chain with Langsmith tracing enabled")

    result = chain.invoke({"topic": "Machine Learning"})

    print(f"Result : {result}")
    print("\nCheck Langshith Dashboard for trace Details.")



@traceable(name="named_runs_demo", tags=["production", "summarization"])
def demo_named_runs():
    """
    Name your runs for easier identification.

    """

    prompt = ChatPromptTemplate.from_template("Summerize: {text}")

    chain = prompt | llm | StrOutputParser

    print("\nNamed Runs Demo:\n")

    result = chain.invoke(
        {"text": "LangSmith provides observability for LLM applications."}
    )

    print(f"Result: {result}")
    print("Run tagged with 'production', 'summarization'")



def demo_trace_with_metadata(user_id: str, request_type: str):
    """
    Add metadata to traces for filtering.
    """

    # metadata is automatically captured

    result = llm.invoke(f"Hello from user : {user_id}")

    return result.content

if __name__ == "__main__":
    demo_basic_traning()
