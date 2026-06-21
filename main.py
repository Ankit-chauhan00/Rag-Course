from dotenv import load_dotenv
from langchain_core import __version__ as core_version  # noqa: E402
from importlib.metadata import version  # noqa: E402
from langchain_google_genai import ChatGoogleGenerativeAI
load_dotenv()

lg_version = version("langgraph")

print(f"LangChain Core Version: {core_version}")
print(f"LangGraph Version: {lg_version}")

def main():
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    response = llm.invoke("HELLO!")
    print(response.content)

    response2 = llm.invoke("Who is current president of Usa")
    print(response2.content)


if __name__ == "__main__":
    main()
