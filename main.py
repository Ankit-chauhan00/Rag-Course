from dotenv import load_dotenv
from langchain_core import __version__ as core_version  # noqa: E402
from importlib.metadata import version  # noqa: E402
from langchain_ollama import ChatOllama  # noqa: E402
load_dotenv()

lg_version = version("langgraph")

print(f"LangChain Core Version: {core_version}")
print(f"LangGraph Version: {lg_version}")

def main():
    llm =ChatOllama(model="llama3.2", temperature=0)
    response = llm.invoke("Say Setup Completed! in One Word")
    print(f"Response from ollama LLM :  {response}")

    print("Setup Completed!")



if __name__ == "__main__":
    main()
