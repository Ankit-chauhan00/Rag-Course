from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage


llm = ChatOllama(model="llama3.2")

response = llm.invoke([
    SystemMessage(content="You are a helpful assistant"),
    HumanMessage(content="What is the capital of France?")
])

print(response.content)