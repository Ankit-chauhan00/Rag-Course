from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

texts = [
    "Paris is the capital of France",
    "Delhi is the capital of India",
    "Tokyo is the capital of Japan",
]


def main():

    client = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    response = client.invoke("What is the meaning of Gooning")

    print(response.content[:100])

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")

    vector = embeddings.embed_query(response.content)

    print(f"Vecton Length: {len(vector)}")
    print(vector[:10])


if __name__ == "__main__":
    main()
