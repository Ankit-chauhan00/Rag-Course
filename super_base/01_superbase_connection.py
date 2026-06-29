import os

from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector
from dotenv import load_dotenv

load_dotenv()

SUPREBASE_URL = os.getenv("SUPERBASE_DATABASE_URL")
host = SUPREBASE_URL.split("@")[1].split(":")[0]
print(host)


def connect_to_suprebase():
    """Connect to suprebase pg Vector"""

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")

    # Note : Suprebase has pgvector pre-installed
    vectorstore = PGVector(
        embeddings=embeddings,
        collection_name="production_docs",
        connection=SUPREBASE_URL,
        use_jsonb=True,
    )

    return vectorstore


def verify_connection(vectorstore: PGVector):
    """Verify the connection works"""

    # add a test document
    test_doc = Document(
        page_content="This is a test document to verify Suprebase setup",
        metadata={"source": "test", "type": "verification"},
    )

    try:
        ids = vectorstore.add_documents([test_doc])
        print(f"Addd the test document: {ids[0]}")

        # Search for it
        result = vectorstore.similarity_search("verify supabase connection", k=1)

        if result:
            print(f"Search works: {result[0].page_content}")

        # Clean up

        # vectorstore.delete(ids)
        # print("Cleanup complete")

        return True

    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    print("Supabase pgvector Connection Test")
    print("=" * 60)

    if not SUPREBASE_URL:
        print("SUPABASE_DATABASE_URL not set")
        return

    host = SUPREBASE_URL.split("@")[1].split(":")[0]
    print(f"Connecting to: {host}")

    vectorstore = connect_to_suprebase()

    if verify_connection(vectorstore):
        print("\nEverything works!")
    else:
        print("\nConnection failed.")




if __name__ == "__main__":
    main()
