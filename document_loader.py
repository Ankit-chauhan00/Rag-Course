import os
import tempfile
from pathlib import Path
from langchain_community.document_loaders import (TextLoader)
from dotenv import load_dotenv

load_dotenv()


def load_text_file():
    #create a temporary file for demonstration
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
        temp_file.write(b"Hello this a text file temporary used for \n a demonstration")
        temp_file_path = temp_file.name

    try:
        #load teh text File using Texture Loader
        loader = TextLoader(temp_file_path)
        document = loader.load()

        # print the loaded document

        print(f"Loaded {len(document)} documents(s)")
        print(f"Content Preview: {document[0].page_content[:100]}...")
        print  (f"Metadata: {document[0].metadata}")

        # for doc in document:
        #     print("Document Content")
        #     print(doc)
        #     print(doc.page_content)

    finally: 
        #clean up the temporary file

        os.remove(temp_file_path)


if __name__ == "__main__":
    load_text_file()