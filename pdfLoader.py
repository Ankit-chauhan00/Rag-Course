from langchain_community.document_loaders import PyPDFLoader


def pdf_loader(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    documet = loader.load()

    print(f"Loaded {len(documet)} document(s) from Pdf")

    for i, doc in enumerate(documet):
        print(f"Document {i + 1} Content Preview: {doc.page_content[:100]}")
        print(f"Metadata: {doc.metadata}")


if __name__ == "__main__":
    pdf_loader("./docs/sample.pdf")
