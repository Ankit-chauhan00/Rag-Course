from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

embedding = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")


# Document with both semantic content and specific identitys

documents = [
    Document(
        page_content="""
        Order ORD-1001 was placed by Ankit Chauhan.
        The customer purchased an Apple MacBook Pro M4 with 16GB RAM and 512GB SSD.
        Payment was completed using a Visa credit card.
        The package was delivered successfully on 12 June 2026.
        """,
        metadata={
            "type": "order",
            "order_id": "ORD-1001",
            "customer": "Ankit Chauhan",
            "status": "Delivered",
        },
    ),
    Document(
        page_content="""
        Refund Policy:
        Customers can request a refund within 30 days after delivery.
        Refunds are only available for damaged, defective, or incorrect products.
        Digital products are non-refundable.
        """,
        metadata={"type": "policy", "category": "refund"},
    ),
    Document(
        page_content="""
        Failed Payment Guide:
        Error Code PAY-403 indicates payment authorization failure.
        Customers should verify card details or contact their bank.
        Retry the payment after a few minutes.
        """,
        metadata={"type": "support", "error_code": "PAY-403"},
    ),
    Document(
        page_content="""
        Product SKU LAP-001
        Apple MacBook Pro M4
        16GB Unified Memory
        512GB SSD
        Space Black
        Price: ₹1,69,900
        """,
        metadata={"type": "product", "sku": "LAP-001", "brand": "Apple"},
    ),
    Document(
        page_content="""
        Product SKU PHN-102
        Samsung Galaxy S26 Ultra
        512GB Storage
        Titanium Gray
        Supports Galaxy AI features.
        """,
        metadata={"type": "product", "sku": "PHN-102", "brand": "Samsung"},
    ),
    Document(
        page_content="""
        Shipping Policy:
        Standard shipping takes 3 to 5 business days.
        Express shipping usually arrives within 24 hours.
        International orders may require customs clearance.
        """,
        metadata={"type": "policy", "category": "shipping"},
    ),
    Document(
        page_content="""
        Customer Support Ticket TKT-7821
        Customer reported that order ORD-1001 was marked delivered
        but the package was not received.
        Investigation was initiated.
        """,
        metadata={"type": "ticket", "ticket_id": "TKT-7821", "priority": "High"},
    ),
    Document(
        page_content="""
        Inventory Report:
        SKU LAP-001 has 24 units remaining.
        SKU PHN-102 has 7 units remaining.
        SKU TAB-220 is currently out of stock.
        """,
        metadata={"type": "inventory"},
    ),
    Document(
        page_content="""
        User Account:
        Username: ankit123
        Email: ankit@example.com
        Membership Tier: Gold
        Loyalty Points: 8400
        """,
        metadata={"type": "user", "username": "ankit123"},
    ),
    Document(
        page_content="""
        Password Reset Instructions:
        If you forgot your password, click the 'Forgot Password' button.
        A password reset email will be sent to your registered email address.
        Reset links expire after 15 minutes.
        """,
        metadata={"type": "help"},
    ),
]


print(f"Loaded {len(documents)} documents")

vector_store = Chroma.from_documents(
    documents, embedding, collection_name="hybrid_test"
)

# create vector retriver

vector_retriver = vector_store.as_retriever(
    search_kwargs={"k": 3}  # Return top 3
)

print("Vector retriver ready")

# BM25 retriver works on a raw text
bm25_retriver = BM25Retriever.from_documents(
    documents,
    k=3,  # return top three
)

print("BM25 retriver ready")


# combine with EnsembeRetriver
ensemble_retriver = EnsembleRetriever(
    retrievers=[bm25_retriver, vector_retriver],
    weights=[0.5, 0.5],  # equal weights
)


print(" Hybrid is ready")


def text_query(query, name, retriever):
    results = retriever.invoke(query)

    print("\n" + "=" * 60)
    print(f"{name} | Query: {query}")
    print("=" * 60)

    for i, doc in enumerate(results, 1):
        print(f"\nResult {i}")
        print(doc.page_content.strip())
        print("Metadata:", doc.metadata)

    return results

queries = [
    "ORD-1001",
    "PAY-403",
    "MacBook Pro",
    "How can I get my money back?",
    "Package delivered but missing",
    "Refund for ORD-1001",
]

if __name__ == "__main__":
    for query in queries:
        print("\n" + "#" * 80)
        print(f"QUERY: {query}")
        print("#" * 80)

        text_query(query, "Vector", vector_retriver)
        text_query(query, "BM25", bm25_retriver)
        text_query(query, "Hybrid", ensemble_retriver)