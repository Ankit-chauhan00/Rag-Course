from dotenv import load_dotenv
from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

document = """
Artificial Intelligence (AI) is the field of computer science that focuses on creating systems capable of performing tasks that typically require human intelligence. These tasks include reasoning, problem-solving, understanding natural language, recognizing images, and learning from data. Modern AI systems often rely on machine learning algorithms trained on large datasets.

Machine Learning is a subset of AI that enables computers to learn patterns from data without being explicitly programmed for every task. There are three primary types of machine learning: supervised learning, unsupervised learning, and reinforcement learning. Supervised learning uses labeled data, while unsupervised learning discovers hidden structures in unlabeled data.

Deep Learning is a specialized branch of machine learning that uses artificial neural networks with multiple layers. These networks excel at image recognition, speech processing, and natural language understanding. Large Language Models such as GPT are built using transformer architectures, which have significantly advanced the field of AI.

Natural Language Processing (NLP) focuses on enabling computers to understand and generate human language. Applications include chatbots, translation systems, text summarization, and sentiment analysis. Recent advances in transformer-based models have greatly improved NLP performance.

Computer Vision is another branch of AI that allows machines to interpret and analyze visual information. Common applications include facial recognition, autonomous driving, medical image analysis, and object detection. Convolutional Neural Networks (CNNs) were once the dominant architecture for computer vision, although Vision Transformers are now becoming increasingly popular.

Databases are essential for storing and retrieving structured information efficiently. Relational databases like PostgreSQL and MySQL organize data into tables connected through relationships. SQL is the standard language used to query and manipulate relational databases.

Vector databases are specifically designed to store high-dimensional embeddings. Instead of matching exact keywords, they perform similarity search using mathematical distance metrics such as cosine similarity or Euclidean distance. Popular vector databases include ChromaDB, Pinecone, Milvus, and Weaviate.

Retrieval-Augmented Generation (RAG) combines information retrieval with large language models. In a RAG pipeline, documents are split into chunks, converted into embeddings, stored in a vector database, and retrieved during user queries. The retrieved context is then provided to the language model to generate accurate responses.

Semantic chunking improves retrieval quality by grouping sentences based on their meaning rather than fixed character or token limits. Unlike traditional chunking, semantic chunking attempts to keep related ideas together, reducing the chance of splitting important concepts across multiple chunks.

Cloud computing enables developers to deploy scalable applications without managing physical hardware. Services like AWS, Microsoft Azure, and Google Cloud Platform provide computing, storage, networking, and AI services on demand. Serverless computing further simplifies deployment by automatically scaling resources based on incoming requests.
""".strip()




def smart_chunker(
    text: str,
    use_semantic: bool = True,
    fallback_chunk_size=500,
) -> list[str]:
    """
    Production chunking with semantic as primary, recursive fallback.
    """

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")

    if use_semantic:
        try:
            chunker = SemanticChunker(
                embeddings,
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=80,
            )

            chunks = chunker.split_text(text)

            # validate if the chunks are too large
            max_chunk_size = 2000
            if any(len(c) > max_chunk_size for c in chunks):
                # fallback to recursive for oversized chunks
                return _recursive_fallback(text, fallback_chunk_size)

            return chunks

        except Exception as e:
            print(f"Semantic chunking failed: {e}, using fallback")
            return _recursive_fallback(text, fallback_chunk_size)


def _recursive_fallback(text: str, chunk_size: int) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=50
    )

    return splitter.split_text(text)

# Usage
chunks = smart_chunker(document, use_semantic=True)
print(f'Created {len(chunks)} semantic chunks')

