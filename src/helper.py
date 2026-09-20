from typing import List

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Centralized configuration shared across app.py, streamlit.py and store_index.py.
PINECONE_INDEX_NAME = "medical-queryai"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384  # dimensionality produced by the embedding model above
GROQ_MODEL_NAME = "openai/gpt-oss-120b"
RETRIEVER_TOP_K = 3


def load_pdf_file(data: str) -> List[Document]:
    """Load all PDF files found in the given directory."""
    loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
    return loader.load()


def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Given a list of Document objects, return a new list of Document objects
    containing only 'source' in metadata and the original page_content.
    """
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(page_content=doc.page_content, metadata={"source": src})
        )
    return minimal_docs


def text_split(extracted_data: List[Document]) -> List[Document]:
    """Split documents into overlapping chunks suitable for embedding."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    return text_splitter.split_documents(extracted_data)


def download_hugging_face_embeddings() -> HuggingFaceEmbeddings:
    """Load the sentence-transformers embedding model (384-dimensional vectors)."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
