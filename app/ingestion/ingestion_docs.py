# app/ingestion/ingest_docs.py
import os
from typing import List

from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.core.vectorstore import get_vectorstore, get_embedding_model
from app.config import settings


def load_docs_from_path(path: str) -> List:
    docs = []
    for root, _, files in os.walk(path):
        for f in files:
            full_path = os.path.join(root, f)
            ext = os.path.splitext(f)[1].lower()
            if ext == ".pdf":
                loader = PyMuPDFLoader(full_path)
            elif ext in [".txt", ".md"]:
                loader = TextLoader(full_path, encoding="utf-8")
            else:
                print(f"Skipping unsupported file: {full_path}")
                continue

            docs.extend(loader.load())
    return docs


def ingest_documents(raw_docs_dir: str):
    print(f"Loading documents from {raw_docs_dir} ...")
    docs = load_docs_from_path(raw_docs_dir)
    print(f"Loaded {len(docs)} documents")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        length_function=len,
    )

    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks")

    embeddings = get_embedding_model()
    vectorstore = get_vectorstore()

    # Chroma has an add_documents method
    vectorstore.add_documents(chunks)
    vectorstore.persist()
    print("Ingestion complete. Vector store updated.")


if __name__ == "__main__":
    raw_dir = "./data/raw_docs"  # put your PDFs here
    ingest_documents(raw_dir)
