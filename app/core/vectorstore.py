# app/core/vectorstore.py
import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from app.config import settings


def get_embedding_model():
    return OpenAIEmbeddings(openai_api_key=settings.openai_api_key)


def get_vectorstore():
    os.makedirs(settings.vector_db_dir, exist_ok=True)
    embeddings = get_embedding_model()
    vectorstore = Chroma(
        persist_directory=settings.vector_db_dir,
        embedding_function=embeddings,
    )
    return vectorstore


def get_doc_retriever():
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": settings.top_k_docs}
    )
    return retriever
