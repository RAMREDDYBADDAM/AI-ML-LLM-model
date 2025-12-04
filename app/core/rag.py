import os
from io import BytesIO
from typing import List, Tuple

from dotenv import load_dotenv
load_dotenv()

# --- Text loading / splitting ---
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# --- Embeddings: BGE ---
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- Vector stores: FAISS (default) or Pinecone (optional) ---
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Pinecone as LCPinecone

# --- Ollama LLM ---
from langchain_community.chat_models import ChatOllama
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

# Optional Pinecone backend
try:
    import pinecone
    _HAS_PINECONE = True
except Exception:
    _HAS_PINECONE = False


def _read_pdf_bytes(data: bytes) -> str:
    """Extract raw text from PDF bytes."""
    pdf = PdfReader(BytesIO(data))
    texts = []
    for p in pdf.pages:
        try:
            texts.append(p.extract_text() or "")
        except Exception:
            texts.append("")
    return "\n".join(texts)


def _to_documents(text: str, splitter: RecursiveCharacterTextSplitter) -> List[Document]:
    """Convert text into LangChain Document chunks."""
    docs = splitter.create_documents([text])
    for d in docs:
        d.metadata.setdefault("source", "uploaded")
    return docs


class RAGService:
    """
    RAG pipeline (Ollama-only):
      - BGE embeddings (BAAI/bge-large-en-v1.5)
      - LangChain RecursiveCharacterTextSplitter
      - FAISS local vector store (default) or Pinecone
      - Ollama LLM for generation
    """

    def __init__(self):
        # ---- Text splitter ----
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=800, chunk_overlap=120
        )

        # ---- Embedding model ----
        self.embedding = HuggingFaceEmbeddings(
            model_name=os.getenv("BGE_MODEL", "BAAI/bge-large-en-v1.5"),
            encode_kwargs={"normalize_embeddings": True},
        )

        # ---- Choose vector store ----
        self.use_pinecone = (
            bool(os.getenv("PINECONE_API_KEY"))
            and bool(os.getenv("PINECONE_INDEX"))
            and _HAS_PINECONE
        )

        if self.use_pinecone:
            pinecone.init(api_key=os.environ["PINECONE_API_KEY"])
            index_name = os.environ["PINECONE_INDEX"]
            self.vs = LCPinecone.from_texts([], embedding=self.embedding, index_name=index_name)
        else:
            self.vs = FAISS.from_texts([""], embedding=self.embedding)
            # clean placeholder
            self.vs.index.reset()
            self.vs.docstore._dict.clear()

        # ---- Ollama model ----
        self.ollama_base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.ollama_model = os.getenv("OLLAMA_MODEL", "llama3:latest")

        # ---- Prompt + chain ----
        self.prompt = ChatPromptTemplate.from_template(
            """You are a helpful AI assistant. Use only the following context to answer the question below.
If you don't know the answer, say you don't know.

<context>
{context}
</context>

Question: {input}
Answer:"""
        )
        self.llm = ChatOllama(
            base_url=self.ollama_base,
            model=self.ollama_model,
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
        )
        self.chain = create_stuff_documents_chain(self.llm, self.prompt)

    # --- Ingest documents ---
    def index_bytes(self, filename: str, content: bytes):
        if filename.lower().endswith(".pdf"):
            text = _read_pdf_bytes(content)
        else:
            text = content.decode("utf-8", errors="ignore")

        docs = _to_documents(text, self.splitter)

        if self.use_pinecone:
            texts = [d.page_content for d in docs]
            self.vs.add_texts(texts=texts, metadatas=[d.metadata for d in docs])
        else:
            self.vs.add_documents(docs)

    # --- Retrieve ---
    def _retrieve(self, query: str, k: int = 5) -> List[Document]:
        return self.vs.similarity_search(query, k=k)

    # --- Generate ---
    def answer(self, query: str) -> Tuple[str, List[str]]:
        top_docs = self._retrieve(query, k=5)
        if not top_docs:
            return "No documents indexed yet.", []

        result = self.chain.invoke({"input": query, "context": top_docs})
        answer = result.content if hasattr(result, "content") else str(result)
        return answer, [d.page_content for d in top_docs]
if __name__ == "__main__":
    rag = RAGService()
    print("✅ RAG service initialized successfully!")

    # Simple text test (you can replace this with any .txt file content)
    text_data = b"Apple is a leading technology company known for the iPhone and MacBook."
    rag.index_bytes("sample.txt", text_data)

    question = "What products is Apple known for?"
    answer, sources = rag.answer(question)
    print("\n🧠 Question:", question)
    print("💬 Answer:", answer)
    print("📚 Context chunks:", len(sources))
