# app/config.py
import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    # OpenAI configuration
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    
    # Ollama configuration (local LLM)
    ollama_enabled: bool = os.getenv("OLLAMA_ENABLED", "true").lower() == "true"
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "mistral")  # Popular models: mistral, neural-chat, dolphin-mixtral
    
    # Database configuration
    database_url: str = os.getenv("DATABASE_URL", "")
    vector_db_dir: str = os.getenv("VECTOR_DB_DIR", "./data/vectorstore")

    # Model & RAG config
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")  # OpenAI model fallback
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))
    top_k_docs: int = 5


settings = Settings()
