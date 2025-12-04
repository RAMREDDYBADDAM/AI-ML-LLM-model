"""LLM wiring supporting both Ollama (local) and OpenAI backends.

Priority order:
1. Ollama (if enabled and available at OLLAMA_BASE_URL)
2. OpenAI (if OPENAI_API_KEY is set)
3. Mock LLM (demo fallback)
"""
from typing import Any
import requests
from app.config import settings

# Try to import Ollama client from langchain
try:
    from langchain_community.chat_models import ChatOllama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


def _check_ollama_health() -> bool:
    """Check if Ollama is running and accessible."""
    if not OLLAMA_AVAILABLE or not settings.ollama_enabled:
        return False
    try:
        resp = requests.get(f"{settings.ollama_base_url}/api/tags", timeout=2)
        return resp.status_code == 200
    except Exception:
        return False


def get_llm() -> Any:
    """
    Returns a configured LLM instance.
    
    Tries in this order:
    1. Ollama (local, free, privacy-friendly)
    2. OpenAI (cloud-based, requires API key)
    3. Mock LLM (demo mode)
    """
    
    # Try Ollama first (local, no API key needed)
    if settings.ollama_enabled and OLLAMA_AVAILABLE:
        try:
            if _check_ollama_health():
                print(f"✅ Using Ollama LLM: {settings.ollama_model}")
                return ChatOllama(
                    base_url=settings.ollama_base_url,
                    model=settings.ollama_model,
                    temperature=float(settings.temperature),
                )
        except Exception as e:
            print(f"⚠️ Ollama initialization failed: {e}")
    
    # Fallback to OpenAI
    if settings.openai_api_key:
        try:
            from langchain_openai import ChatOpenAI
            print(f"✅ Using OpenAI LLM: {settings.llm_model}")
            return ChatOpenAI(
                model=settings.llm_model,
                temperature=float(settings.temperature),
                api_key=settings.openai_api_key,
            )
        except Exception as e:
            print(f"⚠️ OpenAI initialization failed: {e}")
    
    # Final fallback: use mock
    print("⚠️ No LLM backend available, using mock LLM (demo mode)")
    from app.core.mock_llm import get_mock_llm
    return get_mock_llm()
