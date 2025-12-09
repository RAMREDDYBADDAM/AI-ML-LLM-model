<!-- .github/copilot-instructions.md -->
# Copilot / Agent Instructions — Financial RAG Project

Purpose
- Help AI coding agents be productive quickly: runnable entrypoints, key patterns, and configuration.

Quick Start (developer)
- Install deps: `pip install -r requirements.txt`.
- Ingest documents (put PDFs / .txt into `data/raw_docs`):
  - `python app/ingestion/ingestion_docs.py`
- Run the API server (FastAPI app is in `app/core/server.py`):
  - `uvicorn app.core.server:app --reload --host 0.0.0.0 --port 8000`

Important Files / Entry Points
- `app/core/server.py` — primary FastAPI app and `/chat` endpoint. Use as `uvicorn app.core.server:app`.
- `app/core/chains.py` — high-level chains: `run_doc_rag`, `run_sql_analytics`, `run_hybrid`, and `answer_financial_question` (API entry).
- `app/core/router.py` — classification chain that returns `query_type` (DOC | SQL | HYBRID) using LangChain StructuredOutputParser.
- `app/core/vectorstore.py` — Chroma-backed vector store helpers and `get_doc_retriever()` used by RAG flows.
- `app/core/sql_tools.py` — SQL agent creation; expects `DATABASE_URL` for `SQLDatabase.from_uri()`.
- `app/core/rag.py` — alternative RAG pipeline using HuggingFace/FAISS or Pinecone + Ollama; used for local indexing/testing.
- `app/core/plot_generator.py` — **[NEW]** Financial metrics visualization pipeline: extracts company ticker and metric from text, queries PostgreSQL, generates matplotlib plots, returns base64-encoded JSON response.
- `app/ingestion/ingestion_docs.py` — ingestion script: loads PDFs and text, splits into chunks, persists to vectorstore.
- `app/config.py` — single source of runtime config (`settings`) using `pydantic.BaseSettings` and `.env`.

Configuration & Environment
- Uses `app.config.settings`. Check or set in a `.env` file in repo root.
- Relevant env vars discovered in code:
  - `OPENAI_API_KEY` — OpenAI embeddings (used by `OpenAIEmbeddings`).
  - `DATABASE_URL` — SQL DB URI for SQL agent and plot generator (postgres example: `postgresql://user:pass@host:port/db`).
  - `VECTOR_DB_DIR` — local Chroma persistence dir (default `./data/vectorstore`).
  - `BGE_MODEL`, `PINECONE_API_KEY`, `PINECONE_INDEX` — optional Pinecone/embedding model for `app/core/rag.py`.
  - `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `LLM_TEMPERATURE` — used by `app/core/rag.py` if using Ollama locally.

Project Patterns & Conventions (explicit)
- Single `settings` object: import `from app.config import settings` and read `settings.openai_api_key`, `settings.vector_db_dir`, etc.
- Chains + small helpers: high-level orchestration is in `app/core/chains.py`. Keep API-compatible return shape: dict with `answer` and `query_type`.
- Router decides flow: call `classify_query(question)` from `app/core/router.py` to decide DOC/SQL/HYBRID.
- Vectorstore persistence: Chroma persists to `settings.vector_db_dir`. Ingestion uses `vectorstore.add_documents(chunks)` then `vectorstore.persist()`.

Project Caveats (things for agents to watch)
- The FastAPI app used at runtime is `app/core/server.py` (not `app/api/server.py` — the latter is currently empty).
- The package `app` contains `_init_.py` (single underscores) instead of `__init__.py`. This can make `python -m` imports fail; running scripts directly (`python app/ingestion/ingestion_docs.py`) works. Consider fixing filename to `__init__.py` if converting to module invocation.
- `get_llm()` is referenced across `app/core/*` but the `app/core/llm.py` implementation is not present in this tree — check for a missing file or provide a `get_llm()` that returns a properly configured langchain LLM wrapper.

Common Tasks & Examples
- Ingest local docs (from project root):
  ```bash
  pip install -r requirements.txt
  python app/ingestion/ingestion_docs.py
  ```
- Start server and test chat (from project root):
  ```bash
  uvicorn app.core.server:app --reload --port 8000
  curl -X POST "http://localhost:8000/chat" -H "Content-Type: application/json" -d '{"user_id":"u1","question":"What are Apple's main products?"}'
  ```
- Generate financial plot from RAG output:
  ```bash
  curl -X POST "http://localhost:8000/api/plot" -H "Content-Type: application/json" -d '{"user_id":"u1","question":"Show me Apple revenue growth trend"}'
  ```
  Returns: `{"company":"AAPL","metric":"revenue","data_points":N,"is_trend":true,"plot_base64":"<base64-encoded PNG>"}`
- If using SQL features: ensure `DATABASE_URL` points to a populated DB and run queries via the `/chat` endpoint; the SQL agent will generate and run SQL through LangChain's SQL toolkit.
- If using plot generation: ensure `DATABASE_URL` points to a PostgreSQL instance with the financial metrics schema (see `db/schema.sql` for table definitions).

Dependencies
- See `requirements.txt` for the main packages: `fastapi`, `uvicorn`, `langchain`, `langchain-openai`, `langchain-community`, `chromadb`, `sqlalchemy`, `psycopg2-binary`, `python-dotenv`, `pymupdf`, `unstructured`, `matplotlib`.

If you need clarification
- Tell me which area to expand (LLM wiring, DB setup, or fixing package init). I can: add example `app/core/llm.py`, change `_init_.py` to `__init__.py`, or produce a `README.md` with run scripts.
