"""Serve helper for the Financial RAG project.

Features:
- Loads `.env` automatically
- Optional `--ingest` flag to call the ingestion pipeline before serving
- Configurable host, port and reload via CLI

Usage examples:
  # start normally
  python serve.py

  # start with auto-reload on port 8000
  python serve.py --reload --port 8000

  # run ingestion then start
  python serve.py --ingest
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run the Financial RAG FastAPI app")
    p.add_argument("--host", default="127.0.0.1", help="Host to bind (default 127.0.0.1)")
    p.add_argument("--port", type=int, default=8000, help="Port to bind (default 8000)")
    p.add_argument("--reload", action="store_true", help="Enable uvicorn reload (development)")
    p.add_argument("--ingest", action="store_true", help="Run document ingestion before starting the server")
    p.add_argument("--raw-docs", default="./data/raw_docs", help="Path to raw documents for ingestion")
    return p


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    # Load .env if present
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.ingest:
        logging.info("Running document ingestion from %s", args.raw_docs)
        try:
            # Import here to avoid heavy deps at module import time
            from app.ingestion.ingestion_docs import ingest_documents

            ingest_documents(args.raw_docs)
            logging.info("Ingestion finished")
        except Exception as e:
            logging.exception("Ingestion failed: %s", e)
            return 2

    # Start uvicorn programmatically
    try:
        import uvicorn

        logging.info("Starting uvicorn on %s:%s (reload=%s)", args.host, args.port, args.reload)
        uvicorn.run("app.core.server:app", host=args.host, port=args.port, reload=args.reload, log_level="info")
        return 0
    except Exception as e:
        logging.exception("Failed to start server: %s", e)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
