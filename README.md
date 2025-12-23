# Financial RAG Project

A Retrieval-Augmented Generation (RAG) system for financial data analysis with live stock data integration.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the API server
uvicorn app.core.server:app --reload --host 0.0.0.0 --port 8000

# 3. Test the API
curl -X POST "http://localhost:8000/chat" -H "Content-Type: application/json" \
  -d '{"user_id":"u1","question":"Tell me about Apple stock"}'
```

## Data Sources & Fallback Hierarchy

The system uses the following priority for data retrieval:

| Priority | Source | Requirements |
|----------|--------|--------------|
| 1 | **PostgreSQL Database** | `DATABASE_URL` configured and DB reachable |
| 2 | **Yahoo Finance (Live)** | `USE_YAHOO_DATA=true` + `yfinance` installed |
| 3 | **Sample Data (Fallback)** | Only when both above fail |

## Environment Configuration

Create a `.env` file in the project root:

```env
# ============================================================
# YAHOO FINANCE (Recommended - No setup required)
# ============================================================
USE_YAHOO_DATA=true
YAHOO_CACHE_TTL=900  # Cache TTL in seconds (15 min default)
YAHOO_SYMBOLS=AAPL,MSFT,GOOGL,AMZN,NVDA,TSLA,META

# ============================================================
# POSTGRESQL DATABASE (Optional - For custom data)
# ============================================================
# Uncomment to enable database mode:
# DATABASE_URL=postgresql://postgres:postgres@localhost:5432/financial_db

# Individual DB settings (used by csv_to_sql.py)
DB_HOST=localhost
DB_PORT=5432
DB_NAME=financial_db
DB_USER=postgres
DB_PASSWORD=postgres

# ============================================================
# LLM CONFIGURATION
# ============================================================
OLLAMA_ENABLED=true
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=mistral

# ============================================================
# VECTOR STORE
# ============================================================
VECTOR_DB_DIR=./data/vectorstore
```

## Database Setup (Optional)

If you want to use PostgreSQL instead of Yahoo Finance:

```bash
# Load S&P 500 historical data into PostgreSQL
python scripts/csv_to_sql.py --mode replace

# This creates the sp500_data table with columns:
# - date, sp500, dividend, earnings, cpi, etc.
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/chat` | POST | Main chat endpoint for financial Q&A |
| `/api/insights` | GET | Get market insights and analytics |
| `/health` | GET | Health check |

### Chat Request Format

```json
{
  "user_id": "user123",
  "question": "What is Apple's current stock price?"
}
```

### Response Format

```json
{
  "answer": "## 📊 Apple Inc. (AAPL) - Live Data\n\n**Current Price:** $195.50...",
  "query_type": "SQL",
  "source": "yahoo_finance"
}
```

## Testing Live Data

```bash
# Test Yahoo Finance integration
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"user_id":"test","question":"Tell me about NVDA stock"}'

# Expected response includes live price, market cap, P/E ratio from Yahoo Finance
```

## Troubleshooting

### "Live database unavailable" message appears

1. Check if `USE_YAHOO_DATA=true` in `.env`
2. Verify yfinance is installed: `pip install yfinance`
3. Check internet connectivity
4. Try clearing cache: restart the server

### Database connection fails

1. Ensure PostgreSQL is running
2. Verify `DATABASE_URL` format: `postgresql://user:pass@host:port/dbname`
3. Run: `python scripts/csv_to_sql.py --mode replace` to initialize tables

## Project Structure

```
app/
├── core/
│   ├── server.py       # FastAPI application
│   ├── chains.py       # RAG/SQL chain logic
│   ├── yahoo_service.py # Yahoo Finance integration
│   ├── router.py       # Query classification
│   └── vectorstore.py  # Document retrieval
├── config.py           # Settings management
└── ingestion/          # Data ingestion scripts

scripts/
└── csv_to_sql.py       # Load CSV data to PostgreSQL

data/
├── raw_docs/           # PDF/text documents for RAG
└── vectorstore/        # Chroma vector store
```
