from typing import Dict, Any
from app.config import settings
from app.core.router import classify_query
from app.core.llm import get_llm

# Do NOT instantiate LLMs at import time. Use get_llm() inside functions
# so startup / uvicorn --reload doesn't trigger heavy network I/O.

# =============================================================================
# PROFESSIONAL FINANCIAL ANALYST SYSTEM PROMPT
# =============================================================================
_FINANCIAL_ANALYST_SYSTEM_PROMPT = """You are a financial analysis assistant for stakeholders and investors.

You must STRICTLY follow these rules:
- Use ONLY the provided retrieved context
- NEVER invent numbers, dates, or performance metrics
- NEVER state "no data available" without providing a useful summary
- NEVER hallucinate market-wide statistics

WHEN DATA IS PARTIAL:
- Summarize what IS available clearly
- Explicitly scope conclusions (company-level vs market-level)
- Explain limitations briefly and professionally
- Provide cautious, real-world insights based on known financial principles
- Suggest realistic next steps without assuming missing data

RESPONSE STRUCTURE (MANDATORY):
1. **Summary** - Key findings based only on retrieved data
2. **Context & Data Scope** - What data was analyzed, time period, limitations
3. **Practical Insights** - Actionable observations for stakeholders
4. **Data-Grounded Suggestions** - Next steps without predictions or financial advice

TONE:
- Professional and analyst-style
- Neutral and factual
- Helpful, not defensive

DO NOT:
- Ask the user to fetch external sources
- Claim lack of usefulness
- Use phrases like "cannot answer" or "insufficient data" alone
- Provide investment advice or predictions

Your goal is to sound like a real financial analyst explaining insights from available data."""

_GUARDRAIL_DIRECTIVE = (
    "You must never invent or guess financial numbers. If specific data is not in the context, "
    "clearly state what IS available and provide analysis based on that. "
    "Always maintain professional analyst tone and structure your response clearly."
)


def _llm_invoke(messages: list, use_mock: bool = False) -> str:
    """
    Invoke the configured LLM (Ollama, OpenAI, or mock).
    
    For LLM chains, we convert messages to string prompt format.
    """
    if use_mock:
        from app.core.mock_llm import get_mock_llm
        mock_llm = get_mock_llm()
        user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
        return mock_llm.invoke(user_msg).get("content", "No response")
    
    try:
        llm = get_llm()
        # Try using common LLM interface
        if hasattr(llm, "invoke"):
            prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
            result = llm.invoke(prompt)
            if hasattr(result, "content"):
                return result.content
            return str(result)

        if hasattr(llm, "get_response"):
            return llm.get_response(messages)

        if hasattr(llm, "__call__"):
            # Some LangChain wrappers are callable
            prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
            result = llm(prompt)
            if isinstance(result, dict):
                return result.get("content", str(result))
            if hasattr(result, "content"):
                return result.content
            return str(result)

        # Last-resort fallback to mock
        from app.core.mock_llm import get_mock_llm
        mock_llm = get_mock_llm()
        user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
        return mock_llm.invoke(user_msg).get("content", "No response")
    except Exception as e:
        print(f"LLM error: {e}, using mock fallback")
        from app.core.mock_llm import get_mock_llm
        mock_llm = get_mock_llm()
        user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
        return mock_llm.invoke(user_msg).get("content", "No response")



# ------------------------------------------------------
# Document RAG Chain
# ------------------------------------------------------
def run_doc_rag(question: str) -> Dict[str, Any]:
    try:
        from app.core.vectorstore import get_doc_retriever
        retriever = get_doc_retriever()
        docs = retriever.invoke(question)
        context = "\n\n---\n\n".join([d.page_content for d in docs])
    except Exception as e:
        docs = []
        context = f"(Vectorstore unavailable: {str(e)[:80]})"

    # If no docs found, try to build a small company context so answers aren't generic
    if not docs:
        try:
            from app.core.sp500_companies import search_companies
            hits = search_companies(question, 1)
            if hits:
                c = hits[0]
                context = (
                    f"Company: {c.get('name','N/A')} ({c.get('ticker','')})\n"
                    f"Sector: {c.get('sector','N/A')}\n"
                    f"Revenue: {c.get('revenue','N/A')}\n"
                    f"Period: {c.get('period','N/A')}"
                )
        except Exception:
            pass

    system = (
        f"{_FINANCIAL_ANALYST_SYSTEM_PROMPT}\n\n"
        f"ADDITIONAL DIRECTIVE: {_GUARDRAIL_DIRECTIVE}"
    )

    user = f"""RETRIEVED CONTEXT:
{context}

USER QUESTION: {question}

Provide a structured, professional financial analysis response following the mandatory format:
1. Summary
2. Context & Data Scope  
3. Practical Insights
4. Data-Grounded Suggestions"""

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    answer = _llm_invoke(messages)
    return {"answer": answer, "source_count": len(docs), "query_type": "DOC"}


# ------------------------------------------------------
# Database Availability Check
# ------------------------------------------------------
def _check_db_available() -> bool:
    """
    Check if database is reachable by attempting SELECT 1.
    Returns True if DB is available, False otherwise.
    """
    if not settings.database_url:
        return False
    
    try:
        from sqlalchemy import create_engine, text
        engine = create_engine(settings.database_url)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        print(f"[DB CHECK] Database unavailable: {e}")
        return False


def _check_yahoo_available() -> bool:
    """
    Check if Yahoo Finance service is available and enabled.
    Returns True if Yahoo is available, False otherwise.
    """
    try:
        from app.core.yahoo_service import is_yahoo_available
        return is_yahoo_available()
    except ImportError:
        print("[YAHOO CHECK] yfinance not installed")
        return False
    except Exception as e:
        print(f"[YAHOO CHECK] Yahoo Finance unavailable: {e}")
        return False


# ------------------------------------------------------
# SQL Analytics Chain
# ------------------------------------------------------
def run_sql_analytics(question: str) -> Dict[str, Any]:
    """
    SQL chain with proper fallback hierarchy:
    1. Database (if configured and reachable)
    2. Yahoo Finance (if enabled and available)
    3. Sample data fallback (only if both above fail)
    """
    
    # Step 1: Check DB availability
    db_available = _check_db_available()
    yahoo_available = _check_yahoo_available()
    
    print(f"[SQL ANALYTICS] DB available: {db_available}, Yahoo available: {yahoo_available}")
    
    # Try database first if available
    if db_available:
        try:
            from app.core.sql_tools import get_sql_agent
            agent = get_sql_agent()
            result = agent.invoke(question)

            if isinstance(result, dict):
                answer_text = result.get("output", str(result))
            else:
                answer_text = str(result)

            answer_text = answer_text.strip() or "The SQL agent did not return any rows."

            return {
                "answer": f"## 📊 Database Query Result\n\n{answer_text}\n\n*Source: Live PostgreSQL database*",
                "query_type": "SQL",
                "source": "database"
            }

        except Exception as e:
            print(f"[SQL ANALYTICS] DB query failed: {e}")
            # Fall through to Yahoo Finance

    # Step 2: Try Yahoo Finance for live data
    if yahoo_available:
        try:
            return _get_yahoo_finance_response(question)
        except Exception as e:
            print(f"[SQL ANALYTICS] Yahoo Finance error: {e}")
            # Fall through to sample data

    # Step 3: Final fallback to sample data (only when BOTH fail)
    return _get_sample_data_response(question)


def _format_currency(value):
    """Format large numbers as currency strings."""
    if value is None or value == 'N/A':
        return 'N/A'
    try:
        num = float(value)
        if num >= 1e12:
            return f"${num/1e12:.2f}T"
        if num >= 1e9:
            return f"${num/1e9:.2f}B"
        if num >= 1e6:
            return f"${num/1e6:.2f}M"
        return f"${num:,.2f}"
    except:
        return str(value)


def _get_yahoo_finance_response(question: str) -> Dict[str, Any]:
    """Get live data from Yahoo Finance - returns actual data, not boilerplate."""
    from app.core.yahoo_service import yahoo_service, get_company_info
    from datetime import datetime
    
    question_lower = question.lower()
    as_of = datetime.now().isoformat()
    
    # TOP N COMPANIES QUERY - Return actual ranked list
    if any(word in question_lower for word in ['top', 'largest', 'biggest', 'best', 'leading']):
        # Extract N from query (default 10)
        import re
        n_match = re.search(r'top\s*(\d+)', question_lower)
        n = int(n_match.group(1)) if n_match else 10
        
        companies = yahoo_service.get_top_companies_data()
        if companies:
            # Sort by market cap descending
            sorted_companies = sorted(
                [c for c in companies if c.get('market_cap')],
                key=lambda x: x.get('market_cap', 0),
                reverse=True
            )[:n]
            
            # Build concise ranked list
            ranked_list = []
            for i, c in enumerate(sorted_companies, 1):
                price = c.get('current_price')
                cap = c.get('market_cap', 0)
                ranked_list.append({
                    "rank": i,
                    "ticker": c.get('ticker'),
                    "name": c.get('name', '')[:30],
                    "price": f"${price:.2f}" if price else "N/A",
                    "market_cap": _format_currency(cap),
                    "sector": c.get('sector', 'N/A')[:20]
                })
            
            return {
                "answer": {
                    "top_companies": ranked_list,
                    "metric": "market_cap",
                    "count": len(ranked_list),
                    "as_of": as_of,
                    "sources": ["yahoo_finance_api"]
                },
                "query_type": "SQL",
                "source": "yahoo_finance"
            }
    
    # SPECIFIC COMPANY QUERY
    company_keywords = {
        'apple': 'AAPL', 'aapl': 'AAPL',
        'microsoft': 'MSFT', 'msft': 'MSFT',
        'google': 'GOOGL', 'googl': 'GOOGL', 'alphabet': 'GOOGL',
        'amazon': 'AMZN', 'amzn': 'AMZN',
        'tesla': 'TSLA', 'tsla': 'TSLA',
        'nvidia': 'NVDA', 'nvda': 'NVDA',
        'meta': 'META', 'facebook': 'META',
        'berkshire': 'BRK-B',
        'johnson': 'JNJ', 'jnj': 'JNJ',
        'walmart': 'WMT', 'wmt': 'WMT',
        'jpmorgan': 'JPM', 'jpm': 'JPM',
    }
    
    for keyword, ticker in company_keywords.items():
        if keyword in question_lower:
            info = get_company_info(ticker)
            if info and info.get('name'):
                return {
                    "answer": {
                        "ticker": ticker,
                        "name": info.get('name'),
                        "price": info.get('current_price'),
                        "market_cap": _format_currency(info.get('market_cap')),
                        "pe_ratio": round(info.get('pe_ratio', 0), 2) if info.get('pe_ratio') else None,
                        "eps": info.get('eps'),
                        "dividend_yield": f"{info.get('dividend_yield', 0)*100:.2f}%" if info.get('dividend_yield') else None,
                        "52w_high": info.get('52_week_high'),
                        "52w_low": info.get('52_week_low'),
                        "sector": info.get('sector'),
                        "industry": info.get('industry'),
                        "as_of": as_of,
                        "sources": ["yahoo_finance_api"]
                    },
                    "query_type": "SQL",
                    "source": "yahoo_finance"
                }
    
    # MARKET OVERVIEW (default)
    companies = yahoo_service.get_top_companies_data()
    if companies:
        top_5 = sorted(
            [c for c in companies if c.get('market_cap')],
            key=lambda x: x.get('market_cap', 0),
            reverse=True
        )[:5]
        
        overview = [
            {
                "ticker": c.get('ticker'),
                "price": f"${c.get('current_price', 0):.2f}" if c.get('current_price') else "N/A",
                "market_cap": _format_currency(c.get('market_cap'))
            }
            for c in top_5
        ]
        
        return {
            "answer": {
                "market_snapshot": overview,
                "count": len(companies),
                "as_of": as_of,
                "sources": ["yahoo_finance_api"]
            },
            "query_type": "SQL",
            "source": "yahoo_finance"
        }
    
    return _get_sample_data_response(question)


def _get_sample_data_response(question: str) -> Dict[str, Any]:
    """
    Fallback to sample data when BOTH database AND Yahoo Finance are unavailable.
    Returns structured JSON schema response.
    """
    from app.core.sp500_companies import (
        get_sp500_companies,
        get_top_companies_by_revenue,
        search_companies
    )
    from datetime import datetime
    
    question_lower = question.lower()
    as_of = datetime.now().isoformat()
    
    # Base fallback response structure
    def build_fallback_response(summary_items, source_note="sample_data"):
        return {
            "answer": {
                "summary": summary_items,
                "pros": [
                    "Sample data available for reference",
                    "S&P 500 company listings accessible"
                ],
                "cons": [
                    "Live data unavailable - prices not current",
                    "Database connection not configured",
                    "Yahoo Finance API not enabled or unreachable"
                ],
                "suggestions": [
                    "1. Set USE_YAHOO_DATA=true in .env for live prices",
                    "2. Install yfinance: pip install yfinance",
                    "3. Or configure DATABASE_URL for PostgreSQL"
                ],
                "as_of": as_of,
                "sources": [source_note]
            },
            "query_type": "SQL",
            "source": "sample_data"
        }

    if any(word in question_lower for word in ['top', 'largest', 'biggest', 'revenue', 'leader']):
        companies = get_top_companies_by_revenue(5)
        if companies:
            summary_items = [
                f"Live data unavailable; showing last snapshot from sample cache",
                *[f"{c['name']} ({c['ticker']}) - {c.get('sector', 'N/A')}" for c in companies[:3]]
            ]
            return build_fallback_response(summary_items)
    
    # Search for specific company
    search_keywords = ['apple', 'microsoft', 'google', 'amazon', 'tesla', 'nvidia', 'meta', 'berkshire']
    for keyword in search_keywords:
        if keyword in question_lower:
            company = search_companies(keyword, 1)
            if company:
                c = company[0]
                summary_items = [
                    f"Live data unavailable; showing cached info for {c['name']}",
                    f"Ticker: {c['ticker']} | Sector: {c.get('sector', 'N/A')}",
                    "Real-time pricing not available"
                ]
                return build_fallback_response(summary_items)
    
    # Default response
    companies = get_sp500_companies(10)
    if companies:
        tickers = ", ".join([c['ticker'] for c in companies[:5]])
        summary_items = [
            "Live data unavailable; showing sample S&P 500 data",
            f"Available tickers: {tickers}",
            "Ask about specific companies for cached details"
        ]
        return build_fallback_response(summary_items)
    
    return build_fallback_response([
        "No data available to answer",
        "Both live and cached data sources unavailable"
    ])


# ------------------------------------------------------
# Hybrid Chain (SQL + RAG)
# ------------------------------------------------------
def run_hybrid(question: str) -> Dict[str, Any]:
    sql_result = run_sql_analytics(question)
    doc_result = run_doc_rag(question)

    system = (
        f"{_FINANCIAL_ANALYST_SYSTEM_PROMPT}\n\n"
        "SPECIAL INSTRUCTION: You are combining numeric SQL data with document context. "
        "Synthesize both sources into a unified analysis. Highlight key metrics, trends, and qualitative context. "
        f"{_GUARDRAIL_DIRECTIVE}"
    )

    user = f"""USER QUESTION: {question}

SQL NUMERIC ANALYSIS:
{sql_result['answer']}

DOCUMENT/CONTEXT INSIGHTS:
{doc_result['answer']}

Provide an integrated, expert-level financial analysis following the mandatory structure:
1. Summary - Unified findings from both data sources
2. Context & Data Scope - What numeric and qualitative data was analyzed
3. Practical Insights - Combined actionable observations
4. Data-Grounded Suggestions - Next steps based on the integrated analysis"""

    messages = [
        {"role": "system", "content": system},
        {"role": "user",  "content": user},
    ]

    merged = _llm_invoke(messages)

    return {
        "answer": merged,
        "query_type": "HYBRID",
        "parts": {"sql": sql_result, "doc": doc_result},
    }


# ------------------------------------------------------
# LIVE DATA Chain (NEW - Priority Handler)
# ------------------------------------------------------
def run_live_data(question: str) -> Dict[str, Any]:
    """
    Fetch LIVE market data. Always runs FIRST for time-sensitive queries.
    Returns standardized response with timestamp and confidence.
    """
    from datetime import datetime
    
    try:
        from app.core.live_data_service import get_live_response, is_live_data_available
        
        if not is_live_data_available():
            return {
                "answer": {
                    "summary": "Live data service unavailable",
                    "live_price": None,
                    "trend": "Unknown",
                    "reasoning": ["Yahoo Finance service not configured or unreachable"],
                    "timestamp": datetime.now().isoformat(),
                    "confidence": "Low"
                },
                "query_type": "LIVE_DATA",
                "source": "error"
            }
        
        live_result = get_live_response(question)
        
        return {
            "answer": live_result,
            "query_type": "LIVE_DATA",
            "source": live_result.get("source", "yahoo_finance")
        }
        
    except Exception as e:
        return {
            "answer": {
                "summary": f"Error fetching live data: {str(e)[:100]}",
                "live_price": None,
                "trend": "Unknown",
                "reasoning": [str(e)],
                "timestamp": datetime.now().isoformat(),
                "confidence": "Low"
            },
            "query_type": "LIVE_DATA",
            "source": "error"
        }


def run_live_data_with_docs(question: str) -> Dict[str, Any]:
    """
    Combine LIVE market data with document context.
    Use for sentiment/trend queries that need both live data and analysis.
    """
    from datetime import datetime
    
    # Get live data first
    live_result = run_live_data(question)
    
    # Get document context
    doc_result = run_doc_rag(question)
    
    # Combine into unified response
    live_data = live_result.get("answer", {})
    
    return {
        "answer": {
            "summary": live_data.get("summary", "Market data with context"),
            "live_price": live_data.get("live_price"),
            "price_change": live_data.get("price_change"),
            "trend": live_data.get("trend", "Unknown"),
            "reasoning": live_data.get("reasoning", []) + ["Document context included below"],
            "timestamp": live_data.get("timestamp", datetime.now().isoformat()),
            "confidence": live_data.get("confidence", "Medium"),
            "document_context": doc_result.get("answer", "No document context available")
        },
        "query_type": "LIVE_DATA_DOC",
        "source": "yahoo_finance + documents"
    }


# ------------------------------------------------------
# Main Orchestrator for API (UPDATED with Live Data Priority)
# ------------------------------------------------------
def answer_financial_question(question: str) -> Dict[str, Any]:
    """
    Main entry point for all financial queries.
    
    ROUTING PRIORITY:
    1. LIVE_DATA - Real-time market queries
    2. LIVE_DATA_DOC - Live data + document context (sentiment/trend)
    3. SQL - Database analytics
    4. HYBRID - SQL + DOC
    5. DOC - Document-only RAG (fallback)
    """
    classification = classify_query(question)
    qtype = classification.get("query_type", "DOC")

    # LIVE_DATA: Time-sensitive queries get live data FIRST
    if qtype == "LIVE_DATA":
        result = run_live_data(question)
    
    # LIVE_DATA_DOC: Sentiment/trend queries need live data + context
    elif qtype == "LIVE_DATA_DOC":
        result = run_live_data_with_docs(question)
    
    # SQL: Database analytics (historical)
    elif qtype == "SQL":
        result = run_sql_analytics(question)
    
    # HYBRID: SQL + Document context
    elif qtype == "HYBRID":
        result = run_hybrid(question)
    
    # DOC: Document-only RAG (fallback)
    else:
        result = run_doc_rag(question)

    result["router"] = classification
    return result
