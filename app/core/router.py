# app/core/router.py
"""
LIVE DATA ROUTER - ALWAYS checks live data FIRST for market queries.

CRITICAL: This router FORCES live data fetch for any market-related query.
Document RAG is ONLY used for policy/regulation/explanation questions.
"""
from typing import Literal

QueryType = Literal["LIVE_DATA", "LIVE_DATA_DOC", "SQL", "DOC", "HYBRID"]


def classify_query(question: str) -> dict:
    """
    LIVE-DATA-FIRST Router.
    
    ANY query about stocks, companies, prices, market, trends → LIVE_DATA
    ONLY pure explanation/policy questions → DOC
    """
    q = question.lower()
    
    # ================================================================
    # DOCUMENT-ONLY KEYWORDS (Very narrow - only pure explanations)
    # These are the ONLY queries that skip live data
    # ================================================================
    doc_only_keywords = [
        "what is the definition",
        "explain the concept",
        "what does the term",
        "regulatory framework",
        "accounting standard",
        "sec regulation",
        "gaap rule"
    ]
    
    # Check if this is a pure doc query (VERY narrow)
    for kw in doc_only_keywords:
        if kw in q:
            return {
                "query_type": "DOC",
                "reason": "Pure definition/regulatory query - using documents.",
                "requires_live": False,
                "requires_docs": True
            }
    
    # ================================================================
    # EVERYTHING ELSE GETS LIVE DATA
    # This is the DEFAULT behavior now
    # ================================================================
    
    # Sentiment/trend queries get live + doc context
    sentiment_keywords = [
        "trend", "sentiment", "outlook", "should i",
        "bullish", "bearish", "momentum", "invest"
    ]
    
    for kw in sentiment_keywords:
        if kw in q:
            return {
                "query_type": "LIVE_DATA_DOC",
                "reason": "Sentiment query - fetching live data + context.",
                "requires_live": True,
                "requires_docs": True
            }
    
    # DEFAULT: ALL market queries get LIVE DATA
    # This includes: top, companies, stock, price, market, ticker, etc.
    return {
        "query_type": "LIVE_DATA",
        "reason": "Market query - fetching live data from Yahoo Finance.",
        "requires_live": True,
        "requires_docs": False
    }
