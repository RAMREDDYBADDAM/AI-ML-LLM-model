# app/core/router.py
from typing import Literal

QueryType = Literal["DOC", "SQL", "HYBRID"]


def classify_query(question: str) -> dict:
    """
    Lightweight rule-based router used when LangChain prompt parsers are
    unavailable. This is intentionally simple and intended for local testing.
    """
    q = question.lower()
    sql_keywords = [
        "sum",
        "total",
        "average",
        "avg",
        "revenue",
        "profit",
        "loss",
        "growth",
        "percent",
        "percentage",
        "q1",
        "q2",
        "q3",
        "q4",
        "quarter",
        "year",
        "compare",
        "how many",
        "count",
        "what is the",
    ]

    doc_keywords = ["policy", "explain", "explain why", "guidance", "regulation", "summary", "what is"]

    sql_score = sum(1 for k in sql_keywords if k in q)
    doc_score = sum(1 for k in doc_keywords if k in q)

    if sql_score >= 2 and doc_score >= 1:
        qtype = "HYBRID"
        reason = "Detected both numeric/SQL keywords and document/explanation keywords."
    elif sql_score >= 1 and sql_score >= doc_score:
        qtype = "SQL"
        reason = "Detected numeric/aggregation keywords; route to SQL analytics."
    else:
        qtype = "DOC"
        reason = "Defaulting to document retrieval and RAG."

    return {"query_type": qtype, "reason": reason}
