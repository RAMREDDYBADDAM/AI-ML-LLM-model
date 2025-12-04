# app/core/sql_tools.py
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from app.config import settings
from app.core.llm import get_llm


def get_sql_db():
    if not settings.database_url:
        raise ValueError("DATABASE_URL is not configured")
    db = SQLDatabase.from_uri(settings.database_url)
    return db


def get_sql_agent():
    """
    Returns a LangChain SQL agent capable of generating and running SQL queries.
    """
    llm = get_llm()
    db = get_sql_db()
    agent = create_sql_agent(
        llm=llm,
        db=db,
        verbose=True,
        handle_parsing_errors=True,
    )
    return agent
