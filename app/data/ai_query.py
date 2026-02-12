from langchain_groq import ChatGroq
from app.core.schema import DB_SCHEMA
from app.utils.sql_executor import execute_sql
from app.utils.sql_validator import validate_sql, sanitize_sql
import os
from core.model import get_llm

llm = get_llm()

async def generate_sql(user_query: str) -> str:
    prompt = f"""
    You are a PostgreSQL expert.

    Database schema:
    {DB_SCHEMA}

    User question:
    {user_query}

    Rules:
    - ONLY SELECT queries
    - NO INSERT, UPDATE, DELETE
    - Use correct column names
    - Use LIMIT where applicable
    - Output ONLY SQL
    """
    result = await llm.ainvoke(prompt)
    return result.content.strip()


async def run_text_to_sql(user_query: str):
    raw_sql = await generate_sql(user_query)
    sql = sanitize_sql(raw_sql)

    
    await validate_sql(sql)
    rows = await execute_sql(sql)

    explanation = await llm.ainvoke(f"""
    User question:
    {user_query}

    SQL used:
    {sql}

    Query result:
    {rows}

    Explain clearly to a user.
    """)
    
    explanation = explanation.content.strip()

    return {
        "sql": sql,
        "rows": rows,
        "answer": explanation
    }
