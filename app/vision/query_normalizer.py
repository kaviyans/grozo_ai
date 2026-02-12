from langchain_groq import ChatGroq
from app.core.model import get_llm

llm = get_llm()

# ---------- NORMALIZE THE ANALYZED CONTENT ----------
async def normalize_product_query(vision_description: str) -> str:
    """
    Convert verbose vision output into a short search query.
    """
    prompt = f"""
    Extract a SHORT product search query (max 20 - 30 words).

    Include if possible:
    - Brand
    - Product name
    - Category

    Description:
    {vision_description}

    Return ONLY the query.
    """
    query = (await llm.ainvoke(prompt)).content.strip()

    return query[:350]
