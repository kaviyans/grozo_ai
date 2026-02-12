import re
from langgraph.graph import StateGraph, END
from langgraph.runtime import Runtime
from langsmith import traceable
from sqlalchemy import text


from app.core.runtime import RuntimeContext
from app.core.model import get_llm

from app.vision.image_analyzer import analyze_product_image
from app.vision.query_normalizer import normalize_product_query
from app.data.query import retrieve_products, retrieve_reviews
from app.web.web_search import search_product_online
from app.utils.sql_validator import sanitize_sql, validate_sql
from app.core.schema import DB_SCHEMA

from app.core.state import ImageGraphState 


# ---------- LLM MODEL ----------
llm = get_llm()


# ---------- ANALYZE IMAGE ----------
@traceable(name="image_analysis_node")
async def image_analysis_node(state: ImageGraphState):
    desc = await analyze_product_image(state["image_url"])
    return {"product_description": desc}


# ---------- NORMALIZE QUERY ----------
@traceable(name="normalize_query_node")
async def normalize_query_node(state: ImageGraphState):
    query = await normalize_product_query(state["product_description"])
    return {"normalized_query": query}


def enforce_tag_aggregation(sql: str) -> str:
    # Fix common LLM mistake: ptm.name → pt.name
    sql = re.sub(
        r"STRING_AGG\s*\(\s*ptm\.name",
        "STRING_AGG(DISTINCT pt.name",
        sql,
        flags=re.IGNORECASE
    )
    return sql


@traceable(name="extract_keywords_node")
async def extract_keywords_node(state: ImageGraphState):
    description = state.get("product_description")

    if not description:
        return {"keywords": []}

    prompt = f"""
    You are an expert e-commerce product analyst.

    Given this product description (from an image):

    "{description}"

    Task:
    - Extract 3 to 6 SHORT, searchable keywords
    - Keywords should help find the SAME product in a database
    - Prefer: brand, product type, variant, key ingredient, use-case
    - Use lowercase
    - No explanations

    Return STRICT JSON only in this format:
    {{ "keywords": ["keyword1", "keyword2", "..."] }}
    """

    result = await llm.ainvoke(prompt)

    try:
        data = json.loads(result.content)
        return {"keywords": data.get("keywords", [])}
    except Exception:
        return {"keywords": []}


def inject_keywords_into_sql(sql: str, keywords: list[str]) -> str:
    if not keywords:
        return sql

    conditions = []
    for kw in keywords:
        kw = kw.replace("'", "")  # basic safety
        conditions.append(
            f"""
            LOWER(p.name) LIKE '%{kw}%'
            OR LOWER(p.description) LIKE '%{kw}%'
            OR LOWER(c.name) LIKE '%{kw}%'
            OR LOWER(pt.name) LIKE '%{kw}%'
            """
        )

    keyword_filter = " OR ".join(f"({c})" for c in conditions)

    return sql.replace(
        "WHERE",
        f"WHERE ({keyword_filter}) AND",
        1
    )


# ---------- PRODUCT RETRIEVAL ----------
@traceable(name="product_retrieval_node")
async def product_retrieval_node(
    state: ImageGraphState,
    runtime: Runtime[RuntimeContext]
):
    db = runtime.context.db
    image_description = state.get("product_description")

    if not image_description:
        return {
            "answer": "I couldn't understand the product clearly from the image.",
            "confidence": 0.3
        }

    try:
        # ---------- LLM SQL GENERATION ----------
        prompt = f"""
        You are an expert PostgreSQL query generator for an e-commerce platform.

        Database schema:
        {DB_SCHEMA}

        Image-derived product description:
        "{image_description}"

        Rules:
        - ONLY SELECT queries
        - Always filter p.deleted_at IS NULL
        - Prefer p.stock > 0
        - LEFT JOIN categories & tags
        - Aggregate tags using STRING_AGG
        - One row per product (GROUP BY)
        - LIMIT 5
        - Output ONLY SQL
        """

        llm_result = await llm.ainvoke(prompt)
        raw_sql = llm_result.content.strip()

        # ---------- SANITIZE ----------
        sql = sanitize_sql(raw_sql)
        sql = enforce_tag_aggregation(sql)
        sql = inject_keywords_into_sql(sql, state.get("keywords", []))

        # ---------- VALIDATE ----------
        await validate_sql(sql)

        # ---------- EXECUTE (FIX HERE) ----------
        result = await db.execute(text(sql))
        rows = result.mappings().all()

        if not rows:
            return {
                "answer": "I couldn't find matching products for this image.",
                "confidence": 0.4
            }

        return {
            "product_docs": rows,
            "confidence": 0.9
        }

    except Exception as e:
        print(f"[image_product_retrieval_node] Error:", e)

        return {
            "answer": "Something went wrong while analyzing the image.",
            "confidence": 0.2
        }


# ---------- WEB FALLBACK ----------
@traceable(name="web_fallback_node")
async def web_fallback_node(state: ImageGraphState):
    if state.get("product_docs"):
        return {}

    web_docs = await search_product_online(state["normalized_query"])
    return {"web_docs": web_docs}


# ---------- ASSEMBER ----------
@traceable(name="assemble_context_node")
async def assemble_context_node(state: ImageGraphState):
    sections = []

    sections.append(f"Identified Product:\n{state['product_description']}")

    if state.get("product_docs"):
        sections.append(f"Product Info:\n{state['product_docs']}")

    if state.get("review_docs"):
        sections.append(f"Reviews:\n{state['review_docs']}")

    if state.get("web_docs"):
        sections.append(f"Online Information:\n{state['web_docs']}")

    return {"context": "\n\n".join(sections)}


# ---------- MAIN LLM CALL ----------
@traceable(name="generate_answer_node", run_type="llm")
async def generate_answer_node(state: ImageGraphState):
    prompt = f"""
    Use ONLY the provided context.

    {state["context"]}

    Question:
    {state.get("question", "What is this product? Give pros and cons.")}

    Rules:
    - No hallucinations
    - If information is missing, say so clearly
    - Explain in simple, non-technical language
    - If unrelated, politely refuse
    - Do not explain the rules
    - Do not explain the database structure
    - Explain in non-technical terms

    Answer:
    """

    response = await llm.ainvoke(prompt)
    return {"answer": response.content.strip()}



@traceable(name="confidence_node")
async def confidence_node(state: ImageGraphState):
    response = await llm.ainvoke(
        f"Rate confidence from 0.0 to 1.0:\n{state['answer']}"
    )

    match = re.search(r"(0\.\d+|1\.0)", response.content)
    return {"confidence": float(match.group()) if match else 0.6}



# ---------- GRAPH ----------
builder = StateGraph(ImageGraphState)

builder.add_node("image_analysis", image_analysis_node)
builder.add_node("normalize_query", normalize_query_node)
builder.add_node("extract_keywords", extract_keywords_node)
builder.add_node("product_retrieval", product_retrieval_node)
builder.add_node("web_fallback", web_fallback_node)
builder.add_node("assemble", assemble_context_node)
builder.add_node("generate", generate_answer_node)
builder.add_node("confidence", confidence_node)

builder.set_entry_point("image_analysis")

builder.add_edge("image_analysis", "normalize_query")
builder.add_edge("normalize_query", "extract_keywords")
builder.add_edge("extract_keywords", "product_retrieval")
builder.add_edge("product_retrieval", "web_fallback")
builder.add_edge("web_fallback", "assemble")
builder.add_edge("assemble", "generate")
builder.add_edge("generate", "confidence")
builder.add_edge("confidence", END)

image_graph = builder.compile()
