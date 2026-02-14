import re
import json
from sqlalchemy import text
from langgraph.graph import StateGraph, END
from langgraph.runtime import Runtime
from langsmith import traceable

from app.core.runtime import RuntimeContext
from app.core.model import get_llm
from app.core.schema import DB_SCHEMA
from app.core.state import ImageGraphState
from app.core.logging_config import get_graph_logger, log_node_entry, log_node_exit, log_node_error

from app.vision.image_analyzer import analyze_product_image
from app.vision.query_normalizer import normalize_product_query
from app.web.web_search import search_product_online
from app.utils.sql_validator import sanitize_sql, validate_sql


# ---------- LOGGER ----------
logger = get_graph_logger("image_graph")


# ---------- LLM ----------
llm = get_llm()


# ---------- IMAGE ANALYSIS ----------
@traceable(name="image_analysis_node")
async def image_analysis_node(state: ImageGraphState):
    log_node_entry(logger, "image_analysis_node", list(state.keys()))
    try:
        desc = await analyze_product_image(state["image_url"])
        logger.info(f"[image_analysis_node] Product description extracted")
        log_node_exit(logger, "image_analysis_node", ["product_description"])
        return {"product_description": desc}
    except Exception as e:
        log_node_error(logger, "image_analysis_node", e)
        return {"product_description": "Unable to analyze image"}


# ---------- QUERY NORMALIZATION ----------
@traceable(name="normalize_query_node")
async def normalize_query_node(state: ImageGraphState):
    log_node_entry(logger, "normalize_query_node", list(state.keys()))
    try:
        query = await normalize_product_query(state["product_description"])
        logger.info(f"[normalize_query_node] Query normalized")
        log_node_exit(logger, "normalize_query_node", ["normalized_query"])
        return {"normalized_query": query}
    except Exception as e:
        log_node_error(logger, "normalize_query_node", e)
        return {"normalized_query": state.get("product_description", "")}


# ---------- SQL HARDENER ----------
def enforce_tag_aggregation(sql: str) -> str:
    """
    Force safe STRING_AGG usage regardless of LLM mistakes.
    Prevents PostgreSQL datatype errors.
    """

    sql = re.sub(
        r"STRING_AGG\s*\([^)]+\)\s+AS\s+tags",
        "STRING_AGG(DISTINCT pt.name, ', ') AS tags",
        sql,
        flags=re.IGNORECASE
    )

    return sql


# ---------- KEYWORD EXTRACTION ----------
@traceable(name="extract_keywords_node")
async def extract_keywords_node(state: ImageGraphState):
    log_node_entry(logger, "extract_keywords_node", list(state.keys()))
    try:
        description = state.get("product_description")

        if not description:
            logger.info("[extract_keywords_node] No description, returning empty keywords")
            log_node_exit(logger, "extract_keywords_node", ["keywords"])
            return {"keywords": []}

        prompt = f"""
You are an expert e-commerce product classifier.

Product description from image:
"{description}"

Task:
- Extract 3 to 6 highly searchable keywords
- Use only lowercase words
- Focus on product type, brand, variant, attributes
- No explanations

Return STRICT JSON:

{{ "keywords": ["keyword1", "keyword2"] }}
"""

        result = await llm.ainvoke(prompt)

        try:
            data = json.loads(result.content)
            keywords = data.get("keywords", [])
            logger.info(f"[extract_keywords_node] Extracted {len(keywords)} keywords")
            log_node_exit(logger, "extract_keywords_node", ["keywords"])
            return {"keywords": keywords}
        except Exception:
            logger.warning("[extract_keywords_node] Failed to parse keywords JSON")
            log_node_exit(logger, "extract_keywords_node", ["keywords"])
            return {"keywords": []}
    except Exception as e:
        log_node_error(logger, "extract_keywords_node", e)
        return {"keywords": []}


# ---------- SQL KEYWORD INJECTION ----------
def inject_keywords_into_sql(sql: str, keywords: list[str]) -> str:

    if not keywords:
        return sql

    safe_conditions = []

    for kw in keywords:
        kw = re.sub(r"[^a-zA-Z0-9\s]", "", kw)

        safe_conditions.append(f"""
            LOWER(p.name) LIKE '%{kw.lower()}%'
            OR LOWER(p.description) LIKE '%{kw.lower()}%'
            OR LOWER(c.name) LIKE '%{kw.lower()}%'
            OR LOWER(pt.name) LIKE '%{kw.lower()}%'
        """)

    keyword_filter = " OR ".join(f"({c})" for c in safe_conditions)

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
    log_node_entry(logger, "product_retrieval_node", list(state.keys()))
    db = runtime.context.db
    description = state.get("product_description")

    if not description:
        logger.warning("[product_retrieval_node] No product description")
        log_node_exit(logger, "product_retrieval_node", ["answer", "confidence"])
        return {
            "answer": "I couldn't understand the product from the image.",
            "confidence": 0.3
        }

    try:

        prompt = f"""
    You are a PostgreSQL expert for an e-commerce platform.

    Database schema:
    {DB_SCHEMA}

    User intent:
    Find products matching this description:

    "{description}"

    STRICT RULES:
    - Output ONLY SQL
    - SELECT queries only
    - Always filter: p.deleted_at IS NULL
    - Prefer: p.stock > 0
    - LEFT JOIN categories & tags
    - Aggregate tags using STRING_AGG(DISTINCT pt.name, ', ')
    - One row per product
    - LIMIT 5
    """

        llm_result = await llm.ainvoke(prompt)
        raw_sql = llm_result.content.strip()

        sql = sanitize_sql(raw_sql)
        sql = enforce_tag_aggregation(sql)
        sql = inject_keywords_into_sql(sql, state.get("keywords", []))

        await validate_sql(sql)

        result = await db.execute(text(sql))
        rows = result.mappings().all()

        if not rows:
            logger.info("[product_retrieval_node] No matching products found")
            log_node_exit(logger, "product_retrieval_node", ["answer", "confidence"])
            return {
                "answer": "No matching products found.",
                "confidence": 0.4
            }
            
        logger.info(f"[product_retrieval_node] Found {len(rows)} products")
        log_node_exit(logger, "product_retrieval_node", ["product_docs", "num_products"])

        return {
            "product_docs": rows,
            "num_products": len(rows)
        }

    except Exception as e:
        log_node_error(logger, "product_retrieval_node", e)
        return {
            "answer": "Something went wrong while searching for products.",
            "confidence": 0.2
        }


# ---------- WEB FALLBACK ----------
@traceable(name="web_fallback_node")
async def web_fallback_node(state: ImageGraphState):
    log_node_entry(logger, "web_fallback_node", list(state.keys()))
    try:
        if state.get("product_docs"):
            logger.info("[web_fallback_node] Products found, skipping web search")
            log_node_exit(logger, "web_fallback_node", [])
            return {}

        web_docs = await search_product_online(state["normalized_query"])
        logger.info(f"[web_fallback_node] Web search completed")
        log_node_exit(logger, "web_fallback_node", ["web_docs"])
        return {"web_docs": web_docs}
    except Exception as e:
        log_node_error(logger, "web_fallback_node", e)
        return {"web_docs": []}


# ---------- CONTEXT ASSEMBLY ----------
@traceable(name="assemble_context_node")
async def assemble_context_node(state: ImageGraphState):
    log_node_entry(logger, "assemble_context_node", list(state.keys()))
    try:
        sections = []

        sections.append(f"Identified Product:\n{state.get('product_description')}")

        if state.get("product_docs"):
            sections.append(f"Database Products:\n{state['product_docs']}")

        if state.get("web_docs"):
            sections.append(f"Web Results:\n{state['web_docs']}")

        logger.info(f"[assemble_context_node] Assembled {len(sections)} sections")
        log_node_exit(logger, "assemble_context_node", ["context"])
        return {"context": "\n\n".join(sections)}
    except Exception as e:
        log_node_error(logger, "assemble_context_node", e)
        return {"context": "Error assembling context"}


# ---------- ANSWER GENERATION (PRODUCTION PROMPT) ----------
@traceable(name="generate_answer_node", run_type="llm")
async def generate_answer_node(state: ImageGraphState):
    log_node_entry(logger, "generate_answer_node", list(state.keys()))
    try:
        context = state.get("context")
        resolved_query = state.get("question", "What is this product?")
        num_products = state.get("num_products", 0)
        memory = state.get("memory")

        prompt = f"""
    Conversation History:
    {memory if memory else "(new conversation)"}

    Verified Context:
    {context}

    User Question:
    {resolved_query}

    Number of Products Found: {num_products}

    Instructions:

    MULTI-PRODUCT BEHAVIOR:
    - If multiple products exist → compare & differentiate
    - Highlight price, ratings, usefulness
    - Recommend when clear winner exists
    - If ambiguous → ask preference

    STYLE:
    - Use bullet points
    - Use friendly e-commerce tone
    - Keep concise & helpful
    - No hallucinations
    - No database/internal discussion
    - No medical claims
    - If info missing → say clearly

    If edible product:
    Include approx nutrition per 100g (only if relevant)

    Answer:
    """

        response = await llm.ainvoke(prompt)
        logger.info("[generate_answer_node] Answer generated")
        log_node_exit(logger, "generate_answer_node", ["answer"])
        return {"answer": response.content.strip()}
    except Exception as e:
        log_node_error(logger, "generate_answer_node", e)
        return {"answer": "I'm sorry, I couldn't generate a response. Please try again."}


# ---------- CONFIDENCE ----------
@traceable(name="confidence_node")
async def confidence_node(state: ImageGraphState):
    log_node_entry(logger, "confidence_node", list(state.keys()))
    try:
        response = await llm.ainvoke(
            f"Rate confidence from 0.0 to 1.0 for this answer:\n{state['answer']}"
        )

        match = re.search(r"(0\.\d+|1\.0)", response.content)
        confidence = float(match.group()) if match else 0.6
        
        logger.info(f"[confidence_node] Confidence: {confidence}")
        log_node_exit(logger, "confidence_node", ["confidence", "products"])
        return {"confidence": confidence, "products": state.get("product_docs", [])}
    except Exception as e:
        log_node_error(logger, "confidence_node", e)
        return {"confidence": 0.5, "products": state.get("product_docs", [])}


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
