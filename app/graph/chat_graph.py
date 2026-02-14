import re
import asyncio
from langgraph.graph import StateGraph, END
from langgraph.runtime import Runtime
from langchain_core.runnables import RunnableConfig
from langsmith import traceable

from app.core.state import GraphState
from app.core.runtime import RuntimeContext
from app.core.model import get_llm, get_vector_db
from app.core.logging_config import (
    get_graph_logger,
    log_node_entry,
    log_node_exit,
    log_node_error
)

from app.core.schema import DB_SCHEMA

from app.utils.sql_executor import execute_sql
from app.utils.sql_validator import validate_sql, sanitize_sql
from app.core.schema import DB_SCHEMA

# ---------- LOGGER ----------
logger = get_graph_logger("chat_graph")

from app.utils.helper import (
    get_product_details,
    get_product_reviews,
    get_product_ratings,
    get_faqs,
    get_policy_by_type,
    get_available_categories
)

from app.memory.chat_memory import get_history, save_message
from app.memory.short_term import (
    add_stm,
    get_stm,
    set_session_context,
    get_session_context,
    touch_session
)



# ---------- LLM MODEL ----------
llm = get_llm()


# ---------- CHROMA DB ----------
vector_store = get_vector_db()


# ---------- ASYNC VECTOR SEARCH ----------
async def async_vector_search(query: str, k: int = 4):
    """Run vector search in a thread pool to make it async."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: vector_store.similarity_search(query, k=k))


# ---------- INTENT DETECTION ----------
@traceable(name="intent_detection")
async def detect_intent(state: GraphState):
    log_node_entry(logger, "detect_intent", list(state.keys()))
    try:
        prompt = f"""
    You are an intent classification system for an e-commerce assistant.

    Your task is to classify the user's query into exactly ONE of the following intents:

    1. product_info
    - Queries asking about a specific product
    - Price, features, specifications, availability, reviews, ratings
    - Buying intent (e.g., "buy", "price", "best", "recommend")

    2. policy_info
    - Queries about company policies
    - Returns, refunds, shipping, warranty, cancellation, privacy, terms
    - Queries about problems, complaints, or support
    - Damaged product, late delivery, refund issues, login/payment problems

    3. list_products
    - Queries asking to list or browse products
    - "list all products"
    - "show me products"
    - "what products are available"

    4. comparison
    - Queries comparing two or more products
    - "compare iphone vs samsung"
    - "which is better"

    4. normal_query
    - General questions not related to products, policies, or support
    - Greetings, small talk, unrelated topics
    - If the question is vague or ambiguous without clear product/policy intent
    - Follow-up questions that need context resolution
    
    5. Blocked query
    - If the user ask about illegal, harmful, or inappropriate content
    - If the user ask about security, privacy, or personal data
    - If the user ask about db internal, server, or application details
    - If the user query is unrelated to the e-commerce domain
    - If the user query contains hate speech, discrimination, or offensive language


    Classification Rules (IMPORTANT):

    - If a product name is mentioned → product_info
    - If buying or evaluation intent is present → product_info
    - If two or more products are mentioned → comparison
    - If the query mentions problems, issues, or complaints → issue_support
    - If the query is about rules, guarantees, or company procedures → policy_info
    - If none of the above apply → normal_query

    User Query:
    \"\"\"{state["query"]}\"\"\"

    Output Format:
    Return ONLY ONE label from the list below.
    Do NOT add explanations, punctuation, or extra text.

    Valid labels:
    product_info
    policy_info
    comparison
    list_products
    normal_query
    blocked_query
    """
        response = await llm.ainvoke(prompt)
        intent = response.content.strip().lower()
        logger.info(f"[detect_intent] Intent detected: {intent}")
        log_node_exit(logger, "detect_intent", ["intent"], f"intent={intent}")
        return {"intent": intent}
    except Exception as e:
        log_node_error(logger, "detect_intent", e)
        return {"intent": "normal_query"}


# ---------- NORMALIZE QUERY ----------
@traceable(name="normalize_query")
async def normalize_query_llm(query: str) -> str:
    try:
        logger.debug(f"[normalize_query_llm] Normalizing: {query[:50]}...")
        prompt = f"""
        Extract the core product name from the user query.
        Remove intent phrases.
        Return ONLY product keywords.

        Query: {query}
        """
        response = await llm.ainvoke(prompt)
        normalized = response.content.strip().lower()
        logger.debug(f"[normalize_query_llm] Result: {normalized}")
        return normalized
    except Exception as e:
        log_node_error(logger, "normalize_query_llm", e)
        return query.lower()


# ---------- STRUCTURED DB CALLS ----------
@traceable(name="product_db_fetch")
async def product_db_node(
    state: GraphState,
    runtime: Runtime[RuntimeContext]
):
    log_node_entry(logger, "product_db_node", list(state.keys()))
    try:
        query = await normalize_query_llm(state["query"])

        products = await get_product_details(query, runtime)  
        reviews = await get_product_reviews(query, runtime)
        ratings = await get_product_ratings(query, runtime)
        faqs = await get_faqs()

        logger.info(f"[product_db_node] Found {len(products)} products")
        log_node_exit(logger, "product_db_node", ["products", "reviews", "ratings", "faqs"])
        return {
            "products": products, 
            "reviews": reviews,
            "ratings": ratings,
            "faqs": faqs
        }
    except Exception as e:
        log_node_error(logger, "product_db_node", e)
        return {"products": [], "reviews": [], "ratings": None, "faqs": []}


# ---------- MONGODB CALLS ----------
@traceable(name="policy_db_fetch")
async def policy_db_node(state: GraphState):
    log_node_entry(logger, "policy_db_node", list(state.keys()))
    try:
        query = await normalize_query_llm(state["query"])
        
        data = {"policy": await get_policy_by_type(query)}
        logger.info(f"[policy_db_node] Policy found: {bool(data.get('policy'))}")
        log_node_exit(logger, "policy_db_node", ["policy"])
        return data
    except Exception as e:
        log_node_error(logger, "policy_db_node", e)
        return {"policy": None}


# ---------- RAG RETRIEVER ----------
@traceable(name="rag_retriever")
async def rag_node(state: GraphState):
    log_node_entry(logger, "rag_node", list(state.keys()))
    try:
        query = state["query"]

        docs = await async_vector_search(query, k=4)
        rag_text = "\n".join(d.page_content for d in docs)
        
        logger.info(f"[rag_node] Retrieved {len(docs)} documents")
        log_node_exit(logger, "rag_node", ["rag_context"])
        return {"rag_context": rag_text}
    except Exception as e:
        log_node_error(logger, "rag_node", e)
        return {"rag_context": ""}


# ---------- ASSEMBLER ----------
@traceable(name="context_assembler")
async def assemble_context(state: GraphState):
    log_node_entry(logger, "assemble_context", list(state.keys()))
    try:
        sections = []
        products = state.get("products", [])
        if products:
            product_lines = ["PRODUCT DETAILS:"]
            for idx, product in enumerate(products, start=1):
                product_lines.append(f"\n--- Product {idx} ---")
                product_lines.append(f"  Name: {product.get('name', 'N/A')}")
                product_lines.append(f"  Price: ${product.get('selling_price', 'N/A')}")
                if product.get('price') and product.get('price') != product.get('selling_price'):
                    product_lines.append(f"  Original Price: ${product.get('price')}")
                product_lines.append(f"  Rating: {product.get('average_rating', 'No ratings')} ({product.get('rating_count', 0)} reviews)")
                product_lines.append(f"  Category: {product.get('category_name', 'N/A')}")
                if product.get('tags'):
                    product_lines.append(f"  Tags: {product.get('tags')}")
                if product.get('description'):
                    desc = product.get('description', '')[:200] 
                    product_lines.append(f"  Description: {desc}...")
            sections.append("\n".join(product_lines))

        if state.get("ratings"):
            sections.append(f"RATINGS:\n{state['ratings']}")

        if state.get("reviews"):
            sections.append(f"REVIEWS:\n{state['reviews']}")

        if state.get("faqs"):
            sections.append(f"FAQS:\n{state['faqs']}")

        if state.get("policy"):
            sections.append(f"POLICY:\n{state['policy']}")

        if state.get("rag_context"):
            sections.append(f"DOCUMENT CONTEXT:\n{state['rag_context']}")

        logger.info(f"[assemble_context] Assembled {len(sections)} sections")
        log_node_exit(logger, "assemble_context", ["context", "products"])
        return {"context": "\n\n".join(sections), "products": products}
    except Exception as e:
        log_node_error(logger, "assemble_context", e)
        return {"context": "", "products": []}


# ---------- LIST PRODUCTS NODE (PREDEFINED SQL - NO LLM) ----------
@traceable(name="list_products_node")
async def list_products_node(state: GraphState):
    """
    Product listing with human-intent understanding.
    - Try LLM-generated SQL first (emotion-aware)
    - Validate & execute safely
    - Fallback to static list if anything fails
    """
    log_node_entry(logger, "list_products_node", list(state.keys()))
    user_query = state.get("query")
    categories = await execute_sql("SELECT DISTINCT name FROM categories")
    category_names = [row["name"] for row in categories]
    categories_str = ", ".join(category_names)


    try:
        # ---------------- LLM SQL GENERATION ----------------
        prompt = f"""
        You are an expert PostgreSQL query generator for an e-commerce platform.
        
        Below is the db schema:
        {DB_SCHEMA}

        Available categories currently in the database:
        {categories_str}

        User input:
        "{user_query}"

        Interpret human intent & emotion.

        STRICT RULES:

        - You MUST ONLY use categories from the list above
        - NEVER invent categories or generic words like "food", "items", etc.
        - Map intent to the MOST RELEVANT categories from the list

        Examples:

        - hungry → choose edible-related categories from the list
        - thirsty → choose drink-related categories from the list
        - cleaning / clean myself → choose hygiene or cleaning categories
        - sleepy / tired → choose comfort / energy related categories

        Query Requirements:

        - ONLY SELECT queries
        - Always filter p.deleted_at IS NULL
        - Prefer p.stock > 0
        - Use LEFT JOINs for categories & tags
        - Aggregate tags using STRING_AGG
        - Ensure one row per product (GROUP BY)
        - Match AGAINST:
            * p.name
            * p.description
            * c.name
            * ptm.name

        - Always include LIMIT 20
        - Output ONLY SQL
        """

        result = await llm.ainvoke(prompt)
        raw_sql = result.content.strip()

        sql = sanitize_sql(raw_sql)

        try:
            await validate_sql(sql)
        except ValueError as ve:
            logger.warning(f"[list_products_node] SQL validation failed: {ve}")
            raise ve

        rows = await execute_sql(sql)

        if not rows:
            return {
                "answer": "I couldn’t find matching products. Want to try Snacks or Instant Foods?",
                "confidence": 0.6
            }


        # ---------------- FORMAT RESPONSE ----------------
        product_list = []
        for row in rows:
            price = row.get("selling_price") or row.get("price")
            rating = row.get("average_rating") or "No ratings yet"

            product_list.append(
                f"- {row['name']}: ₹{price} (Rating: {rating})"
            )

        logger.info(f"[list_products_node] Listed {len(rows)} products")
        log_node_exit(logger, "list_products_node", ["products", "confidence"])
        return {
            "products": rows,
            "confidence": 0.9
        }

    # ---------------- FALLBACK ----------------
    except Exception as e:
        log_node_error(logger, "list_products_node", e)
        return {
            "answer": "I might not have understood that perfectly 😅",
            "confidence": 0.4
        }


# ---------- COMPARISON NODE (TEXT-TO-SQL WITH STRICT VALIDATION) ----------
@traceable(name="comparison_node")
async def comparison_node(state: GraphState):
    """
    For product comparisons:
    - Generate SQL using LLM
    - Sanitize and validate SQL (SELECT-only)
    - Execute safely
    - Use LLM only for explanation
    """
    log_node_entry(logger, "comparison_node", list(state.keys()))
    query = state["query"]
    
    try:
        prompt = f"""
        You are a PostgreSQL expert for product comparison queries.

        Database schema:
        {DB_SCHEMA}

        User question:
        {query}

        Rules:
        - ONLY SELECT queries
        - NO INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE
        - Use correct column names from schema
        - Always use LIMIT (max 20)
        - Output ONLY the SQL query, no explanations
        - NO markdown code fences
        - Compare products by fetching relevant data for each
        
        If the product is food/edible, include approximate nutrition per 100g and compare base on this:
        - Calories
        - Protein  
        - Carbs
        - Key nutrients
        
        Use bullet points and emojis for readability.
        Keep each product explanation concise (3-4 lines max).
        NEVER make medical claims. Use phrases like "may help" or "good source of".
        Format in clean markdown.
        """
        
        result = await llm.ainvoke(prompt)
        raw_sql = result.content.strip()
        
        sql = sanitize_sql(raw_sql)
        
        try:
            await validate_sql(sql)
        except ValueError as ve:
            logger.warning(f"[comparison_node] SQL validation failed: {ve}")
            sql = """
                SELECT 
                    id, name, selling_price, description,
                    CASE 
                        WHEN rating_count > 0 
                        THEN ROUND((total_rating / NULLIF(rating_count, 0))::numeric, 1)
                        ELSE NULL
                    END AS average_rating,
                    rating_count
                FROM products
                WHERE deleted_at IS NULL
                ORDER BY sold_count DESC
                LIMIT 10
            """
        
        rows = await execute_sql(sql)
        
        if not rows:
            return {
                "answer": "I couldn't find the products you're looking for. Please check the product names and try again.",
                "confidence": 0.5
            }
        
        explanation_prompt = f"""
        User question:
        {query}

        Product data retrieved:
        {rows}

        Provide a clear, helpful comparison of these products.
        Focus on: price, ratings, features, and value.
        Be concise and user-friendly.
        Do NOT mention SQL or database details.
        
        If the product is food/edible, include approximate nutrition per 100g and compare base on this:
        - Calories
        - Protein  
        - Carbs
        - Key nutrients
        
        Use bullet points and emojis for readability.
        Keep each product explanation concise (3-4 lines max).
        NEVER make medical claims. Use phrases like "may help" or "good source of".
        Format in clean markdown.
        """
        
        explanation = await llm.ainvoke(explanation_prompt)
        
        logger.info(f"[comparison_node] Comparison generated successfully")
        log_node_exit(logger, "comparison_node", ["answer", "confidence"])
        return {
            "answer": explanation.content.strip(),
            "confidence": 0.8
        }
        
    except Exception as e:
        log_node_error(logger, "comparison_node", e)
        return {
            "answer": "I'm sorry, I couldn't complete the comparison. Please try rephrasing your question.",
            "confidence": 0.0
        }


# ---------- CONTEXT RESOLUTION (FOLLOW-UP QUERIES) ----------
@traceable(name="resolve_query_context")
async def resolve_query_context(
    query: str,
    user_id: str,
    thread_id: str
) -> tuple[str, dict]:
    """
    Resolve ambiguous queries using session context.
    
    Handles:
    - Pronouns ("this", "that", "it")
    - Follow-up questions ("what about the price?")
    - Context continuation
    
    Returns: (resolved_query, context_used)
    """
    try:
        logger.debug(f"[resolve_query_context] Checking: {query[:50]}...")
        ambiguous_patterns = [
            r"\b(this|that|it|its|the)\s+(product|item|one)\b",
            r"\b(what about|how about|and the|tell me more)\b",
            r"^(price|cost|reviews?|rating|details?)\?*$",
        ]
        
        is_ambiguous = any(
            re.search(pattern, query.lower()) 
            for pattern in ambiguous_patterns
        )
        
        if not is_ambiguous:
            return query, {}
        
        last_product = await get_session_context(user_id, thread_id, "last_product")
        
        if last_product and last_product.get("name"):
            product_name = last_product["name"]
            resolved = f"{query} (referring to {product_name})"
            logger.debug(f"[resolve_query_context] Resolved to: {resolved}")
            return resolved, {"resolved_from": "last_product", "product": product_name}
        
        stm = await get_stm(user_id, thread_id, limit=3)
        for msg in reversed(stm):
            if msg.get("role") == "user":
                return query, {"fallback": "no_context"}
        
        return query, {}
    except Exception as e:
        log_node_error(logger, "resolve_query_context", e)
        return query, {}


# ---------- MAIN LLM CALL ----------
@traceable(name="generate_answer_node", run_type="llm")
async def generate_answer(
    state: GraphState,
    runtime: Runtime[RuntimeContext],
    config: RunnableConfig
):
    log_node_entry(logger, "generate_answer", list(state.keys()))
    try:
        user_id = runtime.context.user_id or "anonymous"
        thread_id = runtime.context.thread_id or config["configurable"].get("thread_id", "default")
        
        await touch_session(user_id, thread_id)

        history = await get_history(user_id, thread_id, scope="user", limit=5)
        stm = await get_stm(user_id, thread_id, limit=5)
    
        memory_messages = []
        for m in stm:
            memory_messages.append(f"{m['role']}: {m['content']}")
        
        if len(stm) < 3:
            for m in history:
                memory_messages.append(f"{m['role']}: {m['content']}")
        
        memory = "\n".join(memory_messages[-5:]) 

        resolved_query, resolution_info = await resolve_query_context(
            state["query"], user_id, thread_id
        )
        
        if resolution_info.get("fallback") == "no_context":
            pass  

        products = state.get("products", [])
        num_products = len(products) if products else 0

        prompt = f"""
        Conversation History (Session-Scoped):
        {memory if memory else "(New conversation)"}

        VERIFIED CONTEXT (Authoritative Data – Highest Priority):
        {state.get("context")}

        User Question:
        {resolved_query}

        Number of Products Found: {num_products}

        INSTRUCTIONS (CRITICAL):

        - You MUST ONLY use information explicitly present in VERIFIED CONTEXT
        - NEVER introduce products, categories, attributes, or facts not present
        - NEVER rely on prior knowledge or guess missing details
        - If VERIFIED CONTEXT does not contain relevant products, say so clearly
        - If product details are incomplete, state what is missing
        - Do NOT invent nutritional values
        - Do NOT invent product names
        - Do NOT assume products exist

        MULTI-PRODUCT GUIDELINES:

        - Reference products EXACTLY as provided
        - Use Product 1 / Product 2 numbering ONLY if multiple products exist in context
        - Highlight differences ONLY if those differences appear in context
        - If multiple products exist but no ranking data is provided, ask for preference

        DOMAIN RULES:

        - Prefer VERIFIED CONTEXT over conversation history
        - Use conversation history only for disambiguation
        - NEVER hallucinate
        - NEVER generalize beyond provided data

        RESPONSE STYLE:

        - Use bullet points and emojis for readability
        - Keep explanations concise
        - Use non-technical language
        - If no products found → respond naturally (do NOT fabricate examples)

        FOOD / NUTRITION RULE (STRICT):

        - Provide nutrition ONLY if nutrition data exists in VERIFIED CONTEXT
        - If absent → say "Nutrition information is not available"

        SAFETY RULES:

        - No medical claims
        - No internal system details
        - No database/schema explanations

        Answer:
        """

        response = await llm.ainvoke(prompt)
        answer = response.content.strip()

        await add_stm(user_id, thread_id, "user", state["query"])
        await add_stm(user_id, thread_id, "assistant", answer)
        
        await save_message(user_id, thread_id, "user", state["query"], scope="user")
        await save_message(user_id, thread_id, "assistant", answer, scope="user")
        
        if products and len(products) > 0:
            await set_session_context(
                user_id, thread_id,
                "last_product",
                products[0]
            )
            await set_session_context(
                user_id, thread_id,
                "products_list",
                {"products": products, "count": len(products)}
            )

        logger.info(f"[generate_answer] Answer generated, {len(products)} products")
        log_node_exit(logger, "generate_answer", ["answer", "products"])
        return {
            "answer": answer,
            "products": products
        }
        
    except Exception as e:
        log_node_error(logger, "generate_answer", e)
        return {
            "answer": "I'm sorry, I'm unable to process your request at the moment.",
            "products": [] 
        }
  
  
@traceable(name="blocked_query_node", run_type="llm")
async def blocked_query_node(
    state: GraphState,
    runtime: Runtime[RuntimeContext],
    config: RunnableConfig
):
    log_node_entry(logger, "blocked_query_node", list(state.keys()))
    try:
        system_prompt = """
    You are a friendly but strict AI assistant for an e-commerce platform.

    Your role:
    - You MUST NOT answer questions about backend logic, database schemas, APIs, server architecture, security rules, internal workflows, admin systems, or confidential implementation details.
    - You MUST NOT answer unrelated questions that are outside shopping, products, orders, or customer support.

    When such a question is detected:
    - Do NOT explain why it is blocked.
    - Do NOT reveal any technical or internal information.
    - Respond in a light-hearted, humorous, and polite tone.
    - Keep the joke short, safe, and friendly.
    - Gently redirect the user back to shopping or product-related help.

    Style rules:
    - Sound playful, not sarcastic or rude.
    - One or two sentences only.
    - No technical terms.
    - No mentions of databases, schemas, backend, or security in your reply.

    Example tone (do not copy verbatim):
    - “Haha, that’s above my pay grade 😄 Let’s get you a great product instead!”
    - “Nice try 😌 I’m here to shop, not spill secrets!”

    Always end by offering help with:
    - finding products
    - comparing items
    - understanding features
    - placing orders
    
    NOTE: Randomize the humor slightly each time while keeping the same friendly tone.
    """
        response = await llm.ainvoke(
            system_prompt + f"\n\nUser Question:\n{state['query']}"
        )

        logger.info(f"[blocked_query_node] Blocked query handled")
        log_node_exit(logger, "blocked_query_node", ["answer", "products"])
        return {
            "answer": response.content.strip(),
            "products": [] 
        }
    except Exception as e:
        log_node_error(logger, "blocked_query_node", e)
        return {
            "answer": "I can help you find great products! What are you looking for?",
            "products": []
        }

# ---------- CONFIDENCE CHECK ----------
@traceable(name="confidence_check")
async def confidence_check(state: GraphState):
    log_node_entry(logger, "confidence_check", list(state.keys()))
    try:
        response = await llm.ainvoke(
            f"Give a confidence score between 0.0 and 1.0:\n{state['answer']}"
        )

        match = re.search(r"(0\.\d+|1\.0)", response.content)
        confidence = float(match.group()) if match else 0.6
        
        logger.info(f"[confidence_check] Confidence: {confidence}")
        log_node_exit(logger, "confidence_check", ["confidence", "answer", "products"])
        return {
            "confidence": confidence,
            "answer": state.get("answer"),
            "products": state.get("products", []) 
        }
    except Exception as e:
        log_node_error(logger, "confidence_check", e)
        return {
            "confidence": 0.5,
            "answer": state.get("answer"),
            "products": state.get("products", [])
        }


# ---------- GRAPH ----------
builder = StateGraph(GraphState)

builder.add_node("intent", detect_intent)
builder.add_node("product_db", product_db_node)
builder.add_node("policy_db", policy_db_node)
builder.add_node("rag", rag_node)
builder.add_node("assemble", assemble_context)
builder.add_node("generate", generate_answer)
builder.add_node("blocked_query", blocked_query_node)
builder.add_node("list_products", list_products_node)
builder.add_node("comparison", comparison_node)
builder.add_node("confidence", confidence_check)

builder.set_entry_point("intent")

builder.add_conditional_edges(
    "intent",
    lambda s: s["intent"],
    {
        "product_info": "product_db",
        "policy_info": "policy_db",
        "list_products": "list_products",
        "comparison": "comparison",
        "normal_query": "list_products",
        "blocked_query": "blocked_query"
    }
)

builder.add_edge("product_db", "assemble")
builder.add_edge("policy_db", "rag")
builder.add_edge("rag", "assemble")
builder.add_edge("assemble", "generate")
builder.add_edge("generate", "confidence")
builder.add_edge("list_products", "generate")
builder.add_edge("comparison", "confidence")
builder.add_edge("blocked_query", END)
builder.add_edge("confidence", END)

chat_graph = builder.compile()
