import re
import asyncio
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableConfig
from langsmith import traceable

from app.core.state import AdminState
from app.core.model import get_llm, get_vector_db
from app.core.admin_schema import (
    ADMIN_DB_SCHEMA,
    ADMIN_INTENTS,
    FORBIDDEN_FIELDS,
    PRODUCT_TAGS
)
from app.utils.admin_sql_validator import (
    validate_admin_sql,
    sanitize_admin_sql,
    mask_result_rows
)
from app.utils.sql_executor import execute_sql
from app.memory.chat_memory import get_history, save_message
from app.memory.short_term import (
    add_stm,
    get_stm,
    touch_session
)
from app.core.logging_config import (
    get_graph_logger,
    log_node_entry,
    log_node_exit,
    log_node_error
)

# ---------- LOGGER ----------
logger = get_graph_logger("admin_graph")

# ---------- LLM MODEL ----------
llm = get_llm()

# ---------- VECTOR DB (for RAG) ----------
vector_db = get_vector_db()


# ---------- ASYNC VECTOR SEARCH ----------
async def async_admin_vector_search(query: str, k: int = 6):
    """Run vector search in a thread pool to make it async."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: vector_db.similarity_search(query, k=k))


# ---------- PRIVACY CHECK (FIRST LINE OF DEFENSE) ----------
@traceable(name="privacy_check_node")
async def privacy_check_node(state: AdminState):
    log_node_entry(logger, "privacy_check_node", list(state.keys()))
    try:
        query_lower = state["query"].lower()
        
        for field in FORBIDDEN_FIELDS:
            if field in query_lower:
                logger.warning(f"[privacy_check_node] Privacy violation: {field}")
                log_node_exit(logger, "privacy_check_node", ["privacy_violation", "is_safe"])
                return {
                    "privacy_violation": f"Access to '{field}' is not permitted",
                    "is_safe": False
                }
        
        privacy_patterns = [
            "password",
            "show me user email",
            "list all emails",
            "phone number",
            "user otp",
            "session token",
            "authentication",
            "login credentials"
        ]
        
        for pattern in privacy_patterns:
            if pattern in query_lower:
                logger.warning(f"[privacy_check_node] Privacy pattern detected: {pattern}")
                log_node_exit(logger, "privacy_check_node", ["privacy_violation", "is_safe"])
                return {
                    "privacy_violation": "This query attempts to access protected user data",
                    "is_safe": False
                }
        
        logger.info("[privacy_check_node] Query is safe")
        log_node_exit(logger, "privacy_check_node", ["is_safe", "privacy_violation"])
        return {"is_safe": True, "privacy_violation": None}
    except Exception as e:
        log_node_error(logger, "privacy_check_node", e)
        return {"is_safe": False, "privacy_violation": "Error checking query safety"}


# ---------- INTENT DETECTION ----------
@traceable(name="detect_admin_intent")
async def detect_admin_intent(state: AdminState):
    log_node_entry(logger, "detect_admin_intent", list(state.keys()))
    try:
        if not state.get("is_safe", True):
            logger.info("[detect_admin_intent] Intent: privacy_blocked")
            log_node_exit(logger, "detect_admin_intent", ["intent"])
            return {"intent": "privacy_blocked"}
        
        prompt = f"""
    You are an intent classifier for an e-commerce admin assistant.
    
    Classify the query into EXACTLY ONE intent:
    
    1. product_management
       - Add, update, delete products
       - Manage pricing, stock, categories, images
       - Product activation/deactivation
    
    2. order_management
       - View orders, order details
       - Update order status
       - Process refunds
       - Payment information
    
    3. user_management
       - View users (aggregated/masked only)
       - Assign roles
       - User activity summaries
       - Account status
    
    4. analytics
       - Sales reports
       - Revenue analysis
       - Top products
       - Category performance
       - Aggregated metrics
    
    5. coupon_management
       - Create/update coupons
       - Discount management
       - Usage tracking
    
    6. loyalty_management
       - Loyalty program configuration
       - Points rules
       - Points adjustments
    
    7. notification_management
       - Send notifications
       - View notification history
    
    8. system_policy
       - Company policies
       - Operational rules
       - System information
    
    9. general_query
       - Greetings, general questions
       - Unclear queries
    
    10. product_listing
       - List available products
    
    Admin Query:
    \"\"\"{state["query"]}\"\"\"
    
    Return ONLY the intent label. No explanation.
    """
        
        response = await llm.ainvoke(prompt)
        intent = response.content.strip().lower().replace(" ", "_")
        
        # Validate intent
        if intent not in ADMIN_INTENTS:
            intent = "general_query"
        
        logger.info(f"[detect_admin_intent] Intent: {intent}")
        log_node_exit(logger, "detect_admin_intent", ["intent"])
        return {"intent": intent}
    except Exception as e:
        log_node_error(logger, "detect_admin_intent", e)
        return {"intent": "general_query"}


# ---------- SQL GENERATION NODE ----------
@traceable(name="generate_admin_sql_node", run_type="llm")
async def generate_admin_sql_node(state: AdminState):
    """
    Generate SQL for database operations.
    Only runs for DB-related intents.
    """
    log_node_entry(logger, "generate_admin_sql_node", list(state.keys()))
    try:
        intent = state.get("intent", "general_query")
        
        if intent in ["system_policy", "general_query", "privacy_blocked"]:
            logger.info(f"[generate_admin_sql_node] Skipping SQL for intent: {intent}")
            log_node_exit(logger, "generate_admin_sql_node", ["generated_sql", "sql_validated"])
            return {"generated_sql": None, "sql_validated": False}
        
        prompt = f"""
    You are a PostgreSQL expert for an e-commerce admin system.
    
    Database Schema:
    {ADMIN_DB_SCHEMA}
    
    Product Tags for semantic meaning of products:
    {PRODUCT_TAGS}
    
    Admin Intent: {intent}
    Admin Query: {state["query"]}
    
    CRITICAL RULES:
    - Generate ONLY the SQL query, no explanations
    - NO markdown code fences
    - Use LIMIT for SELECT queries (max 50)
    - For user data: NEVER select email, phone, password_hash, otp
    - For user queries: Use aggregations (COUNT, SUM, AVG) when possible
    - Always filter deleted_at IS NULL for products
    - Use proper JOINs based on foreign keys
    
    OPERATION RULES BY INTENT:
    - product_management: SELECT allowed
    - order_management: SELECT allowed (no INSERT, UPDATE, DELETE)
    - user_management: SELECT only, mask sensitive data
    - analytics: SELECT only, aggregations preferred
    - coupon_management: SELECT allowed (no INSERT, UPDATE, DELETE)
    - loyalty_management: SELECT allowed (no INSERT, UPDATE, DELETE)
    - notification_management: SELECT allowed (no INSERT, UPDATE, DELETE)

    IF intent == analytics:
    - NEVER select description, created_at, updated_at
    - ALWAYS use COUNT, SUM, AVG, GROUP BY
    
    Generate the SQL:
    """
        
        response = await llm.ainvoke(prompt)
        raw_sql = response.content.strip()
        sql = sanitize_admin_sql(raw_sql)
        
        logger.info(f"[generate_admin_sql_node] Generated SQL: {sql[:100]}...")
        log_node_exit(logger, "generate_admin_sql_node", ["generated_sql", "sql_validated"])
        
        return {"generated_sql": sql, "sql_validated": False}
    except Exception as e:
        log_node_error(logger, "generate_admin_sql_node", e)
        return {"generated_sql": None, "sql_validated": False, "sql_error": "Failed to generate SQL"}


# ---------- SQL VALIDATION NODE ----------
@traceable(name="validate_sql_node")
async def validate_sql_node(state: AdminState):
    """
    Validate generated SQL against security rules.
    """
    log_node_entry(logger, "validate_sql_node", list(state.keys()))
    try:
        sql = state.get("generated_sql")
        intent = state.get("intent", "general_query")
        
        if not sql:
            logger.info("[validate_sql_node] No SQL to validate")
            log_node_exit(logger, "validate_sql_node", ["sql_validated", "sql_error"])
            return {"sql_validated": False, "sql_error": None}
        
        is_valid, error = await validate_admin_sql(sql, intent)
        
        if not is_valid:
            logger.warning(f"[validate_sql_node] Validation failed: {error}")
            log_node_exit(logger, "validate_sql_node", ["sql_validated", "sql_error", "generated_sql"])
            return {
                "sql_validated": False,
                "sql_error": error,
                "generated_sql": None  
            }
        
        logger.info("[validate_sql_node] SQL validated successfully")
        log_node_exit(logger, "validate_sql_node", ["sql_validated", "sql_error"])
        return {"sql_validated": True, "sql_error": None}
    except Exception as e:
        log_node_error(logger, "validate_sql_node", e)
        return {"sql_validated": False, "sql_error": "Validation failed"}


# ---------- SQL EXECUTION NODE ----------
@traceable(name="execute_sql_node")
async def execute_sql_node(state: AdminState):
    """
    Execute validated SQL and mask sensitive results.
    """
    log_node_entry(logger, "execute_sql_node", list(state.keys()))
    try:
        sql = state.get("generated_sql")
        
        if not sql or not state.get("sql_validated"):
            logger.info("[execute_sql_node] No validated SQL to execute")
            log_node_exit(logger, "execute_sql_node", ["db_result"])
            return {"db_result": None}
        
        rows = await execute_sql(sql)
        
        masked_rows = mask_result_rows(rows) if rows else []
        
        logger.info(f"[execute_sql_node] Executed successfully, {len(masked_rows)} rows")
        log_node_exit(logger, "execute_sql_node", ["db_result"])
        return {"db_result": masked_rows}
        
    except Exception as e:
        log_node_error(logger, "execute_sql_node", e)
        return {
            "db_result": None,
            "sql_error": "Database query failed. Please try again."
        }


# ---------- RAG RETRIEVAL NODE ----------
@traceable(name="rag_retrieval_node")
async def rag_retrieval_node(state: AdminState):
    """
    Retrieve documents for policy/system queries.
    """
    log_node_entry(logger, "rag_retrieval_node", list(state.keys()))
    try:
        intent = state.get("intent", "")
        
        if intent not in ["system_policy", "general_query"] and state.get("db_result"):
            logger.info("[rag_retrieval_node] Skipping RAG - DB result available")
            log_node_exit(logger, "rag_retrieval_node", ["documents", "rag_context"])
            return {"documents": [], "rag_context": None}
        
        docs = await async_admin_vector_search(state["query"], k=6)
        rag_context = "\n\n".join(
            f"[{d.metadata.get('type', 'document').upper()}]\n{d.page_content}"
            for d in docs
        )
        
        logger.info(f"[rag_retrieval_node] Retrieved {len(docs)} documents")
        log_node_exit(logger, "rag_retrieval_node", ["documents", "rag_context"])
        return {"documents": docs, "rag_context": rag_context}
    except Exception as e:
        log_node_error(logger, "rag_retrieval_node", e)
        return {"documents": [], "rag_context": None}


DB_INTENTS = {
    "product_listing",
    "product_management",
    "order_management",
    "user_management",
    "analytics",
    "coupon_management",
    "loyalty_management",
    "notification_management",
}

# ---------- CONTEXT ASSEMBLY NODE ----------
@traceable(name="assemble_context_node")
async def assemble_context_node(state: AdminState):
    log_node_entry(logger, "assemble_context_node", list(state.keys()))
    try:
        if state.get("privacy_violation"):
            logger.info("[assemble_context_node] Privacy violation context")
            log_node_exit(logger, "assemble_context_node", ["context"])
            return {"context": f"SECURITY: {state['privacy_violation']}"}
        
        intent = state.get("intent")

        if intent in DB_INTENTS and not state.get("db_result"):
            logger.info("[assemble_context_node] No DB results for DB intent")
            log_node_exit(logger, "assemble_context_node", ["context"])
            return {
                "context": (
                    "DATABASE RESULT: NO RECORDS FOUND.\n"
                    "YOU MUST NOT INVENT PRODUCTS, ORDERS, USERS, PRICES, OR EXAMPLES."
                )
            }

        sections = []

        if state.get("sql_error"):
            sections.append(f"NOTE: {state['sql_error']}")

        if state.get("db_result"):
            sections.append(
                f"DATABASE RESULTS ({len(state['db_result'])} rows):\n{state['db_result']}"
            )

        if state.get("rag_context"):
            sections.append(f"DOCUMENTS:\n{state['rag_context']}")

        logger.info(f"[assemble_context_node] Assembled {len(sections)} sections")
        log_node_exit(logger, "assemble_context_node", ["context"])
        return {"context": "\n\n".join(sections)}
    except Exception as e:
        log_node_error(logger, "assemble_context_node", e)
        return {"context": "Error assembling context"}


# ---------- ANSWER GENERATION NODE ----------
@traceable(name="generate_admin_answer", run_type="llm")
async def generate_admin_answer(state: AdminState, config: RunnableConfig):
    log_node_entry(logger, "generate_admin_answer", list(state.keys()))
    try:
        admin_id = config.get("configurable", {}).get("user_id", "admin")
        thread_id = config.get("configurable", {}).get("thread_id", "default")
        
        await touch_session(admin_id, thread_id)
        
        if state.get("privacy_violation"):
            answer = f"""I apologize, but I cannot process this request.

            **Reason:** {state['privacy_violation']}

            For security and privacy compliance, access to sensitive user data such as emails, phone numbers, passwords, and authentication tokens is restricted.

            If you need this information for a legitimate purpose, please contact your system administrator through the proper channels."""
            
            await save_message(admin_id, thread_id, "user", state["query"], scope="admin")
            await save_message(admin_id, thread_id, "assistant", answer, scope="admin")
            
            logger.info("[generate_admin_answer] Privacy blocked response")
            log_node_exit(logger, "generate_admin_answer", ["answer"])
            return {"answer": answer}
        
        history = await get_history(admin_id, thread_id, scope="admin", limit=3)
        stm = await get_stm(admin_id, thread_id, limit=3)
        
        memory_messages = [f"{m['role']}: {m['content']}" for m in stm]
        if len(stm) < 2:
            memory_messages.extend(f"{m['role']}: {m['content']}" for m in history)
        
        memory = "\n".join(memory_messages[-3:])
        
        prompt = f"""
    You are an ADMIN assistant for an e-commerce platform.
    
    Session History:
    {memory if memory else "(New conversation)"}
    
    Intent: {state.get("intent", "general")}
    
    Context:
    {state.get("context", "(No data)")}
    
    Admin Query:
    {state["query"]}
    
    RESPONSE RULES:
    - Be professional and concise
    - Never mention internal table names or schemas
    - Never expose raw SQL queries
    - Like if no data is found, say "no records found" in a clear manner
    - If data is missing, clearly state what's unavailable
    - For user data, always note that sensitive fields are masked
    - Format numbers and currencies appropriately
    - If the query failed, suggest alternative approaches
    - NEVER fabricate or infer database records
    - ONLY use data explicitly present in DATABASE RESULT
    - If no rows exist, say so clearly and stop
    - DO NOT provide examples, placeholders, or sample data
    
    TABLE FORMATTING RULES (MANDATORY):
    - When displaying multiple records (products, orders, users, analytics, etc.), ALWAYS use markdown tables
    - Use this exact format:
      
      ### Table Title
      | Column1 | Column2 | Column3 |
      |---------|---------|---------|
      | Value1  | Value2  | Value3  |
      
    - For products: Use columns like | # | Name | Price | Stock | Category |
    - For orders: Use columns like | Order ID | Customer | Total | Status | Date |
    - For analytics: Use columns like | Metric | Value | Change |
    - Include a "Total" or "Summary" row at the bottom if appropriate
    - Format currency with ₹ symbol (e.g., ₹120.00)
    - Format stock as numbers without decimals when possible
    - NEVER use bullet points (* or -) for listing database records
    - NEVER use numbered lists (1. 2. 3.) for listing database records
    - Only use bullet points for non-data explanatory text
    
    PRIVACY RULES:
    - Never reveal full email addresses
    - Never reveal phone numbers
    - Never reveal passwords or tokens
    - Use masked format: j***@gmail.com, ******1234
    
    SEMANTIC TAG RULES:
    - If the query mentions concepts like:
    "healthy", "organic", "fitness", "vegan", "protein", "eco", "natural"
    - Then you MUST:
    - Join product_tags_map and product_tags
    - Filter using LOWER(product_tags.name) LIKE '%<keyword>%'
    - NEVER invent products if no rows match
    
    CRITICAL RULE (ABSOLUTE):
    - If you invent data, it is a critical failure.

    Generate a helpful admin response:
    """
        
        response = await llm.ainvoke(prompt)
        answer = response.content.strip()
        
        await add_stm(admin_id, thread_id, "user", state["query"])
        await add_stm(admin_id, thread_id, "assistant", answer)
        
        await save_message(admin_id, thread_id, "user", state["query"], scope="admin")
        await save_message(admin_id, thread_id, "assistant", answer, scope="admin")
        
        logger.info("[generate_admin_answer] Answer generated successfully")
        log_node_exit(logger, "generate_admin_answer", ["answer"])
        return {"answer": answer}
    except Exception as e:
        log_node_error(logger, "generate_admin_answer", e)
        return {"answer": "I'm sorry, I couldn't process that request. Please try again."}


# ---------- CONFIDENCE CHECK NODE ----------
@traceable(name="confidence_check_node")
async def confidence_check_node(state: AdminState):
    """
    Evaluate response confidence.
    """
    log_node_entry(logger, "confidence_check_node", list(state.keys()))
    try:
        response = await llm.ainvoke(
            f"Rate how appropriately this answer addresses the user's query from 0.0 to 1.0.\n"
            f"- For greetings/casual queries, a friendly response = 1.0\n"
            f"- For data queries, accurate data response = 1.0\n"
            f"Return ONLY a decimal number.\n\n"
            f"Query: {state.get('query', '')}\n"
            f"Answer: {state['answer']}"
        )
        
        match = re.search(r"(0\.\d+|1\.0|0|1)", response.content.strip())
        confidence = float(match.group()) if match else 0.8
        confidence = max(0.0, min(1.0, confidence))
        
        logger.info(f"[confidence_check_node] Confidence: {confidence}")
        log_node_exit(logger, "confidence_check_node", ["confidence"])
        return {"confidence": confidence}
    except Exception as e:
        log_node_error(logger, "confidence_check_node", e)
        return {"confidence": 0.8}


# ---------- ROUTING FUNCTIONS ----------
def sql_result_router(state: AdminState) -> str:
    """Route based on SQL execution result."""
    if state.get("db_result"):
        return "has_data"
    return "no_data"


def confidence_router(state: AdminState) -> str:
    """Route based on confidence score."""
    confidence = state.get("confidence", 0.5)
    retry_count = state.get("retry_count", 0)
    
    if confidence >= 0.6 or retry_count >= 2:
        return "end"
    
    return "retry"


@traceable(name="retry_node")
def retry_node(state: AdminState):
    """Increment retry count for low-confidence answers."""
    return {"retry_count": state.get("retry_count", 0) + 1}


builder = StateGraph(AdminState)

builder.add_node("privacy_check", privacy_check_node)
builder.add_node("detect_intent", detect_admin_intent)
builder.add_node("generate_sql", generate_admin_sql_node)
builder.add_node("validate_sql", validate_sql_node)
builder.add_node("execute_sql", execute_sql_node)
builder.add_node("rag_retrieval", rag_retrieval_node)
builder.add_node("assemble_context", assemble_context_node)
builder.add_node("generate_answer", generate_admin_answer)
builder.add_node("confidence_check", confidence_check_node)
builder.add_node("retry", retry_node)

builder.set_entry_point("privacy_check")

builder.add_edge("privacy_check", "detect_intent")

builder.add_conditional_edges(
    "detect_intent",    
    lambda state: state.get("intent", "general_query"),
    {
        "privacy_blocked": "assemble_context",
        "system_policy": "rag_retrieval",
        "general_query": "assemble_context",
        "product_management": "generate_sql",
        "order_management": "generate_sql",
        "user_management": "generate_sql",
        "analytics": "generate_sql",
        "coupon_management": "generate_sql",
        "loyalty_management": "generate_sql",
        "notification_management": "generate_sql",
        "product_listing": "generate_sql",
    }
)

builder.add_edge("generate_sql", "validate_sql")
builder.add_edge("validate_sql", "execute_sql")
builder.add_edge("execute_sql", "assemble_context")

# builder.add_conditional_edges(
#     "execute_sql",
#     sql_result_router,
#     {
#         "has_data": "assemble_context",
#         "no_data": "rag_retrieval"
#     }
# )

builder.add_edge("rag_retrieval", "assemble_context")

builder.add_edge("assemble_context", "generate_answer")
builder.add_edge("generate_answer", "confidence_check")

builder.add_conditional_edges(
    "confidence_check",
    confidence_router,
    {
        "end": END,
        "retry": "retry"
    }
)

builder.add_edge("retry", "assemble_context")

admin_graph = builder.compile()
