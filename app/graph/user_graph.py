from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from langsmith import traceable

from app.core.runtime import RuntimeContext
from app.graph.chat_graph import chat_graph
from app.graph.image_graph import image_graph
from app.core.state import OrchestratorState
from app.core.model import get_llm
from app.core.logging_config import (
    get_graph_logger,
    log_node_entry,
    log_node_exit,
    log_node_error
)

# ---------- LOGGER ----------
logger = get_graph_logger("user_graph")

# ---------- LLM MODEL ----------
llm = get_llm()


# ---------- DETECT THE MODAL ----------
@traceable(name="detect_modality")
async def detect_modality(state: OrchestratorState):
    log_node_entry(logger, "detect_modality", list(state.keys()))
    try:
        if state.get("image_url"):
            logger.info("[detect_modality] Modality: image")
            log_node_exit(logger, "detect_modality", ["modality"])
            return {"modality": "image"}

        if state.get("query"):
            logger.info("[detect_modality] Modality: text")
            log_node_exit(logger, "detect_modality", ["modality"])
            return {"modality": "text"}

        logger.warning("[detect_modality] Modality: fallback")
        log_node_exit(logger, "detect_modality", ["modality"])
        return {"modality": "fallback"}
    except Exception as e:
        log_node_error(logger, "detect_modality", e)
        return {"modality": "fallback"}


# ---------- TEXT-BASED SUB-GRAPH (UNIFIED WITH ORDERS) ----------
@traceable(name="text_subgraph_node")
async def text_subgraph_node(
    state: OrchestratorState,
    runtime: Runtime[RuntimeContext],
    config: RunnableConfig
):
    log_node_entry(logger, "text_subgraph_node", list(state.keys()))
    try:
        user_id = runtime.context.user_id or state.get("user_id") or "anonymous"
        thread_id = runtime.context.thread_id or config["configurable"].get("thread_id", "default")
        
        result = await chat_graph.ainvoke(
            {
                "query": state["query"],
                "user_id": user_id,
                "thread_id": thread_id
            },
            config=config
        )
        
        products = result.get("products", [])
        if products is None:
            products = []
        answer = result.get("answer", "")
        
        logger.info(f"[text_subgraph_node] Result: products count={len(products)}")
        log_node_exit(logger, "text_subgraph_node", ["answer", "products", "confidence"])
        
        return {
            "answer": answer,
            "products": products, 
            "confidence": result.get("confidence", 0.7)
        }
    except Exception as e:
        log_node_error(logger, "text_subgraph_node", e)
        return {
            "answer": "I'm sorry, something went wrong. Please try again.",
            "products": [],
            "confidence": 0.0
        }


# ---------- IMAGE BASED SUB-GRAPH ----------
@traceable(name="image_subgraph_node")
async def image_subgraph_node(
    state: OrchestratorState,
    runtime: Runtime[RuntimeContext],
    config: RunnableConfig
):
    log_node_entry(logger, "image_subgraph_node", list(state.keys()))
    try:
        result = await image_graph.ainvoke(
            {
                "image_url": state["image_url"],
                "question": state.get("query")
            },
            config=config,
            context=runtime.context
        )
        
        logger.info(f"[image_subgraph_node] Products found: {len(result.get('product_docs', []))}")
        log_node_exit(logger, "image_subgraph_node", ["answer", "products", "confidence"])

        return {
            "answer": result["answer"],
            "products": result.get("product_docs", []), 
            "confidence": result.get("confidence", 0.6)
        }
    except Exception as e:
        log_node_error(logger, "image_subgraph_node", e)
        return {
            "answer": "I couldn't process the image. Please try again.",
            "products": [],
            "confidence": 0.0
        }


# ---------- GRAPH ----------
builder = StateGraph(OrchestratorState)

builder.add_node("detect_modality", detect_modality)
builder.add_node("text_graph", text_subgraph_node)
builder.add_node("image_graph", image_subgraph_node)

builder.set_entry_point("detect_modality")

builder.add_conditional_edges(
    "detect_modality",
    lambda s: s["modality"],
    {
        "text": "text_graph",
        "image": "image_graph"
    }
)

builder.add_edge("text_graph", END)
builder.add_edge("image_graph", END)

orchestrator_graph = builder.compile()
