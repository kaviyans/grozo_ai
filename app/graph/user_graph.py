from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from langsmith import traceable

from app.core.runtime import RuntimeContext
from app.graph.chat_graph import chat_graph
from app.graph.image_graph import image_graph
from app.core.state import OrchestratorState
from app.core.model import get_llm


# ---------- LLM MODEL ----------
llm = get_llm()


# ---------- DETECT THE MODAL ----------
@traceable(name="detect_modality")
async def detect_modality(state: OrchestratorState):
    if state.get("image_url"):
        return {"modality": "image"}

    if state.get("query"):
        return {"modality": "text"}

    return {"modality": "fallback"}



# ---------- TEXT-BASED SUB-GRAPH (UNIFIED WITH ORDERS) ----------
@traceable(name="text_subgraph_node")
async def text_subgraph_node(
    state: OrchestratorState,
    runtime: Runtime[RuntimeContext],
    config: RunnableConfig
):
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
    
    print(f"[text_subgraph] Result: answer={answer}..., products count={len(products)}")
    
    return {
        "answer": answer,
        "products": products, 
        "confidence": result.get("confidence", 0.7)
    }


# ---------- IMAGE BASED SUB-GRAPH ----------
@traceable(name="image_subgraph_node")
async def image_subgraph_node(
    state: OrchestratorState,
    runtime: Runtime[RuntimeContext],
    config: RunnableConfig
):
    result = await image_graph.ainvoke(
        {
            "image_url": state["image_url"],
            "question": state.get("query")
        },
        config=config,
        context=runtime.context
    )
    
    print(result.get("product_docs", []))

    return {
        "answer": result["answer"],
        "products": result.get("product_docs", []), 
        "confidence": result.get("confidence", 0.6)
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
