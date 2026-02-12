from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Form
from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.db import get_db
from app.core.runtime import RuntimeContext
from app.utils.thread import generate_thread_id
from app.graph.user_graph import orchestrator_graph
from langchain_core.runnables import RunnableConfig
from app.utils.cloudinary import upload_to_cloudinary

router = APIRouter(prefix="/main", tags=["User API"])

@router.post("/chat")
async def unified_chat(
    query: Optional[str] = Form(None),
    image: Optional[List[UploadFile]] = File(None),
    db: AsyncSession = Depends(get_db)
):
    image_url = None


    if image:
        image_url = await upload_to_cloudinary(image)

    thread_id = await generate_thread_id()
    
    print("Image URL:", image_url[0] if image_url else None)
    print("query :", query)

    result = await orchestrator_graph.ainvoke(
        {
            "query": query,
            "image_url": image_url[0] if image_url else None,
            "thread_id": thread_id
        },
        config=RunnableConfig(
            configurable={"thread_id": thread_id}
        ),
        context=RuntimeContext(
            user_id="anonymous",
            thread_id=thread_id,
            db=db
        )
    )

    products = result.get("products", [])
    if products is None:
        products = []

    return {
        "answer": result["answer"],
        "products": products,  
        "confidence": result["confidence"],
        "thread_id": thread_id
    }
