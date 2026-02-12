from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel
from typing import Optional
import os, tempfile, shutil
from pathlib import Path

# from app.sample.product_research import run_product_analysis
from app.graph.admin_graph import admin_graph
from app.ingestion.ingestion import ingest_document
from app.core.db import get_db
from app.utils.thread import generate_thread_id
from langchain_core.runnables import RunnableConfig

router = APIRouter(prefix="/admin", tags=["Admin API"])

class AdminChatIn(BaseModel):
    query: str
    admin_id: str
    thread_id: Optional[str] = None

DOC_STORE = Path("data/admin_docs")
DOC_STORE.mkdir(parents=True, exist_ok=True)

SUPPORTED_EXTENSIONS = {".pdf", ".doc", ".docx", ".xls", ".xlsx"}

async def get_next_version(filename: str) -> str:
    base = Path(filename).stem
    ext = Path(filename).suffix
    versions = []

    for f in DOC_STORE.glob(f"{base}_v*{ext}"):
        try:
            versions.append(int(f.stem.split("_v")[-1]))
        except:
            pass

    return f"{(max(versions) + 1) if versions else 1}.0"

@router.post("/ingest")
async def admin_ingest_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db)
):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(400, "Unsupported file type")

    version = await get_next_version(file.filename)
    versioned_name = f"{Path(file.filename).stem}_v{version.split('.')[0]}{suffix}"
    stored_path = DOC_STORE / versioned_name

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    shutil.move(tmp_path, stored_path)

    background_tasks.add_task(
        ingest_document,
        str(stored_path),
        {
            "filename": file.filename,
            "doc_type": suffix[1:],
            "version": version,
        },
    )

    return {"status": "ingestion_started", "stored_as": versioned_name}

@router.post("/chat")
async def admin_chat(payload: AdminChatIn):
    if(payload.thread_id):
        thread_id = payload.thread_id
    else:
        thread_id = await generate_thread_id()

    config = RunnableConfig(
        configurable={
            "thread_id": thread_id,
            "user_id": payload.admin_id,
        }
    )

    result = await admin_graph.ainvoke(
        {
            "query": payload.query,
            "documents": [],
            "answer": "",
            "confidence": 0.0,
            "retry_count": 0,
        },
        config=config,
    )

    return {
        "answer": result.get("answer"),
        "confidence": result.get("confidence"),
        "thread_id": thread_id,
        "sources_count": len(result.get("documents", [])),
    }


# @router.post("/products/market-intelligence")
# async def run_product_market_analysis(
#     payload: AdminChatIn
# ):
#     try:
#         result = await run_product_analysis(
#             query=payload.query,
#             admin_id=payload.admin_id,
#             thread_id=payload.thread_id or (await generate_thread_id()),
#             analysis_type="full",
#             time_period=30,
#             limit= 20,
#         )
#         return result
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))