from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.user import router as user_router
from app.api.admin import router as admin_router
from app.sample.margin_api import router as margin_router


app = FastAPI(title="Multimodal E-commerce RAG")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],
)

app.include_router(user_router)
app.include_router(admin_router)
app.include_router(margin_router)


