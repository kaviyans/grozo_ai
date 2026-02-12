from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from functools import lru_cache
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatZhipuAI
import os
from pathlib import Path
load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

BASE_DIR = Path(__file__).resolve().parents[2]

# ---------- CHAT MODEL ----------
@lru_cache
def get_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.2,
        api_key=os.getenv("GROQ_API_KEY")
    )
    
    
# @lru_cache
# def get_llm():
#     return ChatOpenAI(
#         model="gpt-oss-20b",
#         temperature=0.7,
#         openai_api_base="https://Fyra.im/v1",
#         openai_api_key=os.getenv("FYRA_API_KEY"),
#     )
    
# ---------- VISION MODEL ----------  
@lru_cache
def get_vision_llm():
    return ChatGroq(
        model="meta-llama/llama-4-maverick-17b-128e-instruct",
        temperature=0
    )

# ---------- CHROMA DB INSTANCE ----------
@lru_cache
def get_vector_db():
    return Chroma(
        collection_name="admin_docs",
        embedding_function=embeddings,
        persist_directory=str(BASE_DIR / "chroma")
    )