from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from pymongo import MongoClient
from dotenv import load_dotenv
load_dotenv()
import os
import cloudinary

# ---------- POSTGRESQL ----------
engine = create_async_engine(
    os.getenv("POSTGRES_URL"), 
    echo=False, 
    future=True,
    pool_size=10, 
    max_overflow=20
)
AsyncSessionLocal = sessionmaker(bind=engine, class_=AsyncSession, autoflush=False, expire_on_commit=False)
Base = declarative_base()

async def get_db():
    async with AsyncSession(engine, expire_on_commit=False) as session:
        yield session


# ---------- MONGO DB ----------
MONGO_URL = os.getenv("MONGO_URL")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "ecommerce_db")

mongo_client = MongoClient(MONGO_URL)
mongo_db = mongo_client[MONGO_DB_NAME]



# ---------- CLOUDINARY ----------
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key= os.getenv("CLOUDINARY_API_KEY"),
    api_secret= os.getenv("CLOUDINARY_API_SECRET"),
    secure=True
)
