from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from .image_summary import summarize_image
from app.core.logging_config import get_ingestion_logger

# ---------- LOGGER ----------
logger = get_ingestion_logger()

# ---------- TEXT CHUNKING ----------
def chunk_text(docs: List[Document]) -> List[Document]:
    logger.info(f"[chunk_text] Chunking {len(docs)} documents")
    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=900,
            chunk_overlap=120,
            separators=[
                "\n\n## ",
                "\n\n### ",
                "\n\n",
                "\n",
                " "
            ]
        )

        chunks = splitter.split_documents(docs)

        result = [
            Document(
                page_content=c.page_content,
                metadata={**c.metadata, "chunk_type": "text"}
            )
            for c in chunks
        ]
        logger.info(f"[chunk_text] Created {len(result)} text chunks")
        return result
    except Exception as e:
        logger.error(f"[chunk_text] Error chunking text: {e}", exc_info=True)
        return []

# ---------- TABLE CHUNKING ----------
def chunk_tables(rows: List[str], file_path: str) -> List[Document]:
    logger.info(f"[chunk_tables] Chunking {len(rows)} table rows from {file_path}")
    try:
        result = [
            Document(
                page_content=row,
                metadata={
                    "source": file_path,
                    "chunk_type": "table"
                }
            )
            for row in rows
        ]
        logger.info(f"[chunk_tables] Created {len(result)} table chunks")
        return result
    except Exception as e:
        logger.error(f"[chunk_tables] Error chunking tables: {e}", exc_info=True)
        return []

# ---------- IMAGE CHUNKING ----------
async def chunk_images(images, file_path: str) -> List[Document]:
    logger.info(f"[chunk_images] Processing {len(images)} images from {file_path}")
    try:
        docs = []

        for i, img in enumerate(images):
            try:
                summary = await summarize_image(img)

                docs.append(
                    Document(
                        page_content=summary,
                        metadata={
                            "source": file_path,
                            "page": i,
                            "chunk_type": "image"
                        }
                    )
                )
            except Exception as e:
                logger.warning(f"[chunk_images] Image summarization failed for image {i}: {e}")

        logger.info(f"[chunk_images] Created {len(docs)} image chunks")
        return docs
    except Exception as e:
        logger.error(f"[chunk_images] Error chunking images: {e}", exc_info=True)
        return []
