from app.core.model import get_vector_db
from app.core.logging_config import get_ingestion_logger
from .extract import (
    extract_text,
    extract_tables,
    extract_images_and_charts
)
from langchain_core.documents import Document
from .chunking import chunk_text, chunk_tables, chunk_images

# ---------- LOGGER ----------
logger = get_ingestion_logger()

# ---------- CHROMA DB ----------
vector_db = get_vector_db()


# ---------- INGEST FUNCTIONALITY ----------
async def ingest_document(file_path: str, metadata: dict | None = None):
    logger.info(f"[ingest_document] Starting ingestion for: {file_path}")
    try:
        metadata = metadata or {}

        # ---------- EXTRACT ----------
        logger.info(f"[ingest_document] Extracting text from: {file_path}")
        text_docs = extract_text(file_path)
        logger.info(f"[ingest_document] Extracting tables from: {file_path}")
        table_rows = extract_tables(file_path)
        logger.info(f"[ingest_document] Extracting images from: {file_path}")
        images = extract_images_and_charts(file_path)

        # ---------- CHUNK ----------
        logger.info(f"[ingest_document] Chunking content")
        text_chunks = chunk_text(text_docs)
        table_chunks = chunk_tables(table_rows, file_path)
        image_chunks = await chunk_images(images, file_path)

        all_docs = text_chunks + table_chunks + image_chunks

        if not all_docs:
            logger.warning(f"[ingest_document] No content extracted from {file_path}")
            return

        # ---------- INGEST ----------
        logger.info(f"[ingest_document] Adding {len(all_docs)} chunks to vector DB")
        vector_db.add_documents(
            [
                Document(
                    page_content=d.page_content,
                    metadata={
                        **d.metadata,
                        "filename": metadata.get("filename"),
                        "doc_type": metadata.get("doc_type"),
                        "version": metadata.get("version"),
                    },
                )
                for d in all_docs
            ]
        )

        logger.info(f"[ingest_document] Successfully ingested {len(all_docs)} chunks from {metadata.get('filename')}")
    except Exception as e:
        logger.error(f"[ingest_document] Error ingesting {file_path}: {e}", exc_info=True)
        raise
