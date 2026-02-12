from app.core.model import get_vector_db
from .extract import (
    extract_text,
    extract_tables,
    extract_images_and_charts
)
from langchain_core.documents import Document
from .chunking import chunk_text, chunk_tables, chunk_images

# ---------- CHROMA DB ----------
vector_db = get_vector_db()


# ---------- INGEST FUNCTIONALITY ----------
async def ingest_document(file_path: str, metadata: dict | None = None):
    metadata = metadata or {}

    # ---------- EXTRACT ----------
    text_docs = extract_text(file_path)
    table_rows = extract_tables(file_path)
    images = extract_images_and_charts(file_path)

    # ---------- CHUNK ----------
    text_chunks = chunk_text(text_docs)
    table_chunks = chunk_tables(table_rows, file_path)
    image_chunks = await chunk_images(images, file_path)

    all_docs = text_chunks + table_chunks + image_chunks

    if not all_docs:
        print(f"No content extracted from {file_path}")
        return

    # ---------- INGEST ----------
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

    print(f" Ingested {len(all_docs)} chunks from {metadata.get('filename')}")
