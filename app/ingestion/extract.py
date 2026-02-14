import camelot
from pdf2image import convert_from_path
import pytesseract
from langchain_core.documents import Document
from typing import List
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    Docx2txtLoader,
    UnstructuredExcelLoader,
)
from pathlib import Path
from app.core.logging_config import get_ingestion_logger

# ---------- LOGGER ----------
logger = get_ingestion_logger()


pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# ---------- OCR ----------
def ocr_pdf(file_path: str):
    logger.info(f"[ocr_pdf] Running OCR on: {file_path}")
    try:
        images = convert_from_path(file_path)
        docs = []
        
        for i, img in enumerate(images):
            text = pytesseract.image_to_string(img, lang="eng")
            docs.append(
                Document(
                    page_content=text,
                    metadata={"page": i, "source": file_path, "type": "ocr"}
                )
            )
        logger.info(f"[ocr_pdf] OCR completed for {len(images)} pages")
        return docs
    except Exception as e:
        logger.error(f"[ocr_pdf] OCR failed for {file_path}: {e}", exc_info=True)
        return []


def has_text(docs):
    for doc in docs:
        if doc.page_content.strip():
            return True
    return False


# ---------- EXTRACT TEXT FROM DOC ----------
def extract_text(file_path: str) -> List[Document]:
    logger.info(f"[extract_text] Extracting text from: {file_path}")
    try:
        ext = Path(file_path).suffix.lower()

        if ext == ".pdf":
            docs = PyMuPDFLoader(file_path).load()

            if not has_text(docs):
                logger.info("[extract_text] Scanned PDF detected -> running OCR")
                docs = ocr_pdf(file_path)
            else:
                logger.info("[extract_text] Text-based PDF detected -> using native text")

            return docs

        elif ext in [".doc", ".docx"]:
            logger.info("[extract_text] Loading Word document")
            return Docx2txtLoader(file_path).load()

        elif ext in [".xls", ".xlsx"]:
            logger.info("[extract_text] Loading Excel document")
            return UnstructuredExcelLoader(file_path, mode="elements").load()

        else:
            logger.error(f"[extract_text] Unsupported file type: {ext}")
            raise ValueError(f"Unsupported file type: {ext}")
    except Exception as e:
        logger.error(f"[extract_text] Error extracting text from {file_path}: {e}", exc_info=True)
        raise


# ---------- EXTRACT TABLE FROM DOC ----------
def extract_tables(file_path: str) -> List[str]:
    logger.info(f"[extract_tables] Extracting tables from: {file_path}")
    try:
        if Path(file_path).suffix.lower() != ".pdf":
            logger.info("[extract_tables] Not a PDF, skipping table extraction")
            return []

        try:
            tables = camelot.read_pdf(file_path, pages="all", flavor="lattice")  # LINE BASED TABLES
            if not tables:
                tables = camelot.read_pdf(file_path, pages="all", flavor="stream")  # SPACE BASED TABLES

            rows = []
            for table in tables:
                rows.extend(
                    [" | ".join(map(str, row)) for row in table.df.values.tolist()]
                )
            logger.info(f"[extract_tables] Extracted {len(rows)} table rows")
            return rows
        except Exception as e:
            logger.warning(f"[extract_tables] Table extraction failed: {e}")
            return []
    except Exception as e:
        logger.error(f"[extract_tables] Error extracting tables from {file_path}: {e}", exc_info=True)
        return []


# ---------- EXTRACT IMAGE FROM DOC ----------
def extract_images_and_charts(file_path: str):
    logger.info(f"[extract_images_and_charts] Extracting images from: {file_path}")
    try:
        if Path(file_path).suffix.lower() != ".pdf":
            logger.info("[extract_images_and_charts] Not a PDF, skipping image extraction")
            return []
        images = convert_from_path(file_path)
        logger.info(f"[extract_images_and_charts] Extracted {len(images)} images")
        return images
    except Exception as e:
        logger.error(f"[extract_images_and_charts] Error extracting images from {file_path}: {e}", exc_info=True)
        return []
