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



pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# ---------- OCR ----------
def ocr_pdf(file_path: str):
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
    return docs


def has_text(docs):
    for doc in docs:
        if doc.page_content.strip():
            return True
    return False


# ---------- EXTRACT TEXT FROM DOC ----------
def extract_text(file_path: str) -> List[Document]:
    ext = Path(file_path).suffix.lower()

    if ext == ".pdf":
        docs = PyMuPDFLoader(file_path).load()

        if not has_text(docs):
            print("Scanned PDF detected → running OCR")
            docs = ocr_pdf(file_path)
        else:
            print("Text-based PDF detected → using native text")


        return docs

    elif ext in [".doc", ".docx"]:
        return Docx2txtLoader(file_path).load()

    elif ext in [".xls", ".xlsx"]:
        return UnstructuredExcelLoader(file_path, mode="elements").load()

    else:
        raise ValueError(f"Unsupported file type: {ext}")


# ---------- EXTRACT TABLE FROM DOC ----------
def extract_tables(file_path: str) -> List[str]:
    if Path(file_path).suffix.lower() != ".pdf":
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
        return rows
    except Exception as e:
        print("Table extraction failed:", e)
        return []


# ---------- EXTRACT IMAGE FROM DOC ----------
def extract_images_and_charts(file_path: str):
    if Path(file_path).suffix.lower() != ".pdf":
        return []
    return convert_from_path(file_path)
