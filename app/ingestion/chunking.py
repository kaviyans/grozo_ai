from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from .image_summary import summarize_image

# ---------- TEXT CHUNKING ----------
def chunk_text(docs: List[Document]) -> List[Document]:
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

    return [
        Document(
            page_content=c.page_content,
            metadata={**c.metadata, "chunk_type": "text"}
        )
        for c in chunks
    ]

# ---------- TABLE CHUNKING ----------
def chunk_tables(rows: List[str], file_path: str) -> List[Document]:
    return [
        Document(
            page_content=row,
            metadata={
                "source": file_path,
                "chunk_type": "table"
            }
        )
        for row in rows
    ]

# ---------- IMAGE CHUNKING ----------
async def chunk_images(images, file_path: str) -> List[Document]:
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
            print("Image summarization failed:", e)

    return docs
