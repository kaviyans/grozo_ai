# Multimodal E-commerce RAG System

A production-grade Retrieval-Augmented Generation (RAG) system for e-commerce with multimodal support, built using LangGraph, FastAPI, PostgreSQL, and MongoDB.

---

## Project Overview

The Multimodal E-commerce RAG System is designed to provide intelligent, context-aware responses for both end users and administrators within an e-commerce environment. The system supports natural language queries as well as image-based product discovery, enabling flexible and realistic interaction patterns.

The application combines LLM-driven reasoning, vector-based retrieval, structured SQL queries, and vision models to deliver grounded, reliable responses. It separates user-facing workflows from administrative workflows to ensure security, privacy, and operational clarity.

Key capabilities include:

- Text-based product, policy, and FAQ queries  
- Image-based product understanding and matching  
- Retrieval-Augmented Generation over ingested documents  
- SQL-driven analytics and product retrieval  
- Persistent conversational memory  
- Privacy-aware query validation  

---

## Technical Overview / Architecture

### High-Level System Design

The system follows a modular architecture built around LangGraph workflows. Different query modalities (text vs image) are dynamically detected and routed to specialized processing pipelines.

### User Query Flow

**Text Query Pipeline**

1. User submits query  
2. Modality detection determines input type  
3. Intent classification identifies query category  
4. SQL generation or vector retrieval is triggered  
5. Relevant data is retrieved (products, policies, FAQs)  
6. LLM generates grounded response using memory  
7. Confidence scoring applied  
8. Final response returned  

**Image Query Pipeline**

1. User uploads image  
2. Vision model analyzes visual content  
3. Product description generated  
4. Keywords extracted  
5. Database and web matching performed  
6. Reviews and specifications retrieved  
7. Response generated (name, features, pros/cons)  

User-uploaded images are processed transiently and not stored permanently.

---

### Admin Workflow

**Document Ingestion Pipeline**

1. Admin uploads document  
2. Document type detection  
3. Layout-aware parsing (text, tables, images)  
4. Metadata tagging and versioning  
5. Chunking and embedding generation  
6. Storage in Chroma vector database  

**Admin Query Pipeline**

1. Admin submits query  
2. Privacy and safety validation  
3. Intent detection  
4. SQL generation with validation  
5. Query execution  
6. LLM-based response generation  

---

### Core Components

- **API Layer**: FastAPI-based asynchronous backend  
- **Workflow Engine**: LangGraph for orchestrating reasoning flows  
- **LLM Provider**: Groq-hosted LLaMA models  
- **Vision Model**: Multimodal LLaMA vision model  
- **Vector Database**: Chroma for semantic retrieval  
- **Relational Database**: PostgreSQL for structured product data  
- **Memory Store**: MongoDB for conversational context  
- **Image Handling**: Cloudinary for temporary image processing  
- **Web Search**: Tavily integration for fallback retrieval  

---

## Setup & Run Instructions

### Prerequisites

- Python 3.10 or higher  
- PostgreSQL database  
- MongoDB instance  
- Groq API key  
- Cloudinary account  
- Tavily API key  

---

### Installation

```bash
pip install -r requirements.txt
````

---

### Environment Variables

Create a `.env` file:

```env
POSTGRES_URL=postgresql+asyncpg://user:pass@host:port/db
MONGO_URL=mongodb://localhost:27017
MONGO_DB_NAME=ecommerce_db
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
CLOUDINARY_CLOUD_NAME=your_cloud_name
CLOUDINARY_API_KEY=your_cloudinary_key
CLOUDINARY_API_SECRET=your_cloudinary_secret
```

---

### Running the Application

```bash
uvicorn app.main:app --reload --port 8000
```

The API will be available at:

```
http://localhost:8000
```

---

## Key Design Decisions

### Separation of User and Admin Workflows

User interactions and administrative operations are handled by distinct graph workflows. This ensures better security, simpler reasoning logic, and safer query execution.

---

### Multimodal Query Handling

Queries are automatically classified based on modality. Text queries trigger RAG and SQL pipelines, while image queries invoke vision-based product discovery mechanisms.

---

### Hybrid Retrieval Strategy

The system integrates:

* Vector retrieval for unstructured documents
* SQL queries for structured product data
* Web search fallback for missing knowledge

This improves both accuracy and coverage.

---

### Grounded Response Generation

Responses are constrained by retrieved context from vector databases and SQL results. This reduces hallucinations and improves factual reliability.

---

### Memory Architecture

MongoDB is used for persistent conversational memory, allowing the system to maintain continuity across sessions while avoiding tight coupling with the relational database.

---

### Privacy and Query Safety

Administrative queries undergo strict validation. Sensitive fields and unsafe operations are blocked before SQL execution to prevent misuse or data leaks.

---

## Project Structure

```
├── app/
│   ├── main.py
│   ├── api/
│   ├── core/
│   ├── graph/
│   ├── memory/
│   ├── data/
│   ├── ingestion/
│   ├── vision/
│   ├── web/
│   └── utils/
│
├── data/
│   ├── admin_docs/
│   └── chroma/
│
├── chroma/
├── requirements.txt
└── README.md
```

---

## Technology Stack

* Framework: FastAPI
* Workflow Engine: LangGraph
* LLM Provider: Groq (LLaMA models)
* Vision Model: Multimodal LLaMA Vision
* Vector Database: Chroma
* Embeddings: HuggingFace (all-MiniLM-L6-v2)
* Relational Database: PostgreSQL (async)
* Memory Store: MongoDB
* Image Processing: Cloudinary
* Web Search: Tavily
* Document Parsing: PyMuPDF

---

## Performance Considerations

* Modality-aware routing minimizes unnecessary computation
* Vector retrieval reduces LLM token usage
* Asynchronous database operations improve throughput
* Background ingestion prevents API blocking



