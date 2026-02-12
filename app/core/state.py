from typing import TypedDict, List, Optional, Dict, Any, Literal
from app.core.admin_schema import SuggestionType

class GraphState(TypedDict, total=False):
    query: str
    intent: str

    user_id: Optional[str]
    thread_id: Optional[str]

    products: List[Dict[str, Any]]  
    reviews: List[Dict[str, Any]]
    ratings: Dict[str, Any]
    faqs: List[Dict[str, Any]]
    policy: Dict[str, Any]

    rag_context: str

    context: str
    answer: str
    confidence: float

    resolved_query: Optional[str]
    session_context: Optional[Dict[str, Any]]

    image_description: Optional[str]


class OrchestratorState(TypedDict, total=False):
    query: str | None
    image_url: str | None

    user_id: Optional[str]
    thread_id: Optional[str]

    modality: str
    products: Optional[List[Dict[str, Any]]]  
    answer: str
    confidence: float

class ImageGraphState(TypedDict, total=False):
    image_url: str
    question: str

    product_description: str
    normalized_query: str

    product_docs: List[Dict[str, Any]]
    review_docs: List[Dict[str, Any]]
    web_docs: Optional[List[str]]

    context: str
    answer: str
    confidence: float


class AdminState(TypedDict, total=False):
    query: str
    
    admin_id: Optional[str]
    thread_id: Optional[str]
    
    intent: str  
    
    privacy_violation: Optional[str]
    is_safe: bool
    
    generated_sql: Optional[str]
    sql_validated: bool
    sql_error: Optional[str]
    db_result: Optional[List[Dict[str, Any]]]
    
    documents: List[Any]
    rag_context: Optional[str]
    tags: List[str]     
    
    context: str
    answer: str
    confidence: float
    
    retry_count: int
    
