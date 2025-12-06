"""
Pydantic schemas for API request/response models.
"""

from pydantic import BaseModel
from typing import List, Optional, Any


# ==================== Health Schemas ====================

class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    mongodb: str
    openai: str


# ==================== Ingestion Schemas ====================

class IngestResponse(BaseModel):
    """Response model for document ingestion."""
    message: str
    chunks_created: int
    documents_stored: int


# ==================== Question Answering Schemas ====================

class QuestionRequest(BaseModel):
    """Request model for question answering."""
    question: str
    k: int = 5  # Number of documents to retrieve
    use_reranker: bool = True  # Use reranker for precision boost


class SourceInfo(BaseModel):
    """Source information for answers."""
    text: str
    page: Any
    source: Any
    score: float
    rerank_score: Optional[float] = None


class QuestionResponse(BaseModel):
    """Response model for question answering."""
    question: str
    answer: str
    sources: List[SourceInfo]
    routed: bool = False  # True if handled by router without RAG


# ==================== Search Schemas ====================

class SearchRequest(BaseModel):
    """Request model for similarity search."""
    query: str
    k: int = 5
    use_reranker: bool = True  # Use reranker for precision boost


class SearchResult(BaseModel):
    """Single search result."""
    text: str
    score: float
    rerank_score: Optional[float] = None
    metadata: dict


class SearchResponse(BaseModel):
    """Response model for similarity search."""
    query: str
    results: List[SearchResult]


# ==================== Document Schemas ====================

class DocumentCountResponse(BaseModel):
    """Response model for document count."""
    count: int


class DeleteResponse(BaseModel):
    """Response model for delete operations."""
    message: str
