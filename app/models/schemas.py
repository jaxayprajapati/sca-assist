"""
Pydantic schemas for API request/response models.
"""

from pydantic import BaseModel, Field
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


# ==================== Auth Schemas ====================

class TokenRequest(BaseModel):
    """Request model for issuing JWT access tokens."""
    subject: str = Field(..., min_length=1, max_length=256, description="User identifier or email")
    scopes: Optional[List[str]] = Field(default=None, description="Optional permission scopes")
    expires_minutes: Optional[int] = Field(default=None, ge=1, le=10080, description="Token expiry in minutes (max 7 days)")


class TokenResponse(BaseModel):
    """Response model for JWT access tokens."""
    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiry in seconds")
