"""
API Routes for SCA Assist.

This module contains all the API endpoint definitions.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File

from app.models.schemas import (
    HealthResponse,
    IngestResponse,
    QuestionRequest,
    QuestionResponse,
    SearchRequest,
    SearchResponse,
    DocumentCountResponse,
    DeleteResponse
)
from app.services.rag_service import rag_service


# ==================== Routers ====================

health_router = APIRouter(tags=["Health"])
ingest_router = APIRouter(prefix="/ingest", tags=["Ingestion"])
qa_router = APIRouter(tags=["Question Answering"])
search_router = APIRouter(tags=["Search"])
documents_router = APIRouter(prefix="/documents", tags=["Documents"])


# ==================== Health Endpoints ====================

@health_router.get("/")
async def root():
    """Root endpoint."""
    return {"message": "SCA Assist API is running"}


@health_router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check the health of all services."""
    result = rag_service.check_health()
    return HealthResponse(**result)


# ==================== Ingestion Endpoints ====================

@ingest_router.post("/pdf", response_model=IngestResponse)
async def ingest_pdf(file: UploadFile = File(...)):
    """
    Ingest a PDF file: chunk, embed, and store in MongoDB.
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    try:
        content = await file.read()
        result = await rag_service.ingest_pdf(content, file.filename)
        return IngestResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Question Answering Endpoints ====================

@qa_router.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """
    Ask a question and get an answer based on ingested documents (RAG).
    Uses smart query routing and optional reranking for improved precision.
    """
    try:
        result = rag_service.ask_question(
            request.question,
            request.k,
            request.use_reranker,
            use_router=True  # Always use smart routing
        )
        return QuestionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Search Endpoints ====================

@search_router.post("/search", response_model=SearchResponse)
async def similarity_search(request: SearchRequest):
    """
    Perform similarity search on ingested documents.
    Uses optional reranking for improved precision.
    """
    try:
        result = rag_service.similarity_search(
            request.query,
            request.k,
            request.use_reranker
        )
        return SearchResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Document Endpoints ====================

@documents_router.get("/count", response_model=DocumentCountResponse)
async def get_document_count():
    """Get the total number of documents in the vector store."""
    try:
        count = rag_service.get_document_count()
        return DocumentCountResponse(count=count)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@documents_router.delete("", response_model=DeleteResponse)
async def delete_all_documents():
    """Delete all documents from the vector store."""
    try:
        deleted = rag_service.delete_all_documents()
        return DeleteResponse(message=f"Deleted {deleted} documents")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
