"""
FastAPI Application Entry Point for SCA Assist.

This is the main entry point for the application.
Run with: python -m app.main or uvicorn app.main:app --reload
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import (
    health_router,
    ingest_router,
    qa_router,
    search_router,
    documents_router
)
from app.utils.logging import logger


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application instance
    """
    app = FastAPI(
        title="SCA Assist API",
        description="API for document ingestion and RAG-based question answering",
        version="1.0.0"
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(health_router)
    app.include_router(ingest_router)
    app.include_router(qa_router)
    app.include_router(search_router)
    app.include_router(documents_router)

    logger.info("SCA Assist API initialized")

    return app


# Create the app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
