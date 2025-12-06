"""
Reranker Service using OpenAI for improving search precision.

This module provides reranking functionality using OpenAI's GPT model
to score document relevance with structured output.
"""

from typing import List, Dict, Any

from pydantic import BaseModel, Field

from app.services.openai_service import openai_service
from app.utils.logging import logger
from app.utils.prompt_loader import prompt_loader


# ==================== Pydantic Models for Structured Output ====================

class DocumentScore(BaseModel):
    """Model for a single document relevance score."""
    index: int = Field(..., description="Document index (0-based)")
    score: float = Field(..., ge=0, le=10, description="Relevance score (0-10)")


class RerankerResponse(BaseModel):
    """Structured response from the reranker LLM."""
    scores: List[DocumentScore] = Field(..., description="List of document scores")


class RerankerService:
    """Service for reranking search results using OpenAI structured output."""

    def __init__(self):
        """Initialize the reranker service."""
        logger.info("Reranker service initialized (using OpenAI structured output)")

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents based on relevance to query using OpenAI structured output.

        Args:
            query: The search query
            documents: List of documents with 'text' field
            top_k: Number of top results to return

        Returns:
            Reranked list of documents with updated scores
        """
        if not documents:
            return []

        if len(documents) <= top_k:
            return documents

        try:
            # Build document list for scoring
            doc_list = "\n".join([
                f"[Doc {i}]: {doc.get('text', '')[:500]}"
                for i, doc in enumerate(documents)
            ])

            # Load prompts from YAML config
            system_prompt = prompt_loader.get_prompt("reranker", "reranker", "system")
            user_prompt = prompt_loader.format_prompt(
                "reranker", "reranker", "user",
                query=query,
                doc_list=doc_list
            )

            # Get model config from YAML
            model = prompt_loader.get_config("reranker", "reranker", "model", "gpt-4o-mini")
            temperature = prompt_loader.get_config("reranker", "reranker", "temperature", 0)

            # Get structured response from OpenAI
            response = openai_service.chat_structured(
                response_model=RerankerResponse,
                user_message=user_prompt,
                system_message=system_prompt,
                model=model,
                temperature=temperature
            )

            # Convert to scores dict
            scores: Dict[int, float] = {}
            for doc_score in response.scores:
                if 0 <= doc_score.index < len(documents):
                    scores[doc_score.index] = doc_score.score

            logger.info(f"Received {len(scores)} document scores from LLM")

            # Add scores to documents
            scored_docs = []
            for i, doc in enumerate(documents):
                doc_copy = doc.copy()
                doc_copy["original_score"] = doc_copy.get("score", 0)
                doc_copy["rerank_score"] = scores.get(i, 0)
                doc_copy["score"] = scores.get(i, 0)
                scored_docs.append(doc_copy)

            # Sort by rerank score (descending)
            scored_docs.sort(key=lambda x: x["rerank_score"], reverse=True)

            logger.info(f"Reranked {len(documents)} documents, returning top {top_k}")
            return scored_docs[:top_k]

        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return documents[:top_k]


# Default service instance
reranker_service = RerankerService()
