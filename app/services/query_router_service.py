"""
Query Router Service for smart query classification.

This module classifies queries to determine the appropriate handling:
- Greeting: Respond directly without RAG
- Document Query: Proceed with retrieval + reranking
- Off-topic: Reject early without wasting resources
- Clarification: Provide help information
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from app.services.openai_service import openai_service
from app.utils.logging import logger
from app.utils.prompt_loader import prompt_loader


class QueryType(str, Enum):
    """Types of queries the router can classify."""
    GREETING = "greeting"
    DOCUMENT_QUERY = "document_query"
    OFF_TOPIC = "off_topic"
    CLARIFICATION = "clarification"


class RouterResponse(BaseModel):
    """Structured response from the query router."""
    query_type: QueryType = Field(..., description="Classification of the query")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score (0-1)")
    reason: str = Field(..., description="Brief reason for the classification")


class QueryRouterService:
    """Service for classifying and routing queries."""

    def __init__(self):
        """Initialize the router service."""
        logger.info("Query router service initialized")

    def classify(self, query: str) -> RouterResponse:
        """
        Classify a query to determine how it should be handled.

        Args:
            query: The user's query

        Returns:
            RouterResponse with query_type, confidence, and reason
        """
        # Quick checks for obvious cases (save API calls)
        query_lower = query.strip().lower()

        # Check for empty or very short queries
        if len(query_lower) < 2:
            return RouterResponse(
                query_type=QueryType.CLARIFICATION,
                confidence=1.0,
                reason="Query too short"
            )

        # Check for common greetings (fast path)
        greetings = {"hi", "hello", "hey", "hola", "good morning", "good afternoon",
                     "good evening", "thanks", "thank you", "bye", "goodbye"}
        if query_lower in greetings or query_lower.rstrip("!.,") in greetings:
            return RouterResponse(
                query_type=QueryType.GREETING,
                confidence=1.0,
                reason="Common greeting detected"
            )

        # Use LLM for complex classification
        try:
            system_prompt = prompt_loader.get_prompt("router", "router", "system")
            user_prompt = prompt_loader.format_prompt(
                "router", "router", "user",
                query=query
            )

            model = prompt_loader.get_config("router", "router", "model", "gpt-4o-mini")
            temperature = prompt_loader.get_config("router", "router", "temperature", 0)

            response = openai_service.chat_structured(
                response_model=RouterResponse,
                user_message=user_prompt,
                system_message=system_prompt,
                model=model,
                temperature=temperature
            )

            logger.info(f"Query classified as: {response.query_type} (confidence: {response.confidence})")
            return response

        except Exception as e:
            logger.error(f"Router classification failed: {e}")
            # Default to document query on error (let the RAG handle it)
            return RouterResponse(
                query_type=QueryType.DOCUMENT_QUERY,
                confidence=0.5,
                reason="Classification failed, defaulting to document query"
            )

    def get_response(self, query_type: QueryType, document_count: int = 0) -> Optional[str]:
        """
        Get a canned response for non-document queries.

        Args:
            query_type: The type of query
            document_count: Number of documents in the system

        Returns:
            Response string or None if query should proceed to RAG
        """
        if query_type == QueryType.DOCUMENT_QUERY:
            # Check if there are documents
            if document_count == 0:
                return prompt_loader.get_prompt("router", "responses", "no_documents")
            return None  # Proceed to RAG

        # Get response from YAML
        try:
            return prompt_loader.get_prompt("router", "responses", query_type.value)
        except KeyError:
            logger.warning(f"No response template for query type: {query_type}")
            return None

    def should_proceed_to_rag(self, query: str, document_count: int = 0) -> tuple[bool, Optional[str]]:
        """
        Determine if a query should proceed to RAG or be handled directly.

        Args:
            query: The user's query
            document_count: Number of documents in the system

        Returns:
            Tuple of (should_proceed, direct_response)
            - If should_proceed is True, direct_response is None
            - If should_proceed is False, direct_response contains the response
        """
        # Classify the query
        classification = self.classify(query)

        # Get response if not a document query
        response = self.get_response(classification.query_type, document_count)

        if response is not None:
            return False, response

        return True, None


# Default service instance
query_router_service = QueryRouterService()
