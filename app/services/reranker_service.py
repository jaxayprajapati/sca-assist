"""
Reranker Service using OpenAI for improving search precision.

This module provides reranking functionality using OpenAI's GPT model
to score document relevance.
"""

import json
from typing import List, Dict, Any

from app.services.openai_service import openai_service
from app.utils.logging import logger


class RerankerService:
    """Service for reranking search results using OpenAI."""

    def __init__(self):
        """Initialize the reranker service."""
        logger.info("Reranker service initialized (using OpenAI)")

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents based on relevance to query using OpenAI.

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
                f"[Doc {i}]: {doc.get('text', '')[:500]}"  # Limit text length
                for i, doc in enumerate(documents)
            ])

            system_prompt = """You are a relevance scoring expert.
Given a query and a list of documents, score each document's relevance to the query.
Return ONLY a valid JSON array of objects with 'index' and 'score' (0-10 scale).
Higher score = more relevant. Be strict - only highly relevant docs should score above 7.
Example output: [{"index": 0, "score": 8.5}, {"index": 1, "score": 3.2}]"""

            user_prompt = f"""Query: {query}

Documents:
{doc_list}

Score each document's relevance (0-10). Return JSON array only."""

            # Get scores from OpenAI
            response = openai_service.chat(
                user_message=user_prompt,
                system_message=system_prompt
            )

            # Parse JSON response
            scores = self._parse_scores(response, len(documents))

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

    def _parse_scores(self, response: str, num_docs: int) -> Dict[int, float]:
        """
        Parse scores from OpenAI response.

        Args:
            response: JSON string from OpenAI
            num_docs: Number of documents

        Returns:
            Dict mapping document index to score
        """
        scores = {}

        try:
            # Clean response - extract JSON array
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]

            # Parse JSON
            parsed = json.loads(response)

            for item in parsed:
                idx = item.get("index", -1)
                score = item.get("score", 0)
                if 0 <= idx < num_docs:
                    scores[idx] = float(score)

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse reranker response: {e}")
            # Fallback: assign decreasing scores based on original order
            for i in range(num_docs):
                scores[i] = num_docs - i

        return scores


# Default service instance
reranker_service = RerankerService()
