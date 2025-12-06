"""
RAG (Retrieval-Augmented Generation) Service.

This module contains the business logic for document ingestion,
similarity search, and question answering.
"""

import os
import tempfile
from typing import List, Dict, Any, Tuple

from app.utils.logging import logger
from app.services.ingest_service import ingest_service
from app.services.openai_service import openai_service
from app.services.mongodb_service import mongodb_service
from app.services.reranker_service import reranker_service


class RAGService:
    """Service class for RAG operations."""

    def __init__(self):
        """Initialize the RAG service."""
        self.batch_size = 100

    def check_health(self) -> Dict[str, str]:
        """
        Check the health of all dependent services.

        Returns:
            Dictionary with service status
        """
        # Check MongoDB
        try:
            mongodb_service.client.admin.command('ping')
            mongodb_status = "connected"
        except Exception as e:
            mongodb_status = f"error: {str(e)}"

        # Check OpenAI
        try:
            openai_service.get_embedding("test")
            openai_status = "connected"
        except Exception as e:
            openai_status = f"error: {str(e)}"

        status = "healthy" if mongodb_status == "connected" and openai_status == "connected" else "degraded"

        return {
            "status": status,
            "mongodb": mongodb_status,
            "openai": openai_status
        }

    async def ingest_pdf(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """
        Ingest a PDF file: chunk, embed, and store in MongoDB.

        Args:
            file_content: The PDF file content as bytes
            filename: Original filename

        Returns:
            Dictionary with ingestion results
        """
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file_content)
            tmp_path = tmp_file.name

        try:
            logger.info(f"Ingesting PDF: {filename}")

            # Step 1: Load and chunk the PDF
            chunks = ingest_service.load_and_split_pdf(tmp_path)
            logger.info(f"Created {len(chunks)} chunks")

            # Step 2: Extract texts and metadata
            texts = [chunk.page_content for chunk in chunks]
            metadatas = [
                {
                    "source": filename,
                    "page": chunk.metadata.get("page", 0),
                    "chunk_index": i
                }
                for i, chunk in enumerate(chunks)
            ]

            # Step 3: Generate embeddings in batches
            all_embeddings = []
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]
                batch_embeddings = openai_service.get_embeddings(batch_texts)
                all_embeddings.extend(batch_embeddings)

            logger.info(f"Generated {len(all_embeddings)} embeddings")

            # Step 4: Store in MongoDB
            doc_ids = mongodb_service.store_documents_with_embeddings(
                texts=texts,
                embeddings=all_embeddings,
                metadatas=metadatas
            )
            logger.info(f"Stored {len(doc_ids)} documents")

            return {
                "message": f"Successfully ingested {filename}",
                "chunks_created": len(chunks),
                "documents_stored": len(doc_ids)
            }

        finally:
            # Cleanup temp file
            os.unlink(tmp_path)

    def ask_question(self, question: str, k: int = 5, use_reranker: bool = True) -> Dict[str, Any]:
        """
        Answer a question using RAG with optional reranking.

        Args:
            question: The question to answer
            k: Number of documents to retrieve
            use_reranker: Whether to use reranker for precision boost

        Returns:
            Dictionary with answer and sources
        """
        logger.info(f"Question: {question} (reranker: {use_reranker})")

        # Step 1: Generate embedding for the query
        query_embedding = openai_service.get_embedding(question)

        # Step 2: Retrieve relevant documents (fetch more if using reranker)
        fetch_k = 20 if use_reranker else k
        results = mongodb_service.similarity_search(
            query_embedding=query_embedding,
            k=fetch_k
        )

        if not results:
            return {
                "question": question,
                "answer": "No relevant documents found. Please ingest some documents first.",
                "sources": []
            }

        # Step 3: Apply reranking if enabled
        if use_reranker and results:
            logger.info(f"Reranking {len(results)} results...")
            results = reranker_service.rerank(
                query=question,
                documents=results,
                top_k=k
            )
        else:
            results = results[:k]

        # Step 4: Build context from retrieved documents
        context = "\n\n".join([
            f"[Page {r.get('metadata', {}).get('page', 'N/A')}]: {r.get('text', '')}"
            for r in results
        ])

        # Step 5: Generate answer using OpenAI
        system_prompt = """You are a helpful assistant that answers questions based on the provided context.
Use only the information from the context to answer the question.
If the context doesn't contain enough information to answer, say so.
Cite the page numbers when relevant."""

        user_prompt = f"""Context:
{context}

Question: {question}

Please provide a detailed answer based on the context above."""

        answer = openai_service.chat(
            user_message=user_prompt,
            system_message=system_prompt
        )

        # Prepare sources
        sources = [
            {
                "text": r.get("text", "")[:200] + "...",
                "page": r.get("metadata", {}).get("page", "N/A"),
                "source": r.get("metadata", {}).get("source", "N/A"),
                "score": r.get("original_score", r.get("score", 0)),
                "rerank_score": r.get("rerank_score")
            }
            for r in results
        ]

        logger.info("Answer generated successfully")

        return {
            "question": question,
            "answer": answer,
            "sources": sources
        }

    def similarity_search(self, query: str, k: int = 5, use_reranker: bool = True) -> Dict[str, Any]:
        """
        Perform similarity search on ingested documents with optional reranking.

        Args:
            query: The search query
            k: Number of results to return
            use_reranker: Whether to use reranker for precision boost

        Returns:
            Dictionary with search results
        """
        logger.info(f"Search query: {query} (reranker: {use_reranker})")

        # Generate embedding for the query
        query_embedding = openai_service.get_embedding(query)

        # Perform similarity search (fetch more if using reranker)
        fetch_k = 20 if use_reranker else k
        results = mongodb_service.similarity_search(
            query_embedding=query_embedding,
            k=fetch_k
        )

        # Apply reranking if enabled
        if use_reranker and results:
            logger.info(f"Reranking {len(results)} results...")
            results = reranker_service.rerank(
                query=query,
                documents=results,
                top_k=k
            )
        else:
            results = results[:k]

        search_results = [
            {
                "text": r.get("text", ""),
                "score": r.get("original_score", r.get("score", 0)),
                "rerank_score": r.get("rerank_score"),
                "metadata": r.get("metadata", {})
            }
            for r in results
        ]

        return {
            "query": query,
            "results": search_results
        }

    def get_document_count(self) -> int:
        """
        Get the total number of documents in the vector store.

        Returns:
            Document count
        """
        return mongodb_service.count_documents(mongodb_service.VECTOR_COLLECTION)

    def delete_all_documents(self) -> int:
        """
        Delete all documents from the vector store.

        Returns:
            Number of deleted documents
        """
        deleted = mongodb_service.delete_all_documents()
        logger.info(f"Deleted {deleted} documents")
        return deleted


# Default service instance
rag_service = RAGService()
