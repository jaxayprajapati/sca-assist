"""
MongoDB Vector Store Service module.

This module provides a centralized MongoDB service for vector storage operations
using pymongo with connection pooling, vector search, and common CRUD operations.
Designed for storing document embeddings and performing similarity searches.
"""

from typing import Any, Dict, List, Optional

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.operations import SearchIndexModel

from app.config.settings import settings
from app.utils.logging import logger


class MongoDBService:
    """Service class for MongoDB vector store operations."""

    # Default collection for vector storage
    VECTOR_COLLECTION = "pdf-documents"

    # Vector index configuration
    VECTOR_INDEX_NAME = "vector_index"
    EMBEDDING_FIELD = "embedding"
    EMBEDDING_DIMENSIONS = 3072  # OpenAI text-embedding-3-large dimensions

    def __init__(
        self,
        host: str = None,
        port: int = None,
        database: str = "sca_assist"
    ):
        """
        Initialize the MongoDB vector store service.

        Args:
            host: MongoDB host (default: from settings or localhost)
            port: MongoDB port (default: from settings or 27017)
            database: Database name (default: sca_assist)
        """
        # Use MONGODB_URI from settings if available
        mongodb_uri = getattr(settings, "MONGODB_URI", None)
        if mongodb_uri:
            self.uri = mongodb_uri
            self.database_name = database
            self.client: MongoClient = MongoClient(self.uri)
            self.db: Database = self.client[database]
            logger.info(f"MongoDB vector store service initialized: {self.uri}/{database}")
        else:
            self.host = host or "localhost"
            self.port = port or 27017
            self.database_name = database
            self.uri = f"mongodb://{self.host}:{self.port}"
            self.client: MongoClient = MongoClient(self.uri)
            self.db: Database = self.client[database]
            logger.info(f"MongoDB vector store service initialized: {self.uri}/{database}")

    def get_collection(self, collection_name: str) -> Collection:
        """
        Get a collection by name.

        Args:
            collection_name: Name of the collection

        Returns:
            Collection object
        """
        return self.db[collection_name]

    def insert_one(self, collection_name: str, document: Dict[str, Any]) -> str:
        """
        Insert a single document.

        Args:
            collection_name: Name of the collection
            document: Document to insert

        Returns:
            Inserted document ID as string
        """
        collection = self.get_collection(collection_name)
        result = collection.insert_one(document)
        logger.debug(f"Inserted document into {collection_name}: {result.inserted_id}")

        return str(result.inserted_id)

    def insert_many(self, collection_name: str, documents: List[Dict[str, Any]]) -> List[str]:
        """
        Insert multiple documents.

        Args:
            collection_name: Name of the collection
            documents: List of documents to insert

        Returns:
            List of inserted document IDs as strings
        """
        collection = self.get_collection(collection_name)
        result = collection.insert_many(documents)
        logger.debug(f"Inserted {len(result.inserted_ids)} documents into {collection_name}")

        return [str(id) for id in result.inserted_ids]

    def find_one(
        self,
        collection_name: str,
        query: Dict[str, Any],
        projection: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Find a single document.

        Args:
            collection_name: Name of the collection
            query: Query filter
            projection: Fields to include/exclude (optional)

        Returns:
            Document if found, None otherwise
        """
        collection = self.get_collection(collection_name)
        result = collection.find_one(query, projection)
        logger.debug(f"Find one in {collection_name}: {'found' if result else 'not found'}")

        return result

    def find_many(
        self,
        collection_name: str,
        query: Dict[str, Any] = None,
        projection: Optional[Dict[str, Any]] = None,
        limit: int = 0,
        skip: int = 0,
        sort: Optional[List[tuple]] = None
    ) -> List[Dict[str, Any]]:
        """
        Find multiple documents.

        Args:
            collection_name: Name of the collection
            query: Query filter (default: {} for all)
            projection: Fields to include/exclude (optional)
            limit: Maximum number of documents (default: 0 for no limit)
            skip: Number of documents to skip (default: 0)
            sort: List of (field, direction) tuples for sorting

        Returns:
            List of documents
        """
        if query is None:
            query = {}

        collection = self.get_collection(collection_name)
        cursor = collection.find(query, projection)

        if sort:
            cursor = cursor.sort(sort)
        if skip:
            cursor = cursor.skip(skip)
        if limit:
            cursor = cursor.limit(limit)

        results = list(cursor)
        logger.debug(f"Find many in {collection_name}: {len(results)} documents")

        return results

    def update_one(
        self,
        collection_name: str,
        query: Dict[str, Any],
        update: Dict[str, Any],
        upsert: bool = False
    ) -> int:
        """
        Update a single document.

        Args:
            collection_name: Name of the collection
            query: Query filter
            update: Update operations
            upsert: Insert if not found (default: False)

        Returns:
            Number of modified documents
        """
        collection = self.get_collection(collection_name)
        result = collection.update_one(query, {"$set": update}, upsert=upsert)
        logger.debug(f"Updated {result.modified_count} document(s) in {collection_name}")

        return result.modified_count

    def update_many(
        self,
        collection_name: str,
        query: Dict[str, Any],
        update: Dict[str, Any]
    ) -> int:
        """
        Update multiple documents.

        Args:
            collection_name: Name of the collection
            query: Query filter
            update: Update operations

        Returns:
            Number of modified documents
        """
        collection = self.get_collection(collection_name)
        result = collection.update_many(query, {"$set": update})
        logger.debug(f"Updated {result.modified_count} documents in {collection_name}")

        return result.modified_count

    def delete_one(self, collection_name: str, query: Dict[str, Any]) -> int:
        """
        Delete a single document.

        Args:
            collection_name: Name of the collection
            query: Query filter

        Returns:
            Number of deleted documents
        """
        collection = self.get_collection(collection_name)
        result = collection.delete_one(query)
        logger.debug(f"Deleted {result.deleted_count} document from {collection_name}")

        return result.deleted_count

    def delete_many(self, collection_name: str, query: Dict[str, Any]) -> int:
        """
        Delete multiple documents.

        Args:
            collection_name: Name of the collection
            query: Query filter

        Returns:
            Number of deleted documents
        """
        collection = self.get_collection(collection_name)
        result = collection.delete_many(query)
        logger.debug(f"Deleted {result.deleted_count} documents from {collection_name}")

        return result.deleted_count

    def count_documents(self, collection_name: str, query: Dict[str, Any] = None) -> int:
        """
        Count documents matching a query.

        Args:
            collection_name: Name of the collection
            query: Query filter (default: {} for all)

        Returns:
            Number of matching documents
        """
        if query is None:
            query = {}

        collection = self.get_collection(collection_name)
        count = collection.count_documents(query)

        return count

    def create_index(
        self,
        collection_name: str,
        keys: List[tuple],
        unique: bool = False
    ) -> str:
        """
        Create an index on a collection.

        Args:
            collection_name: Name of the collection
            keys: List of (field, direction) tuples
            unique: Whether the index is unique (default: False)

        Returns:
            Index name
        """
        collection = self.get_collection(collection_name)
        index_name = collection.create_index(keys, unique=unique)
        logger.info(f"Created index '{index_name}' on {collection_name}")

        return index_name

    def drop_collection(self, collection_name: str) -> None:
        """
        Drop a collection.

        Args:
            collection_name: Name of the collection to drop
        """
        self.db.drop_collection(collection_name)
        logger.info(f"Dropped collection: {collection_name}")

    def list_collections(self) -> List[str]:
        """
        List all collections in the database.

        Returns:
            List of collection names
        """
        return self.db.list_collection_names()

    def close(self) -> None:
        """Close the MongoDB connection."""
        self.client.close()
        logger.info("MongoDB connection closed")

    # ==================== Vector Store Methods ====================

    def store_document_with_embedding(
        self,
        text: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None,
        collection_name: str = None
    ) -> str:
        """
        Store a document with its embedding vector.

        Args:
            text: The document text content
            embedding: The embedding vector
            metadata: Additional metadata (optional)
            collection_name: Collection name (default: VECTOR_COLLECTION)

        Returns:
            Inserted document ID as string
        """
        if collection_name is None:
            collection_name = self.VECTOR_COLLECTION

        document = {
            "text": text,
            self.EMBEDDING_FIELD: embedding,
            "metadata": metadata or {}
        }

        return self.insert_one(collection_name, document)

    def store_documents_with_embeddings(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        collection_name: str = None
    ) -> List[str]:
        """
        Store multiple documents with their embedding vectors.

        Args:
            texts: List of document text contents
            embeddings: List of embedding vectors
            metadatas: List of metadata dicts (optional)
            collection_name: Collection name (default: VECTOR_COLLECTION)

        Returns:
            List of inserted document IDs as strings
        """
        if collection_name is None:
            collection_name = self.VECTOR_COLLECTION

        if metadatas is None:
            metadatas = [{}] * len(texts)

        documents = [
            {
                "text": text,
                self.EMBEDDING_FIELD: embedding,
                "metadata": metadata
            }
            for text, embedding, metadata in zip(texts, embeddings, metadatas)
        ]

        return self.insert_many(collection_name, documents)

    def create_vector_search_index(
        self,
        collection_name: str = None,
        index_name: str = None,
        dimensions: int = None,
        similarity: str = "cosine"
    ) -> str:
        """
        Create a vector search index for similarity search.

        Note: This requires MongoDB Atlas or a MongoDB instance with
        Atlas Search enabled. For local MongoDB, use manual similarity search.

        Args:
            collection_name: Collection name (default: VECTOR_COLLECTION)
            index_name: Index name (default: VECTOR_INDEX_NAME)
            dimensions: Vector dimensions (default: EMBEDDING_DIMENSIONS)
            similarity: Similarity metric - cosine, euclidean, dotProduct (default: cosine)

        Returns:
            Index name
        """
        if collection_name is None:
            collection_name = self.VECTOR_COLLECTION
        if index_name is None:
            index_name = self.VECTOR_INDEX_NAME
        if dimensions is None:
            dimensions = self.EMBEDDING_DIMENSIONS

        collection = self.get_collection(collection_name)

        # Vector search index definition
        search_index_model = SearchIndexModel(
            definition={
                "fields": [
                    {
                        "type": "vector",
                        "numDimensions": dimensions,
                        "path": self.EMBEDDING_FIELD,
                        "similarity": similarity
                    }
                ]
            },
            name=index_name,
            type="vectorSearch"
        )

        try:
            result = collection.create_search_index(model=search_index_model)
            logger.info(f"Created vector search index '{index_name}' on {collection_name}")
            return result
        except Exception as e:
            logger.warning(f"Could not create vector search index: {e}")
            logger.info("Using manual similarity search instead")
            return None

    def similarity_search(
        self,
        query_embedding: List[float],
        k: int = 5,
        collection_name: str = None,
        filter_query: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform similarity search using vector embeddings.

        Uses MongoDB Atlas Vector Search if available,
        otherwise falls back to manual cosine similarity.

        Args:
            query_embedding: The query embedding vector
            k: Number of results to return (default: 5)
            collection_name: Collection name (default: VECTOR_COLLECTION)
            filter_query: Additional filter criteria (optional)

        Returns:
            List of similar documents with scores
        """
        if collection_name is None:
            collection_name = self.VECTOR_COLLECTION

        collection = self.get_collection(collection_name)

        # Try Atlas Vector Search first
        try:
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": self.VECTOR_INDEX_NAME,
                        "path": self.EMBEDDING_FIELD,
                        "queryVector": query_embedding,
                        "numCandidates": k * 10,
                        "limit": k
                    }
                },
                {
                    "$project": {
                        "text": 1,
                        "metadata": 1,
                        "score": {"$meta": "vectorSearchScore"}
                    }
                }
            ]

            if filter_query:
                pipeline[0]["$vectorSearch"]["filter"] = filter_query

            results = list(collection.aggregate(pipeline))
            logger.debug(f"Vector search returned {len(results)} results")
            return results

        except Exception as e:
            logger.debug(f"Atlas Vector Search not available: {e}")
            # Fallback to manual similarity search
            return self._manual_similarity_search(
                query_embedding, k, collection_name, filter_query
            )

    def _manual_similarity_search(
        self,
        query_embedding: List[float],
        k: int,
        collection_name: str,
        filter_query: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Manual cosine similarity search (fallback for local MongoDB).

        Args:
            query_embedding: The query embedding vector
            k: Number of results to return
            collection_name: Collection name
            filter_query: Additional filter criteria (optional)

        Returns:
            List of similar documents with scores
        """
        import math

        collection = self.get_collection(collection_name)

        query = filter_query or {}
        documents = list(collection.find(query))

        if not documents:
            return []

        # Calculate cosine similarity for each document
        def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            norm1 = math.sqrt(sum(a * a for a in vec1))
            norm2 = math.sqrt(sum(b * b for b in vec2))
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return dot_product / (norm1 * norm2)

        scored_docs = []
        for doc in documents:
            if self.EMBEDDING_FIELD in doc:
                score = cosine_similarity(query_embedding, doc[self.EMBEDDING_FIELD])
                scored_docs.append({
                    "_id": doc["_id"],
                    "text": doc.get("text", ""),
                    "metadata": doc.get("metadata", {}),
                    "score": score
                })

        # Sort by score descending and return top k
        scored_docs.sort(key=lambda x: x["score"], reverse=True)
        results = scored_docs[:k]

        logger.debug(f"Manual similarity search returned {len(results)} results")
        return results

    def get_document_by_text(
        self,
        text: str,
        collection_name: str = None
    ) -> Optional[Dict[str, Any]]:
        """
        Find a document by its text content.

        Args:
            text: The text to search for
            collection_name: Collection name (default: VECTOR_COLLECTION)

        Returns:
            Document if found, None otherwise
        """
        if collection_name is None:
            collection_name = self.VECTOR_COLLECTION

        return self.find_one(collection_name, {"text": text})

    def delete_all_documents(self, collection_name: str = None) -> int:
        """
        Delete all documents from the vector collection.

        Args:
            collection_name: Collection name (default: VECTOR_COLLECTION)

        Returns:
            Number of deleted documents
        """
        if collection_name is None:
            collection_name = self.VECTOR_COLLECTION

        return self.delete_many(collection_name, {})


# Default service instance
mongodb_service = MongoDBService()
