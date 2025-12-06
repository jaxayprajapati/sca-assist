"""
Document Ingestion and Chunking Service.

This module provides document loading and recursive text splitting
functionality for processing documents into chunks suitable for
embedding and retrieval.
"""

from typing import List, Optional

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from app.utils.logging import logger


class IngestService:
    """Service class for document ingestion and chunking."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        separators: Optional[List[str]] = None
    ):
        """
        Initialize the ingest service.

        Args:
            chunk_size: Maximum size of each chunk (default: 1000)
            chunk_overlap: Overlap between chunks (default: 200)
            separators: List of separators for splitting (default: None uses standard separators)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Default separators for recursive splitting
        if separators is None:
            separators = ["\n\n", "\n", ". ", " ", ""]

        self.separators = separators

        # Initialize the text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len
        )

        logger.info(
            f"Ingest service initialized with chunk_size={chunk_size}, "
            f"chunk_overlap={chunk_overlap}"
        )

    def load_pdf(self, file_path: str) -> List[Document]:
        """
        Load a PDF file.

        Args:
            file_path: Path to the PDF file

        Returns:
            List of Document objects
        """
        logger.info(f"Loading PDF: {file_path}")
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} pages from PDF")

        return documents

    def load_text(self, file_path: str, encoding: str = "utf-8") -> List[Document]:
        """
        Load a text file.

        Args:
            file_path: Path to the text file
            encoding: File encoding (default: utf-8)

        Returns:
            List of Document objects
        """
        logger.info(f"Loading text file: {file_path}")
        loader = TextLoader(file_path, encoding=encoding)
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} documents from text file")

        return documents

    def load_directory(
        self,
        directory_path: str,
        glob_pattern: str = "**/*.pdf",
        show_progress: bool = True
    ) -> List[Document]:
        """
        Load all matching files from a directory.

        Args:
            directory_path: Path to the directory
            glob_pattern: Pattern to match files (default: **/*.pdf)
            show_progress: Show loading progress (default: True)

        Returns:
            List of Document objects
        """
        logger.info(f"Loading directory: {directory_path} with pattern: {glob_pattern}")

        # Determine loader class based on pattern
        if "*.pdf" in glob_pattern:
            loader_cls = PyPDFLoader
        else:
            loader_cls = TextLoader

        loader = DirectoryLoader(
            directory_path,
            glob=glob_pattern,
            loader_cls=loader_cls,
            show_progress=show_progress
        )
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} documents from directory")

        return documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks using recursive character splitting.

        Args:
            documents: List of Document objects to split

        Returns:
            List of chunked Document objects
        """
        logger.info(f"Splitting {len(documents)} documents into chunks")
        chunks = self.text_splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks")

        return chunks

    def split_text(self, text: str) -> List[str]:
        """
        Split a text string into chunks.

        Args:
            text: Text string to split

        Returns:
            List of text chunks
        """
        logger.debug("Splitting text into chunks")
        chunks = self.text_splitter.split_text(text)
        logger.debug(f"Created {len(chunks)} chunks from text")

        return chunks

    def load_and_split_pdf(self, file_path: str) -> List[Document]:
        """
        Load a PDF file and split it into chunks.

        Args:
            file_path: Path to the PDF file

        Returns:
            List of chunked Document objects
        """
        documents = self.load_pdf(file_path)
        chunks = self.split_documents(documents)

        return chunks

    def load_and_split_text(
        self,
        file_path: str,
        encoding: str = "utf-8"
    ) -> List[Document]:
        """
        Load a text file and split it into chunks.

        Args:
            file_path: Path to the text file
            encoding: File encoding (default: utf-8)

        Returns:
            List of chunked Document objects
        """
        documents = self.load_text(file_path, encoding)
        chunks = self.split_documents(documents)

        return chunks

    def load_and_split_directory(
        self,
        directory_path: str,
        glob_pattern: str = "**/*.pdf"
    ) -> List[Document]:
        """
        Load all files from a directory and split into chunks.

        Args:
            directory_path: Path to the directory
            glob_pattern: Pattern to match files (default: **/*.pdf)

        Returns:
            List of chunked Document objects
        """
        documents = self.load_directory(directory_path, glob_pattern)
        chunks = self.split_documents(documents)

        return chunks


# Default service instance
ingest_service = IngestService()
