"""
OpenAI Service module using LangChain.

This module provides a centralized OpenAI service for interacting with
OpenAI models through LangChain, supporting chat completions and embeddings.
"""

from typing import List, Optional

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from app.config.settings import settings
from app.utils.logging import logger


class OpenAIService:
    """Service class for OpenAI operations using LangChain."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ):
        """
        Initialize the OpenAI service.

        Args:
            model: The OpenAI model to use (default: gpt-4o-mini)
            temperature: Sampling temperature (0-2, default: 0.7)
            max_tokens: Maximum tokens in response (default: None)
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Initialize ChatOpenAI
        self.llm = ChatOpenAI(
            api_key=settings.OPENAI_API_KEY,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )

        # Initialize Embeddings (using latest OpenAI embedding model)
        self.embeddings = OpenAIEmbeddings(
            api_key=settings.OPENAI_API_KEY,
            model="text-embedding-3-large"
        )

        logger.info(f"OpenAI service initialized with model: {model}")

    def chat(
        self,
        user_message: str,
        system_message: Optional[str] = None
    ) -> str:
        """
        Send a chat message and get a response.

        Args:
            user_message: The user's message
            system_message: Optional system prompt

        Returns:
            The AI's response as a string
        """
        messages = []

        if system_message:
            messages.append(SystemMessage(content=system_message))

        messages.append(HumanMessage(content=user_message))

        logger.debug(f"Sending chat request to {self.model}")
        response = self.llm.invoke(messages)
        logger.debug("Chat response received")

        return response.content

    def chat_with_history(
        self,
        messages: List[dict],
        system_message: Optional[str] = None
    ) -> str:
        """
        Send a chat with conversation history.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            system_message: Optional system prompt

        Returns:
            The AI's response as a string
        """
        langchain_messages = []

        if system_message:
            langchain_messages.append(SystemMessage(content=system_message))

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "user":
                langchain_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                langchain_messages.append(AIMessage(content=content))
            elif role == "system":
                langchain_messages.append(SystemMessage(content=content))

        logger.debug(f"Sending chat request with {len(messages)} messages")
        response = self.llm.invoke(langchain_messages)
        logger.debug("Chat response received")

        return response.content

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        logger.debug(f"Generating embeddings for {len(texts)} texts")
        embeddings = self.embeddings.embed_documents(texts)
        logger.debug("Embeddings generated")

        return embeddings

    def get_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        logger.debug("Generating embedding for single text")
        embedding = self.embeddings.embed_query(text)
        logger.debug("Embedding generated")

        return embedding


# Default service instance
openai_service = OpenAIService()
