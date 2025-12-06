"""
OpenAI Service module using LangChain.

This module provides a centralized OpenAI service for interacting with
OpenAI models through LangChain, supporting chat completions and embeddings.
"""

from typing import List, Optional, Type, TypeVar

from pydantic import BaseModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from app.config.settings import settings
from app.utils.logging import logger

# TypeVar for generic structured output
T = TypeVar("T", bound=BaseModel)


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

    def chat_structured(
        self,
        response_model: Type[T],
        user_message: str,
        system_message: Optional[str] = None,
        temperature: Optional[float] = None,
        model: Optional[str] = None
    ) -> T:
        """
        Send a chat message and get a structured response as a Pydantic model.

        Args:
            response_model: Pydantic model class for the response structure
            user_message: The user's message
            system_message: Optional system prompt
            temperature: Optional temperature override (default: 0 for consistency)
            model: Optional model override (default: uses service's model)

        Returns:
            Parsed response as the specified Pydantic model
        """
        # Create LLM with overrides if specified
        temp = temperature if temperature is not None else 0
        llm_model = model if model is not None else self.model
        llm = ChatOpenAI(
            api_key=settings.OPENAI_API_KEY,
            model=llm_model,
            temperature=temp,
            max_tokens=self.max_tokens
        )

        # Bind structured output to LLM
        structured_llm = llm.with_structured_output(response_model)

        # Build messages
        messages = []
        if system_message:
            messages.append(SystemMessage(content=system_message))
        messages.append(HumanMessage(content=user_message))

        logger.debug(f"Sending structured chat request to {llm_model}")
        response = structured_llm.invoke(messages)
        logger.debug(f"Structured response received: {response_model.__name__}")

        return response

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
