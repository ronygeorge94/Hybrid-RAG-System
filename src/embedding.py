"""
Embedding Model Utilities

This module provides utilities for initializing and configuring embedding models,
specifically supporting OpenAI embeddings for document vectorization in RAG systems.
"""

import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()


def get_embedding_model(model_name: str) -> OpenAIEmbeddings:
    """
    Initialize and return an OpenAI embeddings model instance.

    This function creates an authenticated OpenAIEmbeddings object using the
    provided model name and API key from environment variables.

    Args:
        model_name: The name of the OpenAI embedding model to use (e.g., 'text-embedding-ada-002').

    Returns:
        An initialized OpenAIEmbeddings instance ready for use.

    Raises:
        ValueError: If the OPENAI_API_KEY environment variable is not set.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")
    return OpenAIEmbeddings(model=model_name, openai_api_key=api_key)