import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

# Load environment variables (API Keys)
load_dotenv()

def get_llm(provider: str, model_name: str, temperature: float = 0.1):
    """
    Factory function to initialize a Chat Model based on the provider.
    
    Args:
        provider: The LLM provider ("openai", "anthropic", "google")
        model_name: The specific model ID (e.g., "gpt-4o-mini")
        temperature: Controls randomness (0 = deterministic, 1 = creative)
    """
    
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in .env")
        return ChatOpenAI(
            model=model_name, 
            temperature=temperature, 
            openai_api_key=api_key
        )

    elif provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in .env")
        return ChatAnthropic(
            model=model_name, 
            temperature=temperature, 
            anthropic_api_key=api_key
        )

    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")