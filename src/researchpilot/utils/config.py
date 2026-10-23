import os

from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()  # Load environment variables from .env file


def get_tavily_client() -> TavilyClient:
    """Create and return a TavilyClient instance using the API key from environment variables."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY is not set in the environment variables.")
    return TavilyClient(api_key=api_key)


def get_gemini_api_key() -> str:
    """Retrieve the Gemini API key from environment variables."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set in the environment variables.")
    return api_key


CHARACTER_LIMIT = 1500
PAGE_CONTENT_LIMIT = 6000
