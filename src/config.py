import os
from typing import Optional
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv(override=True)


class AgentConfig:
    """Central configuration for Groq LLM, Tavily Search, Memory, and Context limits."""

    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "").strip("\"' ")
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "").strip("\"' ")
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY", "").strip("\"' ")

    # Groq Model settings
    DEFAULT_GROQ_MODEL: str = os.getenv("GROQ_MODEL_NAME", "openai/gpt-oss-120b")
    FALLBACK_GROQ_MODELS: list = [
        "openai/gpt-oss-120b",
        "qwen/qwen3.8-27b",
        "openai/gpt-oss-20b",
    ]

    TEMPERATURE: float = 0.0  # Zero temperature for deterministic, hallucination-free generation
    MAX_TOKENS: int = 4096

    # Tavily Search Settings
    MAX_SEARCH_RESULTS: int = int(os.getenv("MAX_SEARCH_RESULTS", 5))
    SEARCH_DEPTH: str = os.getenv("SEARCH_DEPTH", "advanced")  # "basic" or "advanced"

    # Context & Memory Management Settings
    MAX_CONTEXT_TOKENS: int = 6000  # Budget for search context passed to synthesis
    MAX_HISTORY_TURNS: int = 6  # Recent verbatim conversation turns preserved
    SUMMARY_THRESHOLD: int = 4  # Threshold after which older messages are summarized
    PERSISTENCE_DIR: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "sessions")

    @staticmethod
    def mask_secret(secret: Optional[str]) -> str:
        """Safely mask secret keys for logging/diagnostics (e.g., gsk_1234...abcd)."""
        if not secret:
            return "[Not Set]"
        if len(secret) <= 10:
            return "***"
        return f"{secret[:7]}...{secret[-4:]}"

    @classmethod
    def validate(cls) -> dict:
        """Validate required API keys and return status dict."""
        status = {
            "groq_configured": bool(cls.GROQ_API_KEY and cls.GROQ_API_KEY != "your_groq_api_key_here"),
            "groq_key_masked": cls.mask_secret(cls.GROQ_API_KEY) if cls.GROQ_API_KEY else "Missing",
            "tavily_configured": bool(cls.TAVILY_API_KEY and cls.TAVILY_API_KEY != "your_tavily_api_key_here"),
            "tavily_key_masked": cls.mask_secret(cls.TAVILY_API_KEY) if cls.TAVILY_API_KEY else "Missing",
            "model": cls.DEFAULT_GROQ_MODEL,
        }
        return status
