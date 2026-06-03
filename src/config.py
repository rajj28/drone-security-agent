"""
config.py — Centralized configuration and environment management for Drone Security Analyst Agent.

Loads all environment variables, constants, and directory paths using pydantic for validation.
"""

import os
from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings
from pydantic import Field, ValidationError
from dotenv import load_dotenv

# Load .env file
load_dotenv()

class Settings(BaseSettings):
    # Google Gemini (primary LLM / VLM)
    GEMINI_API_KEY: str = Field(..., env="GEMINI_API_KEY")
    GEMINI_MODEL: str = Field("gemini-2.5-pro", env="GEMINI_MODEL")
    GEMINI_FALLBACK_MODEL: str = Field("gemini-2.5-flash", env="GEMINI_FALLBACK_MODEL")
    # Production: try Flash first to avoid 4× Pro retries (~30s) on every 429
    GEMINI_PREFER_FLASH: bool = Field(True, env="GEMINI_PREFER_FLASH")
    # When true, never call Pro (avoids burning quota on fallback after Flash 429)
    GEMINI_FLASH_ONLY: bool = Field(False, env="GEMINI_FLASH_ONLY")
    API_MIN_INTERVAL_SEC: float = Field(12.0, env="API_MIN_INTERVAL_SEC")
    GEMINI_EMBEDDING_MODEL: str = Field("text-embedding-004", env="GEMINI_EMBEDDING_MODEL")
    GEMINI_EMBEDDING_DIMENSION: int = Field(768, env="GEMINI_EMBEDDING_DIMENSION")

    # Legacy OpenAI fields (optional — unused when Gemini is configured)
    OPENAI_API_KEY: str = Field("", env="OPENAI_API_KEY")
    OPENAI_EMBEDDING_MODEL: str = Field("text-embedding-004", env="OPENAI_EMBEDDING_MODEL")
    OPENAI_EMBEDDING_DIMENSION: int = Field(768, env="OPENAI_EMBEDDING_DIMENSION")
    
    # Agent LLM Provider (for Q&A - allows using free alternatives to Gemini)
    AGENT_LLM_PROVIDER: str = Field("gemini", env="AGENT_LLM_PROVIDER")  # gemini, groq, ollama, openai
    GROQ_API_KEY: str = Field("", env="GROQ_API_KEY")  # Free tier: 20 req/min, 1M tokens/day
    OLLAMA_BASE_URL: str = Field("http://localhost:11434", env="OLLAMA_BASE_URL")
    OLLAMA_MODEL: str = Field("llama3.1", env="OLLAMA_MODEL")

    # Pinecone (flytbase: llama-text-embed-v2 integrated inference, 768 dim)
    PINECONE_API_KEY: str = Field(..., env="PINECONE_API_KEY")
    PINECONE_INDEX_NAME: str = Field("flytbase", env="PINECONE_INDEX_NAME")
    PINECONE_CLOUD: str = Field("aws", env="PINECONE_CLOUD")
    PINECONE_REGION: str = Field("us-east-1", env="PINECONE_REGION")
    PINECONE_DIMENSION: int = Field(768, env="PINECONE_DIMENSION")
    PINECONE_METRIC: str = Field("cosine", env="PINECONE_METRIC")
    PINECONE_USE_INTEGRATED: bool = Field(True, env="PINECONE_USE_INTEGRATED")
    PINECONE_NAMESPACE: str = Field("drone-security", env="PINECONE_NAMESPACE")
    PINECONE_TEXT_FIELD: str = Field("text", env="PINECONE_TEXT_FIELD")
    PINECONE_HOST: str = Field("", env="PINECONE_HOST")

    # LangChain (optional - only needed for tracing)
    LANGCHAIN_API_KEY: str = Field("", env="LANGCHAIN_API_KEY")
    LANGCHAIN_TRACING_V2: bool = Field(False, env="LANGCHAIN_TRACING_V2")
    LANGCHAIN_PROJECT: str = Field("drone-security-agent", env="LANGCHAIN_PROJECT")
    
    # MongoDB (optional - for context persistence)
    MONGODB_URI: str = Field("", env="MONGODB_URI")
    
    # Hugging Face (for Cloud Enhanced Analyzer)
    HF_API_TOKEN: str = Field("", env="HF_API_TOKEN")
    USE_CLOUD_ANALYZER: bool = Field(True, env="USE_CLOUD_ANALYZER")

    # One video = one session (see session_bootstrap.py)
    SESSION_ID: str = Field("", env="SESSION_ID")
    USE_MONGO_CONTEXT: bool = Field(False, env="USE_MONGO_CONTEXT")

    # App Config
    DATA_DIR: Path = Field(Path("data"), env="DATA_DIR")
    FRAMES_DIR: Path = Field(Path("data/frames"), env="FRAMES_DIR")
    EXTRACTED_DIR: Path = Field(Path("data/extracted"), env="EXTRACTED_DIR")
    OUTPUTS_DIR: Path = Field(Path("outputs"), env="OUTPUTS_DIR")
    MAX_FRAMES: int = Field(25, env="MAX_FRAMES")
    VIDEO_FILE: Path = Field(Path("data/video.mp4"), env="VIDEO_FILE")
    VIDEO_DURATION_SECONDS: int = Field(3599, env="VIDEO_DURATION_SECONDS")
    VIDEO_FPS: int = Field(15, env="VIDEO_FPS")
    VIDEO_START_UNIX: int = Field(1675267200, env="VIDEO_START_UNIX")
    ALERT_THRESHOLD: float = Field(0.75, env="ALERT_THRESHOLD")

    # Output subdirs
    TELEMETRY_DIR: Path = Field(Path("outputs/telemetry"))
    ANALYSIS_DIR: Path = Field(Path("outputs/analysis"))
    ALERTS_DIR: Path = Field(Path("outputs/alerts"))
    INDEX_DIR: Path = Field(Path("outputs/index"))
    SESSION_DIR: Path = Field(Path("outputs/session"))

    # Locations
    LOCATIONS: List[str] = [
        "Main Gate", "Perimeter North", "Garage", "Back Entrance",
        "Restricted Zone", "Driveway", "Side Entrance"
    ]

    class Config:
        env_file = ".env"
        case_sensitive = False

# Singleton settings instance
try:
    settings = Settings()
except ValidationError as e:
    print(f"ERROR: Config validation error: {e}")
    raise

from src.session_bootstrap import apply_session_layout

apply_session_layout(settings.SESSION_ID or None)

# Ensure all output directories exist
for d in [
    settings.EXTRACTED_DIR,
    settings.OUTPUTS_DIR,
    settings.TELEMETRY_DIR,
    settings.ANALYSIS_DIR,
    settings.ALERTS_DIR,
    settings.INDEX_DIR,
    settings.SESSION_DIR,
    settings.INDEX_DIR / "query_results"
]:
    os.makedirs(d, exist_ok=True)
