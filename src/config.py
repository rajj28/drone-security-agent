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
    # OpenAI
    OPENAI_API_KEY: str = Field(..., env="OPENAI_API_KEY")
    OPENAI_EMBEDDING_MODEL: str = Field("text-embedding-3-large", env="OPENAI_EMBEDDING_MODEL")
    OPENAI_EMBEDDING_DIMENSION: int = Field(1024, env="OPENAI_EMBEDDING_DIMENSION")

    # Pinecone
    PINECONE_API_KEY: str = Field(..., env="PINECONE_API_KEY")
    PINECONE_INDEX_NAME: str = Field("drone-security-frames", env="PINECONE_INDEX_NAME")
    PINECONE_CLOUD: str = Field("aws", env="PINECONE_CLOUD")
    PINECONE_REGION: str = Field("us-east-1", env="PINECONE_REGION")
    PINECONE_DIMENSION: int = Field(1024, env="PINECONE_DIMENSION")
    PINECONE_METRIC: str = Field("cosine", env="PINECONE_METRIC")

    # LangChain (optional - only needed for tracing)
    LANGCHAIN_API_KEY: str = Field("", env="LANGCHAIN_API_KEY")
    LANGCHAIN_TRACING_V2: bool = Field(False, env="LANGCHAIN_TRACING_V2")
    LANGCHAIN_PROJECT: str = Field("drone-security-agent", env="LANGCHAIN_PROJECT")
    
    # MongoDB (optional - for context persistence)
    MONGODB_URI: str = Field("", env="MONGODB_URI")
    
    # Hugging Face (for Cloud Enhanced Analyzer)
    HF_API_TOKEN: str = Field("", env="HF_API_TOKEN")
    USE_CLOUD_ANALYZER: bool = Field(False, env="USE_CLOUD_ANALYZER")

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
