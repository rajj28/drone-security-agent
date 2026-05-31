"""
production_config.py - Production configuration for security video analysis system
"""

import os
from typing import Optional
from pydantic import validator
from pydantic_settings import BaseSettings

class ProductionSettings(BaseSettings):
    """Production settings with validation"""
    
    # Database Configuration
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_NAME: str = "security_analysis"
    DB_USER: str = "security_user"
    DB_PASSWORD: str = "test_password"
    
    # Redis Configuration
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None
    
    # MinIO Configuration
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_SECURE: bool = False
    MINIO_BUCKET_NAME: str = "security-videos"
    
    # OpenAI Configuration
    OPENAI_API_KEY: str = "test_key"
    
    # Pinecone Configuration
    PINECONE_API_KEY: str = "test_key"
    PINECONE_ENV: str = "us-west1-gcp"
    PINECONE_INDEX_NAME: str = "video-frames"
    
    # Security Configuration
    JWT_SECRET: str = "test_jwt_secret"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 30
    
    # Email Configuration
    SMTP_HOST: str = "smtp.gmail.com"
    SMTP_PORT: int = 587
    SMTP_USER: str = "test@example.com"
    SMTP_PASSWORD: str = "test_password"
    SMTP_TLS: bool = True
    
    # Processing Configuration
    MAX_CONCURRENT_VIDEOS: int = 4
    FRAME_EXTRACTION_INTERVAL: int = 1  # 1 frame per second
    BATCH_SIZE: int = 10
    MAX_VIDEO_SIZE_MB: int = 1000  # 1GB
    
    # Storage Configuration
    STORAGE_RETENTION_DAYS: int = 30
    MAX_STORAGE_GB: int = 1000
    
    # Monitoring Configuration
    PROMETHEUS_PORT: int = 8000
    LOG_LEVEL: str = "INFO"
    
    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8080
    API_WORKERS: int = 4
    
    # Grafana Configuration
    GRAFANA_USER: str = "admin"
    GRAFANA_PASSWORD: str = "admin"
    
        
    class Config:
        env_file = ".env.production"
        case_sensitive = True

# Create production settings instance
production_settings = ProductionSettings()

# Database URLs
DATABASE_URL = f"postgresql://{production_settings.DB_USER}:{production_settings.DB_PASSWORD}@{production_settings.DB_HOST}:{production_settings.DB_PORT}/{production_settings.DB_NAME}"

# Redis URL
REDIS_URL = f"redis://{production_settings.REDIS_HOST}:{production_settings.REDIS_PORT}/{production_settings.REDIS_DB}"

# MinIO URLs
MINIO_VIDEO_URL = f"http://{production_settings.MINIO_ENDPOINT}/{production_settings.MINIO_BUCKET_NAME}"

# Logging Configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        },
        "detailed": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s",
        },
    },
    "handlers": {
        "default": {
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "formatter": "detailed",
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "logs/security_analysis.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
        },
    },
    "loggers": {
        "": {
            "level": production_settings.LOG_LEVEL,
            "handlers": ["default", "file"],
        },
        "uvicorn": {
            "level": "INFO",
            "handlers": ["default", "file"],
            "propagate": False,
        },
        "sqlalchemy.engine": {
            "level": "WARNING",
            "handlers": ["default", "file"],
            "propagate": False,
        },
    },
}

# Performance Tuning
PERFORMANCE_CONFIG = {
    "database_pool_size": 20,
    "database_max_overflow": 30,
    "redis_connection_pool_size": 50,
    "video_processing_workers": 4,
    "frame_analysis_workers": 8,
    "alert_processing_workers": 2,
}

# Security Configuration
SECURITY_CONFIG = {
    "cors_origins": ["*"],
    "cors_methods": ["GET", "POST", "PUT", "DELETE"],
    "cors_headers": ["*"],
    "rate_limit_per_minute": 100,
    "max_request_size_mb": 100,
}

# Storage Configuration
STORAGE_CONFIG = {
    "video_storage_path": "data/videos",
    "frame_storage_path": "data/frames",
    "temp_storage_path": "data/temp",
    "backup_storage_path": "data/backups",
    "compression_enabled": True,
    "encryption_enabled": False,
}
