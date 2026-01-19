"""
Configuration Management for Multilingual RAG System

This module handles all configuration settings loaded from environment variables.
It uses Pydantic for validation and type safety.

Usage:
    from app.config import settings
    
    api_key = settings.OPENROUTER_API_KEY
    model = settings.MODEL_ANALYZER
"""

from pydantic_settings import BaseSettings
from pydantic import Field, validator, AnyHttpUrl
from typing import List, Optional, Dict, Any
from pathlib import Path
import os


class Settings(BaseSettings):
    """
    Application settings with validation.
    All settings are loaded from environment variables or .env file.
    """
    
    # =============================================================================
    # APPLICATION SETTINGS
    # =============================================================================
    APP_NAME: str = "Multilingual RAG System"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"
    
    # API Configuration
    API_V1_PREFIX: str = "/api/v1"
    RATE_LIMIT_PER_MINUTE: int = 60
    
    # CORS Configuration
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8000"]
    
    @validator("CORS_ORIGINS", pre=True)
    def parse_cors_origins(cls, v):
        """Parse comma-separated CORS origins"""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    # =============================================================================
    # OPENROUTER CONFIGURATION (LLM Gateway)
    # =============================================================================
    OPENROUTER_API_KEY: str = Field(..., description="OpenRouter API key")
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    
    # Model Selection (Can be changed without code changes!)
    MODEL_ROUTER: str = "google/gemini-2.0-flash-exp:free"
    MODEL_PLANNER: str = "google/gemini-2.0-flash-exp:free"
    MODEL_ANALYZER: str = "anthropic/claude-3.5-sonnet"
    MODEL_SYNTHESIZER: str = "anthropic/claude-3.5-sonnet"
    MODEL_VALIDATOR: str = "google/gemini-flash-1.5"
    
    # Model Temperatures
    TEMP_ROUTER: float = 0.1
    TEMP_PLANNER: float = 0.7
    TEMP_ANALYZER: float = 0.3
    TEMP_SYNTHESIZER: float = 0.2
    TEMP_VALIDATOR: float = 0.1
    
    # Model Configuration Helper
    @property
    def MODEL_CONFIGS(self) -> Dict[str, Dict[str, Any]]:
        """Get all model configurations"""
        return {
            "router": {
                "model": self.MODEL_ROUTER,
                "temperature": self.TEMP_ROUTER,
                "max_tokens": 500,
            },
            "planner": {
                "model": self.MODEL_PLANNER,
                "temperature": self.TEMP_PLANNER,
                "max_tokens": 1000,
            },
            "analyzer": {
                "model": self.MODEL_ANALYZER,
                "temperature": self.TEMP_ANALYZER,
                "max_tokens": 4000,
            },
            "synthesizer": {
                "model": self.MODEL_SYNTHESIZER,
                "temperature": self.TEMP_SYNTHESIZER,
                "max_tokens": 2000,
            },
            "validator": {
                "model": self.MODEL_VALIDATOR,
                "temperature": self.TEMP_VALIDATOR,
                "max_tokens": 1000,
            },
        }
    
    # =============================================================================
    # QDRANT CLOUD (Vector Database)
    # =============================================================================
    QDRANT_URL: str = Field(..., description="Qdrant Cloud URL")
    QDRANT_API_KEY: Optional[str] = Field(None, description="Qdrant API key")
    QDRANT_COLLECTION_NAME: str = "multilingual_docs"
    QDRANT_TIMEOUT: int = 60
    
    # =============================================================================
    # POSTGRESQL (Supabase)
    # =============================================================================
    DATABASE_URL: str = Field(..., description="PostgreSQL connection string")
    DATABASE_POOL_SIZE: int = 5
    DATABASE_MAX_OVERFLOW: int = 10
    
    # =============================================================================
    # REDIS (Upstash - Optional)
    # =============================================================================
    REDIS_URL: Optional[str] = Field(None, description="Redis connection string")
    REDIS_ENABLED: bool = False
    
    @validator("REDIS_ENABLED", pre=True, always=True)
    def set_redis_enabled(cls, v, values):
        """Enable Redis only if URL is provided"""
        return bool(values.get("REDIS_URL"))
    
    # =============================================================================
    # EMBEDDING CONFIGURATION
    # =============================================================================
    EMBEDDING_MODEL: str = "intfloat/multilingual-e5-large"
    EMBEDDING_DIMENSION: int = 1024
    EMBEDDING_BATCH_SIZE: int = 32
    EMBEDDING_DEVICE: str = "cpu"  # "cpu" or "cuda"
    
    @validator("EMBEDDING_DEVICE")
    def validate_device(cls, v):
        """Validate device is cpu or cuda"""
        if v not in ["cpu", "cuda"]:
            raise ValueError("EMBEDDING_DEVICE must be 'cpu' or 'cuda'")
        return v
    
    # =============================================================================
    # DOCUMENT PROCESSING
    # =============================================================================
    MAX_UPLOAD_SIZE_MB: int = 50
    ALLOWED_EXTENSIONS: List[str] = ["pdf", "docx", "txt", "csv", "md"]
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 128
    
    @validator("ALLOWED_EXTENSIONS", pre=True)
    def parse_extensions(cls, v):
        """Parse comma-separated extensions"""
        if isinstance(v, str):
            return [ext.strip() for ext in v.split(",")]
        return v
    
    @property
    def MAX_UPLOAD_SIZE_BYTES(self) -> int:
        """Convert MB to bytes"""
        return self.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    
    # =============================================================================
    # RETRIEVAL SETTINGS
    # =============================================================================
    TOP_K_RETRIEVAL: int = 25
    TOP_K_RERANK: int = 5
    MIN_SIMILARITY_SCORE: float = 0.7
    
    @validator("MIN_SIMILARITY_SCORE")
    def validate_similarity_score(cls, v):
        """Validate score is between 0 and 1"""
        if not 0 <= v <= 1:
            raise ValueError("MIN_SIMILARITY_SCORE must be between 0 and 1")
        return v
    
    # =============================================================================
    # SECURITY
    # =============================================================================
    JWT_SECRET_KEY: str = Field(..., description="Secret key for JWT")
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    @validator("JWT_SECRET_KEY")
    def validate_secret_key(cls, v):
        """Ensure secret key is strong enough"""
        if len(v) < 32:
            raise ValueError("JWT_SECRET_KEY must be at least 32 characters")
        return v
    
    # =============================================================================
    # SUPPORTED LANGUAGES
    # =============================================================================
    SUPPORTED_LANGUAGES: List[str] = [
        "en", "hi", "bn", "te", "mr", "ta", "ur", "gu", 
        "kn", "ml", "or", "pa", "as", "mai", "sa", "ks",
        "ne", "sd", "kok", "doi", "mni", "sat", "bo"
    ]
    
    @validator("SUPPORTED_LANGUAGES", pre=True)
    def parse_languages(cls, v):
        """Parse comma-separated languages"""
        if isinstance(v, str):
            return [lang.strip() for lang in v.split(",")]
        return v
    
    # Language name mapping
    @property
    def LANGUAGE_NAMES(self) -> Dict[str, str]:
        """Map language codes to full names"""
        return {
            "en": "English", "hi": "Hindi", "bn": "Bengali",
            "te": "Telugu", "mr": "Marathi", "ta": "Tamil",
            "ur": "Urdu", "gu": "Gujarati", "kn": "Kannada",
            "ml": "Malayalam", "or": "Odia", "pa": "Punjabi",
            "as": "Assamese", "mai": "Maithili", "sa": "Sanskrit",
            "ks": "Kashmiri", "ne": "Nepali", "sd": "Sindhi",
            "kok": "Konkani", "doi": "Dogri", "mni": "Manipuri",
            "sat": "Santali", "bo": "Bodo"
        }
    
    # =============================================================================
    # DIRECTORIES
    # =============================================================================
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    UPLOAD_DIR: Path = BASE_DIR / "uploads"
    LOGS_DIR: Path = BASE_DIR / "logs"
    DATA_DIR: Path = BASE_DIR / "data"
    
    @validator("UPLOAD_DIR", "LOGS_DIR", "DATA_DIR", pre=True, always=True)
    def create_directories(cls, v):
        """Create directories if they don't exist"""
        if isinstance(v, str):
            v = Path(v)
        v.mkdir(parents=True, exist_ok=True)
        return v
    
    # =============================================================================
    # MODEL CONFIGURATION
    # =============================================================================
    
    class Config:
        """Pydantic configuration"""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"  # Ignore extra fields


# =============================================================================
# GLOBAL SETTINGS INSTANCE
# =============================================================================

def get_settings() -> Settings:
    """
    Get application settings.
    This function can be used as a FastAPI dependency.
    
    Returns:
        Settings instance
    
    Example:
        from fastapi import Depends
        from app.config import get_settings, Settings
        
        @app.get("/")
        def root(settings: Settings = Depends(get_settings)):
            return {"app": settings.APP_NAME}
    """
    return Settings()


# Create global settings instance
settings = get_settings()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_model_config(agent_name: str) -> Dict[str, Any]:
    """
    Get LLM configuration for a specific agent.
    
    Args:
        agent_name: Name of agent (router, planner, analyzer, etc.)
    
    Returns:
        Dictionary with model, temperature, and other config
    
    Example:
        config = get_model_config("analyzer")
        # {'model': 'anthropic/claude-3.5-sonnet', 'temperature': 0.3, ...}
    """
    config = settings.MODEL_CONFIGS.get(agent_name)
    if not config:
        raise ValueError(f"Unknown agent: {agent_name}")
    
    return {
        **config,
        "api_key": settings.OPENROUTER_API_KEY,
        "base_url": settings.OPENROUTER_BASE_URL,
    }


def is_language_supported(lang_code: str) -> bool:
    """
    Check if a language is supported.
    
    Args:
        lang_code: ISO 639-1 language code (e.g., 'en', 'hi')
    
    Returns:
        True if language is supported
    """
    return lang_code in settings.SUPPORTED_LANGUAGES


def get_language_name(lang_code: str) -> str:
    """
    Get full language name from code.
    
    Args:
        lang_code: ISO 639-1 language code
    
    Returns:
        Full language name or 'Unknown'
    """
    return settings.LANGUAGE_NAMES.get(lang_code, "Unknown")


# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "detailed": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "json": {
            "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
            "format": "%(asctime)s %(name)s %(levelname)s %(message)s",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "detailed",
            "filename": settings.LOGS_DIR / "app.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
        },
        "error_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "detailed",
            "filename": settings.LOGS_DIR / "error.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "level": "ERROR",
        },
    },
    "root": {
        "level": settings.LOG_LEVEL,
        "handlers": ["console", "file", "error_file"],
    },
    "loggers": {
        "uvicorn": {"level": "INFO"},
        "fastapi": {"level": "INFO"},
        "qdrant_client": {"level": "WARNING"},
    },
}


# =============================================================================
# VALIDATION ON IMPORT
# =============================================================================

if __name__ == "__main__":
    # Test configuration
    print("=" * 80)
    print("CONFIGURATION VALIDATION")
    print("=" * 80)
    
    print(f"\n✅ App Name: {settings.APP_NAME}")
    print(f"✅ Version: {settings.APP_VERSION}")
    print(f"✅ Environment: {settings.ENVIRONMENT}")
    print(f"✅ Debug Mode: {settings.DEBUG}")
    
    print(f"\n🤖 LLM Models:")
    print(f"  - Router: {settings.MODEL_ROUTER}")
    print(f"  - Analyzer: {settings.MODEL_ANALYZER}")
    print(f"  - Synthesizer: {settings.MODEL_SYNTHESIZER}")
    
    print(f"\n🗄️  Databases:")
    print(f"  - Qdrant URL: {settings.QDRANT_URL}")
    print(f"  - PostgreSQL: Connected")
    print(f"  - Redis: {'Enabled' if settings.REDIS_ENABLED else 'Disabled'}")
    
    print(f"\n🌍 Languages: {len(settings.SUPPORTED_LANGUAGES)} supported")
    print(f"  - {', '.join(settings.SUPPORTED_LANGUAGES[:10])}...")
    
    print(f"\n📁 Directories:")
    print(f"  - Base: {settings.BASE_DIR}")
    print(f"  - Uploads: {settings.UPLOAD_DIR}")
    print(f"  - Logs: {settings.LOGS_DIR}")
    
    print(f"\n🔧 Model Config Example:")
    config = get_model_config("analyzer")
    print(f"  - Model: {config['model']}")
    print(f"  - Temperature: {config['temperature']}")
    
    print("\n" + "=" * 80)
    print("✅ Configuration loaded successfully!")
    print("=" * 80)
