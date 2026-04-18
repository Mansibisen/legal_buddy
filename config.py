"""
Configuration for Multi-Modal Legal System
"""
import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings"""
    
    # API Keys
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    
    # Vector Store Configuration
    CHROMA_DB_PATH: str = "./chroma_db_legal"
    CHROMA_COLLECTION_NAME: str = "legal_documents"
    EMBEDDING_MODEL: str = "text-embedding-3-small"  # OpenAI embedding model
    
    # Document Processing
    CHUNK_SIZE: int = 1024  # Characters per chunk
    CHUNK_OVERLAP: int = 200  # Overlap between chunks
    SECTION_AWARE_CHUNKING: bool = True  # Use semantic chunking
    
    # PDF Processing
    PDF_EXTRACTION_METHOD: str = "unstructured"  # Options: "marker", "unstructured", "pypdf"
    MAX_PAGES: int = 500  # Max pages to process per document
    EXTRACT_IMAGES: bool = True  # Extract and process images
    
    # Vision Model Configuration
    VISION_MODEL: str = "gpt-4o-mini"  # OpenAI vision model
    VISION_ENABLED: bool = True
    IMAGE_COMPRESSION_QUALITY: int = 85
    IMAGE_MAX_DIMENSION: int = 2048
    
    # Data Cleaning
    REMOVE_BOILERPLATE: bool = True
    REMOVE_PAGE_NUMBERS: bool = True
    NORMALIZE_WHITESPACE: bool = True
    LANGUAGE: str = "en"  # Primary language of documents
    
    # Dataset
    DATASET_NAME: str = "pile-of-law"
    DATASET_CACHE_DIR: str = "./data/cache"
    RAW_DATA_DIR: str = "./data/raw"
    PROCESSED_DATA_DIR: str = "./data/processed"
    
    # Batch Processing
    BATCH_SIZE: int = 10
    MAX_WORKERS: int = 4
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "./logs/legal_system.log"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Global settings instance
settings = Settings()

# Create necessary directories
Path(settings.DATASET_CACHE_DIR).mkdir(parents=True, exist_ok=True)
Path(settings.RAW_DATA_DIR).mkdir(parents=True, exist_ok=True)
Path(settings.PROCESSED_DATA_DIR).mkdir(parents=True, exist_ok=True)
Path(settings.CHROMA_DB_PATH).mkdir(parents=True, exist_ok=True)
Path("./logs").mkdir(parents=True, exist_ok=True)
