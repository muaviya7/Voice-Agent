from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
import os
from pathlib import Path

class Settings(BaseSettings):
    """Application settings"""
    
    # API Keys
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    google_api_key: str = Field(..., env="GOOGLE_API_KEY")
    kimi_api_key: str = Field(..., env="KIMI_API_KEY")
    deepseek_api_key: str = Field(..., env="DEEPSEEK_API_KEY")
    
    # Application
    app_name: str = Field(default="Virtuans Voice Agent", env="APP_NAME")
    app_env: str = Field(default="development", env="APP_ENV")
    
    # ChromaDB
    chroma_persist_directory: str = Field(default="./db/chromadb_store", env="CHROMA_PERSIST_DIRECTORY")
    chroma_collection_name: str = Field(default="sunmarke_content", env="CHROMA_COLLECTION_NAME")
    
    # Scraping
    target_website: str = Field(default="https://www.sunmarke.com/", env="TARGET_WEBSITE")
    max_pages: int = Field(default=100, env="MAX_PAGES")
    
    # RAG
    chunk_size: int = Field(default=1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, env="CHUNK_OVERLAP")
    top_k_results: int = Field(default=5, env="TOP_K_RESULTS")
    
    # Voice
    stt_model: str = Field(default="whisper-1", env="STT_MODEL")
    tts_voice: str = Field(default="alloy", env="TTS_VOICE")
    audio_sample_rate: int = Field(default=16000, env="AUDIO_SAMPLE_RATE")
    
    # Paths
    @property
    def base_dir(self) -> Path:
        return Path(__file__).parent.parent
    
    @property
    def data_dir(self) -> Path:
        return self.base_dir / "data"
    
    @property
    def scraped_content_dir(self) -> Path:
        return self.data_dir / "scraped_content"
    
    @property
    def audio_cache_dir(self) -> Path:
        return self.data_dir / "audio_cache"
    
    @property
    def logs_dir(self) -> Path:
        return self.base_dir / "logs"
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Global settings instance
settings = Settings()
