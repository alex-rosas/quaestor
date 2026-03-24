"""Central configuration for Quaestor.

All settings are loaded from environment variables (via a .env file).
Never hardcode values here — add a new field with a default or make it required.
"""

from enum import Enum
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProvider(str, Enum):
    GROQ = "groq"
    OLLAMA = "ollama"
    TOGETHER = "together"


class EmbeddingProvider(str, Enum):
    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- LLM ---
    llm_provider: LLMProvider = Field(
        default=LLMProvider.GROQ,
        description="Which LLM backend to use: groq | ollama | together",
    )
    groq_api_key: str = Field(default="", description="Groq API key")
    groq_model: str = Field(
        default="llama-3.1-70b-versatile",
        description="Groq model name",
    )
    together_api_key: str = Field(default="", description="Together.ai API key")
    together_model: str = Field(
        default="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        description="Together.ai model name",
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama server base URL",
    )
    ollama_llm_model: str = Field(
        default="llama3.1:8b",
        description="Ollama model name for LLM inference",
    )
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=1024, gt=0)

    # --- Embeddings ---
    embedding_provider: EmbeddingProvider = Field(
        default=EmbeddingProvider.OLLAMA,
        description="Which embedding backend to use: ollama | huggingface",
    )
    ollama_embedding_model: str = Field(
        default="nomic-embed-text",
        description="Ollama model name for embeddings",
    )
    huggingface_embedding_model: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="HuggingFace model name for embeddings (fallback)",
    )

    # --- Vector store ---
    chroma_persist_dir: Path = Field(
        default=Path(".chroma"),
        description="Directory where ChromaDB persists its data",
    )
    chroma_collection_name: str = Field(
        default="quaestor",
        description="ChromaDB collection name",
    )

    # --- Chunking ---
    chunk_size: int = Field(
        default=512,
        gt=0,
        description="Token count for fixed-size chunking (Phase 1 baseline)",
    )
    chunk_overlap: int = Field(
        default=50,
        ge=0,
        description="Token overlap between consecutive chunks",
    )

    # --- Retrieval ---
    retrieval_top_k: int = Field(
        default=5,
        gt=0,
        description="Number of chunks to retrieve per query",
    )

    # --- Data paths ---
    data_raw_dir: Path = Field(
        default=Path("data/raw"),
        description="Directory for raw source documents (git-ignored)",
    )
    data_processed_dir: Path = Field(
        default=Path("data/processed"),
        description="Directory for processed/chunked documents (git-ignored)",
    )

    # --- SEC EDGAR downloader ---
    sec_requester_name: str = Field(
        default="",
        description="Name sent in the SEC EDGAR User-Agent header",
    )
    sec_requester_email: str = Field(
        default="",
        description="Email sent in the SEC EDGAR User-Agent header",
    )

    @field_validator("chunk_overlap")
    @classmethod
    def overlap_less_than_chunk(cls, v: int, info: object) -> int:
        """Overlap must be smaller than the chunk size."""
        data = info.data if hasattr(info, "data") else {}
        chunk_size = data.get("chunk_size", 512)
        if v >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({v}) must be less than chunk_size ({chunk_size})"
            )
        return v


# Module-level singleton — import this everywhere.
settings = Settings()
