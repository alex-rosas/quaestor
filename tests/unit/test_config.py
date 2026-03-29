"""Unit tests for src/quaestor/config.py.

Tests are isolated from the local .env file and the OS environment by
instantiating Settings with _env_file=None and using monkeypatch to
scrub any relevant env vars that might bleed in from the shell.
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from quaestor.config import EmbeddingProvider, LLMProvider, Settings, settings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fresh_settings(monkeypatch: pytest.MonkeyPatch, **overrides) -> Settings:
    """Return a Settings instance that ignores .env and the real environment.

    All Quaestor-relevant env vars are deleted before instantiation so tests
    are fully hermetic. Any keyword arguments are forwarded as constructor
    overrides that take the highest precedence in pydantic-settings.
    """
    env_vars_to_clear = [
        "LLM_PROVIDER",
        "GROQ_API_KEY",
        "GROQ_MODEL",
        "TOGETHER_API_KEY",
        "TOGETHER_MODEL",
        "OLLAMA_BASE_URL",
        "OLLAMA_LLM_MODEL",
        "LLM_TEMPERATURE",
        "LLM_MAX_TOKENS",
        "EMBEDDING_PROVIDER",
        "OLLAMA_EMBEDDING_MODEL",
        "HUGGINGFACE_EMBEDDING_MODEL",
        "CHROMA_PERSIST_DIR",
        "CHROMA_COLLECTION_NAME",
        "CHUNK_SIZE",
        "CHUNK_OVERLAP",
        "RETRIEVAL_TOP_K",
        "DATA_RAW_DIR",
        "DATA_PROCESSED_DIR",
        "SEC_REQUESTER_NAME",
        "SEC_REQUESTER_EMAIL",
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "LANGFUSE_HOST",
    ]
    for var in env_vars_to_clear:
        monkeypatch.delenv(var, raising=False)

    return Settings(_env_file=None, **overrides)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

class TestSettingsSingleton:
    def test_singleton_is_settings_instance(self):
        """The module-level `settings` must be a Settings instance."""
        assert isinstance(settings, Settings)


# ---------------------------------------------------------------------------
# Default values
# ---------------------------------------------------------------------------

class TestDefaults:
    def test_llm_provider_defaults_to_groq(self, monkeypatch):
        cfg = fresh_settings(monkeypatch)
        assert cfg.llm_provider == LLMProvider.GROQ

    def test_llm_provider_string_value_is_groq(self, monkeypatch):
        """LLMProvider is a str-enum; its value should equal the literal 'groq'."""
        cfg = fresh_settings(monkeypatch)
        assert cfg.llm_provider == "groq"

    def test_groq_model_default(self, monkeypatch):
        cfg = fresh_settings(monkeypatch)
        assert cfg.groq_model == "llama-3.3-70b-versatile"

    def test_groq_api_key_default_is_empty(self, monkeypatch):
        cfg = fresh_settings(monkeypatch)
        assert cfg.groq_api_key == ""

    def test_embedding_provider_defaults_to_ollama(self, monkeypatch):
        cfg = fresh_settings(monkeypatch)
        assert cfg.embedding_provider == EmbeddingProvider.OLLAMA

    def test_ollama_embedding_model_default(self, monkeypatch):
        cfg = fresh_settings(monkeypatch)
        assert cfg.ollama_embedding_model == "nomic-embed-text"

    def test_chunk_size_default(self, monkeypatch):
        cfg = fresh_settings(monkeypatch)
        assert cfg.chunk_size == 512

    def test_chunk_overlap_default(self, monkeypatch):
        cfg = fresh_settings(monkeypatch)
        assert cfg.chunk_overlap == 50

    def test_retrieval_top_k_default(self, monkeypatch):
        cfg = fresh_settings(monkeypatch)
        assert cfg.retrieval_top_k == 5

    def test_llm_temperature_default(self, monkeypatch):
        cfg = fresh_settings(monkeypatch)
        assert cfg.llm_temperature == 0.0

    def test_llm_max_tokens_default(self, monkeypatch):
        cfg = fresh_settings(monkeypatch)
        assert cfg.llm_max_tokens == 1024

    def test_chroma_collection_name_default(self, monkeypatch):
        cfg = fresh_settings(monkeypatch)
        assert cfg.chroma_collection_name == "quaestor"

    def test_ollama_base_url_default(self, monkeypatch):
        cfg = fresh_settings(monkeypatch)
        assert cfg.ollama_base_url == "http://localhost:11434"


# ---------------------------------------------------------------------------
# Path types
# ---------------------------------------------------------------------------

class TestPathTypes:
    def test_chroma_persist_dir_is_path(self, monkeypatch):
        """chroma_persist_dir must be a Path, not a bare string."""
        cfg = fresh_settings(monkeypatch)
        assert isinstance(cfg.chroma_persist_dir, Path)

    def test_chroma_persist_dir_default_value(self, monkeypatch):
        cfg = fresh_settings(monkeypatch)
        assert cfg.chroma_persist_dir == Path(".chroma")

    def test_data_raw_dir_is_path(self, monkeypatch):
        cfg = fresh_settings(monkeypatch)
        assert isinstance(cfg.data_raw_dir, Path)

    def test_data_processed_dir_is_path(self, monkeypatch):
        cfg = fresh_settings(monkeypatch)
        assert isinstance(cfg.data_processed_dir, Path)

    def test_chroma_persist_dir_env_override_is_path(self, monkeypatch):
        """Even when set via env var the field must still be coerced to Path."""
        monkeypatch.setenv("CHROMA_PERSIST_DIR", "/tmp/test_chroma")
        cfg = Settings(_env_file=None)
        assert isinstance(cfg.chroma_persist_dir, Path)
        assert cfg.chroma_persist_dir == Path("/tmp/test_chroma")


# ---------------------------------------------------------------------------
# chunk_overlap validator
# ---------------------------------------------------------------------------

class TestChunkOverlapValidator:
    def test_overlap_equal_to_chunk_size_raises(self, monkeypatch):
        """chunk_overlap == chunk_size must raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            fresh_settings(monkeypatch, chunk_size=512, chunk_overlap=512)
        assert "chunk_overlap" in str(exc_info.value)

    def test_overlap_greater_than_chunk_size_raises(self, monkeypatch):
        """chunk_overlap > chunk_size must raise ValidationError."""
        with pytest.raises(ValidationError):
            fresh_settings(monkeypatch, chunk_size=256, chunk_overlap=300)

    def test_overlap_less_than_chunk_size_is_valid(self, monkeypatch):
        """chunk_overlap < chunk_size must succeed."""
        cfg = fresh_settings(monkeypatch, chunk_size=512, chunk_overlap=51)
        assert cfg.chunk_overlap == 51

    def test_overlap_zero_is_valid(self, monkeypatch):
        """Zero overlap is a legitimate setting."""
        cfg = fresh_settings(monkeypatch, chunk_size=512, chunk_overlap=0)
        assert cfg.chunk_overlap == 0

    def test_overlap_one_below_chunk_size_is_valid(self, monkeypatch):
        """Exactly chunk_size - 1 is the maximum valid overlap."""
        cfg = fresh_settings(monkeypatch, chunk_size=512, chunk_overlap=511)
        assert cfg.chunk_overlap == 511

    def test_error_message_contains_both_values(self, monkeypatch):
        """The ValidationError message must name both the bad overlap and chunk_size."""
        with pytest.raises(ValidationError) as exc_info:
            fresh_settings(monkeypatch, chunk_size=100, chunk_overlap=200)
        error_text = str(exc_info.value)
        assert "200" in error_text
        assert "100" in error_text


# ---------------------------------------------------------------------------
# LLMProvider enum acceptance
# ---------------------------------------------------------------------------

class TestLLMProviderEnum:
    def test_ollama_provider_accepted(self, monkeypatch):
        cfg = fresh_settings(monkeypatch, llm_provider="ollama")
        assert cfg.llm_provider == LLMProvider.OLLAMA

    def test_together_provider_accepted(self, monkeypatch):
        cfg = fresh_settings(monkeypatch, llm_provider="together")
        assert cfg.llm_provider == LLMProvider.TOGETHER

    def test_invalid_provider_raises(self, monkeypatch):
        with pytest.raises(ValidationError):
            fresh_settings(monkeypatch, llm_provider="openai")


# ---------------------------------------------------------------------------
# Environment variable override
# ---------------------------------------------------------------------------

class TestEnvOverrides:
    def test_llm_provider_from_env(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        cfg = Settings(_env_file=None)
        assert cfg.llm_provider == LLMProvider.OLLAMA

    def test_chunk_size_from_env(self, monkeypatch):
        monkeypatch.setenv("CHUNK_SIZE", "256")
        monkeypatch.setenv("CHUNK_OVERLAP", "10")
        cfg = Settings(_env_file=None)
        assert cfg.chunk_size == 256

    def test_case_insensitive_env_var(self, monkeypatch):
        """Settings are case-insensitive — lowercase env vars must work too."""
        monkeypatch.setenv("llm_provider", "together")
        cfg = Settings(_env_file=None)
        assert cfg.llm_provider == LLMProvider.TOGETHER


# ---------------------------------------------------------------------------
# Langfuse observability
# ---------------------------------------------------------------------------

class TestLangfuse:
    def test_langfuse_disabled_by_default(self, monkeypatch):
        """langfuse_enabled must be False when keys are not set."""
        cfg = fresh_settings(monkeypatch)
        assert cfg.langfuse_enabled is False

    def test_langfuse_enabled_when_both_keys_set(self, monkeypatch):
        """langfuse_enabled must be True when both public and secret keys are non-empty."""
        cfg = fresh_settings(
            monkeypatch,
            langfuse_public_key="pk-lf-test",
            langfuse_secret_key="sk-lf-test",
        )
        assert cfg.langfuse_enabled is True

    def test_langfuse_disabled_when_only_public_key_set(self, monkeypatch):
        """langfuse_enabled must be False when only one key is present."""
        cfg = fresh_settings(monkeypatch, langfuse_public_key="pk-lf-test")
        assert cfg.langfuse_enabled is False

    def test_langfuse_disabled_when_only_secret_key_set(self, monkeypatch):
        """langfuse_enabled must be False when only the secret key is present."""
        cfg = fresh_settings(monkeypatch, langfuse_secret_key="sk-lf-test")
        assert cfg.langfuse_enabled is False

    def test_langfuse_host_default(self, monkeypatch):
        """langfuse_host must default to localhost:3000 for self-hosted setup."""
        cfg = fresh_settings(monkeypatch)
        assert cfg.langfuse_host == "http://localhost:3000"

    def test_langfuse_host_override(self, monkeypatch):
        """langfuse_host can be overridden to point at Langfuse Cloud."""
        cfg = fresh_settings(monkeypatch, langfuse_host="https://cloud.langfuse.com")
        assert cfg.langfuse_host == "https://cloud.langfuse.com"
