from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Deployment configuration, read from the environment at startup."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    qdrant_url: str
    qdrant_api_key: str | None = None

    # Every ingest request must resolve inside this directory.
    ingest_root: Path = Path(".")

    # Must match the embedding model that produces the vectors.
    embedding_dimensions: int = 1536
    default_top_k: int = 3

    @field_validator("qdrant_api_key", mode="after")
    @classmethod
    def blank_key_is_none(cls, value: str | None) -> str | None:
        # QDRANT_API_KEY= in .env arrives as "", which the client reads as
        # "a key is present" and warns about sending it over plain http.
        return value or None


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
