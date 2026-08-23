from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Ceiling on results per query. Bounds the request field and the default
# below, so a config value cannot walk past the documented API limit.
MAX_TOP_K = 20


class Settings(BaseSettings):
    """Deployment configuration, read from the environment at startup."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    qdrant_url: str
    qdrant_api_key: str | None = None
    openai_api_key: str

    # Every ingest request must resolve inside this directory.
    ingest_root: Path = Path(".")

    # Not constants: a deployment may want its own collection (staging vs
    # production against one Qdrant), and top_k is a knob worth turning
    # without a redeploy.
    qdrant_collection: str = "docs"
    default_top_k: int = Field(default=3, ge=1, le=MAX_TOP_K)

    # Zero until the eval picks a real floor: a baseline needs to see the
    # whole score distribution, not a pre-filtered slice of it.
    default_score_threshold: float = Field(default=0.0, ge=0.0, le=1.0)

    @field_validator("qdrant_api_key", mode="after")
    @classmethod
    def blank_key_is_none(cls, value: str | None) -> str | None:
        """Treat an empty QDRANT_API_KEY as absent rather than present."""
        # QDRANT_API_KEY= in .env arrives as "", which the client reads as
        # "a key is present" and warns about sending it over plain http.
        return value or None


@lru_cache
def get_settings() -> Settings:
    """Read once per process."""
    return Settings()  # type: ignore[call-arg]
