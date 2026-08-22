from openai import AsyncOpenAI
from openai.types import CreateEmbeddingResponse

from ..settings import get_settings
from ..utils.embedding_model import EMBEDDING_DIMENSIONS, EMBEDDING_MODEL

OPENAI_TIMEOUT: float = 500
OPENAI_MAX_RETRIES: int = 5

_openai_client: AsyncOpenAI | None = None


async def get_openai_client() -> AsyncOpenAI:
    """The shared client, built on first use."""
    global _openai_client
    if _openai_client is None:
        settings = get_settings()
        _openai_client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            max_retries=OPENAI_MAX_RETRIES,
            timeout=OPENAI_TIMEOUT,
        )
    return _openai_client


async def close_openai_client():
    """Close it, so the next call builds a fresh one."""
    global _openai_client
    if _openai_client is not None:
        await _openai_client.close()
        _openai_client = None


async def embed(texts: list[str]) -> CreateEmbeddingResponse:
    """Embed a batch. Returns the whole response so the caller can read usage."""
    # No try/except on purpose: only the caller knows whether a failed batch
    # means abort or continue. app.py maps these to status codes.
    return await (await get_openai_client()).embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
        dimensions=EMBEDDING_DIMENSIONS,
    )
