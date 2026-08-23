import httpx2
from openai import AsyncOpenAI
from openai.types import CreateEmbeddingResponse

from ..settings import get_settings
from ..utils.embedding_model import EMBEDDING_DIMENSIONS, EMBEDDING_MODEL

# Generous against real embedding latency (single-digit seconds) while still
# bounding a hung socket. Streamed completions will need a longer read.
OPENAI_TIMEOUT = httpx2.Timeout(connect=10.0, read=60.0, write=30.0, pool=10.0)
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
