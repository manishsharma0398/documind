from openai import AsyncOpenAI
from openai.types import CreateEmbeddingResponse

from ..settings import get_settings
from ..utils.embedding_model import EMBEDDING_DIMENSIONS, EMBEDDING_MODEL

OPENAI_TIMEOUT: float = 500
OPENAI_MAX_RETRIES: int = 5

_openai_client: AsyncOpenAI | None = None


async def get_openai_client() -> AsyncOpenAI:
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
    global _openai_client
    if _openai_client is not None:
        await _openai_client.close()
        _openai_client = None


async def embed(texts: list[str]) -> CreateEmbeddingResponse:
    # No try/except here on purpose. This module cannot decide what a failure
    # means -- only the ingest loop knows whether to skip the batch, abort the
    # run, or record it and carry on. Swallowing here would hand the caller a
    # None where it expects a response, and surface as an AttributeError a long
    # way from the cause. The app-level handlers in app.py map these to status
    # codes for request-scoped calls.
    return await (await get_openai_client()).embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
        dimensions=EMBEDDING_DIMENSIONS,
    )
