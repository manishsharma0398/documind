from collections.abc import AsyncIterator, Iterable
from itertools import islice
from typing import NamedTuple

from ..clients.openai_client import embed
from .embedding_model import EMBEDDING_DIMENSIONS
from .models import Chunk, EmbeddedChunk

# 256 x TOKEN_SIZE = 102k tokens, against a ~300k per-request cap. Raising
# TOKEN_SIZE past ~1170 breaks that.
BATCH = 256


class EmbeddedBatch(NamedTuple):
    """One embedded batch and what the API charged for it."""

    chunks: list[EmbeddedChunk]
    tokens: int

    # 1-based, so the last batch's number is how many it took.
    number: int


async def batch_embed(chunks: Iterable[Chunk]) -> AsyncIterator[EmbeddedBatch]:
    """Embed in batches, yielding each so the caller can upsert and discard.

    Iterable in, generator out: an embedded chunk is ~49KB, so holding the
    corpus would be ~288MB at 6k chunks. Errors propagate.
    """
    stream = iter(chunks)
    number = 0
    while window := list(islice(stream, BATCH)):
        number += 1
        response = await embed([c.text for c in window])
        # The API documents data as input-ordered and ships an index on every
        # item. Pairing on the index costs nothing and needs no such promise.
        ordered = sorted(response.data, key=lambda item: item.index)
        yield EmbeddedBatch(
            chunks=[
                EmbeddedChunk(**chunk.model_dump(), embedding=item.embedding)
                for chunk, item in zip(window, ordered, strict=True)
            ],
            tokens=response.usage.total_tokens,
            number=number,
        )


async def embed_query(text: str) -> list[float]:
    """Embed one query, checking the API gave back what we asked for.

    A short or mis-sized response would otherwise surface as an IndexError or
    a silent dimension mismatch at query time.
    """
    response = await embed([text])
    if len(response.data) != 1:
        raise ValueError(f"expected 1 embedding, got {len(response.data)}")

    vector = response.data[0].embedding
    if len(vector) != EMBEDDING_DIMENSIONS:
        raise ValueError(
            f"embedding has {len(vector)} dimensions, "
            f"collection expects {EMBEDDING_DIMENSIONS}"
        )
    return vector
