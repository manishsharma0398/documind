from collections.abc import AsyncIterator, Iterable
from itertools import islice
from typing import NamedTuple

from ..clients.openai_client import embed
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
        yield EmbeddedBatch(
            chunks=[
                EmbeddedChunk(**chunk.model_dump(), embedding=item.embedding)
                for chunk, item in zip(window, response.data)
            ],
            tokens=response.usage.total_tokens,
            number=number,
        )
