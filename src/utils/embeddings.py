from collections.abc import AsyncIterator, Iterable
from itertools import islice
from typing import NamedTuple

from ..clients.openai_client import embed
from .models import Chunk, EmbeddedChunk

# Chunks per embedding request. Safe by derivation from TOKEN_SIZE, not by
# guesswork: 256 x 400 = 102k tokens against a ~300k per-request cap, and 256
# inputs against a 2048 cap. Raising TOKEN_SIZE past ~1170 breaks the first
# bound.
BATCH = 256


class EmbeddedBatch(NamedTuple):
    """One embedded batch, alongside what the API charged for it.

    A NamedTuple rather than a Pydantic model: every chunk inside has already
    been validated on construction, and re-validating 256 of them per batch
    buys nothing.
    """

    chunks: list[EmbeddedChunk]
    tokens: int

    # 1-based position, so a caller can log progress during a multi-minute run
    # and the last batch's `number` is how many it took.
    number: int


async def batch_embed(chunks: Iterable[Chunk]) -> AsyncIterator[EmbeddedBatch]:
    """Embed in batches, yielding each one so the caller can upsert and discard.

    Takes an iterable rather than a list so the whole pipeline can stream:
    chunk_docs yields, this slices a window off that stream, and nothing ever
    holds the corpus. Materialising here would defeat the generator upstream.

    Deliberately not returning a list either. A 1536-dimension vector plus its
    text is roughly 49KB, so accumulating the whole corpus would hold ~288MB at
    6k chunks and ~484MB at 10k -- growing with the corpus rather than with the
    batch. Yielding keeps peak memory at one batch regardless of corpus size.

    An empty input needs no guard: the first slice is empty and the loop never
    runs.

    Errors from `embed` propagate. Only the ingest loop knows whether a failed
    batch means abort the run or record it and continue.
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
