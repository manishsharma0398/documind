from collections import defaultdict
from typing import NamedTuple

from qdrant_client.models import FieldCondition, Filter, MatchAny

from ..clients.qdrant import scroll_all
from .chunking import file_hash
from .models import Document


class IndexedSource(NamedTuple):
    """What the index currently holds for one source file."""

    # Every distinct file_hash recorded against it. More than one means an
    # interrupted run left chunks from two different pipelines or revisions.
    hashes: set[str]

    # Every distinct chunk_total recorded against it. More than one means the
    # same thing from the other direction.
    totals: set[int]

    # How many points are actually indexed for it right now. Not `count`:
    # NamedTuple subclasses tuple, which already has a count() method, and a
    # field of that name shadows it with an int.
    point_count: int


async def indexed_sources(
    collection_name: str,
    sources: list[str],
) -> dict[str, IndexedSource]:
    """What the index holds for each of `sources`.

    Scoped to the sources the caller asked about, not the whole collection.
    A request may name a subfolder, and reading every payload to use a
    fraction of them costs the size of the index rather than the size of the
    request -- a gap that widens as the collection grows. `source` carries a
    payload index (see ensure_source_index), so the filter is cheap.

    The bound worth knowing: this builds one MatchAny term per source, so a
    request naming tens of thousands of files would want chunking. At corpus
    scale it is one filter.

    Sets rather than single values, because one source can legitimately carry
    two of either. A large note is 70+ chunks and a batch is 256, so a file's
    chunks can straddle two embed batches.
    """
    hashes: dict[str, set[str]] = defaultdict(set)
    totals: dict[str, set[int]] = defaultdict(set)
    counts: dict[str, int] = defaultdict(int)

    if not sources:
        return {}

    for payload in await scroll_all(
        collection_name,
        with_payload=["source", "file_hash", "chunk_total"],
        scroll_filter=Filter(
            must=[FieldCondition(key="source", match=MatchAny(any=sources))]
        ),
    ):
        # Every point is written from a validated Chunk, so all three are
        # always present. A KeyError here means something wrote outside the
        # model, which should be loud rather than silently skipped.
        source = payload["source"]
        hashes[source].add(payload["file_hash"])
        totals[source].add(payload["chunk_total"])
        counts[source] += 1

    return {
        source: IndexedSource(hashes[source], totals[source], counts[source])
        for source in hashes
    }


def partition_documents(
    docs: list[Document],
    indexed: dict[str, IndexedSource],
) -> tuple[list[Document], list[Document]]:
    """Split documents into (unchanged, needs_ingest).

    A document is unchanged only when all three agree:

    - the index holds exactly one hash for it, and it matches the file on disk
    - the index holds exactly one chunk_total for it
    - that many points are actually present

    The count is what the hash alone cannot tell us. Old points are deleted
    before new ones are written, so a run that dies mid-file leaves that file's
    surviving chunks all carrying the *new* hash. A hash-only check reads that
    as complete and never rebuilds it, leaving the document silently truncated
    in the index -- retrievable, plausible, and missing its tail.

    Anything else -- absent, mismatched, mixed, or short -- is rebuilt.
    Rebuilding needlessly costs one embedding call; skipping wrongly leaves the
    index quietly wrong.
    """
    unchanged: list[Document] = []
    needs_ingest: list[Document] = []

    for doc in docs:
        entry = indexed.get(doc.source)
        if (
            entry is not None
            and entry.hashes == {file_hash(doc.text)}
            and entry.totals == {entry.point_count}
        ):
            unchanged.append(doc)
        else:
            needs_ingest.append(doc)

    return unchanged, needs_ingest
