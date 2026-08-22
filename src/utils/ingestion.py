from collections import defaultdict
from typing import NamedTuple

from qdrant_client.models import FieldCondition, Filter, MatchAny

from ..clients.qdrant import scroll_all
from .chunking import file_hash
from .models import Document


class IndexedSource(NamedTuple):
    """What the index currently holds for one source file."""

    hashes: set[str]

    totals: set[int]

    point_count: int


async def indexed_sources(
    collection_name: str,
    sources: list[str],
) -> dict[str, IndexedSource]:
    """What the index holds for each of `sources`.

    Scoped to the request: `source` carries a payload index, so the cost is the
    size of the request rather than of the collection.
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
        # Written from a validated Chunk, so a KeyError means something wrote
        # outside the model -- which should be loud.
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

    Unchanged means one matching hash, one chunk_total, and that many points
    present. The count is what the hash cannot tell us -- old points are
    deleted first, so a run that dies mid-file leaves the new hash behind.
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
