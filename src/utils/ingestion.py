from collections import defaultdict

from ..clients.qdrant import scroll_all
from .chunking import file_hash
from .models import Document


async def indexed_hashes(collection_name: str) -> dict[str, set[str]]:
    """Every indexed source mapped to the hashes recorded against it.

    A set rather than a single value, because one source can legitimately carry
    two. A large note is 70+ chunks and a batch is 256, so a file's chunks can
    straddle two embed batches; a run that dies in between leaves some chunks at
    the new hash and some at the old. Collapsing that to one value would mark a
    half-written file as complete and never rebuild it.
    """
    hashes: dict[str, set[str]] = defaultdict(set)
    for payload in await scroll_all(
        collection_name, with_payload=["source", "file_hash"]
    ):
        source, digest = payload.get("source"), payload.get("file_hash")
        if source and digest:
            hashes[source].add(digest)
    return hashes


def partition_documents(
    docs: list[Document],
    indexed: dict[str, set[str]],
) -> tuple[list[Document], list[Document]]:
    """Split documents into (unchanged, needs_ingest).

    A document is unchanged only when the index holds exactly one hash for it
    and that hash matches. Anything else -- absent, mismatched, or two hashes
    from an interrupted run -- is rebuilt. Erring toward rebuilding costs an
    embedding call; erring the other way leaves a corrupt index in place.
    """
    unchanged: list[Document] = []
    needs_ingest: list[Document] = []
    for doc in docs:
        if indexed.get(doc.source) == {file_hash(doc.text)}:
            unchanged.append(doc)
        else:
            needs_ingest.append(doc)
    return unchanged, needs_ingest
