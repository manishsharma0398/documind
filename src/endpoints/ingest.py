from uuid import uuid4

from fastapi import APIRouter
from qdrant_client.models import FieldCondition, Filter, MatchAny, PointStruct

from ..clients.qdrant import (
    delete_collection_data,
    ensure_collection,
    upsert_collection,
)
from ..settings import get_settings
from ..utils.chunking import chunk_docs
from ..utils.embedding_model import EMBEDDING_COST_PER_MILLION
from ..utils.embeddings import batch_embed
from ..utils.filesystem import get_files_from_folder
from ..utils.ingestion import indexed_sources, partition_documents
from ..utils.logger import logger
from ..utils.models import Document, IngestRequest

ingest_router = APIRouter()


@ingest_router.post("")
async def ingest(payload: IngestRequest):
    """Index a folder, rebuilding only what changed since the last run."""
    collection = get_settings().qdrant_collection
    found = get_files_from_folder(
        payload.folder_path, payload.extensions, payload.exclude
    )
    await ensure_collection(collection)

    # Read even when reindexing: an empty result is how we know there is
    # nothing to delete. Scoped to this request, since it may name a subfolder.
    indexed = await indexed_sources(collection, [doc.source for doc in found])
    unchanged: list[Document]
    if payload.reindex:
        unchanged, to_ingest = [], found
    else:
        unchanged, to_ingest = partition_documents(found, indexed)

    if not to_ingest:
        logger.info(f"nothing to do: all {len(unchanged)} documents are current")
        return {
            "documents": 0,
            "skipped": len(unchanged),
            "chunks": 0,
            "batches": 0,
            "tokens": 0,
            "cost_usd": 0.0,
        }

    # Only the files being rebuilt, so an interrupted run leaves them absent
    # rather than duplicated, and the next run rebuilds them. Scoped even on a
    # reindex: a request may name a subfolder.
    if indexed:
        await delete_collection_data(
            collection,
            criteria=Filter(
                must=[
                    FieldCondition(
                        key="source",
                        match=MatchAny(any=[doc.source for doc in to_ingest]),
                    )
                ]
            ),
        )

    # Counted as batches go past: consuming the generator to len() it would
    # leave nothing to embed.
    counted = 0
    chunk_count = 0
    billed = 0
    batches = 0

    async for batch in batch_embed(chunk_docs(to_ingest)):
        await upsert_collection(
            collection,
            points=[
                PointStruct(
                    id=str(uuid4()),
                    payload=embed.model_dump(exclude={"embedding"}),
                    vector=embed.embedding,
                )
                for embed in batch.chunks
            ],
        )
        billed += batch.tokens
        batches = batch.number
        chunk_count += len(batch.chunks)
        # Our own count, from tiktoken, for the drift check below.
        counted += sum(chunk.total_tokens for chunk in batch.chunks)

    cost = billed / 1_000_000 * EMBEDDING_COST_PER_MILLION
    logger.info(
        f"ingested {len(to_ingest)} of {len(found)} documents as {chunk_count} "
        f"chunks in {batches} batches; skipped {len(unchanged)} unchanged; "
        f"{billed} tokens billed, ${cost:.4f}"
    )

    # Same tokeniser both sides, so these should agree exactly. Divergence
    # means TOKEN_SIZE is not a ceiling and BATCH is no longer provably safe.
    if billed != counted:
        logger.warning(
            f"token count drift: tiktoken counted {counted}, "
            f"the API billed {billed} (difference {billed - counted})"
        )

    return {
        "documents": len(to_ingest),
        "skipped": len(unchanged),
        "chunks": chunk_count,
        "batches": batches,
        "tokens": billed,
        "cost_usd": round(cost, 6),
    }
