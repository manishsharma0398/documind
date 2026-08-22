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
from ..utils.ingestion import indexed_hashes, partition_documents
from ..utils.logger import logger
from ..utils.models import IngestRequest

ingest_router = APIRouter()


@ingest_router.post("")
async def ingest(payload: IngestRequest):
    collection = get_settings().qdrant_collection
    found = get_files_from_folder(payload.folder_path, payload.extensions)
    await ensure_collection(collection)

    # Read the index even when reindexing: the hashes are ignored for the
    # partition, but their absence is how we know the collection is empty and
    # the delete below can be skipped.
    indexed = await indexed_hashes(collection)
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

    # Clear the old points for the files being rebuilt, before writing new ones.
    # Only these files are touched, so an interrupted run leaves them absent
    # rather than duplicated -- and absent means the next run finds no hash for
    # them and rebuilds. Unchanged files are never at risk.
    #
    # Nothing to clear when the collection is empty, which is every first run.
    # Scoped to the sources in this request even on a reindex: a request may
    # name a subfolder, and wiping the collection would take the rest with it.
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

    chunks = list(chunk_docs(to_ingest))

    # Our own count, from tiktoken, before anything is sent.
    counted = sum(chunk.total_tokens for chunk in chunks)
    billed = 0
    batches = 0

    async for batch in batch_embed(chunks):
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

    cost = billed / 1_000_000 * EMBEDDING_COST_PER_MILLION
    logger.info(
        f"ingested {len(to_ingest)} of {len(found)} documents as {len(chunks)} "
        f"chunks in {batches} batches; skipped {len(unchanged)} unchanged; "
        f"{billed} tokens billed, ${cost:.4f}"
    )

    # text-embedding-3-small tokenises with cl100k_base, which is what
    # tiktoken.encoding_for_model returns, so these should agree exactly.
    # Divergence means TOKEN_SIZE is not the ceiling it claims to be, BATCH is
    # no longer provably under the per-request cap, and every cost figure
    # drifts -- all of it silently.
    if billed != counted:
        logger.warning(
            f"token count drift: tiktoken counted {counted}, "
            f"the API billed {billed} (difference {billed - counted})"
        )

    return {
        "documents": len(to_ingest),
        "skipped": len(unchanged),
        "chunks": len(chunks),
        "batches": batches,
        "tokens": billed,
        "cost_usd": round(cost, 6),
    }
