from http import HTTPStatus

from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    Distance,
    Filter,
    PayloadSchemaType,
    PayloadSelector,
    PointStruct,
    ScoredPoint,
    UpdateResult,
    VectorParams,
)

from ..settings import get_settings
from ..utils.embedding_model import EMBEDDING_DIMENSIONS
from ..utils.logger import logger

_qdrant_client: AsyncQdrantClient | None = None


async def get_qdrant_client() -> AsyncQdrantClient:
    """The shared client, built on first use."""
    global _qdrant_client
    if _qdrant_client is None:
        settings = get_settings()
        _qdrant_client = AsyncQdrantClient(
            url=settings.qdrant_url, api_key=settings.qdrant_api_key
        )
    return _qdrant_client


async def close_qdrant_client():
    """Close it, so the next call builds a fresh one."""
    global _qdrant_client
    if _qdrant_client is not None:
        await _qdrant_client.close()
        _qdrant_client = None


async def does_collection_exist(collection_name: str) -> bool:
    return await (await get_qdrant_client()).collection_exists(collection_name)


async def create_collection(collection_name: str, vector_size: int) -> None:
    """Create a collection sized for the embedding model."""
    client = await get_qdrant_client()
    await client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            distance=Distance.COSINE,
            size=vector_size,
        ),
    )


async def ensure_source_index(collection_name: str) -> None:
    """Index `source`, the field every delete and scroll filters on."""
    client = await get_qdrant_client()
    await client.create_payload_index(
        collection_name=collection_name,
        field_name="source",
        field_schema=PayloadSchemaType.KEYWORD,
        wait=True,
    )


async def ensure_collection(collection_name: str, vector_size: int | None = None):
    """Create the collection and its payload index if they are missing.

    A 409 means another worker won the race, which is a success here.
    """
    if await does_collection_exist(collection_name):
        logger.info(f"collection {collection_name} already exist...")
        await ensure_source_index(collection_name)
        return

    logger.info(f"collection {collection_name} doesn't exist creating it...")
    try:
        await create_collection(collection_name, vector_size or EMBEDDING_DIMENSIONS)
    except UnexpectedResponse as exc:
        if exc.status_code != HTTPStatus.CONFLICT:
            raise
        logger.info(
            f"collection {collection_name} was created concurrently, continuing"
        )
    else:
        logger.info(f"collection {collection_name} created successfully")
    await ensure_source_index(collection_name)


async def scroll_all(
    collection_name: str,
    with_payload: list[str],
    scroll_filter: Filter | None = None,
    page_size: int = 1000,
) -> list[dict]:
    """Payloads for the matching points, one page at a time.

    with_vectors=False is what makes this cheap. Pass a filter when the caller
    knows which points it wants: unfiltered the cost is the whole collection.
    """
    client = await get_qdrant_client()
    payloads: list[dict] = []
    offset = None
    while True:
        points, offset = await client.scroll(
            collection_name=collection_name,
            limit=page_size,
            offset=offset,
            scroll_filter=scroll_filter,
            with_payload=with_payload,
            with_vectors=False,
        )
        payloads.extend(point.payload or {} for point in points)
        if offset is None:
            return payloads


async def upsert_collection(
    collection_name: str,
    points: list[PointStruct],
    wait: bool = True,
) -> UpdateResult:
    """Write points, replacing any that share an id."""
    client = await get_qdrant_client()
    return await client.upsert(
        collection_name=collection_name,
        wait=wait,
        points=points,
    )


async def query_collection(
    collection_name: str,
    query: list[float],
    with_payload: PayloadSelector | bool | list[str],
    query_filter: Filter | None = None,
    top_k: int | None = None,
    score_threshold: float | None = None,
) -> list[ScoredPoint]:
    """Nearest points to `query`, optionally filtered and score-limited."""
    client = await get_qdrant_client()
    points = await client.query_points(
        collection_name,
        query=query,
        with_payload=with_payload,
        query_filter=query_filter,
        score_threshold=score_threshold,
        limit=top_k or get_settings().default_top_k,
    )

    return points.points


async def delete_collection_data(
    collection_name: str,
    criteria: Filter,
    wait: bool = True,
) -> UpdateResult:
    """Delete every point matching `criteria`."""
    client = await get_qdrant_client()
    return await client.delete(
        collection_name,
        points_selector=criteria,
        wait=wait,
    )
