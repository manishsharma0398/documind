from http import HTTPStatus

from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    Distance,
    Filter,
    PointStruct,
    ScoredPoint,
    UpdateResult,
    VectorParams,
)

from ..settings import get_settings
from ..utils.constants import DEFAULT_TOP_K, EMBEDDING_DIMENSIONS
from ..utils.logger import logger

_qdrant_client: AsyncQdrantClient | None = None


async def get_qdrant_client() -> AsyncQdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        settings = get_settings()
        _qdrant_client = AsyncQdrantClient(
            url=settings.qdrant_url, api_key=settings.qdrant_api_key
        )
    return _qdrant_client


async def close_qdrant_client():
    global _qdrant_client
    if _qdrant_client is not None:
        await _qdrant_client.close()
        _qdrant_client = None


async def does_collection_exist(collection_name: str) -> bool:
    return await (await get_qdrant_client()).collection_exists(collection_name)


async def create_collection(collection_name: str, vector_size: int) -> None:
    client = await get_qdrant_client()
    await client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            distance=Distance.COSINE,
            size=vector_size,
        ),
    )


async def ensure_collection(collection_name: str, vector_size: int | None = None):
    if await does_collection_exist(collection_name):
        logger.info(f"collection {collection_name} already exist...")
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


async def upsert_collection(
    collection_name: str,
    points: list[PointStruct],
    wait: bool = True,
) -> UpdateResult:
    client = await get_qdrant_client()
    return await client.upsert(
        collection_name=collection_name,
        wait=wait,
        points=points,
    )


async def query_collection(
    collection_name: str,
    query: list[float],
    query_filter: Filter | None = None,
    top_k: int | None = None,
    score_threshold: float | None = None,
    with_payload: bool = True,
) -> list[ScoredPoint]:
    client = await get_qdrant_client()
    points = await client.query_points(
        collection_name,
        query=query,
        with_payload=with_payload,
        query_filter=query_filter,
        score_threshold=score_threshold,
        limit=top_k or DEFAULT_TOP_K,
    )

    return points.points


async def delete_collection_data(
    collection_name: str,
    criteria: Filter,
    wait: bool = True,
) -> UpdateResult:
    client = await get_qdrant_client()
    return await client.delete(
        collection_name,
        points_selector=criteria,
        wait=wait,
    )
