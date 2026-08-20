import os

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

from ..utils.logger import logger

_client: AsyncQdrantClient | None = None


async def get_qdrant_client() -> AsyncQdrantClient:
    global _client
    if _client is None:
        url = os.getenv("QDRANT_URL")
        if not url:
            raise ValueError("QDRANT_URL env required")
        _client = AsyncQdrantClient(url=url, api_key=os.getenv("QDRANT_API_KEY"))
    return _client


async def close_qdrant_client():
    global _client
    if _client is not None:
        await _client.close()
        _client = None


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


async def ensure_collection(collection_name: str, vector_size: int = 1536):
    if await does_collection_exist(collection_name):
        logger.info(f"collection {collection_name} already exist...")
        return

    logger.info(f"collection {collection_name} doesn't exist creating it...")
    try:
        await create_collection(collection_name, vector_size)
    except UnexpectedResponse as exc:
        if exc.status_code != 409:
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
    top_k: int = 3,
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
        limit=top_k,
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
