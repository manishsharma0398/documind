import os

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import VectorParams, Distance
from qdrant_client.http.exceptions import UnexpectedResponse

from src.utils.logger import logger

_client: AsyncQdrantClient | None = None


async def get_qdrant_client() -> AsyncQdrantClient:
    global _client
    if _client is None:
        _client = AsyncQdrantClient(url=os.getenv("QDRANT_URL"))
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


async def ensure_collection(collection_name: str):
    if await does_collection_exist(collection_name):
        logger.info(f"collection {collection_name} already exist...")
        return

    logger.info(f"collection {collection_name} doesn't exist creating it...")
    try:
        await create_collection(collection_name, 1536)
    except UnexpectedResponse as exc:
        if exc.status_code != 409:
            raise
        logger.info(
            f"collection {collection_name} was created concurrently, continuing"
        )
    else:
        logger.info(f"collection {collection_name} created successfully")
