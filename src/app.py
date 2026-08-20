from contextlib import asynccontextmanager

from fastapi import FastAPI

from .clients.qdrant import (
    close_qdrant_client,
    get_qdrant_client,
)
from .endpoints.ingest import ingest_router
from .endpoints.retrieve import retrieve_router
from .utils.logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    # create connections
    await get_qdrant_client()
    yield
    # close connections
    await close_qdrant_client()


def create_fast_api_app() -> FastAPI:
    logger.info("Init App")
    app = FastAPI(title="Documind", lifespan=lifespan)

    app.include_router(ingest_router, prefix="/ingest", tags=["Ingest"])
    app.include_router(retrieve_router, prefix="/retrieve", tags=["Retrieve"])

    return app
