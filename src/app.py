from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from qdrant_client.http.exceptions import ApiException, UnexpectedResponse

from .clients.qdrant import (
    close_qdrant_client,
    get_qdrant_client,
)
from .endpoints.ingest import ingest_router
from .endpoints.retrieve import retrieve_router
from .utils.exceptions import InvalidRequest
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

    @app.exception_handler(InvalidRequest)
    async def invalid_request(request: Request, exc: InvalidRequest):
        logger.warning(f"invalid request to {request.url.path}: {exc}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST, content={"detail": str(exc)}
        )

    @app.exception_handler(ApiException)
    async def qdrant_unavailable(request: Request, exc: ApiException):
        logger.exception(f"qdrant request failed for {request.url.path}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": "vector store unavailable"},
        )

    @app.exception_handler(UnexpectedResponse)
    async def qdrant_http_error(request: Request, exc: UnexpectedResponse):
        # A 4xx means we sent Qdrant a bad request: our bug, not the caller's,
        # and not something a retry will fix.
        if exc.status_code and exc.status_code < status.HTTP_500_INTERNAL_SERVER_ERROR:
            logger.exception(
                f"qdrant rejected request for {request.url.path} "
                f"({exc.status_code} {exc.reason_phrase})"
            )
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": "internal error"},
            )
        return await qdrant_unavailable(request, exc)

    return app
