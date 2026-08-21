from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from openai import APIConnectionError, APIError, APIStatusError
from qdrant_client.http.exceptions import ApiException, UnexpectedResponse

from .clients.openai_client import close_openai_client, get_openai_client
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
    await get_openai_client()
    yield
    # close connections
    await close_qdrant_client()
    await close_openai_client()


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

    @app.exception_handler(APIConnectionError)
    async def openai_unavailable(request: Request, exc: APIError):
        # Registered on APIConnectionError, which also covers APITimeoutError.
        # Annotated APIError, the common parent, because the status handler
        # below delegates 5xx here -- and unlike qdrant's UnexpectedResponse,
        # APIStatusError is a sibling of APIConnectionError, not a subclass.
        # Neither carries a status or request id worth reporting here.
        logger.exception(
            "openai unreachable",
            extra={"path": request.url.path},
        )
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": "embedding service unavailable"},
        )

    @app.exception_handler(APIStatusError)
    async def openai_http_error(request: Request, exc: APIStatusError):
        # Starlette dispatches on the MRO, so this one handler covers every
        # status error: 400, 401, 403, 404, 422, 429 and 5xx.
        #
        # request_id is the field worth having -- it is what OpenAI support
        # asks for, and nothing else identifies the failed call. The batch
        # itself is never logged: for embeddings the request body is the
        # corpus, and a failed batch in the log store is document text in the
        # log store.
        context = {
            "path": request.url.path,
            "status": exc.status_code,
            "request_id": exc.request_id,
        }

        if exc.status_code == status.HTTP_429_TOO_MANY_REQUESTS:
            # Reached only after the client exhausted its own retries, so this
            # is real saturation rather than a blip. Expected and handled --
            # no traceback needed.
            logger.warning("openai rate limited", extra=context)
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={"detail": "embedding service unavailable"},
            )

        if exc.status_code in (
            status.HTTP_401_UNAUTHORIZED,
            status.HTTP_403_FORBIDDEN,
        ):
            # Not transient: every request fails until the key is fixed. Its
            # own level so it is greppable and cannot be mistaken for a blip.
            logger.error("openai rejected our credentials", extra=context)
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": "internal error"},
            )

        if exc.status_code < status.HTTP_500_INTERNAL_SERVER_ERROR:
            # A 4xx means we sent a bad request -- an oversized batch, too many
            # inputs, a bad model name. Our bug, and not one a retry fixes.
            logger.exception("openai rejected our request", extra=context)
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": "internal error"},
            )

        return await openai_unavailable(request, exc)

    return app
