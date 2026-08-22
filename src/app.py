import re
import time
from contextlib import asynccontextmanager
from uuid import uuid4

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
from .utils.logger import logger, request_id


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Open the clients on startup, close them on shutdown."""
    # create connections
    await get_qdrant_client()
    await get_openai_client()
    yield
    # close connections
    await close_qdrant_client()
    await close_openai_client()


# Echoed back and written to logs, so a caller cannot inject through it.
_SAFE_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def _request_id(request: Request) -> str:
    """Honour an upstream id when it is safe, otherwise mint one."""
    supplied = request.headers.get("x-request-id", "")
    if _SAFE_REQUEST_ID.match(supplied):
        return supplied
    return uuid4().hex[:12]


def create_fast_api_app() -> FastAPI:
    """Build the app: routers, request logging, and error handling."""
    logger.info("Init App")
    app = FastAPI(title="Documind", lifespan=lifespan)

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        """Log the request in, and the response out.

        Two lines because a request can take minutes; the first is the only
        sign the server is working on something. Bodies are never logged.
        """
        request_id.set(_request_id(request))
        route = {"method": request.method, "path": request.url.path}
        client = request.client.host if request.client else None
        logger.info("incoming request", extra={**route, "client": client})

        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            # Without this the one request worth seeing is the only one with
            # no outgoing line.
            logger.exception(
                "outgoing response",
                extra={
                    **route,
                    "status": 500,
                    "duration_ms": round((time.perf_counter() - started) * 1000, 1),
                },
            )
            raise

        logger.info(
            "outgoing response",
            extra={
                **route,
                "status": response.status_code,
                "duration_ms": round((time.perf_counter() - started) * 1000, 1),
            },
        )
        # So a caller can quote it when reporting a problem.
        response.headers["x-request-id"] = request_id.get()
        return response

    app.include_router(ingest_router, prefix="/ingest", tags=["Ingest"])
    app.include_router(retrieve_router, prefix="/retrieve", tags=["Retrieve"])

    @app.exception_handler(InvalidRequest)
    async def invalid_request(request: Request, exc: InvalidRequest):
        """The caller asked for something we cannot act on."""
        logger.warning(f"invalid request to {request.url.path}: {exc}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST, content={"detail": str(exc)}
        )

    @app.exception_handler(ApiException)
    async def qdrant_unavailable(request: Request, exc: ApiException):
        """Qdrant could not be reached."""
        logger.exception(f"qdrant request failed for {request.url.path}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": "vector store unavailable"},
        )

    @app.exception_handler(UnexpectedResponse)
    async def qdrant_http_error(request: Request, exc: UnexpectedResponse):
        """Qdrant answered with an error."""
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
        """The embedding API could not be reached."""
        # Also covers APITimeoutError, and the 5xx case delegated from below.
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
        """The embedding API answered with an error."""
        # Starlette dispatches on the MRO, so this covers every status error.
        # request_id is what OpenAI support asks for. The batch is never
        # logged: for embeddings the request body is the corpus.
        context = {
            "path": request.url.path,
            "status": exc.status_code,
            "request_id": exc.request_id,
        }

        if exc.status_code == status.HTTP_429_TOO_MANY_REQUESTS:
            # Only reached after the client exhausted its retries.
            logger.warning("openai rate limited", extra=context)
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={"detail": "embedding service unavailable"},
            )

        if exc.status_code in (
            status.HTTP_401_UNAUTHORIZED,
            status.HTTP_403_FORBIDDEN,
        ):
            # Not transient: every request fails until the key is fixed.
            logger.error("openai rejected our credentials", extra=context)
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": "internal error"},
            )

        if exc.status_code < status.HTTP_500_INTERNAL_SERVER_ERROR:
            # A 4xx means we sent something malformed. Our bug, not a retry.
            logger.exception("openai rejected our request", extra=context)
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": "internal error"},
            )

        return await openai_unavailable(request, exc)

    return app
