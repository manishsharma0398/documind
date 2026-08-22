import logging
from contextvars import ContextVar
from typing import Any

from pythonjsonlogger.json import JsonFormatter

# Task-local, so concurrent requests each see their own id.
request_id: ContextVar[str] = ContextVar("request_id", default="-")


class RequestIdFilter(logging.Filter):
    """Attach the current request id to every record, ours or a library's."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id.get()
        return True


class MessageFirstFormatter(JsonFormatter):
    """Message first, timestamp last."""

    def add_fields(
        self,
        log_data: dict[str, Any],
        record: logging.LogRecord,
        message_dict: dict[str, Any],
    ) -> None:
        super().add_fields(log_data, record, message_dict)
        # extra= keys are merged after the format string, so move it now.
        if "timestamp" in log_data:
            log_data["timestamp"] = log_data.pop("timestamp")


def _handler() -> logging.StreamHandler:
    """A JSON handler that stamps every record with the request id."""
    handler = logging.StreamHandler()
    handler.setFormatter(
        MessageFirstFormatter(
            "%(message)s %(levelname)s %(name)s %(request_id)s %(timestamp)s",
            timestamp=True,
        )
    )
    handler.addFilter(RequestIdFilter())
    return handler


# Named, not root: a handler on root formats every library's output as ours.
logger = logging.getLogger("documind")
logger.setLevel(logging.INFO)
logger.addHandler(_handler())
logger.propagate = False

# Chatty at INFO and duplicated by our own logging.
for _noisy in ("httpx", "httpcore", "watchfiles", "uvicorn.access"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

# Library problems still need somewhere structured to go, with the request id.
root = logging.getLogger()
root.setLevel(logging.WARNING)
root.addHandler(_handler())
