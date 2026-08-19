import logging

from pythonjsonlogger.json import JsonFormatter

logger = logging.getLogger()
logger.setLevel(logging.INFO)

log_handler = logging.StreamHandler()
log_handler.setFormatter(
    JsonFormatter("%(timestamp)s %(levelname)s %(message)s", timestamp=True)
)

logger.addHandler(log_handler)
