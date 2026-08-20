from pathlib import Path
from uuid import uuid4

from ..settings import get_settings
from .constants import (
    DENY_NAMES,
    DENY_SUFFIXES,
    MAX_FILE_BYTES,
    MAX_FILES,
    MAX_TOTAL_BYTES,
    SKIP_DIRS,
    SKIP_NAMES,
    SKIP_SUFFIXES,
    SNIFF_BYTES,
)
from .exceptions import InvalidRequest
from .logger import logger
from .models import Document


def ingest_root() -> Path:
    """Directory that every ingest request must stay inside."""
    return get_settings().ingest_root.expanduser().resolve()


def should_skip_file(file_name: str) -> bool:
    """True for secrets we refuse to read and generated files not worth reading."""
    name = file_name.lower()
    if name in DENY_NAMES or name.startswith(".env."):
        return True
    if name in SKIP_NAMES:
        return True
    suffix = Path(name).suffix
    return suffix in DENY_SUFFIXES or suffix in SKIP_SUFFIXES


def read_text_file(path: Path) -> str | None:
    """Decoded contents, or None when the file is binary or unreadable."""
    try:
        with path.open("rb") as fh:
            head = fh.read(SNIFF_BYTES)
            if b"\x00" in head:
                return None
            return (head + fh.read()).decode("utf-8")
    except (UnicodeDecodeError, OSError):
        return None


def get_files_from_folder(
    dir: str, extensions: list[str] | None = None
) -> list[Document]:
    if not dir:
        raise InvalidRequest("path is required")
    directory: Path = Path(dir).expanduser()
    if not directory.is_dir():
        raise InvalidRequest(f"{dir} is not a directory")
    if not extensions:
        raise InvalidRequest("extensions is required")
    resolved = directory.resolve()
    if any(part in SKIP_DIRS for part in resolved.parts):
        raise InvalidRequest(f"{dir} is inside an excluded directory")
    root = ingest_root()
    if not resolved.is_relative_to(root):
        raise InvalidRequest(f"{dir} is outside the ingest root")
    docs: list[Document] = []
    skipped = 0
    total_bytes = 0
    for root, dirs, filenames in directory.walk():
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for file in filenames:
            if should_skip_file(file):
                continue
            if "*" not in extensions and not any(
                file.endswith(ext) for ext in extensions
            ):
                continue
            path = Path(root) / file
            try:
                if path.stat().st_size > MAX_FILE_BYTES:
                    skipped += 1
                    continue
            except OSError:  # broken symlink, race, permissions
                skipped += 1
                continue
            text = read_text_file(path)
            if text is None:
                skipped += 1
                continue
            total_bytes += len(text)
            if len(docs) >= MAX_FILES or total_bytes > MAX_TOTAL_BYTES:
                raise InvalidRequest(
                    f"{dir} exceeds ingest limits "
                    f"({MAX_FILES} files / {MAX_TOTAL_BYTES // 1_000_000}MB); "
                    "point at a narrower folder"
                )
            docs.append(
                Document(
                    document_id=uuid4(),
                    file_ext=path.suffix,
                    file_name=path.name,
                    text=text,
                    source=str(path.relative_to(directory)),
                )
            )
    if skipped:
        logger.info(f"skipped {skipped} binary or unreadable files in {dir}")
    return docs
