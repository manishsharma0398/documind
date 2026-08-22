from pathlib import Path, PurePosixPath
from uuid import uuid4

from ..settings import get_settings
from .exceptions import InvalidRequest
from .logger import logger
from .models import Document

# Bytes inspected when deciding whether a file is binary.
SNIFF_BYTES = 4096

# Guardrails so one request cannot sweep up an entire filesystem.
MAX_FILES = 2_000
MAX_FILE_BYTES = 2_000_000
MAX_TOTAL_BYTES = 50_000_000


SKIP_DIRS = {
    ".venv",
    "node_modules",
    ".git",
    "__pycache__",
    ".next",
    "dist",
    "build",
    "target",
    "coverage",
    ".cache",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "qdrant_storage",
    ".ssh",
    ".aws",
    ".gnupg",
    ".kube",
}

# Never ingested, whatever extensions the caller asks for.
DENY_NAMES = {
    ".env",
    ".npmrc",
    ".netrc",
    ".pgpass",
    ".htpasswd",
    ".git-credentials",
    "credentials",
    "id_rsa",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
}
DENY_SUFFIXES = {
    ".pem",
    ".key",
    ".pfx",
    ".p12",
    ".jks",
    ".keystore",
    ".ppk",
    ".gpg",
    ".asc",
    ".kdbx",
}


# Generated lock files: pure noise, never worth embedding.
SKIP_NAMES = {
    "package-lock.json",
    "npm-shrinkwrap.json",
    "pnpm-lock.yaml",
    "bun.lockb",
    ".terraform.lock.hcl",
}

SKIP_SUFFIXES = {".lock"}


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


def is_excluded(source: str, patterns: list[str]) -> bool:
    """True when `source` matches any caller-supplied glob.

    full_match, not match, so a pattern describes the whole path: "CLAUDE.md"
    is the one at the root, "**/CLAUDE.md" is any of them.
    """
    if not patterns:
        return False
    path = PurePosixPath(source)
    return any(path.full_match(pattern) for pattern in patterns)


def get_files_from_folder(
    dir: str,
    extensions: list[str] | None = None,
    exclude: list[str] | None = None,
) -> list[Document]:
    """Every readable file under `dir` that the caller asked for.

    Refuses anything outside INGEST_ROOT, and skips credentials, vendor
    directories, binaries and empty files whatever the extensions say.
    """
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
    # Not named `root`: the walk below rebinds that name, and this value is
    # still needed inside the loop.
    ingest_base = ingest_root()
    if not resolved.is_relative_to(ingest_base):
        raise InvalidRequest(f"{dir} is outside the ingest root")
    docs: list[Document] = []
    skipped = 0
    empty = 0
    excluded = 0
    total_bytes = 0
    # Walk the resolved path, not the caller's: a relative folder_path like
    # "./" yields relative roots, and relative_to(ingest_base) below needs
    # an absolute path to subtract.
    for root, dirs, filenames in resolved.walk():
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for file in filenames:
            if should_skip_file(file):
                continue
            if "*" not in extensions and not any(
                file.endswith(ext) for ext in extensions
            ):
                continue
            path = Path(root) / file
            # Before the read, so an excluded file costs no I/O.
            source = str(path.relative_to(ingest_base))
            if is_excluded(source, exclude or []):
                excluded += 1
                continue
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
            # No chunks means no points means no hash, so it would be
            # rebuilt on every run forever.
            if not text.strip():
                empty += 1
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
                    # Relative to the ingest root, not the requested folder:
                    # it is the delete key, the eval join key, and part of the
                    # breadcrumb, so it must not shift between requests.
                    source=source,
                )
            )
    if skipped:
        logger.info(f"skipped {skipped} binary or unreadable files in {dir}")
    if empty:
        logger.info(f"skipped {empty} empty files in {dir}")
    if excluded:
        logger.info(f"excluded {excluded} files in {dir} by caller pattern")
    return docs
