import json
from collections.abc import Iterator
from functools import lru_cache
from hashlib import sha256
from pathlib import PurePosixPath

import tiktoken
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from .embedding_model import EMBEDDING_MODEL
from .models import Chunk, Document

# A ceiling, not a target: the header split runs first, so chunks average
# ~178 tokens. Small values leave the breadcrumb dominating each chunk.
TOKEN_SIZE = 400
TEXT_OVERLAP = int(TOKEN_SIZE * 0.1)

# Chunks with less real content than this are heading-only sections. Kept in
# the index but excludable at query time via content_tokens.
MIN_CHUNK_TOKENS = 100

# Bump when chunking changes in a way the constants below miss, or every
# file_hash stays valid and re-ingest keeps chunks built by the old code.
CHUNKER_VERSION = 1

MARKDOWN_EXTS = {".md", ".markdown", ".mdx"}
MARKDOWN_HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3"), ("####", "h4")]

# Separator between breadcrumb segments, and between the breadcrumb and the text.
SECTION_SEPARATOR = " > "

# Deepest headings only: h1 is usually the document title, already in
# file_name, and a long "###" heading can run to 85 tokens of breadcrumb.
MAX_HEADER_DEPTH = 2

# Chunk, the headers in scope where it was cut, and the breadcrumb built from
# them. Flattening to list[str] is what loses the section association.
SplitChunk = tuple[str, dict[str, str], str | None]


@lru_cache(maxsize=1)
def _encoding():
    return tiktoken.encoding_for_model(EMBEDDING_MODEL)


@lru_cache(maxsize=None)
def _token_splitter(chunk_size: int) -> RecursiveCharacterTextSplitter:
    """Cached per size, since each section reserves a different breadcrumb."""
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size,
        chunk_overlap=TEXT_OVERLAP,
        model_name=EMBEDDING_MODEL,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
    )


@lru_cache(maxsize=1)
def _markdown_splitter() -> MarkdownHeaderTextSplitter:
    # The breadcrumb already carries the heading, and always the deepest one,
    # so this only strips the duplicate. Was 73% of chunks, 6% of tokens.
    return MarkdownHeaderTextSplitter(
        headers_to_split_on=MARKDOWN_HEADERS,
        strip_headers=True,
    )


def _pipeline_fingerprint() -> str:
    """Every setting that changes chunk output, for the hash to cover."""
    # Canonical JSON, not a joined string: joining is not injective, so two
    # configurations could collide. Nothing unordered may go in -- set order
    # varies per process.
    config = {
        "chunker_version": CHUNKER_VERSION,
        "embedding_model": EMBEDDING_MODEL,
        "token_size": TOKEN_SIZE,
        "text_overlap": TEXT_OVERLAP,
        "max_header_depth": MAX_HEADER_DEPTH,
        "section_separator": SECTION_SEPARATOR,
        "markdown_headers": MARKDOWN_HEADERS,
    }
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _normalise(text: str) -> str:
    """Normalise line endings and strip the BOM before hashing."""
    return text.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")


def file_hash(text: str) -> str:
    """Skip-key for one file: its content plus the pipeline that shaped it.

    Content, not mtime: a checkout rewrites mtimes without changing a byte.
    """
    payload = json.dumps(
        {"pipeline": _pipeline_fingerprint(), "content": _normalise(text)},
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def folder_path_of(source: str) -> str | None:
    """Folders containing a source file, e.g. "ai > 07-rag-pipelines".

    The strongest disambiguator in a multi-domain corpus: "caching" appears in
    nine of them, "terraform > caching" in one. None for a file at the root.
    """
    folders = PurePosixPath(source).parent.parts
    return SECTION_SEPARATOR.join(folders) if folders else None


def build_section_path(folders: str | None, headers: dict[str, str]) -> str | None:
    """Join the folder path and headers into one breadcrumb, outermost first.

    Only the deepest MAX_HEADER_DEPTH headings survive; the folder path always
    does. `headers` is sparse, so ordering comes from MARKDOWN_HEADERS.
    """
    present = [
        value for _, key in MARKDOWN_HEADERS if (value := headers.get(key, "").strip())
    ]
    segments = ([folders] if folders else []) + present[-MAX_HEADER_DEPTH:]
    return SECTION_SEPARATOR.join(segments) if segments else None


def with_breadcrumb(text: str, section_path: str | None) -> str:
    """Prefix a chunk with its breadcrumb so the embedding sees it.

    After the token split, never before: otherwise only chunk 1 of a section
    carries it and the rest are orphaned prose.
    """
    if not section_path:
        return text
    return f"{section_path}\n\n{text}"


def _breadcrumb_tokens(section_path: str | None) -> int:
    """Tokens the breadcrumb will add, so the splitter can reserve them."""
    if not section_path:
        return 0
    return len(_encoding().encode(f"{section_path}\n\n"))


def split_doc(text: str, file_extension: str, folders: str | None) -> list[SplitChunk]:
    """Split one document into chunks, each with the breadcrumb for its section.

    Header split first so chunks follow the document's structure, then a token
    split because a section can be far larger than the budget.
    """
    if file_extension.lower() not in MARKDOWN_EXTS:
        # No headers to carry, but the shape has to match the markdown branch:
        # the folder path is still a breadcrumb, and still has to be reserved.
        section_path = build_section_path(folders, {})
        budget = max(
            MIN_CHUNK_TOKENS,
            TOKEN_SIZE - _breadcrumb_tokens(section_path),
        )
        return [
            (chunk, {}, section_path)
            for chunk in _token_splitter(budget).split_text(text)
        ]

    # Header split first so chunks follow the document's structure, then a
    # token split because a section can be far larger than TOKEN_SIZE.
    sections = _markdown_splitter().split_text(text)
    out: list[SplitChunk] = []
    for section in sections:
        # Copy the metadata: one dict is shared by every chunk of a section,
        # and a shared mutable payload is a bug waiting to happen downstream.
        headers = dict(section.metadata)
        section_path = build_section_path(folders, headers)
        # Per section, never hoisted: reserving the breadcrumb here is the only
        # reason TOKEN_SIZE is a real ceiling. The floor must stay above
        # TEXT_OVERLAP or the splitter rejects it.
        budget = max(
            MIN_CHUNK_TOKENS,
            TOKEN_SIZE - _breadcrumb_tokens(section_path),
        )
        for chunk in _token_splitter(budget).split_text(section.page_content):
            out.append((chunk, headers, section_path))

    return out


def chunk_docs(docs: list[Document]) -> Iterator[Chunk]:
    """Yield chunks one at a time so the consumer can embed and discard."""
    for doc in docs:
        folders = folder_path_of(doc.source)
        # Once per document, not per chunk: every chunk of a file shares it.
        doc_hash = file_hash(doc.text)
        splitted_doc = split_doc(doc.text, doc.file_ext, folders)
        chunk_total = len(splitted_doc)
        for i, (chunk, headers, section) in enumerate(splitted_doc):
            text = with_breadcrumb(chunk, section)
            yield Chunk(
                text=text,
                document_id=str(doc.document_id),
                source=doc.source,
                chunk_index=i,
                chunk_total=chunk_total,
                file_hash=doc_hash,
                file_ext=doc.file_ext,
                file_name=doc.file_name,
                # On the stored text, not the bare chunk: batch sizing, cost
                # and context budget all derive from it.
                total_tokens=len(_encoding().encode(text)),
                content_tokens=len(_encoding().encode(chunk)),
                headers=headers,
                section=section,
            )
