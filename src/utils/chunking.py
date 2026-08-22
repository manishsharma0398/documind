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

# 100 left too little room: the section breadcrumb was ~29% of every embedded
# chunk, and top_k=3 gave the model ~300 tokens to answer from. At 400 the
# breadcrumb is ~17% and top_k=3 is ~510 tokens.
#
# Chunks come out at ~178 tokens on average, well under this number: the
# markdown header split runs first, and most sections are shorter than the
# budget. This is a ceiling, not a target.
TOKEN_SIZE = 400
TEXT_OVERLAP = int(TOKEN_SIZE * 0.1)

# Chunks with less real content than this are heading-only sections. Kept in
# the index but excludable at query time via content_tokens.
MIN_CHUNK_TOKENS = 100

# Bump when chunking changes in a way the constants below do not capture --
# a different splitter, strip_headers flipping, a new breadcrumb layout.
# Without a bump, every file_hash stays valid and a re-ingest silently keeps
# chunks built by the old code.
CHUNKER_VERSION = 1

MARKDOWN_EXTS = {".md", ".markdown", ".mdx"}
MARKDOWN_HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3"), ("####", "h4")]

# Separator between breadcrumb segments, and between the breadcrumb and the text.
SECTION_SEPARATOR = " > "

# How many heading levels the breadcrumb keeps, counting from the deepest.
# The deepest heading is the one that disambiguates; h1 is usually just the
# document title, which is already stored in file_name. Capping also contains
# notes that use a whole interview question as an "###" heading — those ran to
# 85 tokens of breadcrumb on every chunk of the section.
MAX_HEADER_DEPTH = 2

# A piece of a document, the headers that were in scope where it was cut, and
# the breadcrumb built from them. split_doc returns these instead of bare
# strings, because flattening to list[str] is what loses the section
# association. The breadcrumb is carried out rather than rebuilt by the caller
# because split_doc has to build it anyway -- see the budget below.
SplitChunk = tuple[str, dict[str, str], str | None]


@lru_cache(maxsize=1)
def _encoding():
    return tiktoken.encoding_for_model(EMBEDDING_MODEL)


@lru_cache(maxsize=None)
def _token_splitter(chunk_size: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size,
        chunk_overlap=TEXT_OVERLAP,
        model_name=EMBEDDING_MODEL,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
    )


@lru_cache(maxsize=1)
def _markdown_splitter() -> MarkdownHeaderTextSplitter:
    # strip_headers=True because the breadcrumb already carries them. A
    # section's body always begins with its deepest heading, and
    # build_section_path always keeps the deepest, so what is stripped here is
    # always the redundant copy -- never a heading the chunk would otherwise
    # lose. Sections nested two levels deep were repeating both.
    #
    # Measured on a 411-document corpus: 73% of chunks restated their own
    # heading, costing 6% of all tokens. Repeating a term inside one chunk does
    # not reinforce its embedding so much as pull the vector toward the heading
    # and away from the body.
    return MarkdownHeaderTextSplitter(
        headers_to_split_on=MARKDOWN_HEADERS,
        strip_headers=True,
    )


def _pipeline_fingerprint() -> str:
    """Digest of everything that changes what chunk_docs produces.

    Folded into every file_hash so retuning the pipeline invalidates the whole
    index automatically. The embedding model is included because vectors from a
    different model are not comparable, even for identical text.
    """
    # Canonical JSON rather than a joined string: joining on a separator is
    # not injective -- ["a|b", "c"] and ["a", "b|c"] both render "a|b|c", so
    # two different configurations could share a fingerprint. sort_keys makes
    # the output independent of declaration order.
    #
    # Nothing unordered may go in here. A set renders in an order that varies
    # between processes, because string hashing is randomised per run.
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
    """Strip differences that carry no meaning but change the bytes.

    read_text_file opens in binary and decodes, so Python's universal-newline
    translation never runs: a file touched by a Windows editor or by
    core.autocrlf keeps its CRLF and would hash as changed. The BOM is the same
    story -- an editor toggles it and the whole corpus re-embeds.
    """
    return text.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")


def file_hash(text: str) -> str:
    """Skip-key for one file: its content plus the pipeline that shaped it.

    Content, not mtime -- a git checkout rewrites mtimes without changing a
    byte, and would force a pointless re-embed of the whole corpus.
    """
    payload = json.dumps(
        {"pipeline": _pipeline_fingerprint(), "content": _normalise(text)},
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def folder_path_of(source: str) -> str | None:
    """Folders containing a source file, e.g. "ai > 07-rag-pipelines".

    This is the strongest disambiguator in a multi-domain corpus: "caching"
    appears in nine of them, "terraform > caching" appears in one. The full
    folder path rather than just the first segment, because it also separates
    chapter 7's chunking notes from chapter 8's, and because picking one
    segment would be an arbitrary rule.

    `source` is relative to the ingest root, so this does not change when a
    request narrows to a subfolder.

    Returns None for a file sitting directly in the ingest root.
    """
    folders = PurePosixPath(source).parent.parts
    return SECTION_SEPARATOR.join(folders) if folders else None


def build_section_path(folders: str | None, headers: dict[str, str]) -> str | None:
    """Join the folder path and headers into one breadcrumb, outermost first.

    Iterates MARKDOWN_HEADERS rather than the dict, so ordering comes from the
    declaration (h1 -> h4) and not from insertion order. headers is sparse:
    text under "## Setup" with no "###" carries only h1 and h2.

    Only the deepest MAX_HEADER_DEPTH headings survive; the folder path is
    always kept, because it is what separates "caching" in terraform from
    "caching" in sql.
    """
    present = [
        value for _, key in MARKDOWN_HEADERS if (value := headers.get(key, "").strip())
    ]
    segments = ([folders] if folders else []) + present[-MAX_HEADER_DEPTH:]
    return SECTION_SEPARATOR.join(segments) if segments else None


def with_breadcrumb(text: str, section_path: str | None) -> str:
    """Prefix a chunk with its breadcrumb so the embedding sees it.

    This is the part that fixes retrieval. strip_headers=False keeps the heading
    in the *section*, but once a section is token-split only the first chunk
    still contains it — chunks 2..n are orphaned prose. Prefixing every chunk
    re-anchors them to where they came from.

    Runs after the token split, never before: before, the breadcrumb would land
    in chunk 1 only, which is the bug this exists to fix.
    """
    if not section_path:
        return text
    return f"{section_path}\n\n{text}"


def _breadcrumb_tokens(section_path: str | None) -> int:
    """Tokens with_breadcrumb will prepend, so the splitter can reserve them.

    Measures exactly what gets prefixed, separator included, and returns 0 when
    nothing will be. Reserving a different amount than is prepended is how
    TOKEN_SIZE stops being a ceiling.
    """
    if not section_path:
        return 0
    return len(_encoding().encode(f"{section_path}\n\n"))


def split_doc(text: str, file_extension: str, folders: str | None) -> list[SplitChunk]:
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
        # Per section, never hoisted: each section has a different breadcrumb,
        # so each gets a different budget. Reserving the breadcrumb here is the
        # only reason TOKEN_SIZE is a real ceiling -- the alternative is
        # splitting to TOKEN_SIZE and then prefixing, which overshoots it by
        # however long the breadcrumb happens to be.
        #
        # MIN_CHUNK_TOKENS floors it so a pathological breadcrumb cannot drive
        # the budget to nothing. It is also coupled to TEXT_OVERLAP: the
        # splitter rejects an overlap greater than or equal to its chunk size,
        # so the floor must stay above it.
        budget = max(
            MIN_CHUNK_TOKENS,
            TOKEN_SIZE - _breadcrumb_tokens(section_path),
        )
        for chunk in _token_splitter(budget).split_text(section.page_content):
            out.append((chunk, headers, section_path))

    return out


def chunk_docs(docs: list[Document]) -> Iterator[Chunk]:
    """Yield chunks one at a time rather than returning a list.

    The consumer embeds and upserts in batches and discards as it goes. Holding
    every chunk here would be survivable; holding every *embedded* chunk is not,
    at 1536 floats each. Streaming from this end is what lets that end stay flat.
    """
    for doc in docs:
        folders = folder_path_of(doc.source)
        # Once per document, not per chunk: every chunk of a file shares it.
        doc_hash = file_hash(doc.text)
        splitted_doc = split_doc(doc.text, doc.file_ext, folders)
        # split_doc returns a list, so the total is known before the first
        # chunk is yielded and every chunk of the document can carry it.
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
                # Counted on the stored text, not the bare chunk: the
                # breadcrumb averages ~29 tokens, about 16% of the corpus, and
                # every number downstream -- embedding batch sizing, cost,
                # retrieval context budget -- depends on this being true.
                total_tokens=len(_encoding().encode(text)),
                content_tokens=len(_encoding().encode(chunk)),
                headers=headers,
                section=section,
            )
