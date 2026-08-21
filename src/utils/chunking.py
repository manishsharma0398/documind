from functools import lru_cache
from pathlib import PurePosixPath

import tiktoken
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from .constants import EMBEDDING_MODEL, TEXT_OVERLAP, TOKEN_SIZE
from .models import Chunk, Document

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

# A piece of a document plus the headers that were in scope where it was cut.
# split_doc returns these instead of bare strings, because flattening to
# list[str] is what loses the section association.
SplitChunk = tuple[str, dict[str, str]]


@lru_cache(maxsize=1)
def _token_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=TOKEN_SIZE,
        chunk_overlap=TEXT_OVERLAP,
        model_name=EMBEDDING_MODEL,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
    )


@lru_cache(maxsize=1)
def _markdown_splitter() -> MarkdownHeaderTextSplitter:
    # strip_headers=False keeps the heading in the chunk text, so the section
    # title stays part of what gets embedded.
    return MarkdownHeaderTextSplitter(
        headers_to_split_on=MARKDOWN_HEADERS,
        strip_headers=False,
    )


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


def split_doc(text: str, file_extension: str) -> list[SplitChunk]:
    token_splitter = _token_splitter()
    if file_extension.lower() not in MARKDOWN_EXTS:
        # No headers to carry, but the shape has to match the markdown branch.
        return [(chunk, {}) for chunk in token_splitter.split_text(text)]

    # Header split first so chunks follow the document's structure, then a
    # token split because a section can be far larger than TOKEN_SIZE.
    sections = _markdown_splitter().split_text(text)
    return [
        # Copy the metadata: one dict is shared by every chunk of a section,
        # and a shared mutable payload is a bug waiting to happen downstream.
        (chunk, dict(section.metadata))
        for section in sections
        for chunk in token_splitter.split_text(section.page_content)
    ]


def chunk_docs(docs: list[Document]) -> list[Chunk]:
    chunks: list[Chunk] = []
    encoding = tiktoken.encoding_for_model(EMBEDDING_MODEL)
    for doc in docs:
        folders = folder_path_of(doc.source)
        splitted_doc = split_doc(doc.text, doc.file_ext)
        for i, (chunk, headers) in enumerate(splitted_doc):
            section = build_section_path(folders, headers)
            text = with_breadcrumb(chunk, section)
            chunks.append(
                Chunk(
                    text=text,
                    document_id=str(doc.document_id),
                    source=doc.source,
                    chunk_index=i,
                    file_ext=doc.file_ext,
                    file_name=doc.file_name,
                    # Counted on the stored text, not the bare chunk: the
                    # breadcrumb is 10-15% of a 100-token chunk, and every
                    # number downstream (embedding cost, context budget)
                    # depends on this being true.
                    token_count=len(encoding.encode(text)),
                    headers=headers,
                    section=section,
                )
            )

    return chunks
