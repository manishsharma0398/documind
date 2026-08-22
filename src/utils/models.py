from uuid import UUID

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    """What to index, and what to leave out."""

    folder_path: str
    extensions: list[str] = Field(min_length=1)

    # Globs against `source`. Corpus policy is the caller's, not ours, and an
    # eval harness needs to pin it.
    exclude: list[str] = Field(default_factory=list)

    # Rebuild regardless of hash, for when the index is wrong in a way the
    # hash cannot see.
    reindex: bool = False


class Document(BaseModel):
    """One source file, read but not yet split."""

    text: str
    source: str
    file_name: str
    file_ext: str
    document_id: UUID


class Chunk(BaseModel):
    """A slice of a document, ready to embed."""

    text: str
    source: str
    file_name: str
    file_ext: str
    document_id: str
    total_tokens: int
    content_tokens: int
    chunk_index: int

    # Content plus the pipeline settings, so retuning the chunker invalidates
    # it. A content-only digest would skip the files that need rebuilding.
    file_hash: str

    # Compared against the points actually indexed: a hash proves the content,
    # only a count proves the write finished.
    chunk_total: int

    # Sparse, and empty for non-markdown. Structured so Qdrant can filter on it.
    headers: dict[str, str] = Field(default_factory=dict)

    # Flattened for display, e.g. "terraform > Providers > State caching".
    section: str | None = None


class EmbeddedChunk(Chunk):
    """A chunk with its vector."""

    embedding: list[float]
