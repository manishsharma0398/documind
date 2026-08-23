from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, Field, StringConstraints

from ..settings import MAX_TOP_K


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


class RetrieveRequest(BaseModel):
    """A question, and how much to return for it."""

    question: Annotated[str, StringConstraints(strip_whitespace=True)] = Field(
        description="Question to be asked",
        min_length=1,
        max_length=1000,
    )
    top_k: int | None = Field(
        description="How many top results to be fetched",
        default=None,
        ge=1,
        le=MAX_TOP_K,
    )
    score_threshold: float | None = Field(
        description="Fetch the results only if above this score",
        default=None,
        ge=0.0,
        le=1.0,
    )


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


class RetrieveResult(BaseModel):
    """One matching chunk and how close it scored."""

    # Cosine similarity, so only comparable within one embedding model.
    score: float
    text: str
    source: str
    section: str | None = None
    chunk_index: int


class RetrieveResponse(BaseModel):
    """The matches, plus the settings actually used to find them.

    The two knobs are echoed because they fall back to `Settings` when the
    caller omits them, and an eval run has to record what it measured against.
    """

    results: list[RetrieveResult]

    # Never null: both fall back to a setting, so a value was always applied.
    top_k: int = Field(description="How many results were fetched")
    score_threshold: float = Field(description="Score floor applied")
