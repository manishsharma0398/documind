from uuid import UUID

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    folder_path: str
    extensions: list[str] = Field(min_length=1)

    # Rebuild every matched file, whatever its hash says. The escape hatch for
    # when the index is wrong in a way the hash cannot see -- a half-finished
    # schema change, a payload field added by hand, a collection restored from
    # a backup.
    reindex: bool = False


class Document(BaseModel):
    text: str
    source: str
    file_name: str
    file_ext: str
    document_id: UUID


class Chunk(BaseModel):
    text: str
    source: str
    file_name: str
    file_ext: str
    document_id: str
    total_tokens: int
    content_tokens: int
    chunk_index: int

    # Identifies the source file AND the pipeline that produced this chunk, so
    # an unchanged file can be skipped on re-ingest. It is not a plain content
    # digest: retuning the chunker changes the output without changing a byte
    # on disk, and a content-only hash would skip exactly the files that need
    # rebuilding. See chunking.file_hash.
    file_hash: str

    # Markdown headers in scope where this chunk was cut. Sparse: a chunk under
    # "## Setup" with no "###" carries only h1 and h2. Empty for non-markdown.
    # Kept structured so Qdrant can filter on it later.
    headers: dict[str, str] = Field(default_factory=dict)

    # The same thing flattened for display and citation, e.g.
    # "terraform > Providers > State caching". None when there is nothing to show.
    section: str | None = None


class EmbeddedChunk(Chunk):
    embedding: list[float]
