from uuid import UUID

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    folder_path: str
    extensions: list[str] = Field(min_length=1)


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

    # Markdown headers in scope where this chunk was cut. Sparse: a chunk under
    # "## Setup" with no "###" carries only h1 and h2. Empty for non-markdown.
    # Kept structured so Qdrant can filter on it later.
    headers: dict[str, str] = Field(default_factory=dict)

    # The same thing flattened for display and citation, e.g.
    # "terraform > Providers > State caching". None when there is nothing to show.
    section: str | None = None
