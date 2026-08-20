from uuid import UUID

from pydantic import BaseModel, Field


class Document(BaseModel):
    text: str
    source: str
    file_name: str
    file_ext: str
    document_id: UUID


class IngestRequest(BaseModel):
    folder_path: str
    extensions: list[str] = Field(min_length=1)
