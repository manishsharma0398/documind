from fastapi import APIRouter

from ..utils.filesystem import get_files_from_folder
from ..utils.models import IngestRequest

ingest_router = APIRouter()


@ingest_router.post("")
def ingest(payload: IngestRequest):
    docs = get_files_from_folder(payload.folder_path, payload.extensions)
    return docs
