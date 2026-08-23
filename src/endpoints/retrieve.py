from fastapi import APIRouter

from ..clients.qdrant import query_collection
from ..settings import get_settings
from ..utils.embeddings import embed_query
from ..utils.logger import logger
from ..utils.models import (
    RetrieveRequest,
    RetrieveResponse,
    RetrieveResult,
)

retrieve_router = APIRouter()


@retrieve_router.post("", response_model=RetrieveResponse)
async def retrieve(payload: RetrieveRequest):
    """Chunks matching a question, ranked by similarity."""
    settings = get_settings()
    question = payload.question
    # Resolved here rather than inside the query, so the response can report
    # what was actually applied instead of what the caller happened to send.
    top_k = payload.top_k or settings.default_top_k
    score_threshold = (
        settings.default_score_threshold
        if payload.score_threshold is None
        else payload.score_threshold
    )

    qdrant_data = await query_collection(
        collection_name=settings.qdrant_collection,
        query=await embed_query(question),
        top_k=top_k,
        score_threshold=score_threshold,
        query_filter=None,
        with_payload=["text", "source", "section", "chunk_index"],
    )
    retrieved_data: list[RetrieveResult] = []
    for data in qdrant_data:
        result = data.payload
        if result is None:
            continue
        # The stored text is prefixed with the breadcrumb so it embeds with its
        # context. `section` carries it already, so strip it rather than repeat it.
        section = (result["section"] or "") + "\n\n"
        main_text = result["text"].removeprefix(section)
        retrieved_data.append(
            RetrieveResult(
                score=data.score,
                text=main_text,
                source=result["source"],
                section=result["section"],
                chunk_index=result["chunk_index"],
            )
        )
    # The middleware logs latency and status; what it cannot see is whether the
    # hits were strong. Results come back ranked, so first and last is the span.
    scores = [result.score for result in retrieved_data]
    span = f"{scores[0]:.3f}-{scores[-1]:.3f}" if scores else "none"
    logger.info(
        f"retrieved {len(retrieved_data)} chunks for {question[:120]!r}; "
        f"scores {span}, top_k {top_k}, threshold {score_threshold}"
    )

    return RetrieveResponse(
        results=retrieved_data,
        top_k=top_k,
        score_threshold=score_threshold,
    )
