"""Search endpoints for pure retrieval queries."""

from fastapi import APIRouter, HTTPException

import schemas
import services

router = APIRouter(prefix="/search", tags=["Search"])


@router.post("/query", response_model=schemas.SearchResponse, summary="Execute search query")
async def query(request: schemas.SearchRequest):
    """Execute pure retrieval search and return raw chunks.

    Unlike Chat, this endpoint returns only retrieved chunks without
    LLM-generated answers. Useful for browsing and exploring documents.

    Args:
        request: SearchRequest with query, top_k, rerank options

    Returns:
        SearchResponse with files summary and chunk details
    """
    try:
        result = services.SearchService.query(
            query_str=request.query,
            top_k=request.top_k,
            rerank=request.rerank,
            use_pre_retrieval=request.use_pre_retrieval,
            env_file=request.env_file,
        )
        return schemas.SearchResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))