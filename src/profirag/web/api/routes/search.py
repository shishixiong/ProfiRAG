"""Search endpoints for pure retrieval queries."""

from fastapi import APIRouter, HTTPException

from profirag.web.api import schemas, services

router = APIRouter(prefix="/search", tags=["Search"])


@router.post("/query", response_model=schemas.SearchResponse, summary="Execute search query")
async def query(request: schemas.SearchRequest):
    try:
        result = services.SearchService.query(
            query_str=request.query,
            top_k=request.top_k,
            rerank=request.rerank,
            use_pre_retrieval=request.use_pre_retrieval,
            retrieve_mode=request.retrieve_mode.value,
            env_file=request.env_file,
        )
        return schemas.SearchResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
