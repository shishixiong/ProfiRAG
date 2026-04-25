"""Chat/Q&A endpoints for RAG queries."""

from fastapi import APIRouter, HTTPException

import schemas
import services

router = APIRouter(prefix="/chat", tags=["Chat/Q&A"])


@router.post("/query", response_model=schemas.ChatResponse, summary="Execute RAG query")
async def query(request: schemas.ChatRequest):
    """Execute RAG query and return response with source references."""
    try:
        result = services.ChatService.query(
            query_str=request.query,
            top_k=request.top_k,
            mode=request.mode.value,  # Convert enum to string
            env_file=request.env_file,
        )
        return schemas.ChatResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))