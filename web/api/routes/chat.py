"""Chat/Q&A endpoints for RAG queries."""

from fastapi import APIRouter, HTTPException

import schemas
import services

router = APIRouter(prefix="/chat", tags=["Chat/Q&A"])


@router.post("/query", response_model=schemas.ChatResponse, summary="Execute RAG query")
async def query(request: schemas.ChatRequest):
    """Execute RAG query and return response with source references."""
    try:
        conversation_dict = None
        if request.conversation:
            conversation_dict = {
                "session_id": request.conversation.session_id,
                "continue": request.conversation.continue_session,
            }

        result = services.ChatService.query(
            query_str=request.query,
            top_k=request.top_k,
            mode=request.mode.value,
            env_file=request.env_file,
            conversation=conversation_dict,
        )
        return schemas.ChatResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/session", summary="Create new conversation session")
async def create_session():
    """Create a new conversation session."""
    session_id = services.ChatService._create_session_id()
    return {"session_id": session_id}


@router.get("/session/{session_id}", summary="Get session info")
async def get_session(session_id: str):
    """Get conversation session info."""
    session = services.ChatService.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.delete("/session/{session_id}", summary="Clear session")
async def clear_session(session_id: str):
    """Clear conversation session."""
    success = services.ChatService.clear_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"success": True, "session_id": session_id}