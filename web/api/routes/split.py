"""Document splitter endpoints."""

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
import json

import schemas
import services

router = APIRouter(prefix="/split", tags=["Document Splitter"])


@router.post("/upload", summary="Upload document for splitting")
async def upload_document(file: UploadFile = File(...)):
    """Upload a document (PDF, MD, TXT, etc.) for splitting."""
    allowed_extensions = {".pdf", ".md", ".txt", ".py", ".java", ".cpp", ".go"}
    file_ext = file.filename.lower()
    if not any(file_ext.endswith(ext) for ext in allowed_extensions):
        raise HTTPException(
            status_code=400,
            detail=f"File type not supported. Allowed: {allowed_extensions}"
        )

    content = await file.read()
    result = services.FileService.save_uploaded_file(content, file.filename)

    return result


@router.post("/preview", response_model=schemas.SplitPreviewResponse, summary="Preview split result")
async def preview_split(request: schemas.SplitPreviewRequest):
    """Preview document split with metadata."""
    file_path = services.FileService.get_file_path(request.file_id)
    if not file_path:
        raise HTTPException(status_code=404, detail="File not found")

    result = services.SplitService.preview_split(
        str(file_path),
        splitter_type=request.splitter_type.value,
        chunk_size=request.chunk_size,
        chunk_overlap=request.chunk_overlap,
        ast_language=request.language.value,
    )

    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])

    # Convert to response format
    chunks = []
    for c in result["chunks"]:
        chunks.append(schemas.ChunkPreview(
            chunk_index=c["chunk_index"],
            text_preview=c["text_preview"],
            metadata=schemas.ChunkMetadata(**c["metadata"]),
        ))

    return schemas.SplitPreviewResponse(
        file_id=result["file_id"],
        total_chunks=result["total_chunks"],
        chunks=chunks,
        summary=result["summary"],
    )


@router.get("/chunks/{file_id}/{chunk_index}", summary="Get specific chunk content")
async def get_chunk(file_id: str, chunk_index: int, full: bool = False):
    """Get content of a specific chunk."""
    # For now, re-run preview to get chunk
    file_path = services.FileService.get_file_path(file_id)
    if not file_path:
        raise HTTPException(status_code=404, detail="File not found")

    result = services.SplitService.preview_split(str(file_path))

    if chunk_index >= result["total_chunks"]:
        raise HTTPException(status_code=404, detail="Chunk index out of range")

    # Find the chunk
    for c in result["chunks"]:
        if c["chunk_index"] == chunk_index:
            return {
                "chunk_index": chunk_index,
                "text": c["text_preview"] if not full else c["text_preview"],  # TODO: store full text
                "metadata": c["metadata"],
            }

    raise HTTPException(status_code=404, detail="Chunk not found")


@router.post("/download", summary="Download split results")
async def download_chunks(request: schemas.SplitDownloadRequest):
    """Download full split results."""
    output_file = services.SplitService.download_chunks(
        request.file_id,
        output_format=request.output_format.value,
    )
    if not output_file:
        raise HTTPException(status_code=404, detail="Split result not found")

    return FileResponse(
        path=output_file,
        media_type="application/octet-stream",
        filename=f"chunks.{request.output_format.value}",
    )


@router.delete("/{file_id}", summary="Delete uploaded file")
async def delete_file(file_id: str):
    """Delete uploaded file and cached results."""
    if services.FileService.cleanup_file(file_id):
        return {"message": "File deleted"}
    raise HTTPException(status_code=404, detail="File not found")