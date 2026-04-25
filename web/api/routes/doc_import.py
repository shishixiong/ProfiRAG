"""Document import endpoints."""

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from typing import List

import schemas
import services

router = APIRouter(prefix="/import", tags=["Document Import"])


@router.post("/upload", response_model=List[schemas.FileInfo], summary="Upload documents for import")
async def upload_documents(files: List[UploadFile] = File(...)):
    """Upload multiple documents for import."""
    results = []
    for file in files:
        content = await file.read()
        result = services.FileService.save_uploaded_file(content, file.filename)
        results.append(schemas.FileInfo(**result))

    return results


@router.post("/start", response_model=schemas.ImportProgress, summary="Start import process")
async def start_import(request: schemas.ImportStartRequest):
    """Start importing documents into vector store."""
    # Get file paths
    file_paths = []
    for file_id in request.file_ids:
        path = services.FileService.get_file_path(file_id)
        if path:
            file_paths.append(str(path))

    if not file_paths:
        raise HTTPException(status_code=400, detail="No valid files to import")

    # Start import
    result = services.ImportService.start_import(
        file_paths=file_paths,
        splitter_type=request.config.splitter_type.value,
        chunk_size=request.config.chunk_size,
        chunk_overlap=request.config.chunk_overlap,
        ast_language=request.config.ast_language.value,
        index_mode=request.config.index_mode.value,
        env_file=request.config.env_file,
        metadata=request.config.metadata,
    )

    return schemas.ImportProgress(
        job_id=result.get("job_id", ""),
        status=result.get("status", "pending"),
        documents_processed=result.get("documents_processed", 0),
        documents_total=result.get("documents_total", 0),
        chunks_created=result.get("chunks_created", 0),
        elapsed_seconds=result.get("elapsed_seconds", 0),
        error=result.get("error"),
    )


@router.get("/progress/{job_id}", response_model=schemas.ImportProgress, summary="Get import progress")
async def get_progress(job_id: str):
    """Get current import job progress."""
    result = services.ImportService.get_progress(job_id)
    if not result:
        raise HTTPException(status_code=404, detail="Job not found")

    return schemas.ImportProgress(
        job_id=job_id,
        status=result.get("status", "unknown"),
        documents_processed=result.get("documents_processed", 0),
        documents_total=result.get("documents_total", 0),
        chunks_created=result.get("chunks_created", 0),
        elapsed_seconds=result.get("elapsed_seconds", 0),
        error=result.get("error"),
    )


@router.get("/stats/{job_id}", response_model=schemas.ImportStats, summary="Get import statistics")
async def get_stats(job_id: str):
    """Get final import statistics."""
    result = services.ImportService.get_stats(job_id)
    if not result:
        raise HTTPException(status_code=404, detail="Job not found or not completed")

    return schemas.ImportStats(**result)


@router.delete("/files/{file_id}", summary="Delete uploaded file")
async def delete_file(file_id: str):
    """Delete an uploaded file."""
    if services.FileService.cleanup_file(file_id):
        return {"message": "File deleted"}
    raise HTTPException(status_code=404, detail="File not found")