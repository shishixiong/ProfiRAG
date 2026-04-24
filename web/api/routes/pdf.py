"""PDF conversion endpoints."""

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from typing import Optional

import schemas
import services

router = APIRouter(prefix="/pdf", tags=["PDF Conversion"])


@router.post("/upload", summary="Upload PDF file")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF file for conversion."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    content = await file.read()
    result = services.FileService.save_uploaded_file(content, file.filename)

    return result


@router.post("/convert/{file_id}", response_model=schemas.PdfConvertResponse, summary="Convert PDF to Markdown")
async def convert_pdf(file_id: str, request: schemas.PdfConvertRequest):
    """Convert uploaded PDF to Markdown."""
    file_path = services.FileService.get_file_path(file_id)
    if not file_path:
        raise HTTPException(status_code=404, detail="File not found")

    result = services.PdfService.convert_pdf(
        str(file_path),
        pages=request.pages,
        write_images=request.write_images,
        exclude_header_footer=request.exclude_header_footer,
        header_footer_min_occurrences=request.header_footer_min_occurrences,
        extract_tables=request.extract_tables,
    )

    return schemas.PdfConvertResponse(**result)


@router.get("/preview/{file_id}", response_model=schemas.PdfPreviewResponse, summary="Get conversion preview")
async def get_preview(file_id: str):
    """Get preview of converted Markdown."""
    result = services.PdfService.get_preview(file_id)
    if not result:
        raise HTTPException(status_code=404, detail="Conversion not found")

    return schemas.PdfPreviewResponse(
        file_id=file_id,
        original_pages=0,  # TODO: get from PDF
        markdown_preview=result["markdown_preview"],
        tables_count=result["tables_count"],
    )


@router.get("/download/{file_id}", summary="Download Markdown file")
async def download_markdown(file_id: str):
    """Download the converted Markdown file."""
    file_dir = services.FileService.get_file_path(file_id)
    if not file_dir:
        raise HTTPException(status_code=404, detail="File not found")

    # Find the output markdown file
    output_dir = file_dir.parent / "output"
    md_files = list(output_dir.glob("*.md"))
    if not md_files:
        raise HTTPException(status_code=404, detail="Markdown file not found")

    return FileResponse(
        path=md_files[0],
        media_type="text/markdown",
        filename=md_files[0].name,
    )


@router.delete("/{file_id}", summary="Delete uploaded file")
async def delete_file(file_id: str):
    """Delete uploaded file and its outputs."""
    if services.FileService.cleanup_file(file_id):
        return {"message": "File deleted"}
    raise HTTPException(status_code=404, detail="File not found")