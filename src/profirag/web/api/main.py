"""ProfiRAG Web Service - FastAPI Backend.

Main entry point for the web service API.
"""

import argparse
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from profirag.web.api.routes import pdf_router, split_router, import_router, chat_router, search_router

STATIC_DIR = Path.cwd() / "web" / "frontend" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("ProfiRAG Web Service starting...")
    yield
    print("ProfiRAG Web Service shutting down...")


app = FastAPI(
    title="ProfiRAG Web Service",
    description="Web service for PDF conversion, document splitting, and import",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(pdf_router, prefix="/api")
app.include_router(split_router, prefix="/api")
app.include_router(import_router, prefix="/api")
app.include_router(chat_router, prefix="/api")
app.include_router(search_router, prefix="/api")


@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "healthy"}


if STATIC_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="frontend")


def run_server():
    """CLI entry point for profirag-web."""
    parser = argparse.ArgumentParser(description="ProfiRAG Web Service")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()

    uvicorn.run(
        "profirag.web.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
