"""ProfiRAG Web Service - FastAPI Backend.

Main entry point for the web service API.
"""

import sys
from pathlib import Path

# Add the api directory to Python path
api_dir = Path(__file__).parent
sys.path.insert(0, str(api_dir))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from routes import pdf_router, split_router, import_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    print("ProfiRAG Web Service starting...")
    yield
    # Shutdown
    print("ProfiRAG Web Service shutting down...")


app = FastAPI(
    title="ProfiRAG Web Service",
    description="Web service for PDF conversion, document splitting, and import",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development, allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(pdf_router)
app.include_router(split_router)
app.include_router(import_router)


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint - API info."""
    return {
        "service": "ProfiRAG Web Service",
        "version": "1.0.0",
        "endpoints": {
            "pdf": "/api/pdf",
            "split": "/api/split",
            "import": "/api/import",
        }
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}