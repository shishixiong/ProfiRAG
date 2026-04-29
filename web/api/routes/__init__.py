"""API routes."""

import sys
from pathlib import Path

# Add the routes directory to Python path
routes_dir = Path(__file__).parent
sys.path.insert(0, str(routes_dir))

from pdf import router as pdf_router
from split import router as split_router
from doc_import import router as import_router
from chat import router as chat_router
from search import router as search_router

__all__ = ["pdf_router", "split_router", "import_router", "chat_router", "search_router"]