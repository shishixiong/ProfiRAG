"""API routes."""

from profirag.web.api.routes.pdf import router as pdf_router
from profirag.web.api.routes.split import router as split_router
from profirag.web.api.routes.doc_import import router as import_router
from profirag.web.api.routes.chat import router as chat_router
from profirag.web.api.routes.search import router as search_router

__all__ = ["pdf_router", "split_router", "import_router", "chat_router", "search_router"]
