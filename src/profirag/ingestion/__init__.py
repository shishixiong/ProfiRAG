"""Data ingestion pipeline"""

from .loaders import DocumentLoader
from .splitters import TextSplitter
from .pipelines import IngestionPipeline

__all__ = ["DocumentLoader", "TextSplitter", "IngestionPipeline"]