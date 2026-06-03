"""doc-agent: Auto-generates and keeps API documentation in sync with source code."""

from .agent import DocAgent
from .extractor import DocExtractor

__version__ = "0.1.0"
__all__ = ["DocAgent", "DocExtractor"]
