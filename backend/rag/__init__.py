"""
RAG (Retrieval-Augmented Generation) System

This package provides a complete RAG implementation with:
- Document processing and chunking
- OpenAI embedding generation
- ChromaDB vector storage with 2 specialized collections
- Agentic retrieval with LangGraph decision-making

Collections:
- academic_knowledge: Academic pathways, institutions, programs
- skill_knowledge: Skills, certifications, training resources
"""

from .config import (
    ACADEMIC_COLLECTION,
    SKILL_COLLECTION,
    EMBEDDING_MODEL,
    SIMILARITY_THRESHOLD,
    TOP_K_RESULTS,
)

__version__ = "1.0.0"
__all__ = [
    "ACADEMIC_COLLECTION",
    "SKILL_COLLECTION",
    "EMBEDDING_MODEL",
    "SIMILARITY_THRESHOLD",
    "TOP_K_RESULTS",
]
