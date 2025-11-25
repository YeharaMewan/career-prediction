"""
RAG System Configuration

This module contains all configuration settings for the RAG (Retrieval-Augmented Generation) system.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directories
BACKEND_DIR = Path(__file__).parent.parent
DATA_DIR = BACKEND_DIR / "data"
KNOWLEDGE_BASE_DIR = BACKEND_DIR / "knowledge_base"

# Dual vector database directories (one for each embedding provider)
CHROMA_PERSIST_DIR_OPENAI = KNOWLEDGE_BASE_DIR / "chroma_openai"
CHROMA_PERSIST_DIR_GEMINI = KNOWLEDGE_BASE_DIR / "chroma_gemini"

# Backward compatibility (points to OpenAI by default)
CHROMA_PERSIST_DIR = CHROMA_PERSIST_DIR_OPENAI

# Collection names (same names in both databases)
ACADEMIC_COLLECTION = "academic_knowledge"
SKILL_COLLECTION = "skill_knowledge"

# Data source directories (shared between both embedding providers)
ACADEMIC_DATA_DIR = DATA_DIR / "academic"
SKILL_DATA_DIR = DATA_DIR / "skills"

# Embedding Provider Configuration
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "fallback")  # openai | gemini | fallback
ENABLE_EMBEDDING_FALLBACK = os.getenv("ENABLE_EMBEDDING_FALLBACK", "true").lower() == "true"

# OpenAI Embeddings Configuration
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"  # 1536 dimensions, cost-effective
OPENAI_EMBEDDING_DIMENSIONS = 1536
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Gemini Embeddings Configuration
GEMINI_EMBEDDING_MODEL = "models/text-embedding-004"  # 768 dimensions, free tier
GEMINI_EMBEDDING_DIMENSIONS = 768
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Legacy config (for backward compatibility)
EMBEDDING_MODEL = OPENAI_EMBEDDING_MODEL
EMBEDDING_DIMENSIONS = OPENAI_EMBEDDING_DIMENSIONS

# Document Processing Configuration
CHUNK_SIZE = 1000  # Characters per chunk
CHUNK_OVERLAP = 200  # Overlap between chunks for context continuity
SEPARATORS = ["\n\n", "\n", " ", ""]  # Hierarchical text splitting

# Retrieval Configuration
SIMILARITY_THRESHOLD = 0.35  # Minimum similarity score (0-1) - Lowered to capture semantically related content for niche careers
TOP_K_RESULTS = 5  # Number of chunks to retrieve - Increased for better coverage
MAX_CONTEXT_LENGTH = 4000  # Maximum characters in retrieved context

# Agentic RAG Configuration
NEEDS_RETRIEVAL_KEYWORDS = [
    "university", "institution", "college", "degree", "program",
    "course", "skill", "certification", "training", "learn",
    "career", "job", "profession", "industry", "sector"
]

# ChromaDB Configuration
CHROMA_SETTINGS = {
    "anonymized_telemetry": False,
    "allow_reset": True,
}

# Distance metrics
DISTANCE_METRIC = "cosine"  # cosine, l2, ip (inner product)

# Logging
LOG_LEVEL = "INFO"
ENABLE_RAG_LOGGING = True

# Metadata fields
METADATA_FIELDS = {
    "source": str,  # PDF filename
    "page": int,  # Page number
    "collection_type": str,  # "academic" or "skill"
    "chunk_id": str,  # Unique chunk identifier
    "total_chunks": int,  # Total chunks in document
    "timestamp": str,  # Ingestion timestamp
}

# PDF Processing
SUPPORTED_PDF_EXTENSIONS = [".pdf"]
MAX_PDF_SIZE_MB = 50  # Maximum PDF file size

# Batch processing
BATCH_SIZE = 100  # Documents per batch for embedding
EMBEDDING_BATCH_SIZE = 50  # Embeddings per API call

# Create directories if they don't exist
for directory in [
    DATA_DIR,
    ACADEMIC_DATA_DIR,
    SKILL_DATA_DIR,
    KNOWLEDGE_BASE_DIR,
    CHROMA_PERSIST_DIR_OPENAI,
    CHROMA_PERSIST_DIR_GEMINI
]:
    directory.mkdir(parents=True, exist_ok=True)

# Validation
embedding_providers_available = []
if OPENAI_API_KEY:
    embedding_providers_available.append("OpenAI")
if GOOGLE_API_KEY:
    embedding_providers_available.append("Gemini")

if not embedding_providers_available:
    print("Warning: No embedding provider API keys found. RAG system will not function.")
    print("  - Set OPENAI_API_KEY for OpenAI embeddings")
    print("  - Set GOOGLE_API_KEY for Gemini embeddings")
else:
    print(f"RAG Configuration loaded successfully")
    print(f"  - Embedding provider: {EMBEDDING_PROVIDER}")
    print(f"  - Available providers: {', '.join(embedding_providers_available)}")
    print(f"  - Fallback enabled: {ENABLE_EMBEDDING_FALLBACK}")
    print(f"  - OpenAI ChromaDB: {CHROMA_PERSIST_DIR_OPENAI}")
    print(f"  - Gemini ChromaDB: {CHROMA_PERSIST_DIR_GEMINI}")
    print(f"  - Academic data directory: {ACADEMIC_DATA_DIR}")
    print(f"  - Skill data directory: {SKILL_DATA_DIR}")
    print(f"  - OpenAI model: {OPENAI_EMBEDDING_MODEL} ({OPENAI_EMBEDDING_DIMENSIONS} dims)")
    print(f"  - Gemini model: {GEMINI_EMBEDDING_MODEL} ({GEMINI_EMBEDDING_DIMENSIONS} dims)")
