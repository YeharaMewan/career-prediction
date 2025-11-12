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
CHROMA_PERSIST_DIR = KNOWLEDGE_BASE_DIR / "chroma"

# Collection names
ACADEMIC_COLLECTION = "academic_knowledge"
SKILL_COLLECTION = "skill_knowledge"

# Data source directories
ACADEMIC_DATA_DIR = DATA_DIR / "academic"
SKILL_DATA_DIR = DATA_DIR / "skills"

# OpenAI Embeddings Configuration
EMBEDDING_MODEL = "text-embedding-3-small"  # 1536 dimensions, cost-effective
EMBEDDING_DIMENSIONS = 1536
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

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
for directory in [DATA_DIR, ACADEMIC_DATA_DIR, SKILL_DATA_DIR, KNOWLEDGE_BASE_DIR, CHROMA_PERSIST_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Validation
if not OPENAI_API_KEY:
    print("Warning: OPENAI_API_KEY not found in environment variables. RAG system will not function.")

print(f"RAG Configuration loaded successfully")
print(f"  - ChromaDB persist directory: {CHROMA_PERSIST_DIR}")
print(f"  - Academic data directory: {ACADEMIC_DATA_DIR}")
print(f"  - Skill data directory: {SKILL_DATA_DIR}")
print(f"  - Embedding model: {EMBEDDING_MODEL}")
