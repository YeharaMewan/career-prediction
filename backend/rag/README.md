# RAG System Documentation

## Overview

This RAG (Retrieval-Augmented Generation) system enhances the Career Planning Multi-Agent System with knowledge base capabilities. It uses **OpenAI embeddings** and **ChromaDB** vector storage to provide grounded, citation-backed information to agents.

## Architecture

### Components

```
rag/
├── config.py              # Configuration settings
├── document_processor.py  # PDF loading and chunking
├── embedding_manager.py   # OpenAI embedding generation
├── vector_store.py        # ChromaDB with 2 collections
└── retriever.py          # Agentic RAG retriever
```

### 2 Specialized Collections

1. **`academic_knowledge`** - Used by `AcademicPathwayAgent`
   - Universities, institutions, degree programs
   - Academic pathways and requirements
   - Admission criteria, scholarships

2. **`skill_knowledge`** - Used by `SkillDevelopmentAgent`
   - Skills, courses, certifications
   - Training resources and platforms
   - Learning paths and roadmaps

---

## Quick Start

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

Required packages:
- `chromadb==0.5.23`
- `pypdf==5.1.0`
- `pymupdf==1.25.6`
- `langchain-openai` (already installed)

### 2. Set OpenAI API Key

Add to `backend/.env`:

```env
OPENAI_API_KEY=sk-your-openai-key-here
```

### 3. Add PDF Documents

Place your PDFs in the appropriate directories:

```bash
backend/data/academic/     # Academic PDFs (universities, programs, etc.)
backend/data/skills/       # Skill PDFs (courses, certifications, etc.)
```

### 4. Ingest Documents

Run the ingestion pipeline:

```bash
# Ingest all collections
python scripts/ingest_knowledge.py --collection all

# Ingest specific collection
python scripts/ingest_knowledge.py --collection academic
python scripts/ingest_knowledge.py --collection skill

# Reset and rebuild
python scripts/ingest_knowledge.py --collection all --reset

# Check status
python scripts/ingest_knowledge.py --status
```

---

## Usage

### Agent Integration

Both `AcademicPathwayAgent` and `SkillDevelopmentAgent` automatically use RAG retrieval:

```python
# Automatic RAG retrieval when agents process tasks
academic_agent = AcademicPathwayAgent()
result = await academic_agent.process_task(state)

# RAG retrieval happens automatically:
# 1. Query: "Academic pathways for software engineer career..."
# 2. Retrieves top 3 relevant chunks from academic_knowledge
# 3. Includes in LLM prompt with citations
```

### Manual Retrieval

You can also use the retriever directly:

```python
from rag.retriever import AgenticRAGRetriever

# Initialize retriever
retriever = AgenticRAGRetriever(
    collection_type="academic",
    similarity_threshold=0.7,
    top_k=3
)

# Retrieve with agentic decision-making
state = retriever.retrieve(
    query="What universities offer computer science programs?",
    include_citations=True
)

if state.needs_retrieval:
    print(f"Retrieved {len(state.retrieved_documents)} documents")
    print(f"Context:\n{state.context}")
```

---

## Configuration

Edit `rag/config.py` to customize:

```python
# Embedding model
EMBEDDING_MODEL = "text-embedding-3-small"  # or text-embedding-3-large

# Chunking parameters
CHUNK_SIZE = 1000  # Characters per chunk
CHUNK_OVERLAP = 200  # Overlap for context

# Retrieval settings
SIMILARITY_THRESHOLD = 0.7  # Min similarity (0-1)
TOP_K_RESULTS = 3  # Number of chunks to retrieve
MAX_CONTEXT_LENGTH = 4000  # Max chars in context

# Agentic keywords (triggers retrieval)
NEEDS_RETRIEVAL_KEYWORDS = [
    "university", "institution", "college", "degree",
    "skill", "certification", "course", "training"
]
```

---

## How It Works

### 1. Document Processing

```
PDF Files → PyMuPDF Loader → Text Extraction → Chunking (1000 chars, 200 overlap)
```

### 2. Embedding Generation

```
Text Chunks → OpenAI API (text-embedding-3-small) → 1536-dim vectors
```

### 3. Vector Storage

```
Embeddings + Metadata → ChromaDB → Persistent Storage (backend/knowledge_base/chroma/)
```

### 4. Agentic Retrieval

```
Agent Query → Decision (needs retrieval?) → Query Embedding → ChromaDB Search →
Similarity Filter (>0.7) → Format Context → Include in LLM Prompt
```

---

## File Structure

```
backend/
├── rag/                          # RAG system
│   ├── __init__.py
│   ├── config.py                 # Configuration
│   ├── document_processor.py     # PDF loading & chunking
│   ├── embedding_manager.py      # OpenAI embeddings
│   ├── vector_store.py           # ChromaDB management
│   ├── retriever.py              # Agentic retriever
│   └── README.md                 # This file
├── data/                         # PDF storage
│   ├── academic/                 # Academic PDFs
│   └── skills/                   # Skill PDFs
├── knowledge_base/               # Persisted data
│   └── chroma/                   # ChromaDB storage
└── scripts/
    └── ingest_knowledge.py       # Ingestion pipeline
```

---

## Command Reference

### Ingestion Commands

```bash
# Ingest all PDFs from both directories
python scripts/ingest_knowledge.py --collection all

# Ingest only academic PDFs
python scripts/ingest_knowledge.py --collection academic

# Ingest only skill PDFs
python scripts/ingest_knowledge.py --collection skill

# Reset collection before ingesting (deletes existing data)
python scripts/ingest_knowledge.py --collection all --reset

# Check vector store status (no ingestion)
python scripts/ingest_knowledge.py --status
```

### Output Example

```
================================================================================
INGESTING ACADEMIC COLLECTION
================================================================================

Step 1: Processing PDFs from backend/data/academic
Loaded 45 pages from university_catalog.pdf
Processed 127 chunks from 3 files

Step 2: Generating embeddings...
Generated 127 embeddings
Embedding cost: $0.0013

Step 3: Storing in ChromaDB...
Added 127 documents to 'academic' collection

================================================================================
INGESTION COMPLETE - ACADEMIC
================================================================================
Documents added: 127
Total in collection: 127
Sources: university_catalog.pdf, programs.pdf, admissions.pdf
Embedding cost: $0.0013
Time elapsed: 12.34 seconds
================================================================================
```

---

## Metadata Structure

Each document chunk includes:

```python
{
    "source_file": "university_catalog.pdf",  # PDF filename
    "page": 5,                                # Page number
    "collection_type": "academic",            # Collection
    "chunk_id": "chunk_42_a8f9d3c1",         # Unique ID
    "chunk_index": 42,                        # Index in sequence
    "total_chunks": 127,                      # Total from source
    "timestamp": "2025-01-15T10:30:00",      # Ingestion time
    "chunk_length": 987,                      # Characters
    "file_path": "/full/path/to/pdf"         # Full path
}
```

---

## Cost Estimation

### OpenAI Embedding Costs

**text-embedding-3-small**: $0.02 per 1M tokens

Example:
- 100 PDF pages = ~500 chunks = ~50,000 words = ~75,000 tokens
- Cost: $0.0015 (less than a penny)

**text-embedding-3-large**: $0.13 per 1M tokens (6.5x more expensive)

### Monitoring Costs

```python
from rag.embedding_manager import EmbeddingManager

manager = EmbeddingManager()
# ... generate embeddings ...

stats = manager.get_stats()
print(f"Cost: ${stats['estimated_cost_usd']}")
```

---

## Troubleshooting

### Issue: "No documents found"

- Check that PDFs exist in `backend/data/academic/` or `backend/data/skills/`
- Verify PDFs are valid and not corrupted
- Check file extensions are `.pdf`

### Issue: "OPENAI_API_KEY not found"

- Add `OPENAI_API_KEY=sk-...` to `backend/.env`
- Restart any running processes

### Issue: "RAG retriever initialization failed"

- Ensure ChromaDB dependencies are installed: `pip install chromadb`
- Check `backend/knowledge_base/chroma/` directory exists and is writable
- Verify collections exist: `python scripts/ingest_knowledge.py --status`

### Issue: "No relevant documents found"

- Check similarity threshold (try lowering to 0.6 in `config.py`)
- Verify PDFs contain relevant content
- Try different query phrasing
- Run ingestion again: `python scripts/ingest_knowledge.py --reset --collection all`

---

## Advanced Usage

### Custom Similarity Threshold

```python
retriever = AgenticRAGRetriever(
    collection_type="academic",
    similarity_threshold=0.6,  # Lower = more results
    top_k=5  # More documents
)
```

### Force Retrieval (Skip Decision)

```python
state = retriever.retrieve(
    query="...",
    force_retrieval=True  # Always retrieve, skip keyword check
)
```

### Metadata Filtering

```python
# Retrieve only from specific source
results = vector_store.query(
    query_embedding=embedding,
    collection_type="academic",
    filter_metadata={"source_file": "university_catalog.pdf"}
)
```

---

## Best Practices

1. **PDF Quality**: Use text-based PDFs, not scanned images
2. **Naming**: Use descriptive PDF filenames (they appear in citations)
3. **Organization**: Separate academic and skill PDFs into correct directories
4. **Updates**: Re-run ingestion with `--reset` when PDFs change
5. **Monitoring**: Check `--status` regularly to verify collection health
6. **Costs**: Monitor embedding costs with `embedding_manager.get_stats()`

---

## Integration with Agents

### Academic Pathway Agent

Retrieves from `academic_knowledge`:
- Universities and institutions
- Degree programs and requirements
- Admission criteria
- Scholarships and funding

### Skill Development Agent

Retrieves from `skill_knowledge`:
- Technical skills and courses
- Certifications and training
- Learning platforms and resources
- Skill development roadmaps

---

## Future Enhancements

Potential improvements:
- [ ] Hybrid search (keyword + semantic)
- [ ] Re-ranking retrieved results
- [ ] Query expansion for better retrieval
- [ ] Automatic PDF updates from web
- [ ] Multi-modal support (images, tables)
- [ ] Feedback loop for retrieval quality
- [ ] Integration with more embedding models

---

## Support

For issues or questions:
1. Check troubleshooting section above
2. Review logs in console output
3. Verify configuration in `rag/config.py`
4. Test with `python scripts/ingest_knowledge.py --status`

---

## Summary

The RAG system provides:
✅ **2 specialized knowledge collections** (academic + skill)
✅ **Agentic decision-making** (only retrieves when needed)
✅ **OpenAI embeddings** (high-quality semantic search)
✅ **ChromaDB persistence** (survives restarts)
✅ **Source citations** (track PDF and page numbers)
✅ **Seamless agent integration** (automatic retrieval)
✅ **Cost-effective** (~$0.001 per 100 pages)

Happy knowledge retrieval! 🚀
