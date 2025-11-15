# Recommended Fixes for RAG System Failure

## Overview
This document provides concrete fixes for the RAG system's failure to retrieve data for niche careers like Music Producer.

---

## Fix #1: Use Configuration Values Instead of Hardcoding (CRITICAL)

### Problem
Agents hardcode `similarity_threshold=0.7` and `top_k=3` instead of using config values.

### Solution
Update both agent files to use config values.

### File 1: `/home/user/career-prediction/backend/agents/workers/academic_pathway.py`

**Current Code (Lines 91-96):**
```python
# Initialize RAG retriever for knowledge base (academic collection)
try:
    self.rag_retriever = AgenticRAGRetriever(
        collection_type="academic",
        similarity_threshold=0.7,  # HARDCODED
        top_k=3                    # HARDCODED
    )
```

**Fixed Code:**
```python
# Initialize RAG retriever for knowledge base (academic collection)
from rag.config import SIMILARITY_THRESHOLD, TOP_K_RESULTS

try:
    self.rag_retriever = AgenticRAGRetriever(
        collection_type="academic",
        similarity_threshold=SIMILARITY_THRESHOLD,  # USE CONFIG (0.35)
        top_k=TOP_K_RESULTS                        # USE CONFIG (5)
    )
```

**Add this import at the top:**
```python
from rag.config import SIMILARITY_THRESHOLD, TOP_K_RESULTS
```

### File 2: `/home/user/career-prediction/backend/agents/workers/skill_development.py`

**Current Code (Lines 59-64):**
```python
# Initialize RAG retriever for knowledge base (skill collection)
try:
    self.rag_retriever = AgenticRAGRetriever(
        collection_type="skill",
        similarity_threshold=0.7,  # HARDCODED
        top_k=3                    # HARDCODED
    )
```

**Fixed Code:**
```python
# Initialize RAG retriever for knowledge base (skill collection)
from rag.config import SIMILARITY_THRESHOLD, TOP_K_RESULTS

try:
    self.rag_retriever = AgenticRAGRetriever(
        collection_type="skill",
        similarity_threshold=SIMILARITY_THRESHOLD,  # USE CONFIG (0.35)
        top_k=TOP_K_RESULTS                        # USE CONFIG (5)
    )
```

**Add this import at the top:**
```python
from rag.config import SIMILARITY_THRESHOLD, TOP_K_RESULTS
```

### Impact
- Lowers threshold from 0.7 to 0.35, allowing niche careers like Music Producer to retrieve documents
- Increases top_k from 3 to 5 for better coverage
- Aligns agents with intended configuration

**Expected Result**: Music Producer should now retrieve 3-5 relevant documents instead of 0.

---

## Fix #2: Enhance PDF Content (IMPORTANT)

### Problem
Only 2 PDF files exist, and they may lack music-specific content for Music Producer career.

### Solution
Add dedicated PDFs for creative industries/music.

### Create Music Production Guide PDF

Create a new file: `/home/user/career-prediction/backend/data/academic/music_production_pathways.pdf`

Should include:
- Music production universities and programs in Sri Lanka
- International institutions for music production
- Audio engineering programs
- Music technology degrees
- Online accredited music production courses

### Create Creative Industry Skills PDF

Create a new file: `/home/user/career-prediction/backend/data/skills/music_and_audio_skills.pdf`

Should include:
- Music production software (DAWs): Ableton Live, Logic Pro, FL Studio, Pro Tools, Reaper
- Audio engineering skills: mixing, mastering, sound design, acoustics
- Music theory and composition
- Music production certifications
- Audio engineering certifications
- Music production online courses and platforms
- Industry-standard tools and workflows

### Re-ingest Documents After Adding PDFs

```bash
cd /home/user/career-prediction/backend

# Option 1: Reset and re-ingest everything
python scripts/ingest_knowledge.py --collection all --reset

# Option 2: Just ingest new files
python scripts/ingest_knowledge.py --collection academic
python scripts/ingest_knowledge.py --collection skill
```

**Expected Result**: ChromaDB will now contain music-specific content that can be retrieved.

---

## Fix #3: Improve Query Formation for Creative Careers (RECOMMENDED)

### Problem
Generic queries may not match niche content well.

### Solution
Add career-aware query variants for creative/niche careers.

### File: `/home/user/career-prediction/backend/agents/workers/academic_pathway.py`

Add a helper method and update the retrieval call to use optimized queries.

### Impact
- Queries will be more semantically aligned with actual content in PDFs
- Better retrieval results for creative, music, and niche careers

---

## Fix #4: Add Logging to Debug Retrieval Issues (GOOD PRACTICE)

### File: `/home/user/career-prediction/backend/agents/workers/academic_pathway.py`

Add enhanced logging in RAG retrieval sections to help debug issues.

---

## Priority of Fixes

| Priority | Fix | Impact | Effort |
|----------|-----|--------|--------|
| **CRITICAL** | #1: Use Config Values | Immediately enable niche career retrieval | 5 min |
| **HIGH** | #2: Add Music PDFs | Provide actual music content to retrieve | 1-2 hours |
| **RECOMMENDED** | #3: Improve Queries | Better semantic matching for creative careers | 30 min |
| **GOOD PRACTICE** | #4: Add Logging | Better debugging for future issues | 15 min |

---

## Recommended Implementation Order

1. **First (Immediate)**: Apply Fix #1 - Change 4 lines in two files to use config values
2. **Second (Soon)**: Apply Fix #4 - Add logging to help debug issues
3. **Third (Important)**: Apply Fix #2 - Add music/creative content PDFs and re-ingest
4. **Fourth (Nice to Have)**: Apply Fix #3 - Improve query formation for niche careers

---

## Verification Checklist

After applying fixes:

- [ ] Fix #1 applied: SIMILARITY_THRESHOLD imported in both agent files
- [ ] Fix #1 applied: TOP_K_RESULTS imported in both agent files
- [ ] Fix #1 applied: Agent initialization uses config values instead of hardcoded
- [ ] Fix #2 applied: New PDFs added to data directories (if applicable)
- [ ] Fix #2 applied: Documents re-ingested
- [ ] Fix #3 applied: Query optimization methods added to agents
- [ ] Fix #4 applied: Enhanced logging added to RAG retrieval sections
- [ ] Tests run: python test_chroma_search.py shows SUCCESS for Music Producer
- [ ] Manual test passes: Music Producer returns 3+ relevant documents

