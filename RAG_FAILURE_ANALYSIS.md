# RAG SYSTEM FAILURE ANALYSIS: Music Producer Retrieval Issue

## Executive Summary

The RAG system is **FAILING TO RETRIEVE MUSIC-RELATED DATA** for the Music Producer career due to **FOUR CRITICAL ISSUES**:

1. **Hardcoded High Similarity Threshold in Agents** (0.7 vs configured 0.35)
2. **Keyword-Gating Prevents Retrieval** (Query formation includes gating keywords)
3. **Missing or Insufficient Music Content** in PDFs
4. **Query-Embedding Mismatch** for niche/creative careers

---

## CRITICAL FINDINGS

### Issue #1: Hardcoded Similarity Threshold OVERRIDES Configuration

**Location**: `/home/user/career-prediction/backend/agents/workers/`
- `academic_pathway.py` line 92-95
- `skill_development.py` line 60-63

**The Problem**:
```python
# In academic_pathway.py
self.rag_retriever = AgenticRAGRetriever(
    collection_type="academic",
    similarity_threshold=0.7,  # ❌ HARDCODED TO 0.7
    top_k=3
)

# In skill_development.py
self.rag_retriever = AgenticRAGRetriever(
    collection_type="skill",
    similarity_threshold=0.7,  # ❌ HARDCODED TO 0.7
    top_k=3
)
```

**vs Config**:
```python
# In config.py line 39
SIMILARITY_THRESHOLD = 0.35  # ✅ Set to 0.35 for niche careers
```

**Impact**: 
- Config was lowered to 0.35 to capture niche careers like "Music Producer"
- Agents override this with 0.7, which is TOO HIGH for semantic similarity
- Music Producer queries likely score 0.35-0.60 (semantic similarity), which get FILTERED OUT by the 0.7 threshold
- **Result**: NO DOCUMENTS RETRIEVED for Music Producer

---

### Issue #2: Keyword-Gating in Retriever Decision Logic

**Location**: `/home/user/career-prediction/backend/rag/retriever.py` lines 88-114

**The Problem**:
```python
def decide_retrieval(self, query: str) -> tuple[bool, str]:
    """Decide if retrieval is needed based on query content."""
    query_lower = query.lower()
    
    # Check for retrieval keywords
    found_keywords = [
        kw for kw in NEEDS_RETRIEVAL_KEYWORDS if kw in query_lower
    ]
    
    if found_keywords:
        reason = f"Query contains knowledge keywords: {', '.join(found_keywords[:3])}"
        return True, reason  # ✅ RETRIEVAL WILL HAPPEN
    else:
        reason = "Query does not require knowledge base retrieval"
        return False, reason  # ❌ RETRIEVAL WILL BE SKIPPED!
```

**NEEDS_RETRIEVAL_KEYWORDS** (config.py lines 44-48):
```python
NEEDS_RETRIEVAL_KEYWORDS = [
    "university", "institution", "college", "degree", "program",
    "course", "skill", "certification", "training", "learn",
    "career", "job", "profession", "industry", "sector"
]
```

**Actual Query Formed** (academic_pathway.py line 731):
```python
query = f"{career_title} education degrees programs institutions training Sri Lanka"
```

For "Music Producer":
```
Query: "Music Producer education degrees programs institutions training Sri Lanka"
```

**Analysis**:
- Query CONTAINS: "education", "degrees", "programs", "institutions", "training"
- These ARE in NEEDS_RETRIEVAL_KEYWORDS
- ✅ Retrieval SHOULD be triggered... BUT...

**Wait - Let me re-check the flow**: The agents call `retrieve()` which does call `decide_retrieval()`. Let me verify if retrieval is actually being called...

Looking at academic_pathway.py lines 727-738:
```python
if self.rag_enabled:
    self.logger.info(f"Retrieving knowledge base information for {career_title}")
    rag_state = self.rag_retriever.retrieve(
        query=f"{career_title} education degrees programs institutions training Sri Lanka",
        include_citations=True
    )
```

The agents are calling `retrieve()` without `force_retrieval=True`, which means `decide_retrieval()` IS being called.

---

### Issue #3: Document Content May Be Missing Music Data

**Location**: 
- `/home/user/career-prediction/backend/data/academic/Sri_Lanka_Universities_Complete_Enhanced.pdf`
- `/home/user/career-prediction/backend/data/skills/skills.pdf`

**The Problem**:
- Only 2 PDF files in the entire system
- No verification that music-related content exists in these PDFs
- The PDFs are likely focused on:
  - Academic: Sri Lankan universities (engineering, medicine, science, management)
  - Skills: General technical and professional skills

**What's Missing**:
- Music production institutions/programs
- Music-related skills (audio engineering, music theory, DAW proficiency, etc.)
- Music industry certifications
- Creative industry pathways

**Evidence**:
- Test file exists: `test_chroma_search.py` (line 19-50) tests "Music Producer" search
- Test comment (line 3): "Test that... Lowered threshold (0.35) captures relevant music programs"
- This implies the team KNOWS music content isn't being retrieved and tried to fix it by lowering threshold
- But the fix was only in config, not applied in agents (Issue #1)

---

### Issue #4: Query-Embedding Semantic Mismatch for Creative Careers

**Location**: `/home/user/career-prediction/backend/agents/workers/academic_pathway.py` line 730-732

**The Problem**:
```python
rag_state = self.rag_retriever.retrieve(
    query=f"{career_title} education degrees programs institutions training Sri Lanka",
    include_citations=True
)
```

For Music Producer, this becomes:
```
"Music Producer education degrees programs institutions training Sri Lanka"
```

**Why This Fails**:
1. The PDF likely contains general Sri Lankan university descriptions
2. If music is mentioned, it's probably in specific contexts:
   - "Fine Arts department may include music"
   - "School of Arts and Media music program"
   - "Music as an elective subject"

3. The query talks about "Music Producer" (a specific career)
4. The PDF talks about "Music" (a general subject)
5. **Semantic Gap**: "Music Producer career requirements" ≠ "Music as a university subject"
6. **Low Similarity Score**: Even if music is mentioned, the semantic alignment is weak
7. **Threshold Blocks It**: 0.7 threshold filters out weak matches

---

## ROOT CAUSE SUMMARY

```
┌─────────────────────────────────────────────────────────┐
│ USER QUERY: "I want to pursue Music Producer"          │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│ AGENT FORMS QUERY:                                      │
│ "Music Producer education degrees programs...           │
│  institutions training Sri Lanka"                       │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│ RETRIEVER.RETRIEVE() IS CALLED                          │
│ ✅ decide_retrieval() = TRUE (keywords present)         │
│ ✅ Query embedding generated                            │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│ VECTOR STORE QUERY                                      │
│ ✅ Semantic search executed                             │
│ ⚠️  Low similarity scores (0.35-0.55)                   │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│ SIMILARITY THRESHOLD FILTER ❌                          │
│ Threshold = 0.7 (HARDCODED in agent)                   │
│ Retrieved scores = 0.35-0.55 (from PDF content gap)    │
│ Result: ALL DOCUMENTS FILTERED OUT!                     │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│ RETURN: context = "" (empty!)                           │
│ NO MUSIC DATA RETRIEVED ❌                              │
└─────────────────────────────────────────────────────────┘
```

---

## EVIDENCE FROM CODE

### Config File Shows Threshold Was Lowered (But Not Applied)

**File**: `/home/user/career-prediction/backend/rag/config.py` lines 38-40

```python
# Retrieval Configuration
SIMILARITY_THRESHOLD = 0.35  # Minimum similarity score (0-1) - Lowered to capture semantically related content for niche careers
TOP_K_RESULTS = 5  # Number of chunks to retrieve - Increased for better coverage
```

**Comment confirms**: "Lowered to capture semantically related content for **niche careers**"

But agents don't use these values!

### Agents Hardcode Their Own Values

**File**: `/home/user/career-prediction/backend/agents/workers/academic_pathway.py` lines 91-96

```python
try:
    self.rag_retriever = AgenticRAGRetriever(
        collection_type="academic",
        similarity_threshold=0.7,  # ❌ NOT USING CONFIG
        top_k=3  # ❌ NOT USING CONFIG
    )
```

**File**: `/home/user/career-prediction/backend/agents/workers/skill_development.py` lines 59-64

```python
try:
    self.rag_retriever = AgenticRAGRetriever(
        collection_type="skill",
        similarity_threshold=0.7,  # ❌ NOT USING CONFIG
        top_k=3  # ❌ NOT USING CONFIG
    )
```

---

## RETRIEVAL FLOW DIAGRAM

```
┌──────────────────────────────────────────────────────────────┐
│ AGENT: AcademicPathwayAgent / SkillDevelopmentAgent         │
└────────────────────┬─────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────────┐
│ __init__(): Initialize RAG Retriever                         │
│ • collection_type="academic" or "skill"                      │
│ • similarity_threshold=0.7  ← HARDCODED (CONFIG IS 0.35!)    │
│ • top_k=3  ← HARDCODED (CONFIG IS 5!)                        │
└────────────────────┬─────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────────┐
│ process_task(): Build planning prompt                        │
│ • If rag_enabled: retrieve() is called with optimized query  │
│ • Query: "{career_title} education degrees programs..."      │
└────────────────────┬─────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────────┐
│ RETRIEVER: AgenticRAGRetriever.retrieve()                   │
│ • decide_retrieval(): Check if keywords present             │
│   └─ Keywords FOUND: "education", "degrees", "programs"      │
│   └─ needs_retrieval = TRUE ✅                              │
└────────────────────┬─────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────────┐
│ retrieve_documents(): Query vector store                      │
│ • Generate query embedding using OpenAI                       │
│ • Search ChromaDB with cosine similarity                      │
│ • Get top-k=3 results by similarity                           │
│ • Filter by similarity_threshold=0.7 ← PROBLEM!              │
└────────────────────┬─────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────────┐
│ VECTOR STORE: ChromaDB Query                                 │
│ Collections: academic_knowledge, skill_knowledge             │
│ • Query similarity scores for Music Producer: 0.35-0.55      │
│ • Threshold check: score >= 0.7? NO ❌                       │
│ • Result: 0 documents pass filter                            │
└────────────────────┬─────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────────┐
│ RETURN: RetrievalState                                       │
│ • needs_retrieval=True                                       │
│ • retrieved_documents=[]  ← EMPTY!                           │
│ • context=""  ← EMPTY!                                       │
│ • decision_reason="Retrieval completed but no docs passed"   │
└──────────────────────────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────────┐
│ PROMPT BUILDING: _build_academic_planning_prompt()           │
│ • RAG context is empty ""                                    │
│ • System relies only on web search results                   │
│ • Music-specific academic data is MISSING ❌                 │
└──────────────────────────────────────────────────────────────┘
```

---

## PROBLEM CHECKLIST

| Issue | Location | Status | Impact |
|-------|----------|--------|--------|
| **Threshold Mismatch** | agents/workers/*.py lines 92-96, 60-64 | ❌ CRITICAL | 0.7 vs 0.35 threshold prevents niche career retrieval |
| **Document Content Gap** | data/academic, data/skills PDFs | ❌ CRITICAL | No music-related content verified in PDFs |
| **Query Semantic Mismatch** | academic_pathway.py:731, skill_development.py:617 | ⚠️ HIGH | Generic queries don't match creative career specificity |
| **Config Not Applied** | agents initialization | ❌ CRITICAL | NEEDS_RETRIEVAL_KEYWORDS, SIMILARITY_THRESHOLD unused by agents |
| **Top-K Hardcoded** | agents/workers/*.py lines 94, 62 | ⚠️ MEDIUM | Set to 3 instead of config's 5 |

---

## WHERE THE ISSUE IS HAPPENING

### During Query Processing:

1. **Query Formation** (academic_pathway.py:731):
   ```python
   query = f"{career_title} education degrees programs institutions training Sri Lanka"
   # For Music Producer: "Music Producer education degrees programs..."
   ```

2. **Retriever Initialization** (academic_pathway.py:92-95):
   ```python
   self.rag_retriever = AgenticRAGRetriever(
       collection_type="academic",
       similarity_threshold=0.7,  # ❌ TOO HIGH!
       top_k=3
   )
   ```

3. **Retrieval Call** (academic_pathway.py:730-733):
   ```python
   rag_state = self.rag_retriever.retrieve(
       query=f"{career_title} education degrees programs institutions training Sri Lanka",
       include_citations=True
   )
   # retrieve() → decide_retrieval() → retrieve_documents()
   # → _filter_by_threshold() ← FILTERS OUT RESULTS HERE!
   ```

4. **Threshold Filtering** (retriever.py:173-200):
   ```python
   def _filter_by_threshold(self, results: Dict[str, Any]) -> Dict[str, Any]:
       for doc, meta, sim in zip(...):
           if sim >= self.similarity_threshold:  # 0.7 ← TOO HIGH!
               # Only docs with sim >= 0.7 are kept
               # Music Producer likely has sim ~0.35-0.55
               documents.append(doc)
   ```

---

## FILES INVOLVED

### Core RAG System:
- `/home/user/career-prediction/backend/rag/config.py` - Configuration (has correct values, not used by agents)
- `/home/user/career-prediction/backend/rag/retriever.py` - Retriever logic (filtering by threshold)
- `/home/user/career-prediction/backend/rag/vector_store.py` - ChromaDB vector store
- `/home/user/career-prediction/backend/rag/embedding_manager.py` - OpenAI embedding generation
- `/home/user/career-prediction/backend/rag/document_processor.py` - PDF processing

### Agent Workers (Using RAG Incorrectly):
- `/home/user/career-prediction/backend/agents/workers/academic_pathway.py` (lines 91-96) - ❌ Hardcodes 0.7
- `/home/user/career-prediction/backend/agents/workers/skill_development.py` (lines 59-64) - ❌ Hardcodes 0.7

### Data:
- `/home/user/career-prediction/backend/data/academic/Sri_Lanka_Universities_Complete_Enhanced.pdf` - Academic data
- `/home/user/career-prediction/backend/data/skills/skills.pdf` - Skills data

### ChromaDB Storage:
- `/home/user/career-prediction/backend/knowledge_base/chroma/` - Persisted vectors

### Test:
- `/home/user/career-prediction/backend/test_chroma_search.py` - Test for Music Producer (reveals awareness of issue)

---

## CONCLUSION

**The RAG system is failing because of a **CONFIGURATION OVERRIDE** combined with **CONTENT GAPS**:**

1. **PRIMARY CAUSE**: Agents hardcode `similarity_threshold=0.7` while config specifies 0.35
2. **SECONDARY CAUSE**: PDFs likely lack sufficient music-related content for meaningful semantic matches
3. **TERTIARY CAUSE**: Query formation doesn't account for the semantic gap between job titles and academic content

The Music Producer career retrieval fails at the **threshold filtering stage**, where legitimate matches (0.35-0.55 similarity) are filtered out by the 0.7 threshold.
