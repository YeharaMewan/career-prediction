# RAG Failure - Code Snippets & Evidence

## 1. CRITICAL: Hardcoded Similarity Threshold in Agents

### Academic Pathway Agent - WRONG
**File**: `/home/user/career-prediction/backend/agents/workers/academic_pathway.py`
**Lines**: 91-96

```python
# Initialize RAG retriever for knowledge base (academic collection)
try:
    self.rag_retriever = AgenticRAGRetriever(
        collection_type="academic",
        similarity_threshold=0.7,    # ❌ HARDCODED - SHOULD USE CONFIG
        top_k=3                       # ❌ HARDCODED - SHOULD USE CONFIG (5)
    )
```

### Skill Development Agent - WRONG
**File**: `/home/user/career-prediction/backend/agents/workers/skill_development.py`
**Lines**: 59-64

```python
# Initialize RAG retriever for knowledge base (skill collection)
try:
    self.rag_retriever = AgenticRAGRetriever(
        collection_type="skill",
        similarity_threshold=0.7,     # ❌ HARDCODED - SHOULD USE CONFIG
        top_k=3                        # ❌ HARDCODED - SHOULD USE CONFIG (5)
    )
```

### Config File - CORRECT VALUES NOT USED
**File**: `/home/user/career-prediction/backend/rag/config.py`
**Lines**: 38-41

```python
# Retrieval Configuration
SIMILARITY_THRESHOLD = 0.35  # Minimum similarity score (0-1) - Lowered to capture semantically related content for niche careers
TOP_K_RESULTS = 5  # Number of chunks to retrieve - Increased for better coverage
MAX_CONTEXT_LENGTH = 4000  # Maximum characters in retrieved context
```

---

## 2. Query Formation for Music Producer

### Academic Pathway Query Formation
**File**: `/home/user/career-prediction/backend/agents/workers/academic_pathway.py`
**Lines**: 728-733

```python
def _build_academic_planning_prompt(...):
    # Perform RAG retrieval from knowledge base
    rag_context = ""
    if self.rag_enabled:
        self.logger.info(f"Retrieving knowledge base information for {career_title}")
        # Optimized query: shorter, keyword-focused for better semantic matching
        rag_state = self.rag_retriever.retrieve(
            query=f"{career_title} education degrees programs institutions training Sri Lanka",
            include_citations=True
        )
        # For Music Producer, this becomes:
        # "Music Producer education degrees programs institutions training Sri Lanka"
        
        if rag_state.context:
            rag_context = rag_state.context
            self.logger.info(f"RAG retrieved {len(rag_state.retrieved_documents)} relevant documents")
        else:
            self.logger.info("No relevant documents found in knowledge base")
```

### Skill Development Query Formation
**File**: `/home/user/career-prediction/backend/agents/workers/skill_development.py`
**Lines**: 613-624

```python
def _build_skill_planning_prompt(...):
    # Perform RAG retrieval from knowledge base
    rag_context = ""
    if self.rag_enabled:
        self.logger.info(f"Retrieving skill knowledge base information for {career_title}")
        # Optimized query: shorter, keyword-focused for better semantic matching
        rag_state = self.rag_retriever.retrieve(
            query=f"{career_title} skills courses certifications training learning resources",
            include_citations=True
        )
        # For Music Producer, this becomes:
        # "Music Producer skills courses certifications training learning resources"
        
        if rag_state.context:
            rag_context = rag_state.context
            self.logger.info(f"RAG retrieved {len(rag_state.retrieved_documents)} relevant documents")
        else:
            self.logger.info("No relevant documents found in skill knowledge base")
```

---

## 3. Retriever Decision & Filtering Logic

### Retriever's Keyword Decision
**File**: `/home/user/career-prediction/backend/rag/retriever.py`
**Lines**: 88-114

```python
def decide_retrieval(self, query: str) -> tuple[bool, str]:
    """
    Decide if retrieval is needed based on query content.
    
    Uses keyword matching to determine if query needs knowledge base.
    """
    query_lower = query.lower()
    
    # Check for retrieval keywords
    found_keywords = [
        kw for kw in NEEDS_RETRIEVAL_KEYWORDS if kw in query_lower
    ]
    
    if found_keywords:
        reason = f"Query contains knowledge keywords: {', '.join(found_keywords[:3])}"
        logger.info(f"Retrieval NEEDED: {reason}")
        return True, reason  # ✅ Will attempt retrieval
    else:
        reason = "Query does not require knowledge base retrieval"
        logger.info(f"Retrieval NOT needed: {reason}")
        return False, reason  # ❌ Will skip retrieval
```

### NEEDS_RETRIEVAL_KEYWORDS
**File**: `/home/user/career-prediction/backend/rag/config.py`
**Lines**: 44-48

```python
NEEDS_RETRIEVAL_KEYWORDS = [
    "university", "institution", "college", "degree", "program",
    "course", "skill", "certification", "training", "learn",
    "career", "job", "profession", "industry", "sector"
]
```

For Music Producer query: "Music Producer education degrees programs institutions training Sri Lanka"
- Contains: "education" ✅, "degrees" ✅, "programs" ✅, "institutions" ✅, "training" ✅
- Result: decide_retrieval() returns TRUE, retrieval will proceed

### Threshold Filtering (THE PROBLEM)
**File**: `/home/user/career-prediction/backend/rag/retriever.py`
**Lines**: 173-200

```python
def _filter_by_threshold(self, results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Filter results by similarity threshold.
    
    THIS IS WHERE MUSIC PRODUCER RESULTS GET FILTERED OUT!
    """
    documents = []
    metadatas = []
    similarities = []
    
    for doc, meta, sim in zip(
        results["documents"], results["metadatas"], results["similarities"]
    ):
        # ❌ PROBLEM: For Music Producer, sim might be 0.35-0.55
        # But self.similarity_threshold is 0.7 (hardcoded in agent)
        if sim >= self.similarity_threshold:  # 0.7 is TOO HIGH
            documents.append(doc)
            metadatas.append(meta)
            similarities.append(sim)
    
    return {
        "documents": documents,
        "metadatas": metadatas,
        "similarities": similarities,
        "count": len(documents),  # ← Returns 0 for Music Producer!
    }
```

### Retriever Entry Point
**File**: `/home/user/career-prediction/backend/rag/retriever.py`
**Lines**: 255-307

```python
def retrieve(
    self,
    query: str,
    force_retrieval: bool = False,
    filter_metadata: Optional[Dict[str, Any]] = None,
    include_citations: bool = True,
) -> RetrievalState:
    """
    Complete agentic retrieval workflow.
    """
    # Initialize state
    state = RetrievalState(query=query, collection_type=self.collection_type)
    
    # Step 1: Decide if retrieval is needed
    if force_retrieval:
        state.needs_retrieval = True
        state.decision_reason = "Forced retrieval"
    else:
        needs_retrieval, reason = self.decide_retrieval(query)
        state.needs_retrieval = needs_retrieval
        state.decision_reason = reason
    
    # Step 2: Retrieve documents if needed
    if state.needs_retrieval:
        results = self.retrieve_documents(query, filter_metadata)
        # ↓ Results contain raw similarity scores (0.0-1.0)
        state.retrieved_documents = results["documents"]
        state.metadatas = results["metadatas"]
        state.similarities = results["similarities"]
        # ↓ retrieve_documents() calls _filter_by_threshold() internally!
        
        # Step 3: Format context
        if results["count"] > 0:  # ← Will be 0 for Music Producer due to threshold
            state.context = self.format_context(...)
        else:
            logger.warning("No documents found above similarity threshold")
            state.context = "No relevant information found in knowledge base."
    else:
        logger.info("Skipping retrieval based on decision")
    
    return state  # ← Returns empty context for Music Producer
```

---

## 4. Retriever Initialization Path

### Agent calls __init__ with hardcoded values
**File**: `/home/user/career-prediction/backend/agents/workers/academic_pathway.py`
**Lines**: 91-96

```python
self.rag_retriever = AgenticRAGRetriever(
    collection_type="academic",
    similarity_threshold=0.7,  # ❌ NOT FROM CONFIG
    top_k=3                    # ❌ NOT FROM CONFIG
)
```

### Retriever stores these values
**File**: `/home/user/career-prediction/backend/rag/retriever.py`
**Lines**: 55-86

```python
def __init__(
    self,
    collection_type: str,
    similarity_threshold: float = SIMILARITY_THRESHOLD,  # Default is 0.35
    top_k: int = TOP_K_RESULTS,                          # Default is 5
    max_context_length: int = MAX_CONTEXT_LENGTH,
):
    """
    Initialize agentic RAG retriever.
    """
    self.collection_type = collection_type
    self.similarity_threshold = similarity_threshold  # ← Gets 0.7 from agent
    self.top_k = top_k  # ← Gets 3 from agent
    self.max_context_length = max_context_length
```

---

## 5. Testing Evidence of the Issue

### Test File Aware of Music Producer Problem
**File**: `/home/user/career-prediction/backend/test_chroma_search.py`
**Lines**: 1-50

```python
"""
Test script to verify Chroma DB search improvements for niche careers.

This script tests that:
1. Lowered threshold (0.35) captures relevant music programs  ← TEAM TRIED TO FIX!
2. Optimized queries produce better semantic matches
3. Top-5 results provide better coverage
"""

def test_music_producer_search():
    """Test search for Music Producer career."""
    
    print("CHROMA DB SEARCH IMPROVEMENTS TEST")
    print(f"Configuration:")
    print(f"  - Similarity Threshold: {SIMILARITY_THRESHOLD}")  # 0.35 in config
    print(f"  - Top-K Results: {TOP_K_RESULTS}")  # 5 in config
    
    # Initialize retriever for academic collection
    academic_retriever = AgenticRAGRetriever(collection_type="academic")
    # ❌ NOTE: When retriever is initialized without params, it uses config defaults (0.35, 5)
    #    But when agents initialize it, they hardcode (0.7, 3)!
    
    # Test 1: Music Producer with optimized query
    career_title = "Music Producer"
    optimized_query = f"{career_title} education degrees programs institutions training Sri Lanka"
    
    result = academic_retriever.retrieve(
        query=optimized_query,
        include_citations=True
    )
    
    if result.retrieved_documents:
        print(f"[SUCCESS] Retrieved {len(result.retrieved_documents)} relevant documents")
    else:
        print("[FAILURE] No documents retrieved!")  ← THIS HAPPENS WITH AGENT's 0.7 THRESHOLD
```

---

## 6. Data Files - No Music Content

### Academic Data Directory
**Path**: `/home/user/career-prediction/backend/data/academic/`

```
-rw-r--r-- 1 root root 507854 Nov 15 04:58 Sri_Lanka_Universities_Complete_Enhanced.pdf
```

Only ONE academic PDF file, specifically about Sri Lankan universities

### Skills Data Directory
**Path**: `/home/user/career-prediction/backend/data/skills/`

```
-rw-r--r-- 1 root root 394464 Nov 15 04:58 skills.pdf
```

Only ONE skills PDF file

### Vector Store Storage
**Path**: `/home/user/career-prediction/backend/knowledge_base/chroma/`

```
-rw-r--r-- 1 root root 2670592 Nov 15 04:58 chroma.sqlite3
drwxr-xr-x 2 root root    4096 Nov 15 04:58 54db6257-cdbe-45d9-b117-9dc093c6f513/  (academic collection)
drwxr-xr-x 2 root root    4096 Nov 15 04:58 f104257d-dfe1-4720-9cf1-31b1d809d9ea/  (skill collection)
```

---

## 7. Query Processing Flow

When a user says: **"I want to pursue Music Producer"**

### 1. Agent Initialization
```python
# In academic_pathway.py __init__
self.rag_retriever = AgenticRAGRetriever(
    collection_type="academic",
    similarity_threshold=0.7,  # ❌ WRONG
    top_k=3                    # ❌ WRONG
)
# Instead of using config: 0.35 and 5
```

### 2. Query Formation
```python
# In academic_pathway.py _build_academic_planning_prompt()
career_title = "Music Producer"
query = f"{career_title} education degrees programs institutions training Sri Lanka"
# Result: "Music Producer education degrees programs institutions training Sri Lanka"
```

### 3. Retriever Call
```python
# In academic_pathway.py process_task()
rag_state = self.rag_retriever.retrieve(
    query="Music Producer education degrees programs institutions training Sri Lanka",
    include_citations=True
)
```

### 4. Inside Retriever.retrieve()
```python
# Step 1: decide_retrieval()
needs_retrieval, reason = self.decide_retrieval(query)
# Query contains: "education", "degrees", "programs", "institutions", "training"
# Result: needs_retrieval = True ✅

# Step 2: retrieve_documents()
results = self.retrieve_documents(query, filter_metadata)
# ChromaDB similarity scores for Music Producer:
# Result 1: similarity=0.45
# Result 2: similarity=0.38
# Result 3: similarity=0.52

# Step 3: _filter_by_threshold()
# For each result:
#   if sim >= self.similarity_threshold (0.7):  # ← HARDCODED TO 0.7
#       keep document
#   else:
#       filter out
# All three results (0.45, 0.38, 0.52) fail the 0.7 check!
# Result: 0 documents pass filter

# Step 4: Return
return RetrievalState(
    needs_retrieval=True,
    retrieved_documents=[],    # ← EMPTY!
    context="No relevant information found..."  # ← EMPTY!
)
```

### 5. In Agent's Prompt Building
```python
if rag_state.context:  # ← Empty string is falsy
    prompt_parts.extend([
        "KNOWLEDGE BASE INFORMATION:",
        rag_context,  # ← EMPTY
    ])
else:
    # ❌ No knowledge base data included
```

---

## SUMMARY TABLE

| Component | Expected (Config) | Actual (Agent) | Impact |
|-----------|-------------------|----------------|--------|
| similarity_threshold | 0.35 | 0.7 | Music data filtered out |
| top_k | 5 | 3 | Fewer results attempted |
| Query trigger | Keyword-based | Keyword-based | Works for Music Producer |
| Retrieval attempt | Should proceed | Does proceed | ✅ |
| Similarity filtering | Should be lenient (0.35) | Strict (0.7) | ❌ FAILURE |
| Final context | With music data | Empty "" | RAG system fails |

