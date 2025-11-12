"""
Test script to verify Chroma DB search improvements for niche careers.

This script tests that:
1. Lowered threshold (0.35) captures relevant music programs
2. Optimized queries produce better semantic matches
3. Top-5 results provide better coverage
"""

import sys
import os

# Add the backend directory to the path
sys.path.insert(0, os.path.dirname(__file__))

from rag.retriever import AgenticRAGRetriever
from rag.config import SIMILARITY_THRESHOLD, TOP_K_RESULTS

def test_music_producer_search():
    """Test search for Music Producer career."""

    print("=" * 80)
    print("CHROMA DB SEARCH IMPROVEMENTS TEST")
    print("=" * 80)
    print()

    print(f"Configuration:")
    print(f"  - Similarity Threshold: {SIMILARITY_THRESHOLD}")
    print(f"  - Top-K Results: {TOP_K_RESULTS}")
    print()

    # Initialize retriever for academic collection
    print("Initializing Academic RAG Retriever...")
    academic_retriever = AgenticRAGRetriever(collection_type="academic")

    # Test 1: Music Producer with optimized query
    print("-" * 80)
    print("TEST 1: Music Producer (Optimized Query)")
    print("-" * 80)

    career_title = "Music Producer"
    optimized_query = f"{career_title} education degrees programs institutions training Sri Lanka"

    print(f"Career: {career_title}")
    print(f"Query: {optimized_query}")
    print()

    result = academic_retriever.retrieve(
        query=optimized_query,
        include_citations=True
    )

    if result.retrieved_documents:
        print(f"[SUCCESS] Retrieved {len(result.retrieved_documents)} relevant documents:")
        print()

        # Check the structure of retrieved_documents
        if isinstance(result.retrieved_documents, list) and len(result.retrieved_documents) > 0:
            first_doc = result.retrieved_documents[0]

            # If documents are dictionaries
            if isinstance(first_doc, dict):
                for i, doc in enumerate(result.retrieved_documents, 1):
                    similarity = doc.get("similarity", 0.0)
                    content = doc.get("content", "")[:200]
                    source = doc.get("metadata", {}).get("source", "Unknown")
                    page = doc.get("metadata", {}).get("page", "N/A")

                    print(f"{i}. Similarity: {similarity:.3f}")
                    print(f"   Source: {source} (Page {page})")
                    print(f"   Content: {content}...")
                    print()
            # If documents are strings (just content)
            else:
                for i, doc in enumerate(result.retrieved_documents, 1):
                    content = str(doc)[:200] if doc else "No content"
                    print(f"{i}. Content: {content}...")
                    print()

        print(f"Context length: {len(result.context)} characters")
        print()
    else:
        print("[FAILURE] No documents retrieved!")
        print("This means the threshold is still too high or query needs further optimization.")
        print()

    # Test 2: Compare with old verbose query
    print("-" * 80)
    print("TEST 2: Music Producer (Old Verbose Query - for comparison)")
    print("-" * 80)

    verbose_query = f"Academic pathways, degrees, programs, and institutions for {career_title} career in Sri Lanka"

    print(f"Query: {verbose_query}")
    print()

    result_verbose = academic_retriever.retrieve(
        query=verbose_query,
        include_citations=True
    )

    if result_verbose.retrieved_documents:
        print(f"[SUCCESS] Retrieved {len(result_verbose.retrieved_documents)} relevant documents")
        print(f"Context length: {len(result_verbose.context)} characters")
        print()
    else:
        print("[INFO] No documents retrieved with verbose query.")
        print()

    # Test 3: Software Engineer (should still work well)
    print("-" * 80)
    print("TEST 3: Software Engineer (High-Confidence Career)")
    print("-" * 80)

    career_title_tech = "Software Engineer"
    tech_query = f"{career_title_tech} education degrees programs institutions training Sri Lanka"

    print(f"Career: {career_title_tech}")
    print(f"Query: {tech_query}")
    print()

    result_tech = academic_retriever.retrieve(
        query=tech_query,
        include_citations=True
    )

    if result_tech.retrieved_documents:
        print(f"[SUCCESS] Retrieved {len(result_tech.retrieved_documents)} relevant documents")
        print(f"Context length: {len(result_tech.context)} characters")
        print()
    else:
        print("[INFO] No documents retrieved for Software Engineer.")
        print()

    # Summary
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    music_success = len(result.retrieved_documents) > 0
    music_count = len(result.retrieved_documents)

    tech_success = len(result_tech.retrieved_documents) > 0
    tech_count = len(result_tech.retrieved_documents)

    print(f"[{'SUCCESS' if music_success else 'FAILURE'}] Music Producer: {music_count} documents retrieved")
    print(f"[{'SUCCESS' if tech_success else 'FAILURE'}] Software Engineer: {tech_count} documents retrieved")
    print()

    if music_success and music_count >= 3:
        print("[SUCCESS] Threshold and query optimization working correctly!")
        print("Music-related programs are now being captured (0.30-0.40 similarity range).")
    elif music_success and music_count > 0:
        print("[PARTIAL SUCCESS] Some music programs retrieved, but could use more coverage.")
    else:
        print("[FAILURE] Music programs still not being retrieved.")
        print("Further investigation needed.")

    print()
    print("=" * 80)


if __name__ == "__main__":
    test_music_producer_search()
