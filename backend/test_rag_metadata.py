"""
Test RAG Metadata Filtering

This script tests whether the RAG retriever can filter by metadata fields
like 'country' and 'institution_type' for Sri Lankan universities.
"""

import logging
import sys
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from rag.retriever import AgenticRAGRetriever

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_metadata_filtering():
    """Test RAG retrieval with metadata filters."""
    
    print("\n" + "="*80)
    print("🧪 TESTING RAG METADATA FILTERING")
    print("="*80 + "\n")
    
    try:
        # Initialize RAG retriever
        print("📦 Initializing RAG retriever...")
        retriever = AgenticRAGRetriever(
            collection_type="academic",
            provider="fallback",
            similarity_threshold=0.35,
            top_k=5
        )
        print("✅ RAG retriever initialized successfully\n")
        
        # Test 1: Retrieve documents with country filter (Sri Lanka)
        print("="*80)
        print("TEST 1: Country Filter - Sri Lanka")
        print("="*80)
        
        query1 = "computer science university programs admission requirements"
        print(f"Query: {query1}")
        print(f"Filter: {{'country': 'Sri Lanka'}}\n")
        
        result1 = retriever.retrieve(
            query=query1,
            force_retrieval=True,
            filter_metadata={"country": "Sri Lanka"},
            include_citations=True
        )
        
        print(f"📊 Results:")
        print(f"   - Documents retrieved: {len(result1.retrieved_documents)}")
        print(f"   - Context length: {len(result1.context)} characters")
        
        if result1.retrieved_documents:
            print(f"\n📄 Document Metadata (first 3):")
            for i, doc in enumerate(result1.retrieved_documents[:3], 1):
                print(f"\n   Document {i}:")
                print(f"   - Country: {doc.metadata.get('country', 'N/A')}")
                print(f"   - University: {doc.metadata.get('university', 'N/A')}")
                print(f"   - Category: {doc.metadata.get('category', 'N/A')}")
                print(f"   - Institution Type: {doc.metadata.get('institution_type', 'N/A')}")
                print(f"   - Source: {doc.metadata.get('source_file', 'N/A')}")
                print(f"   - Content preview: {doc.page_content[:150]}...")
        else:
            print("   ⚠️ No documents found with this filter")
        
        # Test 2: Retrieve documents with country + institution_type filter
        print("\n" + "="*80)
        print("TEST 2: Country + Institution Type Filter")
        print("="*80)
        
        query2 = "engineering university programs"
        print(f"Query: {query2}")
        print(f"Filter: {{'country': 'Sri Lanka', 'institution_type': 'government'}}\n")
        
        result2 = retriever.retrieve(
            query=query2,
            force_retrieval=True,
            filter_metadata={
                "country": "Sri Lanka",
                "institution_type": "government"
            },
            include_citations=True
        )
        
        print(f"📊 Results:")
        print(f"   - Documents retrieved: {len(result2.retrieved_documents)}")
        print(f"   - Context length: {len(result2.context)} characters")
        
        if result2.retrieved_documents:
            print(f"\n📄 Document Metadata (first 3):")
            for i, doc in enumerate(result2.retrieved_documents[:3], 1):
                print(f"\n   Document {i}:")
                print(f"   - Country: {doc.metadata.get('country', 'N/A')}")
                print(f"   - Institution Type: {doc.metadata.get('institution_type', 'N/A')}")
                print(f"   - University: {doc.metadata.get('university', 'N/A')}")
                print(f"   - Source: {doc.metadata.get('source_file', 'N/A')}")
        else:
            print("   ⚠️ No documents found with this filter")
            print("   💡 This means 'institution_type' metadata is NOT available in the vector database")
        
        # Test 3: Check available metadata fields
        print("\n" + "="*80)
        print("TEST 3: Available Metadata Fields")
        print("="*80)
        
        query3 = "university"
        print(f"Query: {query3}")
        print(f"Filter: {{'country': 'Sri Lanka'}}\n")
        
        result3 = retriever.retrieve(
            query=query3,
            force_retrieval=True,
            filter_metadata={"country": "Sri Lanka"},
            include_citations=True
        )
        
        if result3.retrieved_documents:
            print("📋 All available metadata fields in first document:")
            first_doc_metadata = result3.retrieved_documents[0].metadata
            for key, value in first_doc_metadata.items():
                print(f"   - {key}: {value}")
        
        # Summary
        print("\n" + "="*80)
        print("📊 TEST SUMMARY")
        print("="*80)
        print(f"✅ Test 1 (Country filter): {len(result1.retrieved_documents)} documents found")
        print(f"{'✅' if len(result2.retrieved_documents) > 0 else '❌'} Test 2 (Country + Institution Type): {len(result2.retrieved_documents)} documents found")
        
        if len(result2.retrieved_documents) == 0:
            print("\n⚠️ WARNING: 'institution_type' metadata is NOT present in the vector database!")
            print("💡 You need to add 'institution_type' metadata during document ingestion.")
            print("   Update document_processor.py to extract institution_type from PDF content.")
        else:
            print("\n✅ SUCCESS: Metadata filtering is working correctly!")
        
        print("\n" + "="*80 + "\n")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_metadata_filtering()
