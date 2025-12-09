"""Quick RAG Metadata Test"""
from rag.retriever import AgenticRAGRetriever

print("\n" + "="*80)
print("RAG METADATA TEST")
print("="*80 + "\n")

# Initialize with OpenAI
r = AgenticRAGRetriever(
    collection_type='academic', 
    provider='openai',
    top_k=3
)

# Test 1: Country filter only
print("TEST 1: Filter by country='Sri Lanka'\n")
result1 = r.retrieve(
    query='university computer science programs',
    filter_metadata={'country': 'Sri Lanka'},
    force_retrieval=True
)

print(f"Results Found: {len(result1.retrieved_documents)} documents\n")

if result1.retrieved_documents:
    for i in range(min(2, len(result1.retrieved_documents))):
        print(f"Document {i+1}:")
        if i < len(result1.metadatas):
            metadata = result1.metadatas[i]
            print(f"   Country: {metadata.get('country', 'N/A')}")
            print(f"   University: {metadata.get('university', 'N/A')}")
            print(f"   Institution Type: {metadata.get('institution_type', 'NOT PRESENT')}")
            print(f"   Category: {metadata.get('category', 'N/A')}")
            print(f"   All metadata keys: {list(metadata.keys())}")
        print()

# Test 2: Country + Institution Type filter
print("\n" + "="*80)
print("TEST 2: Filter by country='Sri Lanka' AND institution_type='government'\n")
result2 = r.retrieve(
    query='university engineering programs',
    filter_metadata={
        "$and": [
            {'country': 'Sri Lanka'},
            {'institution_type': 'government'}
        ]
    },
    force_retrieval=True
)

print(f"Results Found: {len(result2.retrieved_documents)} documents\n")

if len(result2.retrieved_documents) == 0:
    print("NO DOCUMENTS FOUND")
    print("This means 'institution_type' metadata is NOT in the vector database")
    print("   You need to update document ingestion to add this metadata field\n")
else:
    print("SUCCESS! institution_type metadata exists\n")
    for i in range(min(2, len(result2.retrieved_documents))):
        print(f"Document {i+1}:")
        if i < len(result2.metadatas):
            metadata = result2.metadatas[i]
            print(f"   Institution Type: {metadata.get('institution_type')}")
            print(f"   University: {metadata.get('university', 'N/A')}")
        print()

print("="*80)
