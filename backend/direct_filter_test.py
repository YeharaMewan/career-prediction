"""Direct ChromaDB Filter Test"""
from rag.vector_store import VectorStoreManager
from rag.embedding_manager import EmbeddingManager

print("\n" + "="*80)
print("DIRECT CHROMADB FILTER TEST")
print("="*80 + "\n")

# Initialize
em = EmbeddingManager(provider='gemini')
vs = VectorStoreManager(provider='gemini')

# Test query
query = "university engineering programs"
query_embedding = em.embed_query(query)

print("TEST 1: Filter with country='Sri Lanka' only\n")
results1 = vs.query(
    query_embedding=query_embedding,
    collection_type='academic',
    top_k=3,
    filter_metadata={'country': 'Sri Lanka'}
)
print(f"Results: {results1['count']} documents")
if results1['metadatas']:
    for i, meta in enumerate(results1['metadatas'][:2], 1):
        print(f"\nDoc {i}:")
        print(f"  University: {meta.get('university', 'N/A')}")
        print(f"  Country: {meta.get('country', 'N/A')}")
        print(f"  Institution Type: {meta.get('institution_type', 'N/A')}")

print("\n" + "="*80)
print("TEST 2: Filter with country='Sri Lanka' AND institution_type='government'\n")

# Try different filter syntax
filter_syntax_options = [
    # Option 1: Simple AND
    {'country': 'Sri Lanka', 'institution_type': 'government'},
    
    # Option 2: Explicit $and
    {'$and': [
        {'country': 'Sri Lanka'},
        {'institution_type': 'government'}
    ]},
]

for i, filter_meta in enumerate(filter_syntax_options, 1):
    print(f"\nFilter Syntax {i}: {filter_meta}")
    try:
        results2 = vs.query(
            query_embedding=query_embedding,
            collection_type='academic',
            top_k=3,
            filter_metadata=filter_meta
        )
        print(f"Results: {results2['count']} documents")
        if results2['metadatas']:
            for j, meta in enumerate(results2['metadatas'][:2], 1):
                print(f"  Doc {j}: {meta.get('university', 'N/A')} - {meta.get('institution_type', 'N/A')}")
        else:
            print("  No documents found")
    except Exception as e:
        print(f"  ERROR: {e}")

print("\n" + "="*80)
