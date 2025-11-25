"""
Vector Store Manager

Manages ChromaDB vector store with 2 specialized collections:
- academic_knowledge: Academic pathways, institutions, programs
- skill_knowledge: Skills, certifications, training resources
"""

import logging
from typing import List, Dict, Any, Optional
import uuid

import chromadb
from chromadb.config import Settings
from langchain_core.documents import Document

from .config import (
    CHROMA_PERSIST_DIR,
    CHROMA_PERSIST_DIR_OPENAI,
    CHROMA_PERSIST_DIR_GEMINI,
    ACADEMIC_COLLECTION,
    SKILL_COLLECTION,
    CHROMA_SETTINGS,
    DISTANCE_METRIC,
    TOP_K_RESULTS,
    EMBEDDING_PROVIDER,
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStoreManager:
    """
    Manages ChromaDB vector store with multiple collections and dual provider support.

    Features:
    - Dual persistent storage (OpenAI and Gemini directories)
    - 2 specialized collections per provider (academic, skill)
    - Add, query, delete operations
    - Metadata filtering
    - Similarity search with distance conversion
    - Automatic provider-based directory selection
    """

    def __init__(
        self,
        provider: str = EMBEDDING_PROVIDER,
        persist_directory: Optional[str] = None
    ):
        """
        Initialize ChromaDB client with persistence.

        Args:
            provider: Embedding provider ("openai", "gemini", or "fallback")
            persist_directory: Directory to persist ChromaDB data (overrides provider-based selection)
        """
        self.provider = provider

        # Determine persist directory based on provider
        if persist_directory:
            self.persist_directory = persist_directory
        else:
            self.persist_directory = self._get_persist_directory_for_provider(provider)

        # Initialize persistent ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(**CHROMA_SETTINGS),
        )

        # Collection references
        self.academic_collection = None
        self.skill_collection = None

        logger.info(
            f"VectorStoreManager initialized: provider={self.provider}, "
            f"persist_dir={self.persist_directory}"
        )

    def _get_persist_directory_for_provider(self, provider: str) -> str:
        """
        Get the appropriate persist directory for the given provider.

        Args:
            provider: "openai", "gemini", or "fallback"

        Returns:
            Path to persist directory
        """
        if provider in ["openai", "fallback"]:
            # Fallback uses OpenAI directory by default
            return str(CHROMA_PERSIST_DIR_OPENAI)
        elif provider == "gemini":
            return str(CHROMA_PERSIST_DIR_GEMINI)
        else:
            logger.warning(f"Unknown provider '{provider}', defaulting to OpenAI directory")
            return str(CHROMA_PERSIST_DIR_OPENAI)

    def create_collection(self, collection_name: str) -> chromadb.Collection:
        """
        Create or get a collection with provider metadata.

        Args:
            collection_name: Name of the collection

        Returns:
            ChromaDB collection object
        """
        try:
            collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={
                    "hnsw:space": DISTANCE_METRIC,
                    "embedding_provider": self.provider,
                    "persist_directory": str(self.persist_directory),
                },
            )
            logger.info(
                f"Collection '{collection_name}' ready ({collection.count()} documents) "
                f"[provider: {self.provider}]"
            )
            return collection
        except Exception as e:
            logger.error(f"Error creating collection {collection_name}: {e}")
            raise

    def initialize_collections(self):
        """Initialize both academic and skill collections."""
        self.academic_collection = self.create_collection(ACADEMIC_COLLECTION)
        self.skill_collection = self.create_collection(SKILL_COLLECTION)
        logger.info("Both collections initialized successfully")

    def get_collection(self, collection_type: str) -> chromadb.Collection:
        """
        Get collection by type.

        Args:
            collection_type: "academic" or "skill"

        Returns:
            ChromaDB collection object
        """
        if collection_type == "academic":
            if not self.academic_collection:
                self.academic_collection = self.create_collection(ACADEMIC_COLLECTION)
            return self.academic_collection
        elif collection_type == "skill":
            if not self.skill_collection:
                self.skill_collection = self.create_collection(SKILL_COLLECTION)
            return self.skill_collection
        else:
            raise ValueError(f"Invalid collection type: {collection_type}. Use 'academic' or 'skill'")

    def add_documents(
        self,
        documents: List[Document],
        embeddings: List[List[float]],
        collection_type: str,
    ) -> int:
        """
        Add documents with embeddings to a collection.

        Args:
            documents: List of LangChain Document objects
            embeddings: List of embedding vectors
            collection_type: "academic" or "skill"

        Returns:
            Number of documents added
        """
        if not documents or not embeddings:
            logger.warning("No documents or embeddings to add")
            return 0

        if len(documents) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(documents)} documents but {len(embeddings)} embeddings"
            )

        # Get collection
        collection = self.get_collection(collection_type)

        # Prepare data for ChromaDB
        ids = [str(uuid.uuid4()) for _ in documents]
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        # Add to collection
        try:
            collection.add(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            logger.info(
                f"Added {len(documents)} documents to '{collection_type}' collection"
            )
            return len(documents)
        except Exception as e:
            logger.error(f"Error adding documents to {collection_type}: {e}")
            raise

    def query(
        self,
        query_embedding: List[float],
        collection_type: str,
        top_k: int = TOP_K_RESULTS,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Query a collection with embedding.

        Args:
            query_embedding: Query embedding vector
            collection_type: "academic" or "skill"
            top_k: Number of results to return
            filter_metadata: Optional metadata filters

        Returns:
            Dictionary with results, distances, and metadata
        """
        collection = self.get_collection(collection_type)

        try:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=filter_metadata,
                include=["documents", "metadatas", "distances"],
            )

            # Convert distances to similarity scores (1 - distance for cosine)
            distances = results["distances"][0] if results["distances"] else []
            similarities = [1 - dist for dist in distances]

            # Format results
            formatted_results = {
                "documents": results["documents"][0] if results["documents"] else [],
                "metadatas": results["metadatas"][0] if results["metadatas"] else [],
                "distances": distances,
                "similarities": similarities,
                "ids": results["ids"][0] if results["ids"] else [],
                "count": len(results["documents"][0]) if results["documents"] else 0,
            }

            logger.info(
                f"Query returned {formatted_results['count']} results from '{collection_type}' collection"
            )
            return formatted_results

        except Exception as e:
            logger.error(f"Error querying {collection_type} collection: {e}")
            raise

    def delete_collection(self, collection_type: str):
        """
        Delete a collection.

        Args:
            collection_type: "academic" or "skill"
        """
        collection_name = (
            ACADEMIC_COLLECTION if collection_type == "academic" else SKILL_COLLECTION
        )

        try:
            self.client.delete_collection(name=collection_name)
            logger.info(f"Deleted collection: {collection_name}")

            # Reset collection reference
            if collection_type == "academic":
                self.academic_collection = None
            else:
                self.skill_collection = None

        except Exception as e:
            logger.error(f"Error deleting collection {collection_name}: {e}")
            raise

    def get_collection_stats(self, collection_type: str) -> Dict[str, Any]:
        """
        Get statistics for a collection.

        Args:
            collection_type: "academic" or "skill"

        Returns:
            Dictionary with collection statistics
        """
        collection = self.get_collection(collection_type)

        stats = {
            "name": collection.name,
            "count": collection.count(),
            "collection_type": collection_type,
        }

        # Get sample metadata if collection not empty
        if stats["count"] > 0:
            sample = collection.peek(limit=1)
            if sample["metadatas"]:
                stats["sample_metadata"] = sample["metadatas"][0]

        return stats

    def get_all_stats(self) -> Dict[str, Any]:
        """
        Get statistics for all collections.

        Returns:
            Dictionary with all collection statistics
        """
        stats = {
            "persist_directory": self.persist_directory,
            "collections": {},
        }

        # Academic collection stats
        try:
            stats["collections"]["academic"] = self.get_collection_stats("academic")
        except Exception as e:
            logger.warning(f"Could not get academic collection stats: {e}")
            stats["collections"]["academic"] = {"error": str(e)}

        # Skill collection stats
        try:
            stats["collections"]["skill"] = self.get_collection_stats("skill")
        except Exception as e:
            logger.warning(f"Could not get skill collection stats: {e}")
            stats["collections"]["skill"] = {"error": str(e)}

        # Total count
        total_count = 0
        for coll_stats in stats["collections"].values():
            if "count" in coll_stats:
                total_count += coll_stats["count"]

        stats["total_documents"] = total_count

        return stats


# Example usage
if __name__ == "__main__":
    from .document_processor import DocumentProcessor
    from .embedding_manager import EmbeddingManager
    from .config import ACADEMIC_DATA_DIR, SKILL_DATA_DIR

    # Initialize managers
    doc_processor = DocumentProcessor()
    embedding_manager = EmbeddingManager()
    vector_store = VectorStoreManager()

    # Initialize collections
    vector_store.initialize_collections()

    # Process and add academic documents
    print("\n=== Processing Academic Documents ===")
    academic_docs = doc_processor.process_directory(ACADEMIC_DATA_DIR, "academic")
    if academic_docs:
        docs, embeddings = embedding_manager.embed_documents(academic_docs)
        count = vector_store.add_documents(docs, embeddings, "academic")
        print(f"Added {count} academic documents")

    # Process and add skill documents
    print("\n=== Processing Skill Documents ===")
    skill_docs = doc_processor.process_directory(SKILL_DATA_DIR, "skill")
    if skill_docs:
        docs, embeddings = embedding_manager.embed_documents(skill_docs)
        count = vector_store.add_documents(docs, embeddings, "skill")
        print(f"Added {count} skill documents")

    # Get statistics
    print("\n=== Vector Store Statistics ===")
    stats = vector_store.get_all_stats()
    print(f"Total documents: {stats['total_documents']}")
    for coll_name, coll_stats in stats["collections"].items():
        if "count" in coll_stats:
            print(f"  {coll_name}: {coll_stats['count']} documents")

    # Test query
    print("\n=== Test Query ===")
    query = "What universities offer computer science programs?"
    query_embedding = embedding_manager.embed_query(query)
    results = vector_store.query(query_embedding, "academic", top_k=10)
    print(f"Query: {query}")
    print(f"Found {results['count']} results:")
    for i, (doc, sim) in enumerate(zip(results["documents"], results["similarities"])):
        print(f"\n  Result {i+1} (similarity: {sim:.3f}):")
        print(f"    {doc[:200]}...")
