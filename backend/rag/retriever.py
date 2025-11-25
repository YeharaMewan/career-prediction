"""
Agentic RAG Retriever

Implements agentic retrieval logic with decision-making:
- Decides if retrieval is needed
- Retrieves relevant documents if needed
- Formats context for LLM consumption
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from .embedding_manager import EmbeddingManager
from .vector_store import VectorStoreManager
from .config import (
    SIMILARITY_THRESHOLD,
    TOP_K_RESULTS,
    MAX_CONTEXT_LENGTH,
    NEEDS_RETRIEVAL_KEYWORDS,
    EMBEDDING_PROVIDER,
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RetrievalState:
    """
    State object for agentic RAG retrieval.
    Similar to LangGraph StateGraph pattern.
    """

    query: str
    collection_type: str  # "academic" or "skill"
    needs_retrieval: bool = False
    retrieved_documents: List[str] = field(default_factory=list)
    metadatas: List[Dict[str, Any]] = field(default_factory=list)
    similarities: List[float] = field(default_factory=list)
    context: str = ""
    decision_reason: str = ""


class AgenticRAGRetriever:
    """
    Agentic RAG retriever with decision-making logic and multi-provider support.

    Workflow:
    1. decide_retrieval() - Determines if retrieval is needed
    2. retrieve_documents() - Retrieves relevant chunks if needed
    3. format_context() - Formats retrieved context for LLM

    Features:
    - Automatic provider switching (OpenAI ↔ Gemini)
    - Dual vector database support
    - Fallback-aware embedding generation
    """

    def __init__(
        self,
        collection_type: str,
        provider: str = EMBEDDING_PROVIDER,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
        top_k: int = TOP_K_RESULTS,
        max_context_length: int = MAX_CONTEXT_LENGTH,
    ):
        """
        Initialize agentic RAG retriever with provider support.

        Args:
            collection_type: "academic" or "skill"
            provider: Embedding provider ("openai", "gemini", or "fallback")
            similarity_threshold: Minimum similarity score (0-1)
            top_k: Number of documents to retrieve
            max_context_length: Maximum characters in context
        """
        self.collection_type = collection_type
        self.provider = provider
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k
        self.max_context_length = max_context_length

        # Initialize managers with provider support
        self.embedding_manager = EmbeddingManager(provider=provider)
        self.vector_store = VectorStoreManager(provider=provider)

        # Track active provider (may change during fallback)
        self.active_provider = self.embedding_manager.get_active_provider()

        # Ensure collection exists
        self.vector_store.get_collection(collection_type)

        logger.info(
            f"AgenticRAGRetriever initialized: collection={collection_type}, "
            f"provider={self.active_provider}, threshold={similarity_threshold}, top_k={top_k}"
        )

    def get_active_provider(self) -> str:
        """Get the currently active embedding provider"""
        return self.embedding_manager.get_active_provider()

    def switch_provider(self, new_provider: str):
        """
        Switch to a different embedding provider and vector database.

        This is useful when LLM fallback occurs and we need to switch
        the RAG system to use the corresponding embedding provider.

        Args:
            new_provider: "openai" or "gemini"
        """
        if new_provider == self.active_provider:
            logger.info(f"Already using provider: {new_provider}")
            return

        logger.warning(f"Switching RAG provider: {self.active_provider} → {new_provider}")

        # Reinitialize with new provider
        self.provider = new_provider
        self.embedding_manager = EmbeddingManager(provider=new_provider)
        self.vector_store = VectorStoreManager(provider=new_provider)
        self.active_provider = new_provider

        # Ensure collection exists in new vector store
        self.vector_store.get_collection(self.collection_type)

        logger.info(f"Successfully switched to provider: {new_provider}")

    def decide_retrieval(self, query: str) -> tuple[bool, str]:
        """
        Decide if retrieval is needed based on query content.

        Uses keyword matching to determine if query needs knowledge base.

        Args:
            query: User query

        Returns:
            Tuple of (needs_retrieval, reason)
        """
        query_lower = query.lower()

        # Check for retrieval keywords
        found_keywords = [
            kw for kw in NEEDS_RETRIEVAL_KEYWORDS if kw in query_lower
        ]

        if found_keywords:
            reason = f"Query contains knowledge keywords: {', '.join(found_keywords[:3])}"
            logger.info(f"Retrieval NEEDED: {reason}")
            return True, reason
        else:
            reason = "Query does not require knowledge base retrieval"
            logger.info(f"Retrieval NOT needed: {reason}")
            return False, reason

    def retrieve_documents(
        self,
        query: str,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Retrieve relevant documents from vector store.

        Args:
            query: Query string
            filter_metadata: Optional metadata filters

        Returns:
            Dictionary with retrieved documents and metadata
        """
        logger.info(f"Retrieving documents for query: '{query[:100]}...'")

        # Generate query embedding
        try:
            query_embedding = self.embedding_manager.embed_query(query)
        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            return {
                "documents": [],
                "metadatas": [],
                "similarities": [],
                "count": 0,
            }

        # Query vector store
        try:
            results = self.vector_store.query(
                query_embedding=query_embedding,
                collection_type=self.collection_type,
                top_k=self.top_k,
                filter_metadata=filter_metadata,
            )

            # Filter by similarity threshold
            filtered_results = self._filter_by_threshold(results)

            logger.info(
                f"Retrieved {filtered_results['count']} documents "
                f"(threshold: {self.similarity_threshold})"
            )

            return filtered_results

        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return {
                "documents": [],
                "metadatas": [],
                "similarities": [],
                "count": 0,
            }

    def _filter_by_threshold(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filter results by similarity threshold.

        Args:
            results: Results from vector store query

        Returns:
            Filtered results dictionary
        """
        documents = []
        metadatas = []
        similarities = []

        for doc, meta, sim in zip(
            results["documents"], results["metadatas"], results["similarities"]
        ):
            if sim >= self.similarity_threshold:
                documents.append(doc)
                metadatas.append(meta)
                similarities.append(sim)

        return {
            "documents": documents,
            "metadatas": metadatas,
            "similarities": similarities,
            "count": len(documents),
        }

    def format_context(
        self,
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        similarities: List[float],
        include_citations: bool = True,
    ) -> str:
        """
        Format retrieved documents into context for LLM.

        Args:
            documents: List of document texts
            metadatas: List of metadata dictionaries
            similarities: List of similarity scores
            include_citations: Include source citations

        Returns:
            Formatted context string
        """
        if not documents:
            return ""

        context_parts = []
        total_chars = 0

        for i, (doc, meta, sim) in enumerate(zip(documents, metadatas, similarities)):
            # Create citation
            source = meta.get("source_file", "Unknown")
            page = meta.get("page", "N/A")
            citation = f"[Source: {source}, Page: {page}, Relevance: {sim:.2f}]"

            # Format chunk
            if include_citations:
                chunk_text = f"--- Document {i+1} {citation} ---\n{doc}\n"
            else:
                chunk_text = f"{doc}\n\n"

            # Check if adding this chunk exceeds max length
            if total_chars + len(chunk_text) > self.max_context_length:
                logger.info(
                    f"Reached max context length ({self.max_context_length} chars), "
                    f"using {i} documents"
                )
                break

            context_parts.append(chunk_text)
            total_chars += len(chunk_text)

        context = "\n".join(context_parts)
        logger.info(f"Formatted context: {total_chars} characters from {len(context_parts)} documents")

        return context

    def retrieve(
        self,
        query: str,
        force_retrieval: bool = False,
        filter_metadata: Optional[Dict[str, Any]] = None,
        include_citations: bool = True,
    ) -> RetrievalState:
        """
        Complete agentic retrieval workflow.

        Args:
            query: User query
            force_retrieval: Force retrieval regardless of decision
            filter_metadata: Optional metadata filters
            include_citations: Include source citations in context

        Returns:
            RetrievalState with results
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
            state.retrieved_documents = results["documents"]
            state.metadatas = results["metadatas"]
            state.similarities = results["similarities"]

            # Step 3: Format context
            if results["count"] > 0:
                state.context = self.format_context(
                    results["documents"],
                    results["metadatas"],
                    results["similarities"],
                    include_citations,
                )
            else:
                logger.warning("No documents found above similarity threshold")
                state.context = "No relevant information found in knowledge base."
        else:
            logger.info("Skipping retrieval based on decision")

        return state

    def get_stats(self) -> Dict[str, Any]:
        """
        Get retriever statistics.

        Returns:
            Dictionary with statistics
        """
        collection_stats = self.vector_store.get_collection_stats(self.collection_type)

        stats = {
            "collection_type": self.collection_type,
            "similarity_threshold": self.similarity_threshold,
            "top_k": self.top_k,
            "max_context_length": self.max_context_length,
            "documents_in_collection": collection_stats.get("count", 0),
        }

        return stats


# Example usage
if __name__ == "__main__":
    # Initialize retrievers for both collections
    academic_retriever = AgenticRAGRetriever(collection_type="academic")
    skill_retriever = AgenticRAGRetriever(collection_type="skill")

    # Test queries
    test_queries = [
        ("What universities in Sri Lanka offer computer science programs?", "academic"),
        ("How can I learn Python programming?", "skill"),
        ("What is the weather like today?", "academic"),  # Should not retrieve
    ]

    print("\n=== Testing Agentic RAG Retrieval ===\n")

    for query, coll_type in test_queries:
        retriever = academic_retriever if coll_type == "academic" else skill_retriever

        print(f"\nQuery: {query}")
        print(f"Collection: {coll_type}")

        # Retrieve
        state = retriever.retrieve(query, include_citations=True)

        print(f"Needs Retrieval: {state.needs_retrieval}")
        print(f"Reason: {state.decision_reason}")

        if state.needs_retrieval:
            print(f"Retrieved {len(state.retrieved_documents)} documents")
            if state.context:
                print(f"Context length: {len(state.context)} characters")
                print(f"Context preview:\n{state.context[:300]}...")
            else:
                print("No context generated")

        print("-" * 80)
