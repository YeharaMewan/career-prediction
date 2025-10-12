"""
Web Search Tool using DuckDuckGo Search
Provides real-time web search capabilities for career planning agents
"""
import logging
import time
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from duckduckgo_search import DDGS

logger = logging.getLogger(__name__)


class SearchResult:
    """Represents a single search result."""

    def __init__(self, title: str, url: str, snippet: str, relevance_score: float = 0.0):
        self.title = title
        self.url = url
        self.snippet = snippet
        self.relevance_score = relevance_score
        self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "relevance_score": self.relevance_score,
            "timestamp": self.timestamp.isoformat()
        }


class WebSearchTool:
    """
    Web search utility using DuckDuckGo Search.
    Provides caching and specialized search methods for career planning.
    """

    def __init__(self, cache_duration_minutes: int = 60):
        """
        Initialize the web search tool.

        Args:
            cache_duration_minutes: How long to cache search results (default: 60 minutes)
        """
        self.cache_duration = timedelta(minutes=cache_duration_minutes)
        self.search_cache: Dict[str, tuple[List[SearchResult], datetime]] = {}
        self.ddgs = DDGS()
        self.logger = logging.getLogger(f"{__name__}.WebSearchTool")

    def _get_from_cache(self, query: str) -> Optional[List[SearchResult]]:
        """Get cached search results if available and not expired."""
        if query in self.search_cache:
            results, timestamp = self.search_cache[query]
            if datetime.now() - timestamp < self.cache_duration:
                self.logger.debug(f"Cache hit for query: {query}")
                return results
            else:
                # Cache expired, remove it
                del self.search_cache[query]
        return None

    def _add_to_cache(self, query: str, results: List[SearchResult]):
        """Add search results to cache."""
        self.search_cache[query] = (results, datetime.now())

    def search(
        self,
        query: str,
        max_results: int = 10,
        region: str = "wt-wt",
        use_cache: bool = True
    ) -> List[SearchResult]:
        """
        Perform a web search using DuckDuckGo.

        Args:
            query: Search query string
            max_results: Maximum number of results to return
            region: Region code (default: wt-wt for worldwide, use 'lk-en' for Sri Lanka)
            use_cache: Whether to use cached results

        Returns:
            List of SearchResult objects
        """
        # Check cache first
        if use_cache:
            cached_results = self._get_from_cache(query)
            if cached_results:
                return cached_results[:max_results]

        try:
            self.logger.info(f"Searching DuckDuckGo: {query}")

            # Perform search using duckduckgo_search
            raw_results = self.ddgs.text(
                keywords=query,
                region=region,
                safesearch='moderate',
                max_results=max_results
            )

            # Convert to SearchResult objects
            search_results = []
            for idx, result in enumerate(raw_results):
                # Calculate simple relevance score (higher for earlier results)
                relevance_score = 1.0 - (idx * 0.1)

                search_result = SearchResult(
                    title=result.get('title', ''),
                    url=result.get('href', ''),
                    snippet=result.get('body', ''),
                    relevance_score=max(0.1, relevance_score)
                )
                search_results.append(search_result)

            # Add to cache
            if use_cache:
                self._add_to_cache(query, search_results)

            self.logger.info(f"Found {len(search_results)} results for: {query}")
            return search_results

        except Exception as e:
            self.logger.error(f"Search failed for query '{query}': {str(e)}")
            return []

    def search_universities(
        self,
        career: str,
        country: str = "Sri Lanka",
        max_results: int = 10
    ) -> List[SearchResult]:
        """
        Search for universities offering programs for a specific career.

        Args:
            career: Career name (e.g., "Software Engineer")
            country: Country to search in (default: "Sri Lanka")
            max_results: Maximum number of results

        Returns:
            List of SearchResult objects
        """
        query = f"{career} degree programs universities in {country}"
        region = "lk-en" if country == "Sri Lanka" else "wt-wt"

        return self.search(query, max_results=max_results, region=region)

    def search_scholarships(
        self,
        career: str,
        country: str = "Sri Lanka",
        max_results: int = 8
    ) -> List[SearchResult]:
        """
        Search for scholarship opportunities.

        Args:
            career: Career name
            country: Country to search scholarships for
            max_results: Maximum number of results

        Returns:
            List of SearchResult objects
        """
        query = f"{career} scholarships {country} 2024 2025"
        region = "lk-en" if country == "Sri Lanka" else "wt-wt"

        return self.search(query, max_results=max_results, region=region)

    def search_courses(
        self,
        skill: str,
        platform: Optional[str] = None,
        max_results: int = 10
    ) -> List[SearchResult]:
        """
        Search for online courses for a specific skill.

        Args:
            skill: Skill name (e.g., "Python programming")
            platform: Optional platform name (e.g., "Coursera", "Udemy")
            max_results: Maximum number of results

        Returns:
            List of SearchResult objects
        """
        if platform:
            query = f"{skill} course {platform}"
        else:
            query = f"{skill} online courses Coursera Udemy"

        return self.search(query, max_results=max_results)

    def search_certifications(
        self,
        career: str,
        max_results: int = 8
    ) -> List[SearchResult]:
        """
        Search for professional certifications relevant to a career.

        Args:
            career: Career name
            max_results: Maximum number of results

        Returns:
            List of SearchResult objects
        """
        query = f"{career} professional certifications 2024"

        return self.search(query, max_results=max_results)

    def search_skill_trends(
        self,
        career: str,
        max_results: int = 8
    ) -> List[SearchResult]:
        """
        Search for current skill trends and requirements for a career.

        Args:
            career: Career name
            max_results: Maximum number of results

        Returns:
            List of SearchResult objects
        """
        query = f"{career} required skills 2024 2025 industry trends"

        return self.search(query, max_results=max_results)

    def search_job_requirements(
        self,
        career: str,
        location: str = "worldwide",
        max_results: int = 8
    ) -> List[SearchResult]:
        """
        Search for job requirements and postings to understand current demands.

        Args:
            career: Career name
            location: Location for jobs (default: "worldwide")
            max_results: Maximum number of results

        Returns:
            List of SearchResult objects
        """
        query = f"{career} job requirements skills {location}"

        return self.search(query, max_results=max_results)

    def search_admission_requirements(
        self,
        university: str,
        program: str,
        max_results: int = 5
    ) -> List[SearchResult]:
        """
        Search for admission requirements for a specific university program.

        Args:
            university: University name
            program: Program name
            max_results: Maximum number of results

        Returns:
            List of SearchResult objects
        """
        query = f"{university} {program} admission requirements entry requirements"

        return self.search(query, max_results=max_results)

    def search_tuition_costs(
        self,
        university: str,
        program: Optional[str] = None,
        max_results: int = 5
    ) -> List[SearchResult]:
        """
        Search for tuition costs at a university.

        Args:
            university: University name
            program: Optional program name
            max_results: Maximum number of results

        Returns:
            List of SearchResult objects
        """
        program_part = f"{program} " if program else ""
        query = f"{university} {program_part}tuition fees cost 2024 2025"

        return self.search(query, max_results=max_results)

    def format_results_for_llm(self, results: List[SearchResult], max_snippets: int = 5) -> str:
        """
        Format search results for LLM consumption.

        Args:
            results: List of SearchResult objects
            max_snippets: Maximum number of result snippets to include

        Returns:
            Formatted string with search results
        """
        if not results:
            return "No search results found."

        formatted = "SEARCH RESULTS:\n\n"

        for idx, result in enumerate(results[:max_snippets], 1):
            formatted += f"{idx}. {result.title}\n"
            formatted += f"   URL: {result.url}\n"
            formatted += f"   {result.snippet}\n\n"

        return formatted

    def clear_cache(self):
        """Clear all cached search results."""
        self.search_cache.clear()
        self.logger.info("Search cache cleared")


# Convenience function for quick searches
def quick_search(query: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """
    Quick search function for simple use cases.

    Args:
        query: Search query
        max_results: Maximum results to return

    Returns:
        List of search result dictionaries
    """
    tool = WebSearchTool()
    results = tool.search(query, max_results=max_results)
    return [result.to_dict() for result in results]


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Testing Web Search Tool with DuckDuckGo")
    print("=" * 50)

    tool = WebSearchTool()

    # Test 1: Search for universities
    print("\n1. Searching for Software Engineering universities in Sri Lanka...")
    results = tool.search_universities("Software Engineer", "Sri Lanka", max_results=5)
    for i, result in enumerate(results, 1):
        print(f"   {i}. {result.title}")
        print(f"      {result.url}")

    # Test 2: Search for scholarships
    print("\n2. Searching for scholarships...")
    results = tool.search_scholarships("Engineering", "Sri Lanka", max_results=3)
    for i, result in enumerate(results, 1):
        print(f"   {i}. {result.title}")

    # Test 3: Search for online courses
    print("\n3. Searching for Python courses...")
    results = tool.search_courses("Python programming", platform="Coursera", max_results=3)
    for i, result in enumerate(results, 1):
        print(f"   {i}. {result.title}")

    # Test 4: Format for LLM
    print("\n4. Formatting results for LLM...")
    formatted = tool.format_results_for_llm(results, max_snippets=2)
    print(formatted)

    print("\n" + "=" * 50)
    print("Web Search Tool test completed!")
