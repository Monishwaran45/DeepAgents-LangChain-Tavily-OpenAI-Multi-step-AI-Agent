import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from tavily import TavilyClient
from src.config import AgentConfig

logger = logging.getLogger(__name__)


class SearchResultItem(BaseModel):
    """Normalized search result item from Tavily."""
    title: str
    url: str
    content: str
    score: float = 0.0
    raw_content: Optional[str] = None
    query: Optional[str] = None


class TavilySearchTool:
    """Enhanced Tavily Search Client with multi-query support and normalization."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or AgentConfig.TAVILY_API_KEY
        if not self.api_key or self.api_key == "your_tavily_api_key_here":
            logger.warning("Tavily API Key is not set. Real-time web search will be disabled.")
            self.client = None
        else:
            self.client = TavilyClient(api_key=self.api_key)

    def search(
        self,
        query: str,
        max_results: int = 5,
        search_depth: str = "advanced",
        topic: str = "general",
        include_raw_content: bool = False,
    ) -> List[SearchResultItem]:
        """Execute a single query search via Tavily."""
        if not self.client:
            logger.error("Tavily client is not configured. Returning empty search results.")
            return []

        try:
            response = self.client.search(
                query=query,
                max_results=max_results,
                search_depth=search_depth,
                topic=topic,
                include_raw_content=include_raw_content,
            )

            results: List[SearchResultItem] = []
            for item in response.get("results", []):
                results.append(
                    SearchResultItem(
                        title=item.get("title", "Untitled"),
                        url=item.get("url", ""),
                        content=item.get("content", ""),
                        score=float(item.get("score", 0.0)),
                        raw_content=item.get("raw_content"),
                        query=query,
                    )
                )
            return results

        except Exception as e:
            logger.error(f"Error during Tavily search for query '{query}': {str(e)}")
            return []

    def multi_query_search(
        self,
        queries: List[str],
        max_results_per_query: int = 3,
        search_depth: str = "advanced",
    ) -> List[SearchResultItem]:
        """Execute searches across multiple sub-queries and combine results."""
        all_results: List[SearchResultItem] = []
        for q in queries:
            q_clean = q.strip()
            if not q_clean:
                continue
            sub_results = self.search(
                query=q_clean,
                max_results=max_results_per_query,
                search_depth=search_depth,
            )
            all_results.extend(sub_results)
        return all_results
